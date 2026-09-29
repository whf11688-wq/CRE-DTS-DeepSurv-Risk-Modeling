"""Refinance-gap feature (Technical Design Specification, Section 4.3; Appendix A.6).

For a loan observed at quarter t, the supportable loan L*_t is the largest new loan a lender would make today
against the property, given trailing NOI, a market cap rate, the current refinance rate and underwriting limits.
The refinance gap is the share of the projected balloon balance B_M that could not be refinanced:

    L*_t    = min{ NOI_t / (DSCR_min * k(r_t)),  LTV_max * NOI_t / c_t,  NOI_t / DY_min }
    RefiGap = max(0, B_M - L*_t) / B_M
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cre_dts.config import load_underwriting


def debt_constant(rate, amort_years: int = 30):
    """k(r): annual debt service per dollar of loan for a fully amortizing loan with monthly payments."""
    r = np.asarray(rate, dtype=float)
    m = r / 12.0
    n = 12 * amort_years
    with np.errstate(divide="ignore", invalid="ignore"):
        k = 12.0 * m / (1.0 - (1.0 + m) ** (-n))
    return np.where(m == 0, 1.0 / amort_years, k)


def balloon_balance(balance, coupon, months_to_maturity, amort_years=None):
    """B_M: projected balance at maturity.

    `amort_years=None` (or 0) means interest-only, so the balloon equals the current balance. Otherwise the
    balance amortizes monthly at `coupon` over the stated amortization term until maturity.
    """
    b = np.asarray(balance, dtype=float)
    if not amort_years:
        return b.copy()
    m = np.asarray(coupon, dtype=float) / 12.0
    n = 12 * amort_years
    t = np.asarray(months_to_maturity, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        grow = (1.0 + m) ** t
        pay = b * m / (1.0 - (1.0 + m) ** (-n))
        bal = b * grow - pay * (grow - 1.0) / m
    bal = np.where(m == 0, b * (1.0 - t / n), bal)
    return np.clip(bal, 0.0, None)


def supportable_loan(noi, refi_rate, cap_rate, dscr_min, ltv_max, dy_min, amort_years: int = 30):
    """L*_t: the binding minimum of the DSCR, LTV and debt-yield constraints."""
    noi = np.asarray(noi, dtype=float)
    k = debt_constant(refi_rate, amort_years)
    by_dscr = noi / (np.asarray(dscr_min) * k)
    by_ltv = np.asarray(ltv_max) * noi / np.asarray(cap_rate)
    by_dy = noi / np.asarray(dy_min)
    return np.clip(np.minimum(np.minimum(by_dscr, by_ltv), by_dy), 0, None)


def refi_gap(balloon, supportable):
    """Share of the balloon balance that cannot be refinanced today (0 to 1)."""
    b = np.asarray(balloon, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap = np.maximum(0.0, b - np.asarray(supportable, dtype=float)) / b
    return np.where(b > 0, np.clip(gap, 0.0, 1.0), 0.0)


def add_refi_gap(df: pd.DataFrame, ust10, cfg: dict | None = None) -> pd.DataFrame:
    """Add refi_rate, balloon_bal, supportable_loan and refi_gap columns to a loan-quarter frame.

    Required columns: property_type, noi, cap_rate, balance, coupon, months_to_maturity, amort_years
    (0 or missing = interest-only). `ust10` is a scalar or a column-aligned array of the 10-year Treasury yield.
    Limits and spreads are read from the underwriting configuration by property type (Appendix A.6, A.9).
    """
    cfg = cfg or load_underwriting()
    limits = pd.DataFrame(cfg["property_types"]).T
    unknown = set(df["property_type"]) - set(limits.index)
    if unknown:
        raise KeyError(f"no underwriting limits configured for {sorted(unknown)}")
    lim = limits.loc[df["property_type"]].reset_index(drop=True)
    out = df.reset_index(drop=True).copy()
    out["refi_rate"] = np.asarray(ust10, dtype=float) + lim["spread"].to_numpy(dtype=float)
    amort = out["amort_years"].fillna(0).to_numpy() if "amort_years" in out else np.zeros(len(out))
    out["balloon_bal"] = [
        float(balloon_balance(b, c, t, int(a) if a else None))
        for b, c, t, a in zip(out["balance"], out["coupon"], out["months_to_maturity"], amort)
    ]
    out["supportable_loan"] = supportable_loan(
        out["noi"], out["refi_rate"], out["cap_rate"],
        lim["dscr_min"].to_numpy(dtype=float), lim["ltv_max"].to_numpy(dtype=float),
        lim["dy_min"].to_numpy(dtype=float), amort_years=int(cfg["amort_years"]),
    )
    out["refi_gap"] = refi_gap(out["balloon_bal"], out["supportable_loan"])
    return out
