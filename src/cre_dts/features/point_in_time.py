"""Point-in-time rule and leakage check (Technical Design Specification, Section 3.1; Appendix A.1).

Every report carries two dates: `reporting_period_end` (the period the values describe) and `available_date`
(when the report became publicly available; for ABS-EE, the EDGAR filing date). A feature at quarter t may use
only reports whose `available_date` is on or before the end of quarter t. The join and the leakage check use
`available_date`, never the reporting period. Staleness is measured from `reporting_period_end`.
"""
from __future__ import annotations

import pandas as pd


class LookAheadError(AssertionError):
    """Raised when a feature row uses information published after its observation date."""


def _check_report_dates(reports: pd.DataFrame) -> None:
    missing = {"reporting_period_end", "available_date"} - set(reports.columns)
    if missing:
        raise ValueError(f"reports must carry both dates; missing {sorted(missing)}")
    early = reports["available_date"] < reports["reporting_period_end"]
    if early.any():
        raise ValueError(f"{int(early.sum())} reports are available before their period ends")


def as_of_join(obs: pd.DataFrame, reports: pd.DataFrame, value_cols: list[str],
               max_stale_q: int = 4) -> pd.DataFrame:
    """Attach the latest report that was available at each observation's quarter end.

    obs:      one row per (loan_id, quarter_end)
    reports:  one row per (loan_id, reporting_period_end, available_date) with values
    Staleness is measured from reporting_period_end (how old the information is);
    values older than `max_stale_q` quarters are set to missing and the age is kept
    in `<col>_stale_q` (Appendix A.1, missing-value rule).
    """
    _check_report_dates(reports)
    left = obs.sort_values("quarter_end").reset_index(drop=True)
    right = reports.sort_values("available_date")[
        ["loan_id", "reporting_period_end", "available_date", *value_cols]]
    merged = pd.merge_asof(
        left, right, left_on="quarter_end", right_on="available_date",
        by="loan_id", direction="backward", allow_exact_matches=True,
    )
    age_q = ((merged["quarter_end"] - merged["reporting_period_end"]).dt.days / 91.3125).round()
    for col in value_cols:
        merged[f"{col}_stale_q"] = age_q
        merged.loc[age_q > max_stale_q, col] = pd.NA
    return merged


def assert_no_lookahead(features: pd.DataFrame) -> None:
    """Leakage test: no row may use a report that became available after its observation date."""
    bad = features["available_date"].notna() & (features["available_date"] > features["quarter_end"])
    if bad.any():
        raise LookAheadError(f"{int(bad.sum())} rows use data published after quarter end")
