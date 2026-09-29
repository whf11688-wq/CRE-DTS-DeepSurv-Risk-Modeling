"""Unit tests for the CRE-DTS core components (Technical Design Specification, Appendix B).

All inputs are synthetic or hand-computed. None of these tests uses real loan data.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from cre_dts.config import load_underwriting
from cre_dts.features.point_in_time import LookAheadError, as_of_join, assert_no_lookahead
from cre_dts.features.refi_gap import (add_refi_gap, balloon_balance, debt_constant, refi_gap,
                                       supportable_loan)
from cre_dts.models.track_a_xgb import fit_track_a, predict_raw
from cre_dts.models.track_c_surv import concordance_index, cox_ph_loss, fit_deepsurv, log_risk
from cre_dts.panel.labels import make_labels
from cre_dts.validation.time_cv import expanding_purged_folds

D = pd.Timestamp


# ---------------------------------------------------------------- refinance gap (Section 4.3, Appendix A.6)

def test_debt_constant_known_value():
    # 30-year monthly amortization at 7%: monthly payment per $1 = 0.00665302; x 12 = 0.0798363
    assert float(debt_constant(0.07, 30)) == pytest.approx(0.0798363, rel=1e-5)
    assert float(debt_constant(0.0, 30)) == pytest.approx(1 / 30)


def test_interest_only_loan_balloon_equals_balance():
    assert float(balloon_balance(20_000_000, 0.04, 9, amort_years=None)) == 20_000_000
    # an amortizing loan pays down, so its balloon is below the current balance
    assert float(balloon_balance(20_000_000, 0.04, 60, amort_years=30)) < 20_000_000


def test_refi_gap_worked_example():
    # NOI $1.65M, refi rate 7%, cap rate 8%, limits DSCR 1.35 / LTV 60% / DY 10%
    L = supportable_loan(1_650_000, 0.07, 0.08, 1.35, 0.60, 0.10)
    # DSCR: 1.65M / (1.35 * 0.07983) = 15.31M ; LTV: 0.6 * 20.625M = 12.375M ; DY: 16.5M
    assert L == pytest.approx(12_375_000, rel=1e-6)
    assert refi_gap(20_000_000, L) == pytest.approx(0.38125, rel=1e-6)


def test_refi_gap_zero_when_refinanceable():
    L = supportable_loan(1_650_000, 0.07, 0.08, 1.35, 0.60, 0.10)
    assert float(refi_gap(10_000_000, L)) == 0.0
    assert float(refi_gap(0.0, L)) == 0.0          # no balance, no gap


def test_add_refi_gap_uses_config():
    cfg = load_underwriting()
    loan = pd.DataFrame([dict(property_type="office", noi=1_650_000, cap_rate=0.08, balance=20_000_000,
                              coupon=0.04, months_to_maturity=9, amort_years=0)])
    out = add_refi_gap(loan, ust10=0.0425, cfg=cfg)
    assert out.loc[0, "refi_rate"] == pytest.approx(0.0425 + cfg["property_types"]["office"]["spread"])
    assert out.loc[0, "refi_gap"] == pytest.approx(0.38125, rel=1e-6)
    # a stricter institution-set LTV limit raises the gap; the formula itself is unchanged (Appendix A.9)
    strict = {**cfg, "property_types": {**cfg["property_types"],
                                        "office": {**cfg["property_types"]["office"], "ltv_max": 0.50}}}
    assert add_refi_gap(loan, ust10=0.0425, cfg=strict).loc[0, "refi_gap"] > out.loc[0, "refi_gap"]


# ---------------------------------------------------------------- point-in-time rule (Section 3.1, Appendix A.1)

def _reports():
    return pd.DataFrame({
        "loan_id": ["L1", "L1", "L1"],
        "reporting_period_end": [D("2024-03-31"), D("2024-06-30"), D("2024-09-30")],
        "available_date": [D("2024-04-20"), D("2024-07-22"), D("2024-10-21")],
        "noi": [100.0, 90.0, 80.0],
    })


def test_as_of_join_never_uses_future_reports():
    obs = pd.DataFrame({"loan_id": ["L1"] * 3,
                        "quarter_end": [D("2024-06-30"), D("2024-09-30"), D("2024-12-31")]})
    f = as_of_join(obs, _reports(), ["noi"]).sort_values("quarter_end").reset_index(drop=True)
    assert (f["available_date"] <= f["quarter_end"]).all()
    assert list(f["noi"]) == [100.0, 90.0, 80.0]
    assert_no_lookahead(f)


def test_report_for_quarter_not_used_before_it_is_published():
    # The Q3 report describes the quarter ending Sep 30 but is filed Oct 21, so the Sep 30 observation
    # must still carry the Q2 value.
    obs = pd.DataFrame({"loan_id": ["L1"], "quarter_end": [D("2024-09-30")]})
    f = as_of_join(obs, _reports(), ["noi"])
    assert f.loc[0, "reporting_period_end"] == D("2024-06-30")
    assert f.loc[0, "noi"] == 90.0


def test_leakage_detector_fires():
    bad = pd.DataFrame({"quarter_end": [D("2024-09-30")], "available_date": [D("2024-10-21")]})
    with pytest.raises(LookAheadError):
        assert_no_lookahead(bad)


def test_reports_must_carry_both_dates():
    obs = pd.DataFrame({"loan_id": ["L1"], "quarter_end": [D("2024-09-30")]})
    with pytest.raises(ValueError, match="both dates"):
        as_of_join(obs, _reports().drop(columns=["available_date"]), ["noi"])
    early = _reports()
    early.loc[0, "available_date"] = D("2024-03-01")      # available before its own period ends
    with pytest.raises(ValueError, match="before their period ends"):
        as_of_join(obs, early, ["noi"])


def test_stale_values_set_missing():
    obs = pd.DataFrame({"loan_id": ["L1", "L1"], "quarter_end": [D("2024-12-31"), D("2026-03-31")]})
    f = as_of_join(obs, _reports(), ["noi"], max_stale_q=4).sort_values("quarter_end").reset_index(drop=True)
    assert f.loc[0, "noi"] == 80.0 and f.loc[0, "noi_stale_q"] == 1
    assert pd.isna(f.loc[1, "noi"]) and f.loc[1, "noi_stale_q"] == 6


# ---------------------------------------------------------------- labels (Section 4.1, Appendix A.7)

def test_labels_term_maturity_and_censoring():
    rows = []
    for q in range(0, 6):                                   # A: term default at q=5, matures q=12
        rows.append(dict(loan_id="A", q=q, in_default=(q == 5), maturity_q=12))
    for q in range(0, 9):                                   # B: maturity default at q=8 (= maturity)
        rows.append(dict(loan_id="B", q=q, in_default=(q == 8), maturity_q=8))
    for q in range(0, 7):                                   # C: pays off after q=6 -> censored
        rows.append(dict(loan_id="C", q=q, in_default=False, maturity_q=20))
    lab = make_labels(pd.DataFrame(rows), horizon_q=4).set_index(["loan_id", "q"])
    assert (5 not in lab.loc["A"].index) and (8 not in lab.loc["B"].index)   # no rows at or after default
    assert lab.loc[("A", 0), "y_12m"] == 0 and lab.loc[("A", 1), "y_12m"] == 1
    assert lab.loc[("A", 1), "tau"] == 4 and lab.loc[("A", 1), "event"] == 1
    assert lab.loc[("A", 1), "default_type"] == "term"
    assert lab.loc[("B", 4), "default_type"] == "maturity" and lab.loc[("B", 4), "y_12m"] == 1
    assert (lab.loc["C", "event"] == 0).all() and (lab.loc["C", "y_12m"] == 0).all()
    assert lab.loc[("C", 2), "tau"] == 4 and lab.loc["C", "default_type"].isna().all()


# ---------------------------------------------------------------- purged time-ordered folds (Section 7)

def test_folds_are_purged_and_ordered():
    folds = expanding_purged_folds(first_q=0, last_q=39, min_train_q=8, purge_q=4, valid_q=4)
    assert len(folds) >= 3
    for a, b in zip(folds, folds[1:]):
        assert b.train_end > a.train_end and b.valid_start > a.valid_start
    for f in folds:
        assert f.valid_start - f.train_end - 1 == 4          # purge gap equals the label horizon
        assert f.valid_end - f.valid_start + 1 == 4
        assert f.valid_end <= 39


# ---------------------------------------------------------------- Track A monotone constraint (Section 5.1)

def test_track_a_pd_non_decreasing_in_ltv():
    rng = np.random.default_rng(0)
    n = 3000
    X = pd.DataFrame({"ltv_current": rng.uniform(0.3, 1.1, n), "dscr": rng.uniform(0.8, 2.5, n),
                      "refi_gap": rng.uniform(0, 0.5, n), "months_to_maturity": rng.integers(1, 60, n)})
    logit = -4 + 3 * X["ltv_current"] - 1.2 * X["dscr"] + 3 * X["refi_gap"] + rng.normal(0, 1, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    model = fit_track_a(X, y, params={"n_estimators": 150})
    grid = pd.DataFrame({"ltv_current": np.linspace(0.3, 1.1, 60), "dscr": 1.3, "refi_gap": 0.2,
                         "months_to_maturity": 24})
    pd_ = predict_raw(model, grid)
    assert np.all(np.diff(pd_) >= -1e-9)


# ---------------------------------------------------------------- Track C DeepSurv (Section 5.3)

def _brute_force_cox(lr, t, e):
    total = 0.0
    for i in range(len(t)):
        if e[i]:
            total += lr[i] - np.log(np.exp(lr[t >= t[i]]).sum())
    return -total / max(1, e.sum())


def test_cox_loss_matches_brute_force_with_ties():
    rng = np.random.default_rng(1)
    t = rng.integers(1, 6, 40).astype(float)                # many ties
    e = (rng.random(40) < 0.6).astype(float)
    lr = rng.normal(0, 1, 40)
    got = cox_ph_loss(torch.tensor(lr), torch.tensor(t), torch.tensor(e)).item()
    assert got == pytest.approx(_brute_force_cox(lr, t, e), rel=1e-6)


def test_deepsurv_learns_synthetic_hazard():
    rng = np.random.default_rng(2)
    n = 1500
    X = rng.normal(0, 1, (n, 3))
    true_lr = 1.0 * X[:, 0] - 0.7 * X[:, 1]                 # third feature is noise
    t_event = rng.exponential(1.0 / (0.05 * np.exp(true_lr)))
    t_cens = rng.uniform(0, 40, n)
    time = np.ceil(np.minimum(t_event, t_cens))
    event = (t_event <= t_cens).astype(float)
    model = fit_deepsurv(X, time, event, epochs=150, lr=3e-3, seed=0)
    c = concordance_index(time, event, log_risk(model, X))
    assert c > 0.70
