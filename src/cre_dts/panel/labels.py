"""Default and censoring labels (Technical Design Specification, Section 4.1; Appendix A.7).

A loan is in default at the first quarter in which it is 60 or more days delinquent, in foreclosure or REO, or a
non-performing matured balloon. Payoff, defeasance, and loans still active at the end of the sample are censored.

Input: a loan-quarter panel with columns
    loan_id, q (integer quarter index), in_default (bool, the default condition above), maturity_q (int)
Output: one row per loan-quarter observed before default, with
    default_q     first default quarter (NaN if none)
    y_12m         1 if default_q falls within the next `horizon_q` quarters (q, q + horizon_q]
    tau           quarters from the observation to default or censoring
    event         1 if default is observed, 0 if censored
    default_type  "term" (before maturity) or "maturity" (at or after maturity); None if no default
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_labels(panel: pd.DataFrame, horizon_q: int = 4) -> pd.DataFrame:
    need = {"loan_id", "q", "in_default", "maturity_q"}
    missing = need - set(panel.columns)
    if missing:
        raise ValueError(f"panel is missing {sorted(missing)}")
    p = panel.sort_values(["loan_id", "q"]).reset_index(drop=True)
    first_def = p[p["in_default"].astype(bool)].groupby("loan_id")["q"].min().rename("default_q")
    last_q = p.groupby("loan_id")["q"].max().rename("last_q")
    maturity = p.groupby("loan_id")["maturity_q"].first().rename("mat_q")
    p = p.join(first_def, on="loan_id").join(last_q, on="loan_id").join(maturity, on="loan_id")

    obs = p[p["default_q"].isna() | (p["q"] < p["default_q"])].copy()
    has_def = obs["default_q"].notna()
    obs["y_12m"] = (has_def & (obs["default_q"] > obs["q"])
                    & (obs["default_q"] <= obs["q"] + horizon_q)).astype(int)
    end_q = np.where(has_def, obs["default_q"], obs["last_q"])
    obs["tau"] = (end_q - obs["q"]).astype(int)
    obs["event"] = has_def.astype(int)
    obs["default_type"] = np.where(~has_def, None,
                                   np.where(obs["default_q"] >= obs["mat_q"], "maturity", "term"))
    return obs.drop(columns=["last_q", "mat_q"]).reset_index(drop=True)
