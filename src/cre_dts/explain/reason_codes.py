"""TreeSHAP-based reason codes for Track A (Technical Design Specification, Section 8.1).

XGBoost's built-in TreeSHAP attributions (pred_contribs) give each feature's contribution to a loan's log-odds.
The features that raise the score the most are mapped to a fixed list of human-readable reason codes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

REASONS = {
    "refi_gap": ("R01", lambda v, r: f"Refinance gap of {v:.0%}: balloon balance not refinanceable on current terms"),
    "months_to_maturity": ("R02", lambda v, r: f"Maturity in {v:.0f} months"),
    "ltv_current": ("R03", lambda v, r: f"Current LTV of {v:.0%}"),
    "dscr": ("R04", lambda v, r: f"DSCR of {v:.2f}x"),
    "debt_yield": ("R05", lambda v, r: f"Debt yield of {v:.1%}"),
    "noi_chg_4q": ("R06", lambda v, r: f"NOI change over four quarters of {v:+.0%}"),
    "occupancy": ("R07", lambda v, r: f"Occupancy of {v:.0%}"),
    "rate_chg_since_orig": ("R08", lambda v, r: f"Rates up {v * 100:.1f} points since origination"),
}


def shap_contributions(model: xgb.XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    """Per-feature TreeSHAP contributions in log-odds (bias column dropped)."""
    contrib = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)


def top_reasons(model: xgb.XGBClassifier, X: pd.DataFrame, k: int = 3) -> list[list[dict]]:
    """For each row, the top-k features that RAISE the score, as {code, text, shap} records."""
    contrib = shap_contributions(model, X)
    out = []
    for idx, row in contrib.iterrows():
        ranked = row[row > 0].sort_values(ascending=False)
        recs = []
        for feat, val in ranked.items():
            if feat not in REASONS:
                continue
            code, fmt = REASONS[feat]
            recs.append({"code": code, "text": fmt(X.at[idx, feat], X.loc[idx]), "shap": round(float(val), 3)})
            if len(recs) == k:
                break
        out.append(recs)
    return out
