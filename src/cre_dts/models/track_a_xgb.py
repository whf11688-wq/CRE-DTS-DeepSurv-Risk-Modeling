"""Track A: cross-sectional 12-month PD with monotone XGBoost (Technical Design Specification, Section 5.1).

Monotone constraints are imposed on features with an unambiguous credit direction, so that predicted PD never
falls as leverage or the refinance gap rises, or as coverage improves. Class imbalance is handled with
scale_pos_weight rather than resampling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

MONOTONE = {
    "ltv_current": +1,
    "refi_gap": +1,
    "dscr": -1,
    "debt_yield": -1,
    "occupancy": -1,
}

DEFAULT_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
)


def monotone_vector(features: list[str]) -> tuple[int, ...]:
    """Constraint vector aligned with `features` (0 = unconstrained)."""
    return tuple(MONOTONE.get(f, 0) for f in features)


def fit_track_a(X: pd.DataFrame, y, params: dict | None = None, seed: int = 0) -> xgb.XGBClassifier:
    y = np.asarray(y)
    pos = max(1, int(y.sum()))
    p = {**DEFAULT_PARAMS, **(params or {})}
    model = xgb.XGBClassifier(
        **p, random_state=seed,
        monotone_constraints=monotone_vector(list(X.columns)),
        scale_pos_weight=float((len(y) - pos) / pos),
    )
    model.fit(X, y)
    return model


def predict_raw(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    """Uncalibrated score in (0, 1); calibrate with cre_dts.models.calibration before use as a PD."""
    return model.predict_proba(X)[:, 1]
