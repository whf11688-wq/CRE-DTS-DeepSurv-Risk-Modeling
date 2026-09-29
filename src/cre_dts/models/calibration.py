"""Isotonic calibration on a held-out set (Technical Design Specification, Section 6)."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


class IsotonicCalibrator:
    """Maps a raw model score to a calibrated probability; fitted on data not used to train the model."""

    def __init__(self):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, raw_score, y) -> "IsotonicCalibrator":
        self._iso.fit(np.asarray(raw_score, dtype=float), np.asarray(y, dtype=float))
        return self

    def transform(self, raw_score) -> np.ndarray:
        return self._iso.predict(np.asarray(raw_score, dtype=float))
