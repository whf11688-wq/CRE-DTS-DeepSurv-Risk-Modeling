"""Expanding, purged, time-ordered folds (Technical Design Specification, Section 7; Figure 4).

Each fold trains on all quarters up to `train_end`, skips a purge gap equal to the 12-month label horizon so that
training labels cannot overlap the validation period, then validates on the next `valid_q` quarters.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    train_end: int      # last quarter in the training window (inclusive)
    valid_start: int    # first validation quarter
    valid_end: int      # last validation quarter (inclusive)


def expanding_purged_folds(first_q: int, last_q: int, min_train_q: int = 8,
                           purge_q: int = 4, valid_q: int = 4) -> list[Fold]:
    folds = []
    train_end = first_q + min_train_q - 1
    while True:
        v0 = train_end + purge_q + 1
        v1 = v0 + valid_q - 1
        if v1 > last_q:
            break
        folds.append(Fold(train_end, v0, v1))
        train_end += valid_q
    return folds


def split(df: pd.DataFrame, fold: Fold, q_col: str = "q") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, valid) rows for a fold."""
    train = df[df[q_col] <= fold.train_end]
    valid = df[(df[q_col] >= fold.valid_start) & (df[q_col] <= fold.valid_end)]
    return train, valid
