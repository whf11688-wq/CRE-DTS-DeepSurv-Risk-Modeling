"""Configuration loading (Technical Design Specification, Section 9.1; Appendix A.9)."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UNDERWRITING = REPO_ROOT / "configs" / "underwriting.yaml"


def load_underwriting(path: str | Path | None = None) -> dict:
    """Return the underwriting configuration: amortization term and per-property-type limits."""
    with open(path or DEFAULT_UNDERWRITING, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    required = {"spread", "dscr_min", "ltv_max", "dy_min"}
    for ptype, limits in cfg["property_types"].items():
        missing = required - set(limits)
        if missing:
            raise ValueError(f"{ptype}: missing {sorted(missing)} in underwriting config")
    return cfg
