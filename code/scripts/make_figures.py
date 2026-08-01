#!/usr/bin/env python3
"""Regenerate manuscript and website figures from the locked result rows."""
from __future__ import annotations

import json
from pathlib import Path

from rc_krylov_qec.realstudy import (
    _bar_storage,
    _bar_timing,
    _line_bound,
    _line_direct_vs_forward,
)

ROOT = Path(__file__).resolve().parents[2]
LOCKED = ROOT / "code" / "results" / "locked_results.json"
FIGURES = ROOT / "code" / "manuscript_assets" / "figures"


def main() -> None:
    data = json.loads(LOCKED.read_text(encoding="utf-8"))
    FIGURES.mkdir(parents=True, exist_ok=True)
    _line_direct_vs_forward(
        data["cross_matrix"], FIGURES / "direct_vs_forward"
    )
    _line_bound(data["bound_sweep"], FIGURES / "direct_bound")
    _bar_storage(data["storage"], FIGURES / "direct_storage")
    _bar_timing(data["timing"], FIGURES / "direct_timing")
    print(f"wrote four line/bar figures to {FIGURES}")


if __name__ == "__main__":
    main()
