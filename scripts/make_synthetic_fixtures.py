#!/usr/bin/env python3
"""Generate the offline synthetic fixtures used by tests and the quick start.

    python scripts/make_synthetic_fixtures.py [--out data/synthetic]

Writes ``<out>/spatial`` (miniature Kaggle layout) and ``<out>/temporal``
(UCI-format recordings).  The images and recordings are engineered
renders — clearly synthetic, meaningless clinically, safe to commit-free
regeneration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pd_drawing_analysis.synthetic import (
    generate_spatial_fixtures,
    generate_temporal_fixtures,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    args = parser.parse_args()

    spatial_root = generate_spatial_fixtures(args.out / "spatial")
    temporal_root = generate_temporal_fixtures(args.out / "temporal")
    n_spatial = len(list(spatial_root.rglob("*.png")))
    n_temporal = len(list(temporal_root.glob("*.txt")))
    print(f"Spatial fixtures:  {n_spatial} images under {spatial_root}")
    print(f"Temporal fixtures: {n_temporal} recordings under {temporal_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
