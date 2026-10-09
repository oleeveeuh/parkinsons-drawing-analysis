#!/usr/bin/env python3
"""Build the subject-level data inventory and leakage audit.

Works on either dataset (or both, when both are present locally):

    python scripts/audit_data.py --spiral-root KaggleSpiral --temporal-root "Improved Spiral Test ..."

Writes ``results/data_audit.json`` and prints a summary including the
subject overlap of the vendors' supplied train/test directories.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from pd_drawing_analysis.data import (
    inventory_spatial,
    inventory_temporal,
    supplied_split_leakage_table,
)
from pd_drawing_analysis.reporting import environment_info, write_json_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spiral-root", type=Path, default=None,
                        help="Directory containing the Kaggle spiral/wave layout")
    parser.add_argument("--temporal-root", type=Path, default=None,
                        help="Directory containing PWP (N).txt / Healthy (N).txt files")
    parser.add_argument("--output", type=Path, default=Path("results/data_audit.json"))
    args = parser.parse_args()

    if args.spiral_root is None and args.temporal_root is None:
        parser.error("Provide --spiral-root and/or --temporal-root")

    payload: dict = {"environment": None, "spatial": None, "temporal": None}

    if args.spiral_root:
        inv = inventory_spatial(args.spiral_root)
        summary = inv.summary()
        leakage = supplied_split_leakage_table(inv)
        print("\n=== Spatial dataset (Kaggle Parkinson's Drawings) ===")
        print(f"Images: {len(inv.frame)}  (duplicates on disk: {len(inv.duplicates)})")
        for task, stats in summary["per_task"].items():
            print(
                f"  {task}: {stats['n_images']} images, "
                f"{stats['n_subjects']} subjects "
                f"({stats['n_subjects_by_label']}), "
                f"drawings/subject {stats['drawings_per_subject']}"
            )
        print(f"  subjects shared across tasks: {summary['shared_subjects_across_tasks']}")
        print("\n  Vendor train/test subject overlap (why the supplied split is unusable):")
        print(leakage.to_string(index=False))
        payload["spatial"] = {
            "root": str(args.spiral_root),
            "summary": summary,
            "vendor_split_leakage": leakage.to_dict(orient="records"),
            "duplicate_image_paths": inv.duplicates,
            "images": inv.frame.to_dict(orient="records"),
        }

    if args.temporal_root:
        inv = inventory_temporal(args.temporal_root)
        summary = inv.summary()
        print("\n=== Temporal dataset (UCI spiral drawings) ===")
        print(f"Recordings: {summary['n_recordings']}, subjects: {summary['n_subjects']}")
        print(f"  subjects by label: {summary['n_subjects_by_label']}")
        payload["temporal"] = {
            "root": str(args.temporal_root),
            "summary": summary,
            "recordings": inv.frame.to_dict(orient="records"),
        }

    out = write_json_artifact(payload, args.output)
    print(f"\nAudit artifact written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
