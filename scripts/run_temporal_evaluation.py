#!/usr/bin/env python3
"""Subject-grouped nested-CV evaluation of the temporal (UCI) pipeline.

    python scripts/run_temporal_evaluation.py \
        --data-root "Improved Spiral Test ..." [--include-z] [--synthetic]

Every recording is treated as one participant; folds are split by
participant; all preprocessing is fitted inside each fold; metrics are
participant-level out-of-fold with subject-bootstrap intervals.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pd_drawing_analysis.data import inventory_temporal
from pd_drawing_analysis.evaluation import (
    DEFAULT_MODEL_SPECS,
    XGBOOST_SPEC,
    ModelSpec,
    nested_subject_cv,
)
from pd_drawing_analysis.features import extract_temporal_feature_matrix
from pd_drawing_analysis.metrics import paired_sign_flip_test
from pd_drawing_analysis.reporting import write_json_artifact, write_markdown_summary


def run(root: Path, include_z: bool, models: list[ModelSpec], output_dir: Path,
        n_resamples: int, show_progress: bool = True) -> dict:
    inv = inventory_temporal(root)
    X, names = extract_temporal_feature_matrix(
        inv.frame["path"].tolist(), include_z=include_z, show_progress=show_progress
    )
    y = inv.labels
    subjects = inv.subjects
    print(f"Loaded {len(y)} recordings from {root}; feature matrix {X.shape}")

    results = nested_subject_cv(
        X, y, subjects, model_specs=models, n_resamples=n_resamples,
    )
    import pandas as pd

    results["dataset"] = {
        "name": "UCI Parkinson Disease Spiral Drawings (synthetic fixture)" if "synthetic" in str(root) else
                "UCI Parkinson Disease Spiral Drawings",
        "root": str(root),
        "n_recordings": int(len(y)),
        "n_subjects": int(len(set(subjects.tolist()))),
        "subjects_by_label": {
            str(label): int(count)
            for label, count in pd.Series(subjects).groupby(y).nunique().items()
        },
        "include_z_features": include_z,
        "n_features": int(X.shape[1]),
        "feature_names": names,
    }
    results["protocol"] = {
        "outer_splits": results["outer_splits"],
        "inner_splits": results["inner_splits"],
        "grouping": "subject (one recording per participant)",
        "preprocessing": "variance filter, scaling, PCA fitted inside each fold via sklearn Pipeline",
        "aggregation": "per-recording (one drawing per participant)",
        "uncertainty": "deterministic subject-level stratified bootstrap, percentile intervals",
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Directory with PWP (N).txt / Healthy (N).txt files")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on data/synthetic/temporal instead of real data")
    parser.add_argument("--include-z", action="store_true",
                        help="Include the z-channel features (excluded by default; "
                             "see docs/data_audit.md for why)")
    parser.add_argument("--with-xgboost", action="store_true",
                        help="Also evaluate XGBoost after the simple baselines")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/temporal"))
    args = parser.parse_args()

    root = args.data_root or Path("data/synthetic/temporal")
    if args.synthetic:
        root = Path("data/synthetic/temporal")
    if not root.is_dir():
        raise SystemExit(
            f"Data root {root} not found. Generate fixtures with "
            "`python scripts/make_synthetic_fixtures.py` or point --data-root "
            "at the UCI recordings directory."
        )

    models = list(DEFAULT_MODEL_SPECS)
    if args.with_xgboost:
        models.append(XGBOOST_SPEC)

    results = run(root, args.include_z, models, args.output_dir, args.n_resamples)

    suffix = "synthetic" if args.synthetic else "real"
    tag = "_with_z" if args.include_z else ""
    out_json = write_json_artifact(results, args.output_dir / f"temporal_{suffix}{tag}.json")
    out_md = write_markdown_summary(
        f"Temporal pipeline — subject-grouped nested CV ({suffix}{tag})",
        results,
        args.output_dir / f"temporal_{suffix}{tag}.md",
    )
    print(f"\nArtifacts:\n  {out_json}\n  {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
