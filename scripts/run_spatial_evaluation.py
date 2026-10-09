#!/usr/bin/env python3
"""Subject-grouped nested-CV evaluation of the spatial (Kaggle) pipeline.

    python scripts/run_spatial_evaluation.py --data-root KaggleSpiral
    python scripts/run_spatial_evaluation.py --synthetic

Spiral and wave tasks are evaluated under **identical** deterministic
participant folds (all 28 participants completed both tasks), and the
per-task comparison uses a paired sign-flip permutation test over the
shared participants.  Features are frozen per-image HOG + shape
descriptors by default; ``--cnn`` adds optional ResNet-18 embeddings
(downloads weights once; never used by tests or CI).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from pd_drawing_analysis.data import inventory_spatial
from pd_drawing_analysis.evaluation import (
    DEFAULT_MODEL_SPECS,
    XGBOOST_SPEC,
    ModelSpec,
    evaluate_model_family,
    folds_from_subject_sets,
    subject_group_folds,
)
from pd_drawing_analysis.features import (
    SPATIAL_IMAGE_SIZE,
    cnn_feature_matrix,
    extract_spatial_feature_matrix,
)
from pd_drawing_analysis.metrics import paired_sign_flip_test
from pd_drawing_analysis.reporting import write_json_artifact, write_markdown_summary
from pd_drawing_analysis.subject_ids import assert_subject_disjoint


def extract_task_features(inv, task: str, use_cnn: bool, show_progress: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = inv.frame[inv.frame.task == task].reset_index(drop=True)
    if use_cnn:
        cnn = cnn_feature_matrix(sub["path"], show_progress=show_progress)
        hog = extract_spatial_feature_matrix(sub["path"], show_progress=show_progress)
        X = np.hstack([cnn, hog])
    else:
        X = extract_spatial_feature_matrix(sub["path"], show_progress=show_progress)
    return X, sub["label"].to_numpy().astype(int), sub["subject_id"].to_numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Directory containing the spiral/wave task layout")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run on data/synthetic/spatial instead of real data")
    parser.add_argument("--tasks", nargs="+", default=["spiral", "wave"],
                        choices=["spiral", "wave"])
    parser.add_argument("--cnn", action="store_true",
                        help="Prepend frozen ResNet-18 ImageNet embeddings (optional "
                             "dependency; downloads weights on first use)")
    parser.add_argument("--with-xgboost", action="store_true")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/spatial"))
    args = parser.parse_args()

    root = args.data_root or Path("data/synthetic/spatial")
    if args.synthetic:
        root = Path("data/synthetic/spatial")
    if not root.is_dir():
        raise SystemExit(
            f"Data root {root} not found. Generate fixtures with "
            "`python scripts/make_synthetic_fixtures.py` or point --data-root "
            "at the Kaggle drawing directory."
        )

    inv = inventory_spatial(root)
    if inv.duplicates:
        print(f"Note: {len(inv.duplicates)} byte-identical duplicate images on disk "
              "(duplicate dataset directories); each unique image is used once.")

    # Deterministic participant folds shared by all tasks: participants are
    # folded once at the SUBJECT level, so spiral and wave use the same
    # outer folds wherever their participants coincide.
    union = inv.frame.drop_duplicates("subject_id")
    shared_subject_folds = subject_group_folds(
        union["subject_id"].to_numpy(), union["label"].to_numpy(), n_splits=5
    )

    models = list(DEFAULT_MODEL_SPECS)
    if args.with_xgboost:
        models.append(XGBOOST_SPEC)

    suffix = "synthetic" if args.synthetic else "real"
    feature_tag = "cnn_hog_shape" if args.cnn else "hog_shape"
    results: dict = {
        "dataset": {
            "name": "Kaggle Parkinson's Drawings (synthetic fixture)" if args.synthetic
                    else "Kaggle Parkinson's Drawings",
            "root": str(root),
            "image_size": list(SPATIAL_IMAGE_SIZE),
            "features": feature_tag,
            "per_task": inv.summary()["per_task"],
            "shared_subjects_across_tasks": inv.summary()["shared_subjects_across_tasks"],
            "duplicate_images_ignored": len(inv.duplicates),
        },
        "protocol": {
            "outer_splits": 5,
            "inner_splits": 3,
            "grouping": "class-aware subject id (parkinson:Vxx / healthy:Vxx)",
            "shared_outer_folds_across_tasks": True,
            "preprocessing": "variance filter, scaling, PCA fitted inside each fold via sklearn Pipeline",
            "aggregation": "mean out-of-fold probability per subject, thresholded at 0.5",
            "uncertainty": "deterministic subject-level stratified bootstrap, percentile intervals",
        },
        "tasks": {},
    }

    task_oof: dict[str, pd.DataFrame] = {}
    for task in args.tasks:
        print(f"\n=== Task: {task} ===")
        X, y, subjects = extract_task_features(inv, task, args.cnn, show_progress=True)
        print(f"Feature matrix: {X.shape}")
        task_folds = folds_from_subject_sets(subjects, shared_subject_folds)
        task_results = {"n_images": int(len(y)), "n_subjects": int(len(set(subjects.tolist()))), "models": {}}
        for spec in models:
            print(f"  evaluating {spec.name} ...")
            try:
                task_results["models"][spec.name] = evaluate_model_family(
                    X, y, subjects, spec, task_folds, n_resamples=args.n_resamples
                )
            except ImportError as exc:
                task_results["models"][spec.name] = {"model": spec.name, "skipped": str(exc)}
                print(f"    skipped: {exc}")
        results["tasks"][task] = task_results

        metrics_by_model = {
            name: r["participant_metrics"] for name, r in task_results["models"].items()
            if "participant_metrics" in r
        }
        best_baseline = max(
            (n for n in metrics_by_model if n != "dummy_majority"),
            key=lambda n: metrics_by_model[n]["balanced_accuracy"],
            default=None,
        )
        if best_baseline:
            m = metrics_by_model[best_baseline]
            print(f"  {best_baseline}: participant accuracy {m['accuracy']:.3f} "
                  f"(dummy: {metrics_by_model['dummy_majority']['accuracy']:.3f})")
            task_oof[task] = task_results["models"][best_baseline]["participant_table"]
        else:
            task_oof[task] = pd.DataFrame()

    # Paired spiral-vs-wave comparison on shared participants (exploratory).
    if set(task_oof) == {"spiral", "wave"} and all(not df.empty for df in task_oof.values()):
        merged = task_oof["spiral"].merge(
            task_oof["wave"], on="subject_id", suffixes=("_spiral", "_wave")
        )
        if len(merged):
            correct_spiral = (merged["y_pred_spiral"] == merged["y_true_spiral"]).astype(float)
            correct_wave = (merged["y_pred_wave"] == merged["y_true_wave"]).astype(float)
            results["task_comparison"] = {
                "note": "Exploratory paired comparison on participants who completed both tasks.",
                "n_shared_participants": int(len(merged)),
                "paired_sign_flip_accuracy": paired_sign_flip_test(correct_spiral, correct_wave),
            }

    out_json = write_json_artifact(results, args.output_dir / f"spatial_{suffix}_{feature_tag}.json")
    out_md = write_markdown_summary(
        f"Spatial pipeline — subject-grouped nested CV ({suffix}, {feature_tag})",
        {"dataset": results["dataset"], "protocol": results["protocol"],
         "models": results["tasks"].get(args.tasks[0], {}).get("models", {})},
        args.output_dir / f"spatial_{suffix}_{feature_tag}.md",
    )
    print(f"\nArtifacts:\n  {out_json}\n  {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
