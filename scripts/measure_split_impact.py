#!/usr/bin/env python3
"""Measure the impact of the participant-grouping fix, apples-to-apples.

Same model, features and participant-level aggregation in both arms —
only the split differs:

1. Leaky image-level ``StratifiedKFold`` (the protocol the previous code
   used), repeated over 30 seeds.
2. Participant-grouped ``StratifiedGroupKFold``, same 30 seeds.

Also measures how good the excluded z_range feature *looked* on the
temporal dataset (in-sample threshold vs participant-grouped CV).

    python scripts/measure_split_impact.py \
        --spiral-root KaggleSpiral/drawings \
        --temporal-root "Improved Spiral Test .../data/alldata"

Writes ``results/impact_numbers.json``.  Not part of CI (needs the real
datasets); synthetic fixtures are not sufficient for this comparison.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pd_drawing_analysis.data import (
    inventory_spatial,
    inventory_temporal,
    load_image_gray,
    load_temporal_recording,
)
from pd_drawing_analysis.features import (
    SPATIAL_IMAGE_SIZE,
    TEMPORAL_FEATURE_NAMES,
    extract_temporal_features,
    hog_features,
    shape_features,
)
from pd_drawing_analysis.metrics import aggregate_to_participants
from pd_drawing_analysis.reporting import write_json_artifact

SEED = 42
N_SEEDS = 30


def _pipe() -> Pipeline:
    return Pipeline([
        ("variance", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.95, random_state=SEED)),
        ("clf", LogisticRegression(max_iter=5000, C=0.1, class_weight="balanced")),
    ])


def _participant_accuracy(oof: np.ndarray, subjects: np.ndarray, y: np.ndarray) -> float:
    part = aggregate_to_participants(subjects, y, oof)
    return float(accuracy_score(part["y_true"], part["y_pred"]))


def measure_spatial_split_impact(spiral_root: Path) -> dict:
    inv = inventory_spatial(spiral_root)
    sub = inv.frame[inv.frame.task == "spiral"].reset_index(drop=True)
    X = np.vstack([
        np.concatenate([
            hog_features(load_image_gray(p, size=SPATIAL_IMAGE_SIZE)),
            shape_features(load_image_gray(p, size=SPATIAL_IMAGE_SIZE)),
        ])
        for p in sub["path"]
    ])
    y = sub["label"].to_numpy()
    subjects = sub["subject_id"].to_numpy()

    def oof_for(folds):
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            pipe = _pipe()
            pipe.fit(X[tr], y[tr])
            oof[te] = pipe.predict_proba(X[te])[:, 1]
        return oof

    leaky_parts, grouped_parts, leaky_img, exposures = [], [], [], []
    for seed in range(N_SEEDS):
        leaky_folds = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y))
        oof = oof_for(leaky_folds)
        leaky_img.append(float(accuracy_score(y, (oof >= 0.5).astype(int))))
        leaky_parts.append(_participant_accuracy(oof, subjects, y))
        hits = tot = 0
        for tr, te in leaky_folds:
            train_subjects = set(subjects[tr])
            hits += sum(1 for i in te if subjects[i] in train_subjects)
            tot += len(te)
        exposures.append(hits / tot)

        sf = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
                  .split(X, y, groups=subjects))
        idx = [(np.array([i for i, s in enumerate(subjects) if s in set(subjects[tr])]),
                np.array([i for i, s in enumerate(subjects) if s in set(subjects[te])]))
               for tr, te in sf]
        grouped_parts.append(_participant_accuracy(oof_for(idx), subjects, y))

    lp, gp = np.array(leaky_parts), np.array(grouped_parts)
    return {
        "model": "VarianceThreshold+StandardScaler+PCA(95%)+LogReg(C=0.1, balanced), "
                 "identical in both arms; participant-level aggregation in both arms",
        "seeds": N_SEEDS,
        "leaky_image_level_split": {
            "participant_accuracy_mean": round(float(lp.mean()), 3),
            "participant_accuracy_std": round(float(lp.std()), 3),
            "image_level_headline_accuracy": round(float(np.mean(leaky_img)), 3),
            "advertised_n": int(len(y)),
            "true_participant_n": int(len(set(subjects.tolist()))),
            "subject_in_training_exposure": round(float(np.mean(exposures)), 2),
        },
        "participant_grouped_split": {
            "participant_accuracy_mean": round(float(gp.mean()), 3),
            "participant_accuracy_std": round(float(gp.std()), 3),
        },
        "difference_leaky_minus_grouped": {
            "mean": round(float((lp - gp).mean()), 3),
            "range": [round(float((lp - gp).min()), 3), round(float((lp - gp).max()), 3)],
            "leaky_wins_seeds": f"{int((lp > gp).sum())}/{N_SEEDS}",
        },
        "interpretation": "At n=28 the accuracy delta is small and seed-unstable; the "
                          "reliable effects are unit inflation (102 images vs 28 "
                          "participants) and 97-100% subject exposure in the leaky split.",
    }


def measure_z_artifact(temporal_root: Path) -> dict:
    inv = inventory_temporal(temporal_root)
    Z = np.vstack([extract_temporal_features(load_temporal_recording(p), include_z=True)
                   for p in inv.frame["path"]])
    y = inv.labels
    subjects = inv.subjects
    names = TEMPORAL_FEATURE_NAMES + ["z_mean", "z_std", "z_range"]
    zr = Z[:, names.index("z_range")]

    healthy_max = float(zr[y == 0].max())
    apparent = float((((zr > healthy_max).astype(int)) == y).mean())

    oof = np.full(len(y), np.nan)
    for seed in range(10):
        for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)\
                .split(zr.reshape(-1, 1), y, groups=subjects):
            m = LogisticRegression(max_iter=5000).fit(zr[tr].reshape(-1, 1), y[tr])
            oof[te] = m.predict_proba(zr[te].reshape(-1, 1))[:, 1]
    part = aggregate_to_participants(subjects, y, oof)
    return {
        "apparent_score_naive_threshold_in_sample": round(apparent, 3),
        "honest_participant_grouped_cv_single_feature": {
            "accuracy": round(float(accuracy_score(part["y_true"], part["y_pred"])), 3),
            "auc": round(float(roc_auc_score(part["y_true"], part["mean_proba"])), 3),
        },
        "note": "in-sample threshold reproduces the previous README's 95% claim; "
                "participant-grouped CV on the same single feature is the honest score",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spiral-root", type=Path, default=Path("KaggleSpiral/drawings"))
    parser.add_argument("--temporal-root", type=Path, default=Path(
        "Improved Spiral Test Using Digitized Graphics Tablet for Monitoring "
        "Parkinson's Disease/data/alldata"))
    parser.add_argument("--output", type=Path, default=Path("results/impact_numbers.json"))
    args = parser.parse_args()

    print("Measuring split impact (30 seeds, identical model) ...")
    spatial = measure_spatial_split_impact(args.spiral_root)
    print(f"  leaky -> participant acc {spatial['leaky_image_level_split']['participant_accuracy_mean']}"
          f" ± {spatial['leaky_image_level_split']['participant_accuracy_std']}")
    print(f"  grouped -> participant acc {spatial['participant_grouped_split']['participant_accuracy_mean']}"
          f" ± {spatial['participant_grouped_split']['participant_accuracy_std']}")
    print("Measuring z_range apparent score ...")
    z_artifact = measure_z_artifact(args.temporal_root)
    print(f"  apparent {z_artifact['apparent_score_naive_threshold_in_sample']} -> "
          f"honest {z_artifact['honest_participant_grouped_cv_single_feature']}")

    payload = {"spatial_split_impact": spatial, "z_range_artifact": z_artifact}
    out = write_json_artifact(payload, args.output)
    print(f"\nArtifact written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
