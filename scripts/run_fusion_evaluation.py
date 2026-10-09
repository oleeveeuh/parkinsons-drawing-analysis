#!/usr/bin/env python3
"""Fusion evaluation under subject-grouped nested CV.

Two honest modes:

1. Synthetic paired-modalities (default) — the same synthetic participant
   produces both a recording and a drawing, with the class signal split
   across modalities so fusion is actually required.  Verifies the fusion
   architectures end-to-end:

       python scripts/run_fusion_evaluation.py --synthetic

2. Real-data feature-family fusion ablation — HOG-only vs shape-only vs
   fused descriptors of the same Kaggle drawings (per-participant
   aggregation, identical participant folds):

       python scripts/run_fusion_evaluation.py --ablation-features \
           --spatial-root KaggleSpiral/drawings

Cross-dataset fusion of the two REAL datasets is intentionally not
offered: their participants differ (docs/data_audit.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pd_drawing_analysis.data import inventory_spatial, load_image_gray
from pd_drawing_analysis.features import SPATIAL_IMAGE_SIZE, hog_features, shape_features
from pd_drawing_analysis.fusion import PairedModalityData, assemble_paired_data, evaluate_fusion
from pd_drawing_analysis.subject_ids import parse_kaggle_filename
from pd_drawing_analysis.synthetic import generate_paired_fixtures
from pd_drawing_analysis.reporting import write_json_artifact, write_markdown_summary


def _ablate_feature_families(spatial_root: Path, n_resamples: int) -> tuple[PairedModalityData, dict]:
    """Real-data fusion of HOG and shape feature families.

    Per-participant rows: the mean of each participant's frozen per-image
    features (per-subject arithmetic; nothing is fitted outside folds).
    """
    inv = inventory_spatial(spatial_root)
    hog_rows: dict[str, list[np.ndarray]] = {}
    shape_rows: dict[str, list[np.ndarray]] = {}
    paths = inv.frame["path"].tolist()
    subjects_col = inv.frame["subject_id"].tolist()
    for path, subject in zip(paths, subjects_col):
        img = load_image_gray(path, size=SPATIAL_IMAGE_SIZE)
        hog_rows.setdefault(subject, []).append(hog_features(img))
        shape_rows.setdefault(subject, []).append(shape_features(img))
    subjects = np.array(sorted(hog_rows))
    labels = np.array([1 if s.startswith("parkinson") else 0 for s in subjects])
    data = PairedModalityData(
        subjects=subjects,
        labels=labels,
        modalities={
            "hog": np.vstack([np.mean(hog_rows[s], axis=0) for s in subjects]),
            "shape": np.vstack([np.mean(shape_rows[s], axis=0) for s in subjects]),
        },
    )
    meta = {
        "dataset": {
            "name": "Kaggle Parkinson's Drawings (feature-family fusion ablation)",
            "root": str(spatial_root),
            "n_participants": int(len(subjects)),
            "modalities": {"hog": int(data.modalities["hog"].shape[1]),
                           "shape": int(data.modalities["shape"].shape[1])},
            "aggregation": "per-participant mean of frozen per-image features",
        },
        "scope_note": (
            "Feature-family fusion of descriptors of the SAME drawings. "
            "The temporal and spatial datasets in this repo have different "
            "participants and are not fused anywhere."
        ),
    }
    return data, meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", default=True,
                        help="Synthetic paired-modality demonstration (default)")
    parser.add_argument("--ablation-features", action="store_true",
                        help="Real-data HOG vs shape vs fused ablation")
    parser.add_argument("--spatial-root", type=Path, default=Path("KaggleSpiral/drawings"))
    parser.add_argument("--temporal-root", type=Path, default=None,
                        help="Only used with a real paired dataset, if you ever obtain one")
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/fusion"))
    args = parser.parse_args()

    if args.ablation_features:
        if not args.spatial_root.is_dir():
            raise SystemExit(f"Spatial root {args.spatial_root} not found; run the synthetic "
                             "mode instead or download the authorized dataset.")
        data, meta = _ablate_feature_families(args.spatial_root, args.n_resamples)
        tag = "feature_ablation_real"
    else:
        root = Path("data/synthetic/paired")
        generate_paired_fixtures(root)
        data, info = assemble_paired_data(root / "temporal", root / "images" / "spiral" / "alldata")
        meta = {
            "dataset": {
                "name": "Synthetic paired-modality fixture (methods demonstration)",
                "temporal_dir": str(root / "temporal"),
                "image_dir": str(root / "images" / "spiral" / "alldata"),
                "n_participants": int(len(data.subjects)),
                "signal_split": "half of impaired participants separable only in recordings, "
                                "half only in drawings",
                "notes": info,
            },
            "scope_note": (
                "Methods demonstration on clearly synthetic paired data. The real "
                "datasets in this repo cannot be fused (different participants). "
                "Accuracy numbers here say nothing about Parkinson's disease."
            ),
        }
        tag = "paired_synthetic"

    print(f"Paired participants: {data.subjects.size}; modalities: "
          f"{ {k: v.shape for k, v in data.modalities.items()} }")
    results = evaluate_fusion(data, n_resamples=args.n_resamples)
    results.update(meta)

    out_json = write_json_artifact(results, args.output_dir / f"fusion_{tag}.json")
    out_md = write_markdown_summary(f"Fusion evaluation — {tag}", results,
                                    args.output_dir / f"fusion_{tag}.md")
    for name, r in results["models"].items():
        m = r["participant_metrics"]
        print(f"  {name:22s} acc={m['accuracy']:.3f} bal={m['balanced_accuracy']:.3f} "
              f"f1={m['f1']:.3f}")
    print(f"\nArtifacts:\n  {out_json}\n  {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
