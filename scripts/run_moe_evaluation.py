#!/usr/bin/env python3
"""Mixture-of-experts evaluation under subject-grouped validation.

Three modes:

1. ``--synthetic`` (default) — MoE vs its own single experts under
   subject-grouped nested CV on the two-regime participant fixture.
   Expected honest outcome at this sample size: no advantage; measured,
   not assumed.
2. ``--validate-large`` — implementation validation: the same
   architecture at n=600 on a clean regime structure, where the gate is
   able to specialise experts and the mixture must beat its weak single
   expert.  Validates *mechanics*, says nothing about the real datasets.
3. ``--task-experts --spatial-root …`` — real-data run: one expert per
   drawing task (spiral vs wave features of the SAME 28 participants),
   gate over the concatenated features.  No cross-dataset claim.
4. ``--modality-experts`` — synthetic paired fixtures: one expert per
   *modality* (tabular recording expert vs image expert) for the same
   synthetic participants, demonstrating the architecture's feature-
   slicing with a genuine tabular+image expert pair.

Compare with: ``python scripts/run_fusion_evaluation.py`` (fusion) and
the single-model scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pd_drawing_analysis.data import inventory_spatial, load_image_gray
from pd_drawing_analysis.evaluation import (
    DEFAULT_MODEL_SPECS,
    SEED,
    ModelSpec,
    evaluate_model_family,
    folds_from_subject_sets,
    subject_group_folds,
)
from pd_drawing_analysis.features import SPATIAL_IMAGE_SIZE, hog_features, shape_features
from pd_drawing_analysis.moe import DEFAULT_EXPERTS, ExpertSpec, MixtureOfExperts
from pd_drawing_analysis.reporting import write_json_artifact, write_markdown_summary
from pd_drawing_analysis.synthetic import two_regime_participants

MOE_SPEC = ModelSpec(
    name="moe_em",
    factory=MixtureOfExperts,
    param_grid={"clf__n_em_steps": [3, 8]},
)


def _single_expert_specs():
    """Each MoE expert, evaluated alone under the identical protocol."""
    specs = []
    for expert in DEFAULT_EXPERTS:
        specs.append(
            ModelSpec(
                name=f"single_{expert.name}",
                factory=expert.factory,
                param_grid={},
            )
        )
    return specs


def run_synthetic(n_resamples: int) -> dict:
    """Subject-grouped nested CV: MoE vs its own experts at n=24."""
    from sklearn.pipeline import Pipeline

    X, y, regimes, subjects = two_regime_participants()
    folds = subject_group_folds(subjects, y, n_splits=5)
    index_folds = folds_from_subject_sets(subjects, folds)

    results = {"models": {}}
    for spec in [MOE_SPEC, *_single_expert_specs()]:
        print(f"  evaluating {spec.name} ...")
        results["models"][spec.name] = evaluate_model_family(
            X, y, subjects, spec, index_folds, n_resamples=n_resamples
        )

    gate_summary = _gate_localisation(X, y, regimes)
    results["gate_localisation"] = gate_summary
    results["dataset"] = {
        "name": "Synthetic two-regime participants (methods demonstration)",
        "n_subjects": int(len(y)),
        "regime_counts": {k: int(v) for k, v in zip(*np.unique(regimes, return_counts=True))},
        "notes": "Gate signal is strong by construction; class rules differ per regime.",
    }
    results["scope_note"] = (
        "Mixture-of-experts scaffold evaluated honestly at small n: an advantage over "
        "single experts is NOT expected at this sample size (mixtures are data-hungry); "
        "numbers are reported as measured. See --validate-large for mechanics validation."
    )
    return results


def _gate_localisation(X: np.ndarray, y: np.ndarray, regimes: np.ndarray) -> dict:
    """Fit the MoE on all rows and report how well the gate recovers the
    latent regimes (mechanics diagnostic, not a performance estimate)."""
    moe = MixtureOfExperts().fit(X, y)
    gate = moe.gate_weights(X)
    preferred = gate.argmax(axis=1)
    purity = max(
        float((preferred[regimes == r] == preferred[regimes == r][0]).mean())
        for r in np.unique(regimes)
    )
    return {
        "gate_regime_purity": purity,
        "mean_gate_weights": gate.mean(axis=0).round(3).tolist(),
        "note": "fitted once on all rows; diagnostic only",
    }


def run_validate_large() -> dict:
    """Implementation validation at n=600 (train) on the clean fixture."""
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(SEED)
    X_big, y_big, _, _ = two_regime_participants(n_per_regime=300, seed=SEED)
    Xtr, Xte, ytr, yte = train_test_split(X_big, y_big, test_size=0.3, random_state=SEED, stratify=y_big)

    scores = {}
    for name, model in [
        ("single_logreg_strong_l2", DEFAULT_EXPERTS[0].factory()),
        ("single_tree_depth3", DEFAULT_EXPERTS[2].factory()),
        ("moe_em", MixtureOfExperts()),
    ]:
        model.fit(Xtr, ytr)
        scores[name] = float((model.predict(Xte) == yte).mean())

    gate_purity = _gate_localisation(X_big, y_big, np.array(
        ["A"] * (len(y_big) // 2) + ["B"] * (len(y_big) // 2)
    ))["gate_regime_purity"]

    return {
        "models": {name: {"test_accuracy": acc} for name, acc in scores.items()},
        "gate_regime_purity": gate_purity,
        "dataset": {"name": "Two-regime fixture, n=600 (1200 rows), 30% stratified holdout",
                    "n_train": int(len(ytr)), "n_test": int(len(yte))},
        "scope_note": (
            "Mechanics validation on abundant synthetic data: verifies the EM mixture can "
            "localise experts and outperform its weak single expert when data is sufficient. "
            "NOT an estimate for the real datasets."
        ),
    }


def _participant_task_features(spatial_root: Path):
    """Per-participant HOG+shape features for spiral and wave separately."""
    inv = inventory_spatial(spatial_root)
    per_task: dict[str, dict[str, list[np.ndarray]]] = {}
    for _, row in inv.frame.iterrows():
        img = load_image_gray(row["path"], size=SPATIAL_IMAGE_SIZE)
        feats = np.concatenate([hog_features(img), shape_features(img)])
        per_task.setdefault(row["task"], {}).setdefault(row["subject_id"], []).append(feats)
    tasks = sorted(per_task)
    subjects = np.array(sorted(per_task[tasks[0]]))
    if tasks[1:] and set(per_task[tasks[1]]) != set(subjects):
        raise ValueError("Participants differ between tasks; cannot build per-task experts")
    blocks = [np.vstack([np.mean(per_task[t][s], axis=0) for s in subjects]) for t in tasks]
    labels = np.array([1 if s.startswith("parkinson") else 0 for s in subjects])
    widths = [b.shape[1] for b in blocks]
    return subjects, labels, blocks, widths, tasks


def run_task_experts(spatial_root: Path, n_resamples: int) -> dict:
    """Real data: per-task experts (spiral vs wave) with a learned gate."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    subjects, labels, blocks, widths, tasks = _participant_task_features(spatial_root)
    X = np.hstack(blocks)
    slices = [
        slice(0, widths[0]),
        slice(widths[0], widths[0] + widths[1]),
    ]
    moe_experts = [
        ExpertSpec(f"{tasks[0]}_expert", lambda: LogisticRegression(max_iter=5000, C=0.1),
                   feature_slice=slices[0]),
        ExpertSpec(f"{tasks[1]}_expert", lambda: LogisticRegression(max_iter=5000, C=0.1),
                   feature_slice=slices[1]),
    ]

    # Pipeline without PCA so expert slices index the actual task features.
    def make_moe(params: dict | None = None):
        steps = n_em_steps = (params or {}).get("moe__n_em_steps", 5)
        return Pipeline([
            ("scale", StandardScaler()),
            ("moe", MixtureOfExperts(experts=moe_experts, n_em_steps=int(steps))),
        ])

    class _MoESpec:
        name = "moe_task_experts"
        param_grid: dict = {}

        @staticmethod
        def factory():
            return make_moe()

    _MoESpec.pipeline_builder = staticmethod(make_moe)

    single_task_specs = [
        ModelSpec(name=f"single_{tasks[0]}", factory=lambda: LogisticRegression(max_iter=5000, C=0.1),
                  param_grid={}),
        ModelSpec(name=f"single_{tasks[1]}", factory=lambda: LogisticRegression(max_iter=5000, C=0.1),
                  param_grid={}),
    ]
    pooled_spec = ModelSpec(name="pooled_logreg", factory=lambda: LogisticRegression(max_iter=5000, C=0.1),
                            param_grid={"clf__C": [0.01, 0.1, 1.0]})

    folds = subject_group_folds(subjects, labels, n_splits=5)
    index_folds = folds_from_subject_sets(subjects, folds)

    results = {"models": {}}
    print("  evaluating single-task experts ...")
    for spec in single_task_specs:
        block_idx = 0 if tasks[0] in spec.name else 1
        results["models"][spec.name] = evaluate_model_family(
            blocks[block_idx], labels, subjects, spec, index_folds, n_resamples=n_resamples
        )
    print("  evaluating pooled logreg ...")
    results["models"]["pooled_logreg"] = evaluate_model_family(
        X, labels, subjects, pooled_spec, index_folds, n_resamples=n_resamples
    )
    print("  evaluating MoE (task experts + gate) ...")
    results["models"]["moe_task_experts"] = evaluate_model_family(
        X, labels, subjects, _MoESpec, index_folds, n_resamples=n_resamples,
    )
    # Report the gate's average expert preference.
    moe_model = make_moe().fit(X, labels)
    gate_mean = moe_model.named_steps["moe"].gate_weights(moe_model.named_steps["scale"].transform(X)).mean(axis=0)
    results["gate_mean_weights"] = gate_mean.round(3).tolist()
    results["dataset"] = {
        "name": "Kaggle Parkinson's Drawings — per-task expert MoE (same participants did both tasks)",
        "root": str(spatial_root),
        "n_participants": int(len(subjects)),
        "task_feature_widths": dict(zip(tasks, widths)),
    }
    results["scope_note"] = (
        "Real-data MoE: experts see only their own task's features; the gate sees both. "
        "Same 28 participants completed both tasks, so no cross-dataset claim is involved. "
        "At this sample size parity with a single regularised model is the expected outcome."
    )
    return results


def run_modality_experts(n_resamples: int) -> dict:
    """Synthetic paired fixtures: tabular expert + image expert.

    Every synthetic participant has BOTH a tablet recording and a spiral
    drawing, so a tabular expert and an image expert can legitimately
    specialise in their own modality's feature block while the gate sees
    the concatenation.  Demonstrates the modality-slicing architecture —
    the real datasets cannot supply this pairing.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from pd_drawing_analysis.fusion import assemble_paired_data
    from pd_drawing_analysis.moe import ExpertSpec
    from pd_drawing_analysis.synthetic import generate_paired_fixtures

    root = Path("data/synthetic/paired")
    generate_paired_fixtures(root)
    data, info = assemble_paired_data(root / "temporal", root / "images" / "spiral" / "alldata")
    widths = {name: mat.shape[1] for name, mat in data.modalities.items()}
    X = np.hstack([data.modalities["temporal"], data.modalities["image"]])
    subjects, labels = data.subjects, data.labels
    print(f"Paired participants: {len(subjects)}; blocks: {widths}")

    slices = {
        "tabular": slice(0, widths["temporal"]),
        "image": slice(widths["temporal"], None),
    }
    moe_experts = [
        ExpertSpec("tabular_expert", lambda: LogisticRegression(max_iter=5000, C=0.1),
                   feature_slice=slices["tabular"]),
        ExpertSpec("image_expert", lambda: LogisticRegression(max_iter=5000, C=0.1),
                   feature_slice=slices["image"]),
    ]

    def make_moe(params: dict | None = None):
        steps = int((params or {}).get("moe__n_em_steps", 5))
        return Pipeline([
            ("scale", StandardScaler()),
            ("moe", MixtureOfExperts(experts=moe_experts, n_em_steps=steps)),
        ])

    class _MoESpec:
        name = "moe_modality_experts"
        param_grid = {"moe__n_em_steps": [3, 8]}

        @staticmethod
        def factory():
            return make_moe()

    _MoESpec.pipeline_builder = staticmethod(make_moe)

    def _logreg_factory():
        return LogisticRegression(max_iter=5000, C=0.1)

    single_specs = [
        ModelSpec(name="single_tabular", factory=_logreg_factory, param_grid={}),
        ModelSpec(name="single_image", factory=_logreg_factory, param_grid={}),
    ]
    pooled_spec = ModelSpec(name="pooled_logreg", factory=_logreg_factory,
                            param_grid={"clf__C": [0.01, 0.1, 1.0]})

    folds = subject_group_folds(subjects, labels, n_splits=5)
    index_folds = folds_from_subject_sets(subjects, folds)

    results = {"models": {}}
    print("  evaluating single-modality baselines ...")
    for spec, block in zip(single_specs, [data.modalities["temporal"], data.modalities["image"]]):
        results["models"][spec.name] = evaluate_model_family(
            block, labels, subjects, spec, index_folds, n_resamples=n_resamples
        )
    print("  evaluating pooled logreg ...")
    results["models"]["pooled_logreg"] = evaluate_model_family(
        X, labels, subjects, pooled_spec, index_folds, n_resamples=n_resamples
    )
    print("  evaluating MoE (tabular expert + image expert + gate) ...")
    results["models"]["moe_modality_experts"] = evaluate_model_family(
        X, labels, subjects, _MoESpec, index_folds, n_resamples=n_resamples
    )

    fitted = make_moe().fit(X, labels)
    gate_mean = fitted.named_steps["moe"].gate_weights(
        fitted.named_steps["scale"].transform(X)
    ).mean(axis=0)
    results["gate_mean_weights"] = gate_mean.round(3).tolist()
    results["dataset"] = {
        "name": "Synthetic paired fixtures — modality-sliced MoE (tabular vs image)",
        "n_participants": int(len(subjects)),
        "blocks": {"tabular": widths["temporal"], "image": widths["image"]},
        "expert_slices": {"tabular_expert": "columns 0-24 (recording features)",
                          "image_expert": f"columns {widths['temporal']}-end (HOG+shape)"},
        "notes": info,
    }
    results["scope_note"] = (
        "Architecture demonstration on clearly synthetic paired participants. "
        "The real temporal and image datasets have different participants and "
        "cannot be combined this way; accuracy numbers here are meaningless "
        "beyond verifying the slicing mechanics."
    )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", default=True)
    parser.add_argument("--validate-large", action="store_true")
    parser.add_argument("--task-experts", action="store_true")
    parser.add_argument("--modality-experts", action="store_true",
                        help="Synthetic paired fixtures: tabular expert vs image expert")
    parser.add_argument("--spatial-root", type=Path, default=Path("KaggleSpiral/drawings"))
    parser.add_argument("--n-resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=Path("results/moe"))
    args = parser.parse_args()

    if args.validate_large:
        results = run_validate_large()
        tag = "validate_large"
    elif args.modality_experts:
        results = run_modality_experts(args.n_resamples)
        tag = "modality_experts_synthetic"
    elif args.task_experts:
        if not args.spatial_root.is_dir():
            raise SystemExit(f"Spatial root {args.spatial_root} not found; download the "
                             "authorized dataset (docs/data_sources.md) or use --synthetic.")
        results = run_task_experts(args.spatial_root, args.n_resamples)
        tag = "task_experts_real"
    else:
        results = run_synthetic(args.n_resamples)
        tag = "synthetic"

    out_json = write_json_artifact(results, args.output_dir / f"moe_{tag}.json")
    out_md = write_markdown_summary(f"Mixture of experts — {tag}", results,
                                    args.output_dir / f"moe_{tag}.md")
    for name, r in results["models"].items():
        m = r.get("participant_metrics") or {}
        if m:
            print(f"  {name:28s} acc={m['accuracy']:.3f} bal={m['balanced_accuracy']:.3f}")
        elif "test_accuracy" in r:
            print(f"  {name:28s} test_acc={r['test_accuracy']:.3f}")
    print(f"\nArtifacts:\n  {out_json}\n  {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
