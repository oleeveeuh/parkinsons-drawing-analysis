"""Multimodal and multi-feature fusion under subject-grouped validation.

Three fusion architectures are implemented and tested:

* **early** — concatenate per-participant feature vectors, one
  classifier on top.
* **late** — one classifier per modality, average predicted
  probabilities with a weight selected in the *inner* CV loop.
* **stacked** — per-modality inner-fold out-of-fold probabilities become
  meta-features for a logistic meta-learner.  The meta-learner only ever
  sees inner-training rows of the outer-training set, so outer-evaluation
  participants never influence it.

Honest scope (see ``docs/data_audit.md``): the two REAL datasets in this
repository cannot be fused — they contain different participants.  Fusion
is exercised (a) on synthetic *paired*-modality fixtures, where the same
synthetic participant generates both modalities, and (b) on real data as
*feature-family* fusion (HOG + shape descriptors of the same drawings).
Nothing here supports a cross-dataset fusion claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression

from .evaluation import (
    SEED,
    _inner_folds,
    _inner_score,
    _positive_proba,
    make_model_pipeline,
    subject_group_folds,
)
from .metrics import (
    aggregate_to_participants,
    classification_metrics,
    subject_bootstrap_ci,
)
from .subject_ids import assert_subject_disjoint

__all__ = [
    "PairedModalityData",
    "assemble_paired_data",
    "evaluate_fusion",
    "FUSION_STRATEGIES",
]

# Small, justified grids for the dataset size.
_SINGLETON_GRID = {"clf__C": [0.01, 0.1, 1.0]}
_LATE_WEIGHT_GRID = [0.4, 0.5, 0.6]
_META_GRID = {"C": [0.1, 1.0]}


@dataclass
class PairedModalityData:
    """One feature row per participant, aligned across modalities.

    ``modalities`` maps a modality name to a ``[n_subjects, d]`` matrix.
    Row ``i`` of every matrix belongs to ``subjects[i]``.
    """

    subjects: np.ndarray
    labels: np.ndarray
    modalities: dict[str, np.ndarray]

    def __post_init__(self):
        n = len(self.subjects)
        for name, X in self.modalities.items():
            if X.shape[0] != n:
                raise AssertionError(
                    f"Modality {name!r} has {X.shape[0]} rows for {n} subjects"
                )


def _logreg_pipeline(params: dict | None = None):
    from sklearn.linear_model import LogisticRegression

    class _Spec:  # minimal ModelSpec-compatible view
        name = "logreg_balanced"

        @staticmethod
        def factory():
            return LogisticRegression(max_iter=5000, class_weight="balanced")

    return make_model_pipeline(_Spec, params, pca_variance=0.95)


def _select_on_inner(
    build: Callable[[dict], object],
    grid: list[dict],
    X: np.ndarray,
    y: np.ndarray,
    inner_folds: list,
) -> tuple[dict, float]:
    """Inner-loop selection: balanced accuracy over subject-grouped inner folds."""
    best_params, best_score = {}, -np.inf
    for params in grid:
        scores = []
        for tr, val in inner_folds:
            model = build(params)
            model.fit(X[tr], y[tr])
            proba = _predict_score(model, X[val])
            scores.append(_inner_score(y[val], proba))
        mean = float(np.mean(scores)) if scores else -np.inf
        if mean > best_score:
            best_score, best_params = mean, params
    return best_params, best_score


def _predict_score(model, X: np.ndarray) -> np.ndarray:
    """Positive-class probability with single-class-fold robustness."""
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if len(classes) == 1:
        return np.full(len(X), 1.0 if classes[0] == 1 else 0.0)
    return proba[:, classes.index(1)]


def _run_single_modality_strategy(
    name_of_modality: str,
    data: PairedModalityData,
    outer_folds: list,
    inner_splits: int,
    random_state: int,
) -> tuple[np.ndarray, list[dict]]:
    """Baseline: one modality only (participant-level rows)."""
    X = data.modalities[name_of_modality]
    y = data.labels
    oof = np.full(len(y), np.nan)
    fold_records = []
    for fold_number, (train_idx, eval_idx) in enumerate(outer_folds, start=1):
        assert_subject_disjoint(
            set(data.subjects[train_idx]), set(data.subjects[eval_idx]),
            context=f"single {name_of_modality} outer fold {fold_number}",
        )
        inner = _inner_folds(X[train_idx], y[train_idx], data.subjects[train_idx],
                             inner_splits, random_state)
        params, inner_score = _select_on_inner(
            lambda p: _logreg_pipeline(p), [dict(p) for p in _grid_expand(_SINGLETON_GRID)],
            X[train_idx], y[train_idx], inner,
        )
        model = _logreg_pipeline(params)
        model.fit(X[train_idx], y[train_idx])
        oof[eval_idx] = _predict_score(model, X[eval_idx])
        fold_records.append({"fold": fold_number, "selected_params": params, "inner_cv_score": inner_score})
    return oof, fold_records


def _grid_expand(grid: dict) -> list[dict]:
    keys = list(grid)
    combos: list[dict] = [{}]
    for key in keys:
        combos = [dict(c, **{key: v}) for c in combos for v in grid[key]]
    return combos


def _run_early_fusion(
    data: PairedModalityData,
    outer_folds: list,
    inner_splits: int,
    random_state: int,
) -> tuple[np.ndarray, list[dict]]:
    X = np.hstack(list(data.modalities.values()))
    y = data.labels
    oof = np.full(len(y), np.nan)
    fold_records = []
    for fold_number, (train_idx, eval_idx) in enumerate(outer_folds, start=1):
        assert_subject_disjoint(
            set(data.subjects[train_idx]), set(data.subjects[eval_idx]),
            context=f"early-fusion outer fold {fold_number}",
        )
        inner = _inner_folds(X[train_idx], y[train_idx], data.subjects[train_idx],
                             inner_splits, random_state)
        params, inner_score = _select_on_inner(
            lambda p: _logreg_pipeline(p), _grid_expand(_SINGLETON_GRID),
            X[train_idx], y[train_idx], inner,
        )
        model = _logreg_pipeline(params)
        model.fit(X[train_idx], y[train_idx])
        oof[eval_idx] = _predict_score(model, X[eval_idx])
        fold_records.append({"fold": fold_number, "selected_params": params, "inner_cv_score": inner_score})
    return oof, fold_records


def _modality_oof_within_train(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, inner: list
) -> np.ndarray:
    """Inner-fold out-of-fold probabilities for one modality (stacking input)."""
    meta = np.full(len(y), np.nan)
    for tr, val in inner:
        model = _logreg_pipeline()
        model.fit(X[tr], y[tr])
        meta[val] = _predict_score(model, X[val])
    if np.isnan(meta).any():
        raise AssertionError("Inner OOF coverage gap in stacking")
    return meta


def _run_late_fusion(
    data: PairedModalityData,
    outer_folds: list,
    inner_splits: int,
    random_state: int,
) -> tuple[np.ndarray, list[dict]]:
    names = list(data.modalities)
    y = data.labels
    oof = np.full(len(y), np.nan)
    fold_records = []
    for fold_number, (train_idx, eval_idx) in enumerate(outer_folds, start=1):
        assert_subject_disjoint(
            set(data.subjects[train_idx]), set(data.subjects[eval_idx]),
            context=f"late-fusion outer fold {fold_number}",
        )
        inner = _inner_folds(
            np.zeros((len(train_idx), 1)), y[train_idx], data.subjects[train_idx],
            inner_splits, random_state,
        )
        # Select the mixing weight on inner folds only.
        best_w, best_score = 0.5, -np.inf
        for w in _LATE_WEIGHT_GRID:
            scores = []
            for tr, val in inner:
                parts = []
                for name in names:
                    X = data.modalities[name][train_idx]
                    model = _logreg_pipeline()
                    model.fit(X[tr], y[train_idx][tr])
                    parts.append(_predict_score(model, X[val]))
                blended = w * parts[0] + (1.0 - w) * parts[1]
                scores.append(_inner_score(y[train_idx][val], blended))
            mean = float(np.mean(scores))
            if mean > best_score:
                best_score, best_w = mean, w
        # Refit on the full outer-training set and blend on the eval side.
        eval_parts = []
        for name in names:
            X = data.modalities[name]
            model = _logreg_pipeline()
            model.fit(X[train_idx], y[train_idx])
            eval_parts.append(_predict_score(model, X[eval_idx]))
        oof[eval_idx] = best_w * eval_parts[0] + (1.0 - best_w) * eval_parts[1]
        fold_records.append({"fold": fold_number, "selected_weight": best_w, "inner_cv_score": best_score})
    return oof, fold_records


def _run_stacked_fusion(
    data: PairedModalityData,
    outer_folds: list,
    inner_splits: int,
    random_state: int,
) -> tuple[np.ndarray, list[dict]]:
    names = list(data.modalities)
    y = data.labels
    oof = np.full(len(y), np.nan)
    fold_records = []

    def _meta_model(params):
        return LogisticRegression(max_iter=5000, **params)

    for fold_number, (train_idx, eval_idx) in enumerate(outer_folds, start=1):
        assert_subject_disjoint(
            set(data.subjects[train_idx]), set(data.subjects[eval_idx]),
            context=f"stacked-fusion outer fold {fold_number}",
        )
        inner = _inner_folds(
            np.zeros((len(train_idx), 1)), y[train_idx], data.subjects[train_idx],
            inner_splits, random_state,
        )
        # Meta-features: inner OOF probabilities of each modality, computed
        # strictly inside the outer-training set.
        meta_X = np.column_stack(
            [_modality_oof_within_train(data.modalities[name][train_idx],
                                        y[train_idx], data.subjects[train_idx], inner)
             for name in names]
        )
        meta_y = y[train_idx]
        meta_params, inner_score = _select_on_inner(
            _meta_model, _grid_expand(_META_GRID), meta_X, meta_y, inner,
        )
        meta_model = _meta_model(meta_params)
        meta_model.fit(meta_X, meta_y)
        # Refit base models on the full outer-training set for eval-side meta-features.
        eval_meta = np.column_stack([
            _predict_score(
                _logreg_pipeline().fit(data.modalities[name][train_idx], y[train_idx]),
                data.modalities[name][eval_idx],
            )
            for name in names
        ])
        oof[eval_idx] = meta_model.predict_proba(eval_meta)[:, list(meta_model.classes_).index(1)]
        fold_records.append({"fold": fold_number, "selected_params": meta_params, "inner_cv_score": inner_score})
    return oof, fold_records


FUSION_STRATEGIES: dict[str, Callable] = {
    "early_fusion": _run_early_fusion,
    "late_fusion": _run_late_fusion,
    "stacked_fusion": _run_stacked_fusion,
}


def evaluate_fusion(
    data: PairedModalityData,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = SEED,
    n_resamples: int = 2000,
    include_single_modality_baselines: bool = True,
) -> dict:
    """Paired comparison of fusion strategies and single-modality
    baselines under identical subject-grouped outer folds."""
    subjects = np.asarray(data.subjects)
    labels = np.asarray(data.labels).astype(int)
    if len(set(subjects.tolist())) != len(subjects):
        raise AssertionError("Paired data must have exactly one row per subject")

    subject_folds = subject_group_folds(subjects, labels, outer_splits, random_state)
    # Rows are subjects already: map subject-level folds to trivial index pairs.
    outer_folds = []
    subject_pos = {s: i for i, s in enumerate(subjects)}
    for train_subjects, eval_subjects in subject_folds:
        train_idx = np.array([subject_pos[s] for s in train_subjects])
        eval_idx = np.array([subject_pos[s] for s in eval_subjects])
        assert_subject_disjoint(set(train_subjects.tolist()), set(eval_subjects.tolist()),
                                context="fusion outer fold")
        outer_folds.append((train_idx, eval_idx))

    results: dict[str, dict] = {}
    runs: dict[str, np.ndarray] = {}
    if include_single_modality_baselines:
        for name in data.modalities:
            runs[f"single_{name}"], _ = _run_single_modality_strategy(
                name, data, outer_folds, inner_splits, random_state
            )
    for strategy_name, strategy in FUSION_STRATEGIES.items():
        runs[strategy_name], fold_records = strategy(data, outer_folds, inner_splits, random_state)
        results.setdefault(strategy_name, {})["folds"] = fold_records

    if any(np.isnan(v).any() for v in runs.values()):
        raise AssertionError("Some participants never received an out-of-fold prediction")

    for name, oof in runs.items():
        participants = aggregate_to_participants(subjects, labels, oof)  # 1 row per subject
        metrics = classification_metrics(
            participants["y_true"], participants["y_pred"], participants["mean_proba"],
            n_images=len(labels),
        )
        metrics.ci = subject_bootstrap_ci(
            participants["y_true"], participants["y_pred"], participants["mean_proba"],
            n_resamples=n_resamples, random_state=random_state,
        )
        entry = {"participant_metrics": metrics.to_dict()}
        if name in results:
            entry["folds"] = results[name]["folds"]
        results[name] = entry
    return {
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "random_state": random_state,
        "n_subjects": int(len(subjects)),
        "models": results,
        "oof": {name: v.tolist() for name, v in runs.items()},
    }


def assemble_paired_data(
    temporal_dir,
    image_dir,
    include_z: bool = False,
) -> tuple[PairedModalityData, list[str]]:
    """Build per-participant paired features from a temporal directory
    (one recording per participant) and an image directory (one or more
    drawings per participant, aggregated by per-subject mean of frozen
    per-image features — a per-subject arithmetic transform that fits
    nothing and cannot leak)."""
    from .data import inventory_temporal, load_image_gray, load_temporal_recording
    from .features import (
        extract_temporal_features,
        hog_features,
        shape_features,
        SPATIAL_IMAGE_SIZE,
    )
    from .subject_ids import parse_kaggle_filename
    from pathlib import Path

    temporal_dir, image_dir = Path(temporal_dir), Path(image_dir)
    temporal_inv = inventory_temporal(temporal_dir)

    temporal_rows: dict[str, np.ndarray] = {}
    for _, row in temporal_inv.frame.iterrows():
        features = extract_temporal_features(
            load_temporal_recording(row["path"]), include_z=include_z
        )
        # Re-key to the class-aware V-style id used by the images.
        number = str(row["subject_id"]).split("S")[-1]
        temporal_rows[f"{row['subject_id'].split(':')[0]}:V{int(number):02d}"] = features

    image_rows: dict[str, list[np.ndarray]] = {}
    for png in sorted(image_dir.rglob("*.png")):
        subject = parse_kaggle_filename(png.name).subject_id
        img = load_image_gray(png, size=SPATIAL_IMAGE_SIZE)
        image_rows.setdefault(subject, []).append(
            np.concatenate([hog_features(img), shape_features(img)])
        )

    common = sorted(set(temporal_rows) & set(image_rows))
    if not common:
        raise ValueError(
            "No shared participants between the temporal and image directories; "
            "fusion requires the same participants in both modalities."
        )
    missing = (set(temporal_rows) | set(image_rows)) - set(common)
    if missing:
        print(f"Note: {len(missing)} participants present in only one modality and excluded: "
              f"{sorted(missing)[:5]}{'...' if len(missing) > 5 else ''}")

    image_dim = len(image_rows[common[0]][0])
    X_image = np.vstack([np.mean(image_rows[s], axis=0) for s in common])
    X_temporal = np.vstack([temporal_rows[s] for s in common])
    labels = np.array([1 if s.startswith("parkinson") else 0 for s in common])
    data = PairedModalityData(
        subjects=np.array(common), labels=labels,
        modalities={"temporal": X_temporal, "image": X_image},
    )
    info = [
        f"temporal features: {X_temporal.shape[1]}",
        f"image features per drawing (HOG+shape): {image_dim}",
        f"paired participants: {len(common)}",
    ]
    return data, info
