"""Subject-grouped nested cross-validation.

This module is the corrected evaluation core of the project.  It replaces
the image-level ``StratifiedKFold`` protocol of the previous code, which
allowed ~3 correlated drawings of the same participant to appear on both
sides of every split (see ``docs/data_audit.md``).

Protocol
--------
Outer loop
    ``StratifiedGroupKFold`` over **participants**: performance estimation
    with no participant shared between training and evaluation.
Inner loop
    ``StratifiedGroupKFold`` over participants *within each outer
    training set*: hyperparameter selection, again fully disjoint.
Fitting
    The entire preprocessing chain (variance selection → scaling → PCA →
    classifier) is one scikit-learn ``Pipeline`` fitted inside each inner
    training set.  Validation and evaluation folds never influence
    variance thresholds, scaling statistics, PCA components, or model
    selection.
Aggregation
    Drawing-level out-of-fold predictions are stored with their original
    indices, then aggregated to one prediction per participant before any
    primary metric is computed.
Baselines
    Majority-class ``DummyClassifier``, class-weighted logistic
    regression, and a shallow decision tree are always evaluated;
    XGBoost is optional and only ever reported next to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .metrics import (
    aggregate_to_participants,
    classification_metrics,
    fold_variability,
    subject_bootstrap_ci,
)
from .subject_ids import assert_subject_disjoint

__all__ = [
    "ModelSpec",
    "DEFAULT_MODEL_SPECS",
    "make_model_pipeline",
    "subject_outer_folds",
    "nested_subject_cv",
    "evaluate_model_family",
]

SEED = 42


@dataclass(frozen=True)
class ModelSpec:
    """One model family with a small, justified hyperparameter grid.

    ``pipeline_builder`` (optional) replaces the standard
    variance→scale→PCA pipeline with a custom one (e.g. when a model
    needs unscaled-by-PCA feature blocks, like sliced MoE experts).
    It receives the selected params dict and must return an unfitted
    pipeline; all fitting still happens inside folds.
    """

    name: str
    factory: Any  # callable -> unfitted estimator
    param_grid: dict[str, list]
    requires_extra: str | None = None
    pipeline_builder: Any | None = None


def _pipeline_for(spec: ModelSpec, params: dict | None, pca_variance: float | None) -> Pipeline:
    builder = getattr(spec, "pipeline_builder", None)
    if builder is not None:
        return builder(params or {})
    return make_model_pipeline(spec, params, pca_variance=pca_variance)


def _make_logreg():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(max_iter=5000, class_weight="balanced")


def _make_tree():
    return DecisionTreeClassifier(class_weight="balanced", random_state=SEED)


def _make_dummy():
    return DummyClassifier(strategy="most_frequent")


def _make_xgboost():
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "XGBoost is an optional comparison model. Install it with "
            "pip install '.[xgboost]' (on macOS the OpenMP runtime is also "
            "required: brew install libomp)."
        ) from exc
    return XGBClassifier(
        random_state=SEED,
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,
    )


DEFAULT_MODEL_SPECS: list[ModelSpec] = [
    ModelSpec(
        name="dummy_majority",
        factory=_make_dummy,
        param_grid={},  # no hyperparameters to tune
    ),
    ModelSpec(
        name="logreg_balanced",
        factory=_make_logreg,
        param_grid={"clf__C": [0.01, 0.1, 1.0]},
    ),
    ModelSpec(
        name="decision_tree_depth2_4",
        factory=_make_tree,
        param_grid={"clf__max_depth": [2, 3, 4]},
    ),
]

XGBOOST_SPEC = ModelSpec(
    name="xgboost_shallow",
    factory=_make_xgboost,
    param_grid={"clf__max_depth": [2, 3], "clf__n_estimators": [50, 100]},
    requires_extra="xgboost",
)


def make_model_pipeline(spec: ModelSpec, params: dict | None = None, pca_variance: float | None = 0.95) -> Pipeline:
    """Full preprocessing + model pipeline.

    Everything here is fitted by ``Pipeline.fit`` on the training data
    handed to it — never on the full dataset.  ``pca_variance=None``
    disables PCA (used for the dummy baseline and low-dimensional inputs).
    SMOTE is deliberately **not** used: the spatial dataset is
    approximately balanced and contains repeated drawings per
    participant, and oversampling inside nested CV would fabricate
    near-duplicate minority points drawn from inner-validation
    neighbours.  Class weighting achieves the same goal without
    synthesising observations.
    """
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.preprocessing import StandardScaler

    steps: list[tuple[str, Any]] = []
    if pca_variance is not None:
        steps.append(("variance", VarianceThreshold(threshold=0.0)))
        steps.append(("scale", StandardScaler()))
        steps.append(("pca", PCA(n_components=pca_variance, random_state=SEED)))
    steps.append(("clf", spec.factory()))
    pipeline = Pipeline(steps)
    if params:
        pipeline.set_params(**params)
    return pipeline


def subject_group_folds(subjects: np.ndarray, labels: np.ndarray, n_splits: int,
                        random_state: int = SEED) -> list:
    """Deterministic subject-level stratified folds over **unique subjects**.

    Returns a list of ``(train_subjects, eval_subjects)`` numpy arrays of
    subject ids.  Because folds are defined on subjects — not on rows —
    they can be shared across tasks whose participant sets coincide
    (spiral and wave share all participants), guaranteeing identical
    outer folds for paired comparisons.
    """
    subjects = np.asarray(subjects)
    labels = np.asarray(labels)
    subject_label = {}
    for s, y in zip(subjects, labels):
        if s in subject_label and subject_label[s] != y:
            raise AssertionError(f"Subject {s} has conflicting labels")
        subject_label[s] = y
    unique_subjects = np.array(sorted(subject_label))
    subject_y = np.array([subject_label[s] for s in unique_subjects])
    class_counts = {
        int(value): int(count) for value, count in zip(*np.unique(subject_y, return_counts=True))
    }
    if min(class_counts.values()) < n_splits:
        raise ValueError(
            f"Cannot build {n_splits} outer folds: smallest class has only "
            f"{min(class_counts.values())} participants. Reduce n_splits or add data."
        )
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    subject_folds = []
    for train_pos, eval_pos in splitter.split(
        np.zeros((len(unique_subjects), 1)), subject_y, groups=unique_subjects
    ):
        train_set = unique_subjects[train_pos]
        eval_set = unique_subjects[eval_pos]
        assert_subject_disjoint(set(train_set.tolist()), set(eval_set.tolist()), context="outer fold")
        subject_folds.append((train_set, eval_set))
    return subject_folds


def folds_from_subject_sets(subjects: np.ndarray, subject_folds: list) -> list:
    """Map subject-level folds to ``(train_idx, eval_idx)`` pairs over the
    supplied row array, asserting the boundary stays subject-disjoint."""
    subjects = np.asarray(subjects)
    folds = []
    for train_set, eval_set in subject_folds:
        train_set, eval_set = set(train_set.tolist()), set(eval_set.tolist())
        assert_subject_disjoint(train_set, eval_set, context="mapped outer fold")
        train_idx = np.array([i for i, s in enumerate(subjects) if s in train_set])
        eval_idx = np.array([i for i, s in enumerate(subjects) if s in eval_set])
        folds.append((train_idx, eval_idx))
    return folds


def subject_outer_folds(subjects: np.ndarray, labels: np.ndarray, n_splits: int,
                        random_state: int = SEED) -> list:
    """Deterministic subject-grouped outer folds as index pairs over the
    supplied arrays.  Thin wrapper over :func:`subject_group_folds`."""
    subject_folds = subject_group_folds(subjects, labels, n_splits, random_state)
    return folds_from_subject_sets(np.asarray(subjects), subject_folds)


def _inner_folds(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int, random_state: int):
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, val_idx in splitter.split(X, y, groups=groups):
        assert_subject_disjoint(
            set(groups[train_idx].tolist()), set(groups[val_idx].tolist()), context="inner fold"
        )
        folds.append((train_idx, val_idx))
    return folds


def _grid_candidates(spec: ModelSpec) -> list[dict]:
    keys = list(spec.param_grid)
    if not keys:
        return [{}]
    combos = [{}]
    for key in keys:
        combos = [dict(c, **{key: value}) for c in combos for value in spec.param_grid[key]]
    return combos


def _fit_inner_selection(X_train: np.ndarray, y_train: np.ndarray, groups: np.ndarray,
                         spec: ModelSpec, inner_splits: int, random_state: int) -> tuple[dict, float]:
    """Inner-loop hyperparameter selection on subject-grouped folds."""
    folds = _inner_folds(X_train, y_train, groups, inner_splits, random_state)
    best_params, best_score = {}, -np.inf
    for params in _grid_candidates(spec):
        scores = []
        for tr_idx, val_idx in folds:
            pipeline = _pipeline_for(spec, params, pca_variance=None if spec.name == "dummy_majority" else 0.95)
            pipeline.fit(X_train[tr_idx], y_train[tr_idx])
            proba = _positive_proba(pipeline, X_train[val_idx])
            scores.append(_inner_score(y_train[val_idx], proba))
        mean_score = float(np.mean(scores)) if scores else -np.inf
        if mean_score > best_score:
            best_score, best_params = mean_score, params
    return best_params, best_score


def _inner_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    from sklearn.metrics import balanced_accuracy_score

    if len(np.unique(y_true)) < 2:
        return float((y_true == y_true[0]).mean())
    return float(balanced_accuracy_score(y_true, (proba >= 0.5).astype(int)))


def _positive_proba(pipeline: Pipeline, X: np.ndarray) -> np.ndarray:
    """Positive-class probability, robust to single-class training folds."""
    proba = pipeline.predict_proba(X)
    classes = list(getattr(pipeline, "classes_", []))
    if len(classes) == 1:
        return np.full(len(X), 1.0 if classes[0] == 1 else 0.0)
    return proba[:, classes.index(1)]


def evaluate_model_family(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    spec: ModelSpec,
    outer_folds: list,
    inner_splits: int = 3,
    random_state: int = SEED,
    n_resamples: int = 2000,
) -> dict:
    """Nested subject-grouped evaluation of one model family.

    Returns participant-level metrics (with subject-bootstrap CIs), fold
    variability, out-of-fold predictions keyed by original index, and the
    selected hyperparameters per outer fold.
    """
    y = np.asarray(y).astype(int)
    subjects = np.asarray(subjects)
    oof_proba = np.full(len(y), np.nan)
    oof_subjects = np.empty(len(y), dtype=object)
    fold_records: list[dict] = []
    fold_participant_scores: list[dict] = []

    for fold_number, (train_idx, eval_idx) in enumerate(outer_folds, start=1):
        assert_subject_disjoint(
            set(subjects[train_idx].tolist()), set(subjects[eval_idx].tolist()),
            context=f"outer fold {fold_number}",
        )
        X_train, y_train = X[train_idx], y[train_idx]
        groups_train = subjects[train_idx]

        best_params, best_inner_score = _fit_inner_selection(
            X_train, y_train, groups_train, spec, inner_splits, random_state
        )

        pipeline = _pipeline_for(spec, best_params, pca_variance=None if spec.name == "dummy_majority" else 0.95)
        pipeline.fit(X_train, y_train)

        if spec.name == "dummy_majority":
            # Constant predictor: probability of the positive class is the
            # training positive rate; ROC-AUC is undefined and excluded.
            positive_rate = float(y_train.mean())
            proba_eval = np.full(len(eval_idx), positive_rate)
        else:
            proba_eval = _positive_proba(pipeline, X[eval_idx])

        oof_proba[eval_idx] = proba_eval
        oof_subjects[eval_idx] = subjects[eval_idx]

        participants = aggregate_to_participants(subjects[eval_idx], y[eval_idx], proba_eval)
        fold_metrics = classification_metrics(
            participants["y_true"], participants["y_pred"], participants["mean_proba"],
            n_images=len(eval_idx),
        )
        fold_records.append(
            {
                "fold": fold_number,
                "n_train_images": int(len(train_idx)),
                "n_eval_images": int(len(eval_idx)),
                "n_eval_subjects": int(participants.shape[0]),
                "best_params": {k: (v.item() if hasattr(v, "item") else v) for k, v in best_params.items()},
                "inner_cv_score": best_inner_score,
                "participant_accuracy": fold_metrics.accuracy,
                "participant_balanced_accuracy": fold_metrics.balanced_accuracy,
                "participant_f1": fold_metrics.f1,
            }
        )
        fold_participant_scores.append(
            {
                "accuracy": fold_metrics.accuracy,
                "balanced_accuracy": fold_metrics.balanced_accuracy,
                "f1": fold_metrics.f1,
            }
        )

    if np.isnan(oof_proba).any():
        raise AssertionError("Some samples never received an out-of-fold prediction")

    participants = aggregate_to_participants(subjects, y, oof_proba)
    final = classification_metrics(
        participants["y_true"], participants["y_pred"], participants["mean_proba"], n_images=len(y)
    )
    final.ci = subject_bootstrap_ci(
        participants["y_true"], participants["y_pred"], participants["mean_proba"],
        n_resamples=n_resamples, random_state=random_state,
    )
    if spec.name == "dummy_majority":
        # A constant predictor has no ranking, so any "AUC" would be an
        # artifact of fold constants; report it as undefined.
        final.roc_auc = None
        final.ci.pop("roc_auc", None)

    return {
        "model": spec.name,
        "participant_metrics": final.to_dict(),
        "fold_variability": {
            metric: fold_variability([f[metric] for f in fold_participant_scores])
            for metric in ("accuracy", "balanced_accuracy", "f1")
        },
        "folds": fold_records,
        "oof": {
            "index": list(range(len(y))),
            "subject_id": subjects.tolist(),
            "y_true": y.tolist(),
            "proba": oof_proba.tolist(),
        },
        "participant_table": participants,
    }


def nested_subject_cv(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    model_specs: list[ModelSpec] | None = None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    random_state: int = SEED,
    n_resamples: int = 2000,
) -> dict:
    """Evaluate every requested model family under the same deterministic
    subject-grouped outer folds.

    The outer folds are fixed before any model is fitted, so differences
    between families are not driven by differing splits.  Note that
    *choosing* a model family from these outer results would itself be a
    selection on evaluation data; the honest reading is that all
    families' numbers share the same caveat and the dummy baseline is the
    reference point.
    """
    specs = model_specs if model_specs is not None else DEFAULT_MODEL_SPECS
    folds = subject_outer_folds(subjects, y, outer_splits, random_state)
    results = {"outer_splits": outer_splits, "inner_splits": inner_splits, "random_state": random_state, "models": {}}
    for spec in specs:
        try:
            results["models"][spec.name] = evaluate_model_family(
                X, y, subjects, spec, folds, inner_splits=inner_splits,
                random_state=random_state, n_resamples=n_resamples,
            )
        except ImportError as exc:
            results["models"][spec.name] = {"model": spec.name, "skipped": str(exc)}
    return results
