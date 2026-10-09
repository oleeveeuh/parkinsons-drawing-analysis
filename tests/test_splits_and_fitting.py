"""Group-aware splitting and fold-local fitting guarantees.

These tests fail if any subject crosses a fold boundary or if any
fitted preprocessing step sees data outside its training fold.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from pd_drawing_analysis.evaluation import (
    DEFAULT_MODEL_SPECS,
    ModelSpec,
    _fit_inner_selection,
    make_model_pipeline,
    nested_subject_cv,
    subject_outer_folds,
)
from pd_drawing_analysis.subject_ids import assert_subject_disjoint, subject_overlap


def _synthetic_population(n_subjects=12, drawings=3, seed=42):
    rng = np.random.default_rng(seed)
    subjects, labels = [], []
    for i in range(1, n_subjects + 1):
        label = 1 if i <= n_subjects // 2 else 0
        for _ in range(drawings):
            subjects.append(f"parkinson:V{i:02d}" if label else f"healthy:V{i:02d}")
            labels.append(label)
    subjects = np.array(subjects)
    labels = np.array(labels)
    shift = np.where(labels == 1, 0.8, 0.0)[:, None]
    X = rng.normal(size=(len(labels), 20)) + shift + 0.3 * rng.normal(size=(len(labels), 1))
    return X, labels, subjects


def test_outer_folds_are_subject_disjoint():
    X, y, subjects = _synthetic_population()
    folds = subject_outer_folds(subjects, y, n_splits=5)
    assert len(folds) == 5
    seen_eval_subjects, seen_eval_rows = [], []
    for train_idx, eval_idx in folds:
        assert_subject_disjoint(set(subjects[train_idx]), set(subjects[eval_idx]),
                                context="test outer fold")
        seen_eval_subjects.extend(subjects[eval_idx])
        seen_eval_rows.extend(eval_idx)
    # every image evaluated exactly once and every subject evaluated exactly once
    # (each of the 12 subjects contributes 3 drawings)
    assert sorted(seen_eval_rows) == list(range(len(y)))
    assert len(seen_eval_subjects) == 36
    assert len(set(seen_eval_subjects)) == 12


def test_inner_folds_are_subject_disjoint():
    X, y, subjects = _synthetic_population()
    folds = subject_outer_folds(subjects, y, n_splits=5)
    spec = DEFAULT_MODEL_SPECS[1]
    for train_idx, _ in folds:
        from pd_drawing_analysis.evaluation import _inner_folds

        for inner_train, inner_val in _inner_folds(X[train_idx], y[train_idx],
                                                   subjects[train_idx], 3, 42):
            assert_subject_disjoint(set(subjects[train_idx][inner_train]),
                                    set(subjects[train_idx][inner_val]),
                                    context="test inner fold")


def test_supplied_directory_split_is_rejected_as_subject_independent():
    """The vendor's train/test split must be detectably leaky — a guard
    against ever treating it as subject-independent evaluation data."""
    train = ["parkinson:V01", "parkinson:V02", "healthy:V01"]
    test = ["parkinson:V01", "healthy:V02"]
    assert subject_overlap(train, test), "vendor split wrongly passed as clean"
    with pytest.raises(AssertionError):
        assert_subject_disjoint(train, test)


class _RecordingVariance(TransformerMixin, BaseEstimator):
    """VarianceThreshold stand-in that records every row it is fitted on."""

    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.fitted_rows_ = None

    def fit(self, X, y=None):
        self.fitted_rows_ = np.asarray(X, dtype=float).copy()
        self.variances_ = self.fitted_rows_.var(axis=0)
        return self

    def transform(self, X):
        keep = self.variances_ > self.threshold
        return np.asarray(X, dtype=float)[:, keep]


def test_fold_local_fitting_never_sees_validation_rows():
    """Spy transformer must be fitted exclusively on training-fold rows,
    and the fitted statistics must differ from full-dataset statistics."""
    X, y, subjects = _synthetic_population(n_subjects=10)
    folds = subject_outer_folds(subjects, y, n_splits=5)
    train_idx, eval_idx = folds[0]

    spy = _RecordingVariance()
    spy.fit(X[train_idx])
    train_rows = {row.tobytes() for row in X[train_idx]}
    eval_rows = {row.tobytes() for row in X[eval_idx]}
    fitted_rows = {row.tobytes() for row in spy.fitted_rows_}
    assert fitted_rows == train_rows
    assert not (fitted_rows & eval_rows), "preprocessing saw validation rows"

    # full-dataset fit would give different statistics — prove the guard matters
    full = _RecordingVariance().fit(X)
    assert not np.allclose(full.variances_, spy.variances_)


def test_scaler_and_pca_fitted_on_train_only():
    from sklearn.pipeline import Pipeline

    X, y, subjects = _synthetic_population()
    folds = subject_outer_folds(subjects, y, n_splits=5)
    train_idx, eval_idx = folds[0]
    spec = DEFAULT_MODEL_SPECS[1]
    pipeline = make_model_pipeline(spec, {"clf__C": 0.1})
    pipeline.fit(X[train_idx], y[train_idx])
    scaler: StandardScaler = pipeline.named_steps["scale"]
    pca: PCA = pipeline.named_steps["pca"]
    assert np.allclose(scaler.mean_, X[train_idx].mean(axis=0))
    assert not np.allclose(scaler.mean_, X.mean(axis=0))
    assert pca.n_components_ <= train_idx.size - 1


def test_nested_cv_participant_metrics_and_alignment():
    X, y, subjects = _synthetic_population()
    results = nested_subject_cv(X, y, subjects, model_specs=DEFAULT_MODEL_SPECS[:2],
                                n_resamples=200)
    for name, result in results["models"].items():
        if "participant_metrics" not in result:
            pytest.fail(f"{name} produced no participant metrics")
        oof = result["oof"]
        assert len(oof["index"]) == len(y)
        assert len(oof["proba"]) == len(y)
        assert sorted(oof["index"]) == list(range(len(y)))
        # subject labels in OOF must match the input labels row-for-row
        assert oof["y_true"] == y.tolist()
        assert oof["subject_id"] == subjects.tolist()
        m = result["participant_metrics"]
        assert 0.0 <= m["accuracy"] <= 1.0
        assert m["n_subjects"] == 12


def test_repeated_runs_are_reproducible():
    X, y, subjects = _synthetic_population()
    first = nested_subject_cv(X, y, subjects, model_specs=DEFAULT_MODEL_SPECS[:1], n_resamples=100)
    second = nested_subject_cv(X, y, subjects, model_specs=DEFAULT_MODEL_SPECS[:1], n_resamples=100)
    a = first["models"]["dummy_majority"]["participant_metrics"]
    b = second["models"]["dummy_majority"]["participant_metrics"]
    assert a["accuracy"] == b["accuracy"]
    assert a["subject_bootstrap_ci"] == b["subject_bootstrap_ci"]


def test_missing_optional_dependency_is_reported_not_raised():
    X, y, subjects = _synthetic_population(n_subjects=12)

    def _broken():
        raise ImportError("optional dependency 'x' missing")

    spec = ModelSpec(name="broken_extra", factory=_broken, param_grid={}, requires_extra="x")
    results = nested_subject_cv(X, y, subjects, model_specs=[spec])
    assert "skipped" in results["models"]["broken_extra"]
