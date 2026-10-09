"""Participant aggregation, uncertainty quantification, and baselines."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from pd_drawing_analysis.evaluation import DEFAULT_MODEL_SPECS, nested_subject_cv
from pd_drawing_analysis.metrics import (
    aggregate_to_participants,
    classification_metrics,
    fold_variability,
    paired_sign_flip_test,
    subject_bootstrap_ci,
)


def _participant_frame():
    subjects = np.array(["parkinson:V01", "parkinson:V01", "healthy:V01", "healthy:V02"])
    y_true = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.6, 0.2, 0.4])
    return subjects, y_true, proba


def test_aggregation_treats_drawings_as_one_participant():
    subjects, y_true, proba = _participant_frame()
    agg = aggregate_to_participants(subjects, y_true, proba)
    assert len(agg) == 3  # three participants, not four drawings
    v01 = agg[agg.subject_id == "parkinson:V01"].iloc[0]
    assert v01.n_drawings == 2
    assert v01.mean_proba == pytest.approx(0.75)
    assert v01.y_pred == 1


def test_aggregation_detects_misalignment():
    subjects, y_true, proba = _participant_frame()
    with pytest.raises(AssertionError):
        aggregate_to_participants(subjects, y_true[:-1], proba)


def test_aggregation_detects_conflicting_subject_labels():
    subjects = np.array(["parkinson:V01", "parkinson:V01"])
    with pytest.raises(AssertionError, match="conflicting labels"):
        aggregate_to_participants(subjects, np.array([1, 0]), np.array([0.2, 0.8]))


def test_bootstrap_ci_reproducible_and_participant_level():
    y_true = np.array([1] * 8 + [0] * 7)
    y_pred = np.array([1] * 6 + [0] * 2 + [0] * 6 + [1] * 1)
    y_score = np.array([0.9, 0.8, 0.7, 0.65, 0.6, 0.55, 0.4, 0.3, 0.2, 0.3, 0.25, 0.6, 0.8, 0.5, 0.4])
    ci_a = subject_bootstrap_ci(y_true, y_pred, y_score, n_resamples=500, random_state=7)
    ci_b = subject_bootstrap_ci(y_true, y_pred, y_score, n_resamples=500, random_state=7)
    assert ci_a["accuracy"] == ci_b["accuracy"], "same seed must reproduce intervals"
    lo, hi = ci_a["accuracy"]
    assert 0.0 <= lo <= hi <= 1.0


def test_fold_std_is_not_a_confidence_interval():
    """Fold variability must be reported as such — a distinct structure
    from bootstrap CIs so it can never be mislabelled downstream."""
    variability = fold_variability([0.8, 0.6, 0.7, 0.9, 0.5])
    assert set(variability) == {"mean", "std", "values"}
    assert variability["std"] != 0
    assert "ci" not in variability and "confidence" not in variability


def test_fold_std_clipping_forbidden():
    """Accuracy scores must pass through unclipped, even above 1.0 bounds
    of a naive interval; the API simply never produces [mean±std] intervals."""
    variability = fold_variability([1.0, 1.0, 1.0])
    assert variability["mean"] == 1.0 and variability["std"] == 0.0


def test_paired_sign_flip_test():
    rng = np.random.default_rng(0)
    base = rng.random(28)
    scores_a = base
    scores_b = base + 0.05  # consistent small advantage
    result = paired_sign_flip_test(scores_a, scores_b, n_permutations=1000, random_state=1)
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_pairs"] == 28
    identical = paired_sign_flip_test(base, base)
    assert identical["p_value"] == 1.0  # no difference -> never "significant"


def test_dummy_baseline_evaluation():
    """Majority-class dummy: with a 15/5 split, every fold's training
    majority is the positive class, so pooled participant accuracy is
    exactly the majority share, balanced accuracy 0.5, no AUC."""
    X = np.zeros((20, 2))
    y = np.array([1] * 15 + [0] * 5)
    subjects = np.array([f"s{i:02d}" for i in range(20)])
    results = nested_subject_cv(X, y, subjects, model_specs=[DEFAULT_MODEL_SPECS[0]])
    metrics = results["models"]["dummy_majority"]["participant_metrics"]
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["balanced_accuracy"] == pytest.approx(0.5)
    assert metrics["roc_auc"] is None
    assert metrics["confusion_matrix"] == [[0, 5], [0, 15]]


def test_classification_metrics_sensitivity_specificity():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 1, 0, 0])
    m = classification_metrics(y_true, y_pred, y_true.astype(float))
    assert m.sensitivity == pytest.approx(0.75)
    assert m.specificity == pytest.approx(0.75)
    assert m.confusion_matrix == [[3, 1], [1, 3]]
    assert accuracy_score(y_true, y_pred) == pytest.approx(m.accuracy)
