"""Participant-level metrics and honest uncertainty quantification.

Design rules enforced here:

* Primary metrics are computed at the **participant level**, after
  aggregating each participant's drawing-level out-of-fold predictions.
* Confidence intervals come from a deterministic **subject-level
  bootstrap** (participants are resampled, never individual drawings).
  They are distinct from cross-validated fold variability, which is
  reported separately and never labelled a confidence interval.
* Comparisons between tasks use a **paired sign-flip permutation test**
  over the participants who completed both tasks, and are framed as
  exploratory.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

__all__ = [
    "ParticipantMetrics",
    "aggregate_to_participants",
    "classification_metrics",
    "subject_bootstrap_ci",
    "fold_variability",
    "paired_sign_flip_test",
    "metrics_table",
]


@dataclass
class ParticipantMetrics:
    """Metrics computed from participant-level out-of-fold predictions."""

    n_subjects: int
    n_images: int
    accuracy: float
    balanced_accuracy: float
    f1: float
    roc_auc: float | None
    sensitivity: float
    specificity: float
    confusion_matrix: list
    ci: dict = field(default_factory=dict)  # metric -> (lo, hi), subject bootstrap

    def to_dict(self) -> dict:
        out = {
            "n_subjects": self.n_subjects,
            "n_images": self.n_images,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "confusion_matrix": self.confusion_matrix,
        }
        out["subject_bootstrap_ci"] = {
            k: [float(v[0]), float(v[1])] for k, v in self.ci.items()
        }
        return out


def aggregate_to_participants(
    subjects: np.ndarray,
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate drawing-level out-of-fold predictions to participant level.

    Each participant's predicted probability is the mean of their
    drawings' out-of-fold positive-class probabilities; the participant
    prediction is that mean thresholded at 0.5.  Alignment between
    subjects, labels and probabilities is asserted.
    """
    subjects = np.asarray(subjects)
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    if not (len(subjects) == len(y_true) == len(proba)):
        raise AssertionError(
            f"Prediction misalignment: {len(subjects)} subjects, "
            f"{len(y_true)} labels, {len(proba)} probabilities"
        )
    frame = pd.DataFrame(
        {"subject_id": subjects, "y_true": y_true, "proba": proba}
    )
    # A subject must have a single true label across all their drawings.
    label_conflicts = frame.groupby("subject_id")["y_true"].nunique()
    if (label_conflicts > 1).any():
        bad = label_conflicts[label_conflicts > 1].index.tolist()
        raise AssertionError(f"Subjects with conflicting labels: {bad[:10]}")
    agg = (
        frame.groupby("subject_id", sort=True)
        .agg(y_true=("y_true", "first"), mean_proba=("proba", "mean"), n_drawings=("proba", "size"))
        .reset_index()
    )
    agg["y_pred"] = (agg["mean_proba"] >= threshold).astype(int)
    return agg


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_images: int | None = None,
) -> ParticipantMetrics:
    """Participant-level metrics.  ROC-AUC is reported only when both
    classes are present (otherwise ``None``)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    auc: float | None
    if len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y_true, y_score))
    return ParticipantMetrics(
        n_subjects=int(len(y_true)),
        n_images=int(n_images if n_images is not None else len(y_true)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=auc,
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
    )


def _metric_value(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> float | None:
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    if name == "f1":
        return float(f1_score(y_true, y_pred, zero_division=0))
    if name == "sensitivity":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return float(tp / (tp + fn)) if (tp + fn) else 0.0
    if name == "specificity":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return float(tn / (tn + fp)) if (tn + fp) else 0.0
    if name == "roc_auc":
        if len(np.unique(y_true)) < 2 or len(np.unique(y_score)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    raise KeyError(name)


_METRIC_NAMES = ("accuracy", "balanced_accuracy", "f1", "sensitivity", "specificity", "roc_auc")


def subject_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Deterministic stratified **subject-level** bootstrap confidence
    intervals for participant-level metrics.

    Participants are resampled with replacement, preserving each class's
    count, so class structure is retained.  Resampling participants (not
    drawings) reflects that participants — not images — are the units of
    interest.  Percentile intervals are returned.

    Limitations (documented, not hidden): the bootstrap quantifies
    sampling variability of the point estimate over the participant
    distribution; it cannot overcome the small number of participants,
    and it does not account for the adaptivity of model selection.
    """
    rng = np.random.default_rng(random_state)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    idx_class0 = np.flatnonzero(y_true == 0)
    idx_class1 = np.flatnonzero(y_true == 1)
    if idx_class0.size == 0 or idx_class1.size == 0:
        return {}

    scores: dict[str, list[float]] = {name: [] for name in _METRIC_NAMES}
    for _ in range(n_resamples):
        sample = np.concatenate(
            [
                rng.choice(idx_class0, size=idx_class0.size, replace=True),
                rng.choice(idx_class1, size=idx_class1.size, replace=True),
            ]
        )
        for name in _METRIC_NAMES:
            value = _metric_value(name, y_true[sample], y_pred[sample], y_score[sample])
            if value is not None:
                scores[name].append(value)

    alpha = 1.0 - confidence_level
    ci = {}
    for name, values in scores.items():
        if len(values) < n_resamples * 0.5:
            continue  # metric undefined in most resamples; do not report
        ci[name] = (
            float(np.percentile(values, 100 * alpha / 2)),
            float(np.percentile(values, 100 * (1 - alpha / 2))),
        )
    return ci


def fold_variability(per_fold_scores: list[float]) -> dict:
    """Mean and standard deviation of per-outer-fold participant-level
    scores.

    Fold standard deviation describes *variability across folds* on this
    dataset.  It is NOT a confidence interval and must never be presented
    as one; folds are not independent observations.
    """
    values = [float(v) for v in per_fold_scores if np.isfinite(v)]
    if not values:
        return {"mean": None, "std": None, "values": []}
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)), "values": values}


def paired_sign_flip_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 42,
) -> dict:
    """Two-sided paired sign-flip permutation test on per-participant
    scores (e.g. per-participant correctness under two tasks).

    Appropriate for small paired samples where a t-test's normality
    assumption is untenable.  The result is exploratory: with tens of
    participants only large effects can be detected.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if scores_a.shape != scores_b.shape:
        raise AssertionError("Paired test requires aligned per-participant scores")
    diff = scores_a - scores_b
    observed = float(abs(diff.mean()))
    rng = np.random.default_rng(random_state)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, diff.size))
    null = np.abs((signs * diff).mean(axis=1))
    p_value = float((np.sum(null >= observed - 1e-12) + 1) / (n_permutations + 1))
    return {
        "mean_difference": float(diff.mean()),
        "observed_statistic": observed,
        "p_value": p_value,
        "n_pairs": int(diff.size),
        "n_permutations": n_permutations,
        "random_state": random_state,
    }


def metrics_table(results: dict[str, ParticipantMetrics]) -> str:
    """Render a markdown table of participant-level metrics per model."""
    header = (
        "| Model | Accuracy | Bal. accuracy | F1 | ROC-AUC | Sens. | Spec. |"
        "\n|---|---|---|---|---|---|---|"
    )
    rows = []
    for name, m in results.items():
        auc = f"{m.roc_auc:.3f}" if m.roc_auc is not None else "n/a"
        rows.append(
            f"| {name} | {m.accuracy:.3f} | {m.balanced_accuracy:.3f} | {m.f1:.3f} "
            f"| {auc} | {m.sensitivity:.3f} | {m.specificity:.3f} |"
        )
    return "\n".join([header, *rows])
