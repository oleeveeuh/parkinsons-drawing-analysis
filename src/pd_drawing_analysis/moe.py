"""Mixture-of-experts scaffold with leakage-safe nested-CV evaluation.

Implements the classical Jacobs–Jordan mixture of experts for binary
classification, adapted to scikit-learn estimators:

* **Experts** — K probabilistic classifiers, optionally restricted to
  their own feature slice (so one expert can specialise in one task's
  features and another in a different block, e.g. spiral vs wave).
* **Gate** — a multinomial softmax regression over the full feature
  vector producing per-sample expert weights.
* **EM training** (two steps per iteration, all on training rows):
    * M-step: refit each expert weighted by its responsibilities;
      refit the gate on responsibility-derived pseudo-labels.
    * E-step: responsibilities  r_ik ∝ g_k(x_i) · p_k(y_i | x_i),
      normalised per row (uniform fallback when a row underflows, and an
      unweighted refit fallback when an expert's responsibility mass
      vanishes — both documented rather than silent).

Scope and honesty notes
-----------------------
* The MoE consumes *per-participant* rows and is evaluated inside the
  same subject-grouped nested CV as everything else; it never sees
  evaluation rows during EM.
* With 28–40 participants a MoE has far more parameters than data; the
  expected real-data outcome is parity with a single regularised model
  at best.  The module exists to scaffold and test the *architecture*
  (fixture where specialisation is required, real per-task-expert run),
  not to claim accuracy gains.  The repository name's historical "MoE"
  referred to a removed v1; this module is a clean-room rebuild.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from .evaluation import SEED

__all__ = [
    "ExpertSpec",
    "MixtureOfExperts",
    "DEFAULT_EXPERTS",
]

_RESP_FLOOR = 1e-9


@dataclass(frozen=True)
class ExpertSpec:
    """One expert: a factory plus (optionally) its feature slice."""

    name: str
    factory: Callable[[], BaseEstimator]
    feature_slice: slice | None = None  # None = all features


def _logreg_expert(c: float) -> Callable[[], LogisticRegression]:
    return lambda: LogisticRegression(max_iter=5000, C=c)


DEFAULT_EXPERTS = [
    ExpertSpec("logreg_strong_l2", _logreg_expert(0.01)),
    ExpertSpec("logreg_weak_l2", _logreg_expert(1.0)),
    ExpertSpec("tree_depth3", lambda: DecisionTreeClassifier(max_depth=3, random_state=SEED)),
]

# Per-task experts (e.g. spiral vs wave) are constructed with explicit
# feature slices at the call site, once the task-block widths are known.


class MixtureOfExperts(ClassifierMixin, BaseEstimator):
    """Softmax-gated mixture of probabilistic experts trained by EM.

    Parameters
    ----------
    experts : list[ExpertSpec]
        Expert definitions (name, estimator factory, optional slice).
    n_em_steps : int
        Number of EM iterations (small grid values only; the dataset is tiny).
    gate_C : float
        Inverse regularisation of the softmax gate.
    feature_slices : list[slice] | None
        Optional per-expert slices of the input columns; ``None`` uses
        every expert's own ``feature_slice``.
    """

    def __init__(
        self,
        experts: list[ExpertSpec] | None = None,
        n_em_steps: int = 5,
        gate_C: float = 1.0,
        init: str = "kmeans",
        random_state: int = SEED,
    ):
        self.experts = experts
        self.n_em_steps = n_em_steps
        self.gate_C = gate_C
        self.init = init
        self.random_state = random_state

    # ------------------------------------------------------------------ #
    def _resolve_experts(self):
        return self.experts if self.experts is not None else DEFAULT_EXPERTS

    def _fit_expert(self, spec: ExpertSpec, X: np.ndarray, y: np.ndarray, weights: np.ndarray):
        X_expert = X[:, spec.feature_slice] if spec.feature_slice is not None else X
        model = spec.factory()
        if float(weights.sum()) < 1e-6:
            # An expert lost all responsibility mass; refit unweighted rather
            # than crash (documented fallback, deterministic).
            model.fit(X_expert, y)
        else:
            model.fit(X_expert, y, sample_weight=weights)
        return model

    def _expert_positive_proba(self, expert_model, spec: ExpertSpec, X: np.ndarray) -> np.ndarray:
        X_expert = X[:, spec.feature_slice] if spec.feature_slice is not None else X
        proba = expert_model.predict_proba(X_expert)
        classes = list(expert_model.classes_)
        if len(classes) == 1:
            return np.full(len(X), 1.0 if classes[0] == 1 else 0.0)
        return proba[:, classes.index(1)]

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("MixtureOfExperts requires both classes in the training fold")
        experts = self._resolve_experts()
        k = len(experts)
        n = len(y)

        rng = np.random.default_rng(self.random_state)
        # Initial responsibilities. Uniform init makes every expert identical
        # at the first M-step and EM collapses onto whichever expert scores
        # the highest likelihood (observed empirically). Following the local
        # -experts literature, the default partitions the INPUT space first:
        # k-means clusters seed responsibilities so experts start localised.
        if self.init == "kmeans":
            from sklearn.cluster import KMeans

            partitions = KMeans(n_clusters=k, n_init=10, random_state=self.random_state).fit_predict(X)
            responsibilities = np.eye(k)[partitions]
        elif self.init == "uniform":
            responsibilities = np.full((n, k), 1.0 / k)
        else:
            raise ValueError(f"Unknown init {self.init!r}; use 'kmeans' or 'uniform'")
        gate: LogisticRegression | None = None
        fitted_experts: list = []

        for step in range(int(self.n_em_steps)):
            # --- M-step: experts weighted by responsibilities ---
            fitted_experts = [
                self._fit_expert(spec, X, y, responsibilities[:, i])
                for i, spec in enumerate(experts)
            ]
            # --- E-step: responsibilities from gate × expert likelihood ---
            expert_like = np.column_stack([
                self._expert_positive_proba(model, spec, X)
                for model, spec in zip(fitted_experts, experts)
            ])
            likelihood = np.where(y[:, None] == 1, expert_like, 1.0 - expert_like)
            likelihood = np.clip(likelihood, _RESP_FLOOR, None)
            if gate is None:
                gate_probs = np.full((n, k), 1.0 / k)
            else:
                gate_probs = self._gate_probs_aligned(gate, X, k)
            raw = gate_probs * likelihood
            row_sums = raw.sum(axis=1, keepdims=True)
            responsibilities = np.where(row_sums > _RESP_FLOOR, raw / np.maximum(row_sums, _RESP_FLOOR),
                                        1.0 / k)
            # --- M-step: gate on responsibility pseudo-labels ---
            pseudo = responsibilities.argmax(axis=1)
            if len(np.unique(pseudo)) < 2:
                # Degenerate: every row prefers one expert. Keep the gate
                # degenerate-but-defined so predict_proba stays valid.
                pseudo = pseudo.copy()
                pseudo[rng.integers(0, n)] = 1 - pseudo[0]
            gate = LogisticRegression(max_iter=5000, C=self.gate_C)
            gate.fit(X, pseudo, sample_weight=responsibilities.max(axis=1))

        self.experts_ = fitted_experts
        self.expert_specs_ = experts
        self.gate_ = gate
        self.classes_ = np.array([0, 1])
        self.responsibilities_ = responsibilities
        self.n_iter_ = int(self.n_em_steps)
        return self

    # ------------------------------------------------------------------ #
    def _gate_probs_aligned(self, gate: LogisticRegression, X: np.ndarray, k: int) -> np.ndarray:
        """Gate probabilities realigned to expert indices.

        The softmax may cover fewer classes than experts on a given
        iteration (an expert can lose every pseudo-label); missing
        experts get zero weight so shapes stay (n, k) and rows sum to 1.
        """
        raw = gate.predict_proba(X)
        n = len(X)
        col_by_class = {int(c): i for i, c in enumerate(gate.classes_)}
        return np.column_stack([
            raw[:, col_by_class[k_idx]] if k_idx in col_by_class else np.zeros(n)
            for k_idx in range(k)
        ])

    def gate_weights(self, X: np.ndarray) -> np.ndarray:
        """Per-sample softmax gate distribution over experts (rows sum to 1)."""
        check_is_fitted(self, "gate_")
        k = len(self.experts_)
        return self._gate_probs_aligned(self.gate_, np.asarray(X, dtype=float), k)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "experts_")
        X = np.asarray(X, dtype=float)
        gate_probs = self.gate_weights(X)
        expert_p1 = np.column_stack([
            self._expert_positive_proba(model, spec, X)
            for model, spec in zip(self.experts_, self.expert_specs_)
        ])
        mixture_p1 = np.sum(gate_probs * expert_p1, axis=1)
        mixture_p1 = np.clip(mixture_p1, 0.0, 1.0)
        return np.column_stack([1.0 - mixture_p1, mixture_p1])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
