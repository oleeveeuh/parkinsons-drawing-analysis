"""Mixture-of-experts scaffold tests.

Assertions target *mechanics* (gate normalisation, expert slices, EM
convergence, leakage-safe integration) and an implementation validation
at ample sample size — not superiority over single models, which these
tests document as absent at small n.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"

from pd_drawing_analysis.moe import ExpertSpec, MixtureOfExperts
from pd_drawing_analysis.synthetic import two_regime_participants

TWO_EXPERTS = [
    ExpertSpec("lin", lambda: LogisticRegression(max_iter=5000, C=0.01)),
    ExpertSpec("tree", lambda: DecisionTreeClassifier(max_depth=3, random_state=0)),
]


@pytest.fixture(scope="module")
def small_fixture():
    return two_regime_participants(n_per_regime=12)


def test_gate_rows_normalised_and_mixture_valid(small_fixture):
    X, y, _, _ = small_fixture
    moe = MixtureOfExperts(experts=TWO_EXPERTS).fit(X, y)
    gate = moe.gate_weights(X)
    assert gate.shape == (len(y), 2)
    assert np.allclose(gate.sum(axis=1), 1.0)
    assert (gate >= 0).all()
    proba = moe.predict_proba(X)
    assert proba.shape == (len(y), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba >= 0) & (proba <= 1)).all()
    assert set(np.unique(moe.predict(X))) <= {0, 1}


def test_gate_localises_latent_regimes(small_fixture):
    """The signature dimensions make regimes recoverable; the gate must
    find them (mechanics check for the EM/kmeans initialisation)."""
    X, y, regimes, _ = small_fixture
    moe = MixtureOfExperts(experts=TWO_EXPERTS).fit(X, y)
    preferred = moe.gate_weights(X).argmax(axis=1)
    purity = max(
        float((preferred[regimes == r] == preferred[regimes == r][0]).mean())
        for r in np.unique(regimes)
    )
    assert purity >= 0.9


def test_expert_feature_slices_are_honoured():
    """Each expert must receive exactly its slice of the columns."""
    seen_widths = []

    class _WidthProbe:
        def fit(self, X, y, sample_weight=None):
            seen_widths.append(X.shape[1])
            self.classes_ = np.array([0, 1])
            self.p1_ = float(np.mean(y))
            return self

        def predict_proba(self, X):
            p = np.full(len(X), self.p1_)
            return np.column_stack([1 - p, p])

    X = np.zeros((20, 7))
    y = np.array([0, 1] * 10)
    experts = [
        ExpertSpec("left", lambda: _WidthProbe(), feature_slice=slice(0, 3)),
        ExpertSpec("right", lambda: _WidthProbe(), feature_slice=slice(3, 7)),
    ]
    MixtureOfExperts(experts=experts, n_em_steps=2).fit(X, y)
    assert seen_widths == [3, 4, 3, 4]  # per EM step: expert 0 gets cols 0-2, expert 1 gets cols 3-6


def test_single_class_training_fold_raises():
    X = np.zeros((10, 3))
    with pytest.raises(ValueError, match="both classes"):
        MixtureOfExperts(experts=TWO_EXPERTS).fit(X, np.zeros(10, dtype=int))


def test_unknown_init_rejected(small_fixture):
    X, y, _, _ = small_fixture
    with pytest.raises(ValueError, match="init"):
        MixtureOfExperts(experts=TWO_EXPERTS, init="bogus").fit(X, y)


def test_large_sample_validation_beats_weak_expert():
    """Implementation validation at n=1200 rows: the mixture must localise
    experts (high gate purity) and beat its weakest single expert.  This
    validates mechanics; it makes no claim about the real datasets."""
    X, y, regimes, _ = two_regime_participants(n_per_regime=300)
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    moe = MixtureOfExperts(experts=TWO_EXPERTS).fit(Xtr, ytr)
    weak = LogisticRegression(max_iter=5000, C=0.01).fit(Xtr, ytr)
    moe_acc = float((moe.predict(Xte) == yte).mean())
    weak_acc = float((weak.predict(Xte) == yte).mean())
    gate_purity = max(
        float((moe.gate_weights(X).argmax(axis=1)[regimes == r]
               == moe.gate_weights(X).argmax(axis=1)[regimes == r][0]).mean())
        for r in np.unique(regimes)
    )
    assert gate_purity >= 0.9
    assert moe_acc > weak_acc


def test_synthetic_script_end_to_end(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_moe_evaluation.py"), "--synthetic",
         "--n-resamples", "200", "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    import json

    payload = json.loads((tmp_path / "out" / "moe_synthetic.json").read_text())
    assert "moe_em" in payload["models"]
    for name, r in payload["models"].items():
        metrics = r["participant_metrics"]
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert len(r["oof"]["proba"]) == metrics["n_images"]


def test_task_expert_script_on_fixture(spatial_root, tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_moe_evaluation.py"), "--task-experts",
         "--spatial-root", str(spatial_root), "--n-resamples", "200",
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    import json

    payload = json.loads((tmp_path / "out" / "moe_task_experts_real.json").read_text())
    assert {"single_spiral", "single_wave", "pooled_logreg", "moe_task_experts"} <= set(payload["models"])
    assert len(payload["gate_mean_weights"]) == 2


def test_modality_expert_script_on_paired_fixtures(tmp_path):
    """Tabular+image expert pair on synthetic paired participants: verifies
    the feature-slice wiring end to end (each expert sees only its block)."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_moe_evaluation.py"), "--modality-experts",
         "--n-resamples", "200", "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    import json

    payload = json.loads((tmp_path / "out" / "moe_modality_experts_synthetic.json").read_text())
    assert {"single_tabular", "single_image", "pooled_logreg", "moe_modality_experts"} <= set(payload["models"])
    assert payload["dataset"]["blocks"] == {"tabular": 25, "image": payload["dataset"]["blocks"]["image"]}
    for name, res in payload["models"].items():
        oof = res["oof"]
        assert len(oof["proba"]) == payload["dataset"]["n_participants"]
        assert not np.isnan(np.asarray(oof["proba"])).any()
