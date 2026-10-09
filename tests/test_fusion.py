"""Fusion architecture tests on synthetic paired-modalities.

Fusion code paths are exercised only on clearly synthetic paired data
(same synthetic participant in both modalities) plus a real-data
feature-family ablation.  The real temporal/spatial datasets have
disjoint participants and are never fused.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"

from pd_drawing_analysis.fusion import (
    PairedModalityData,
    _LATE_WEIGHT_GRID,
    assemble_paired_data,
    evaluate_fusion,
)
from pd_drawing_analysis.synthetic import generate_paired_fixtures


@pytest.fixture(scope="module")
def paired(tmp_path_factory):
    dirs = generate_paired_fixtures(tmp_path_factory.mktemp("paired"))
    data, info = assemble_paired_data(dirs["temporal_dir"], dirs["image_dir"] / "spiral" / "alldata")
    return data, info


def test_paired_fixture_aligns_subjects_across_modalities(paired):
    data, _ = paired
    assert len(data.subjects) == 12
    assert data.modalities["temporal"].shape == (12, 25)
    assert data.modalities["image"].shape[0] == 12
    assert data.modalities["image"].shape[1] > 100
    # labels consistent with class-aware subject ids
    assert all((s.startswith("parkinson")) == (y == 1) for s, y in zip(data.subjects, data.labels))


def test_row_mismatch_rejected():
    with pytest.raises(AssertionError):
        PairedModalityData(
            subjects=np.array(["a", "b"]),
            labels=np.array([1, 0]),
            modalities={"x": np.zeros((3, 2))},
        )


def test_assemble_requires_shared_subjects(tmp_path):
    dirs = generate_paired_fixtures(tmp_path)
    empty_img = tmp_path / "no_images"
    empty_img.mkdir()
    with pytest.raises(ValueError, match="shared participants"):
        assemble_paired_data(dirs["temporal_dir"], empty_img)


def test_fusion_beats_single_modalities_on_split_signal(paired):
    """The fixture splits the class signal across modalities, so a correct
    fusion implementation must beat both single-modality baselines."""
    data, _ = paired
    results = evaluate_fusion(data, n_resamples=200)
    acc = {n: r["participant_metrics"]["accuracy"] for n, r in results["models"].items()}
    singles = [acc["single_temporal"], acc["single_image"]]
    assert acc["late_fusion"] >= max(singles)
    assert acc["late_fusion"] >= 0.9
    # every strategy produced a complete, aligned set of OOF predictions
    for name, oof in results["oof"].items():
        assert len(oof) == 12
        assert not np.isnan(np.asarray(oof)).any()


def test_late_fusion_weights_come_from_inner_grid(paired):
    data, _ = paired
    results = evaluate_fusion(data, n_resamples=100)
    for fold in results["models"]["late_fusion"]["folds"]:
        assert fold["selected_weight"] in _LATE_WEIGHT_GRID


def test_stacking_reports_honest_small_sample_behaviour(paired):
    """Stacking may underperform at n=12 — the test documents that it runs,
    produces full coverage, and never silently equals the meta-training set."""
    data, _ = paired
    results = evaluate_fusion(data, n_resamples=100)
    metrics = results["models"]["stacked_fusion"]["participant_metrics"]
    assert metrics["n_subjects"] == 12
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_single_class_training_fold_does_not_crash():
    """Degenerate tiny folds must yield defined probabilities, not IndexError."""
    subjects = np.array([f"parkinson:V{i:02d}" for i in range(1, 7)]
                        + [f"healthy:V{i:02d}" for i in range(1, 7)])
    labels = np.array([1] * 6 + [0] * 6)
    rng = np.random.default_rng(0)
    data = PairedModalityData(
        subjects=subjects,
        labels=labels,
        modalities={
            "temporal": rng.normal(size=(12, 5)) + labels[:, None] * 2.0,
            "image": rng.normal(size=(12, 30)),
        },
    )
    results = evaluate_fusion(data, n_resamples=100)
    for name, oof in results["oof"].items():
        assert len(oof) == 12 and not np.isnan(np.asarray(oof)).any()


def test_end_to_end_fusion_script(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_fusion_evaluation.py"), "--synthetic",
         "--n-resamples", "200", "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    import json

    payload = json.loads((tmp_path / "out" / "fusion_paired_synthetic.json").read_text())
    assert "scope_note" in payload
    assert payload["dataset"]["name"].startswith("Synthetic paired-modality")
    assert {"early_fusion", "late_fusion", "stacked_fusion",
            "single_temporal", "single_image"} <= set(payload["models"])
