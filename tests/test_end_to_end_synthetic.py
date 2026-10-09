"""Deterministic-feature unit checks and end-to-end synthetic evaluations."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"

from pd_drawing_analysis.data import inventory_spatial, inventory_temporal
from pd_drawing_analysis.features import (
    SHAPE_FEATURE_NAMES,
    extract_spatial_features,
    extract_temporal_features,
)
from pd_drawing_analysis.data import load_image_gray


def test_shape_features_deterministic_and_finite(spatial_root):
    path = next((spatial_root / "spiral" / "training" / "parkinson").glob("*.png"))
    a = extract_spatial_features(path)
    b = extract_spatial_features(path)
    assert a.shape == b.shape
    assert np.array_equal(a, b), "feature extraction must be deterministic"
    assert np.isfinite(a).all()
    assert a.shape[1] if a.ndim > 1 else len(a) > len(SHAPE_FEATURE_NAMES)


def test_impaired_and_healthy_features_differ(spatial_root):
    pd_img = next((spatial_root / "spiral" / "training" / "parkinson").glob("*.png"))
    hc_img = next((spatial_root / "spiral" / "training" / "healthy").glob("*.png"))
    assert not np.array_equal(extract_spatial_features(pd_img), extract_spatial_features(hc_img))


def test_temporal_features_finite(temporal_root):
    inv = inventory_temporal(temporal_root)
    from pd_drawing_analysis.data import load_temporal_recording

    feats = [extract_temporal_features(load_temporal_recording(p)) for p in inv.frame["path"]]
    matrix = np.vstack(feats)
    assert np.isfinite(matrix).all()
    assert matrix.shape[0] == 12


def test_z_features_excluded_by_default(temporal_root):
    from pd_drawing_analysis.data import load_temporal_recording

    rec = load_temporal_recording(next(temporal_root.glob("PWP (1).txt")))
    assert extract_temporal_features(rec).shape[0] == 25
    assert extract_temporal_features(rec, include_z=True).shape[0] == 28


def test_synthetic_end_to_end_temporal(temporal_root, tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_temporal_evaluation.py"),
         "--data-root", str(temporal_root), "--n-resamples", "200",
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    artifact = tmp_path / "out" / "temporal_real.json"
    assert artifact.exists()
    import json

    payload = json.loads(artifact.read_text())
    assert payload["dataset"]["n_subjects"] == 12
    for name, model in payload["models"].items():
        assert "participant_metrics" in model or "skipped" in model
    assert "dummy_majority" in payload["models"]


def test_synthetic_end_to_end_spatial(spatial_root, tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_spatial_evaluation.py"),
         "--data-root", str(spatial_root), "--n-resamples", "200",
         "--output-dir", str(tmp_path / "out")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    artifact = tmp_path / "out" / "spatial_real_hog_shape.json"
    assert artifact.exists()
    import json

    payload = json.loads(artifact.read_text())
    assert set(payload["tasks"]) == {"spiral", "wave"}
    assert payload["dataset"]["shared_subjects_across_tasks"] == 12
    for task, block in payload["tasks"].items():
        metrics = block["models"]["logreg_balanced"]["participant_metrics"]
        assert metrics["n_subjects"] == 12
    # shared outer folds + paired comparison present
    assert payload["protocol"]["shared_outer_folds_across_tasks"] is True
    assert "task_comparison" in payload


def test_audit_script_runs_on_fixtures(spatial_root, temporal_root, tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "audit_data.py"),
         "--spiral-root", str(spatial_root), "--temporal-root", str(temporal_root),
         "--output", str(tmp_path / "audit.json")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    import json

    payload = json.loads((tmp_path / "audit.json").read_text())
    assert payload["spatial"]["summary"]["shared_subjects_across_tasks"] == 12
    assert payload["temporal"]["summary"]["n_recordings"] == 12


def test_env_info_script():
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "env_info.py")],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    import json

    info = json.loads(result.stdout)
    assert "python" in info and "packages" in info
