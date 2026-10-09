"""Dataset inventory, input validation, and vendor-split leakage detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pd_drawing_analysis.data import (
    inventory_spatial,
    inventory_temporal,
    load_image_gray,
    load_temporal_recording,
    supplied_split_leakage_table,
)
from pd_drawing_analysis.subject_ids import subject_overlap


def test_spatial_inventory_counts(spatial_root):
    inv = inventory_spatial(spatial_root)
    # Each drawing is written to both vendor splits (mirroring the real
    # dataset's participant repetition): 2 tasks x 12 subjects x 2 drawings x 2 splits = 96
    assert len(inv.frame) == 96
    assert inv.frame["subject_id"].nunique() == 12
    assert inv.frame.groupby("task")["subject_id"].nunique().eq(12).all()
    # both classes present in every task
    assert inv.frame.groupby("task")["label"].nunique().eq(2).all()


def test_spatial_vendor_split_overlap_detected(spatial_root):
    """The generator mirrors the vendor layout, so the audit must flag the
    overlap between supplied training/testing subjects."""
    inv = inventory_spatial(spatial_root)
    table = supplied_split_leakage_table(inv)
    overlap = table["shared_subjects"].sum()
    assert overlap > 0, "audit failed to detect vendor split leakage"


def test_temporal_inventory_counts(temporal_root):
    inv = inventory_temporal(temporal_root)
    assert len(inv.frame) == 12  # 6 PWP + 6 Healthy
    assert inv.frame["subject_id"].nunique() == 12
    assert inv.labels.sum() == 6


def test_temporal_filename_without_class_rejected(tmp_path):
    (tmp_path / "Unknown (1).txt").write_text("1;2;3;4;5;6\n")
    with pytest.raises(ValueError, match="Cannot infer class"):
        inventory_temporal(tmp_path)


def test_temporal_recording_validation(temporal_root):
    inv = inventory_temporal(temporal_root)
    df = load_temporal_recording(inv.frame["path"].iloc[0])
    assert list(df.columns)[:6] == ["x", "y", "z", "pressure", "grip_angle", "timestamp"]
    assert (np.diff(df["timestamp"].to_numpy()) > 0).all()


def test_temporal_recording_rejects_malformed(tmp_path):
    bad = tmp_path / "PWP (1).txt"
    bad.write_text("1;2;3\n4;5;6\n")  # too few columns
    with pytest.raises(ValueError, match="columns"):
        load_temporal_recording(bad)


def test_image_validation_rejects_blank(tmp_path):
    """A blank/degenerate image must raise, never yield zero features."""
    from PIL import Image

    blank = tmp_path / "V01PE01.png"
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8)).save(blank)
    with pytest.raises(ValueError):
        load_image_gray(blank)


def test_image_validation_rejects_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image_gray(tmp_path / "missing.png")


def test_class_conflict_between_directory_and_filename_rejected(spatial_root, tmp_path):
    """A file whose name disagrees with its directory must fail loudly."""
    import shutil

    from pd_drawing_analysis.subject_ids import parse_kaggle_filename

    src = next((spatial_root / "spiral" / "training" / "parkinson").glob("*.png"))
    dst_dir = tmp_path / "spiral" / "training" / "healthy"
    dst_dir.mkdir(parents=True)
    dst = dst_dir / src.name  # parkinson-coded name in healthy folder
    shutil.copy(src, dst)
    with pytest.raises(ValueError, match="conflict"):
        inventory_spatial(tmp_path)
