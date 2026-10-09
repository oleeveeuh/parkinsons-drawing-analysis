"""Subject-ID parsing and class-aware namespacing."""

from __future__ import annotations

import pytest

from pd_drawing_analysis.subject_ids import (
    assert_subject_disjoint,
    class_aware_subject_id,
    parse_kaggle_filename,
    subject_overlap,
)


@pytest.mark.parametrize(
    ("filename", "expected_subject", "expected_label"),
    [
        ("V01PE02.png", "parkinson:V01", 1),
        ("V01HE02.png", "healthy:V01", 0),
        ("V55HO15.png", "healthy:V55", 0),
        ("V03PO09.png", "parkinson:V03", 1),
    ],
)
def test_parse_kaggle_filename(filename, expected_subject, expected_label):
    parsed = parse_kaggle_filename(filename)
    assert parsed.subject_id == expected_subject
    label_value = {"healthy": 0, "parkinson": 1}[parsed.label]
    assert label_value == expected_label


@pytest.mark.parametrize("filename", ["image01.png", "V1PE2.jpg", "XXPE02.png", "V01XX02.png", "notes.txt"])
def test_parse_kaggle_filename_rejects_malformed(filename):
    with pytest.raises(ValueError):
        parse_kaggle_filename(filename)


def test_class_aware_namespacing_prevents_merging():
    """Same numeric code in different classes must stay distinct."""
    assert class_aware_subject_id("parkinson", "01") != class_aware_subject_id("healthy", "01")


def test_namespaced_ids_are_1_to_1_with_people_on_synthetic_fixture(spatial_root):
    """Each namespaced id maps to exactly one label in the inventory."""
    from pd_drawing_analysis.data import inventory_spatial

    inv = inventory_spatial(spatial_root)
    per_subject_labels = inv.frame.groupby("subject_id")["label"].nunique()
    assert (per_subject_labels == 1).all()
    assert set(inv.frame["subject_id"]) == {
        f"parkinson:V{i:02d}" for i in range(1, 7)
    } | {f"healthy:V{i:02d}" for i in range(1, 7)}


def test_overlap_detection(leaky_split_subjects):
    train, test = leaky_split_subjects
    assert subject_overlap(train, test) == {"parkinson:V01", "healthy:V01"}


def test_assert_subject_disjoint_raises(leaky_split_subjects):
    train, test = leaky_split_subjects
    with pytest.raises(AssertionError, match="leakage"):
        assert_subject_disjoint(train, test, context="vendor split")


def test_assert_subject_disjoint_passes_on_clean_split():
    assert_subject_disjoint(["parkinson:V01", "healthy:V02"], ["healthy:V01", "parkinson:V03"])
