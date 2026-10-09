"""Shared offline fixtures.

All tests run on deterministic synthetic data generated in temporary
directories.  No dataset downloads, no pretrained weights, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pd_drawing_analysis.synthetic import (  # noqa: E402
    generate_spatial_fixtures,
    generate_temporal_fixtures,
)


@pytest.fixture(scope="session")
def spatial_root(tmp_path_factory) -> Path:
    return generate_spatial_fixtures(tmp_path_factory.mktemp("spatial"))


@pytest.fixture(scope="session")
def temporal_root(tmp_path_factory) -> Path:
    return generate_temporal_fixtures(tmp_path_factory.mktemp("temporal"))


@pytest.fixture()
def leaky_split_subjects():
    """Subjects mimicking the vendor split: every 'test' subject also in 'train'."""
    train = ["parkinson:V01", "parkinson:V02", "healthy:V01"]
    test = ["parkinson:V01", "healthy:V02", "healthy:V01"]
    return train, test


@pytest.fixture()
def rng():
    return np.random.default_rng(42)
