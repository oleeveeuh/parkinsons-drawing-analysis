"""Feature extraction for drawings.

Two feature families are implemented:

* **Spatial (per-image, frozen):** HOG + simple shape descriptors.  These
  are computed per image from pixels only — they use no fitted statistics
  from the study population — so caching them is leakage-free.  All
  *dataset-fitted* preprocessing (variance selection, scaling, PCA) lives
  in :mod:`pd_drawing_analysis.evaluation` and is fitted inside
  cross-validation folds only.

* **Temporal (per-recording, frozen):** handcrafted descriptors of tablet
  recordings (velocity, pressure, tremor-band power, ...).  Also purely
  per-recording and therefore safe to precompute.

An optional frozen CNN (torchvision ResNet-18, ImageNet weights) exists
for the spatial path.  It is never imported or downloaded by tests or CI;
using it requires installing the ``cnn`` extra and passing ``--cnn``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_image_gray, load_temporal_recording

__all__ = [
    "SHAPE_FEATURE_NAMES",
    "TEMPORAL_FEATURE_NAMES",
    "hog_features",
    "shape_features",
    "extract_spatial_features",
    "extract_spatial_feature_matrix",
    "extract_temporal_features",
    "extract_temporal_feature_matrix",
    "cnn_feature_matrix",
]

# Image side used for all spatial features.  Recorded here so cache keys
# and documentation can reference the exact preprocessing configuration.
SPATIAL_IMAGE_SIZE = (128, 128)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys",
)

SHAPE_FEATURE_NAMES = [
    "mean_intensity",
    "std_intensity",
    "edge_density",
    "ink_ratio",
    "aspect_ratio",
    "extent",
    "solidity_proxy",
    "radius_mean",
    "radius_std",
    "radius_cv",
    "stroke_width_proxy",
    "radial_smoothness",
]

TEMPORAL_FEATURE_NAMES = [
    "duration_norm",
    "sample_rate_norm",
    "n_points_log",
    "velocity_mean",
    "velocity_std",
    "velocity_max",
    "acceleration_mean",
    "acceleration_std",
    "jerk_mean",
    "direction_changes_rate",
    "path_length",
    "displacement_ratio",
    "pressure_mean",
    "pressure_std",
    "pressure_cv",
    "pressure_rate",
    "grip_angle_std",
    "on_paper_ratio",
    "tremor_power_4_8hz",
    "tremor_ratio_4_8hz",
    "speed_cv",
    "pause_ratio",
    "radius_mean",
    "radius_std",
    "radial_smoothness",
]


# --------------------------------------------------------------------------
# Spatial features
# --------------------------------------------------------------------------


def hog_features(img: np.ndarray) -> np.ndarray:
    """Histogram of oriented gradients for a grayscale image in [0, 1]."""
    from skimage.feature import hog

    return hog(
        img,
        **HOG_PARAMS,
        feature_vector=True,
    ).astype(np.float64)


def _ink_mask(img: np.ndarray) -> np.ndarray:
    """Boolean ink mask using a per-image Otsu threshold.

    The Kaggle scans mix bold digital strokes with faint pencil scans
    (some images never reach intensity 0.5), so a fixed threshold fails;
    Otsu adapts per image and uses no dataset-level statistics, keeping
    the feature map frozen.
    """
    from skimage.filters import threshold_otsu

    try:
        threshold = threshold_otsu(img)
    except ValueError:  # constant image; already rejected upstream
        raise
    ink = img < threshold
    if not ink.any():
        raise ValueError(
            "Drawing has no ink pixels below its Otsu threshold "
            "(blank or inverted image)"
        )
    return ink


def shape_features(img: np.ndarray) -> np.ndarray:
    """Twelve deterministic geometric descriptors of a drawing.

    All thresholds are computed per image (Otsu) or fixed constants —
    nothing is fitted on the dataset, so the function is a frozen
    per-image map.
    """
    ink = _ink_mask(img)

    ys, xs = np.nonzero(ink)
    cy, cx = ys.mean(), xs.mean()
    height, width = img.shape
    r = np.hypot(ys - cy, xs - cx) + 1e-9
    center_r = r.mean()

    # Gradient-magnitude proxy for edges / stroke width.
    gy, gx = np.gradient(img)
    grad = np.hypot(gx, gy)
    strong_edges = grad > grad.mean() + grad.std()

    # Radial signal smoothness: residual of radius around its local trend.
    order = np.argsort(np.arctan2(ys - cy, xs - cx))
    r_sorted = r[order]
    if len(r_sorted) > 8:
        kernel = 8
        smoothed = np.convolve(r_sorted, np.ones(kernel) / kernel, mode="same")
        radial_smoothness = float(np.mean(np.abs(r_sorted - smoothed)) / center_r)
    else:
        radial_smoothness = 0.0

    bbox_area = max((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1), 1)
    convex_proxy = float((r.max() ** 2 - r.std() ** 2) / (r.max() ** 2 + 1e-9))

    values = [
        float(img.mean()),                                     # mean_intensity
        float(img.std()),                                      # std_intensity
        float(strong_edges.mean()),                            # edge_density
        float(ink.mean()),                                     # ink_ratio
        float((xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)),  # aspect_ratio
        float(ink.sum() / bbox_area),                          # extent
        convex_proxy,                                          # solidity_proxy
        float(r.mean()),                                       # radius_mean
        float(r.std()),                                        # radius_std
        float(r.std() / center_r),                             # radius_cv
        float(ink.sum() / max(strong_edges.sum(), 1)),         # stroke_width_proxy
        radial_smoothness,                                     # radial_smoothness
    ]
    return np.asarray(values, dtype=np.float64)


def extract_spatial_features(path: str | Path) -> np.ndarray:
    """HOG + shape features for one drawing image."""
    img = load_image_gray(path, size=SPATIAL_IMAGE_SIZE)
    return np.concatenate([hog_features(img), shape_features(img)])


def extract_spatial_feature_matrix(paths, show_progress: bool = True) -> np.ndarray:
    """Extract frozen per-image features for a list of paths.

    Raises on the first unreadable/degenerate image (never substitutes
    zero rows silently).
    """
    from tqdm import tqdm

    iterator = tqdm(paths, desc="features", disable=not show_progress)
    return np.vstack([extract_spatial_features(p) for p in iterator])


def _cache_key(path: Path, model_name: str, weights_tag: str, size: tuple[int, int]) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    config = f"{model_name}|{weights_tag}|{size[0]}x{size[1]}|gray-resize-bilinear-0to1"
    return hashlib.sha256(h.hexdigest().encode() + config.encode()).hexdigest()


def cnn_feature_matrix(
    paths,
    cache_dir: str | Path = "cache/cnn_features",
    model_name: str = "resnet18",
    show_progress: bool = True,
) -> np.ndarray:
    """Frozen ImageNet CNN embeddings (optional dependency).

    Uses the current torchvision weights API (``weights=...``), sets the
    network to evaluation mode and disables gradients.  Embeddings are
    cached on disk keyed by the image content hash plus the exact model,
    weights tag and preprocessing configuration, so the cache can never
    silently mix configurations.

    Requires the ``cnn`` extra (torch + torchvision).  The one-time
    weights download happens here — explicitly, never at import time.
    """
    try:
        import torch
        from torch import nn
        from torchvision import models
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "CNN features require the optional 'cnn' extra: pip install '.[cnn]'. "
            "The default pipeline uses offline HOG + shape features only."
        ) from exc

    weights_enum = models.get_model_weights(model_name)
    weights = weights_enum.DEFAULT
    weights_tag = str(weights).replace("Weights.", "/")
    transform_mean = (0.485, 0.456, 0.406)
    transform_std = (0.229, 0.224, 0.225)

    from skimage.transform import resize as _resize

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    backbone = getattr(models, model_name)(weights=weights)
    backbone = nn.Sequential(*list(backbone.children())[:-1])
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad_(False)

    from tqdm import tqdm

    rows = []
    for path in tqdm(list(paths), desc="cnn", disable=not show_progress):
        path = Path(path)
        key = _cache_key(path, model_name, weights_tag, SPATIAL_IMAGE_SIZE)
        cache_file = cache_dir / f"{key}.npy"
        if cache_file.is_file():
            rows.append(np.load(cache_file))
            continue
        img = load_image_gray(path, size=SPATIAL_IMAGE_SIZE)
        rgb = np.repeat((img * 255.0)[None, ...], 3, axis=0) / 255.0
        rgb = (rgb - np.array(transform_mean)[:, None, None]) / np.array(transform_std)[:, None, None]
        tensor = torch.from_numpy(rgb[None, ...].astype(np.float32))
        with torch.no_grad():
            embedding = backbone(tensor).squeeze().numpy()
        np.save(cache_file, embedding.astype(np.float32))
        rows.append(embedding.astype(np.float32))
    return np.vstack(rows)


# --------------------------------------------------------------------------
# Temporal features
# --------------------------------------------------------------------------


def extract_temporal_features(recording: pd.DataFrame, include_z: bool = False) -> np.ndarray:
    """Handcrafted descriptors of one tablet recording.

    Timestamps are normalised per recording (divided by their own median
    positive difference) before velocities are computed.  The raw UCI
    files mix timestamp units between recordings, and per-recording
    normalisation removes that acquisition artifact — the same class of
    artifact that made the unnormalised z-channel perfectly separate the
    classes (see ``docs/data_audit.md``).

    ``include_z`` re-adds the z-channel features.  They are excluded by
    default because they were removed after *observing* near-perfect class
    separation — an exploratory, post-hoc data-quality decision that must
    not silently shape headline results.  z-derived features can be
    evaluated transparently with ``--include-z``.
    """
    x = recording["x"].to_numpy(dtype=float)
    y = recording["y"].to_numpy(dtype=float)
    p = recording["pressure"].to_numpy(dtype=float)
    g = recording["grip_angle"].to_numpy(dtype=float)
    t = recording["timestamp"].to_numpy(dtype=float)

    dt = np.diff(t)
    positive = dt[dt > 0]
    if positive.size == 0:
        raise ValueError("Recording has no strictly increasing timestamps")
    dt_norm = dt / np.median(positive)
    # Some UCI recordings contain duplicated timestamps; time statistics
    # are computed over strictly positive intervals only.
    dt_pos = dt_norm[dt_norm > 0]
    median_dt = float(np.median(dt_pos))

    dist = np.hypot(np.diff(x), np.diff(y))
    with np.errstate(invalid="ignore", divide="ignore"):
        vel = dist / dt_norm
        acc = np.diff(vel) / dt_norm[1:]
        jerk = np.diff(acc) / dt_norm[2:]
    vel = vel[np.isfinite(vel)]
    acc = acc[np.isfinite(acc)]
    jerk = jerk[np.isfinite(jerk)]

    duration = dt_pos.sum()
    speed_cv = float(vel.std() / (vel.mean() + 1e-9))
    moving = vel > np.percentile(vel, 25) if vel.size else np.array([True])
    pause_ratio = float(1.0 - moving.mean())

    # Tremor-band power: FFT of the velocity residual around its trend,
    # on the normalised time axis (cycles per sample).
    if vel.size >= 16:
        vel_centered = vel - np.linspace(vel[0], vel[-1], vel.size)
        spectrum = np.abs(np.fft.rfft(vel_centered)) ** 2
        freqs = np.fft.rfftfreq(vel.size)
        band = (freqs >= 4 / vel.size) & (freqs <= 8 / vel.size) if vel.size >= 8 else np.zeros_like(freqs, bool)
        tremor_power = float(spectrum[band].sum())
        tremor_ratio = float(spectrum[band].sum() / (spectrum.sum() + 1e-9))
    else:
        tremor_power, tremor_ratio = 0.0, 0.0

    r = np.hypot(x - x.mean(), y - y.mean()) + 1e-9
    order = np.argsort(np.arctan2(y - y.mean(), x - x.mean()))
    r_sorted = r[order]
    smoothed = np.convolve(r_sorted, np.ones(8) / 8, mode="same") if r_sorted.size > 8 else r_sorted

    base = [
        float(duration),
        float(1.0 / (np.median(positive) + 1e-9)),
        float(np.log10(len(x) + 1)),
        float(vel.mean() if vel.size else 0.0),
        float(vel.std() if vel.size else 0.0),
        float(vel.max() if vel.size else 0.0),
        float(acc.mean() if acc.size else 0.0),
        float(acc.std() if acc.size else 0.0),
        float(jerk.mean() if jerk.size else 0.0),
        float(_direction_change_rate(x, y)),
        float(dist.sum()),
        float(np.hypot(x[-1] - x[0], y[-1] - y[0]) / (dist.sum() + 1e-9)),
        float(p.mean()),
        float(p.std()),
        float(p.std() / (p.mean() + 1e-9)),
        float(np.mean(np.abs(np.diff(p))) / (median_dt + 1e-9)),
        float(g.std()),
        float((p > np.percentile(p, 5)).mean()),
        tremor_power,
        tremor_ratio,
        speed_cv,
        pause_ratio,
        float(r.mean()),
        float(r.std()),
        float(np.mean(np.abs(r_sorted - smoothed)) / (r.mean() + 1e-9)),
    ]

    if include_z:
        z = recording["z"].to_numpy(dtype=float)
        z_feats = [
            float(z.mean()),
            float(z.std()),
            float(z.max() - z.min()),
        ]
        base.extend(z_feats)
    return np.asarray(base, dtype=np.float64)


def _direction_change_rate(x: np.ndarray, y: np.ndarray) -> float:
    dx, dy = np.diff(x), np.diff(y)
    if dx.size < 2:
        return 0.0
    dot = dx[1:] * dx[:-1] + dy[1:] * dy[:-1]
    cross = dx[1:] * dy[:-1] - dy[1:] * dx[:-1]
    angles = np.abs(np.arctan2(cross, dot))
    return float(angles.mean())


def extract_temporal_feature_matrix(
    paths,
    include_z: bool = False,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Frozen per-recording temporal features for a list of file paths."""
    from tqdm import tqdm

    names = list(TEMPORAL_FEATURE_NAMES) + (["z_mean", "z_std", "z_range"] if include_z else [])
    rows = []
    for path in tqdm(paths, desc="temporal features", disable=not show_progress):
        recording = load_temporal_recording(path)
        rows.append(extract_temporal_features(recording, include_z=include_z))
    return np.vstack(rows), names
