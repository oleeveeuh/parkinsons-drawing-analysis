"""Deterministic synthetic fixtures.

Generates clearly synthetic stand-ins for both datasets so the whole
pipeline can be verified offline with no downloads:

* Synthetic "drawings": Archimedean spirals and horizontal wave strokes.
  Synthetic "impaired" drawings get tremor wiggle and jitter; the signal
  is engineered to be learnable but not trivial.
* Synthetic "tablet recordings": point sequences in the UCI
  ``X;Y;Z;Pressure;GripAngle;Timestamp`` format, with 5 Hz tremor and
  slower stroke speed for the synthetic impaired class.

These fixtures exist to test *software correctness*, not to say anything
about Parkinson's disease.  Any accuracy measured on them is meaningless.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "generate_spatial_fixtures",
    "generate_temporal_fixtures",
    "generate_paired_fixtures",
    "two_regime_participants",
    "SYNTHETIC_SEED",
]

SYNTHETIC_SEED = 42


def _draw_spiral(size: int, rng: np.random.Generator, impaired: bool) -> np.ndarray:
    from PIL import Image

    img = np.ones((size, size), dtype=np.float64)
    c = size / 2
    n_points = 2500
    theta = np.linspace(0, 5.5 * 2 * np.pi, n_points)
    radius = 3.0 + theta / (2 * np.pi) * (size * 0.42) / 5.5
    x = c + radius * np.cos(theta)
    y = c + radius * np.sin(theta)
    if impaired:
        tremor = 1.6 + 0.8 * rng.random()
        x = x + tremor * np.sin(theta * 9.0) + rng.normal(0, 0.9, n_points)
        y = y + tremor * np.cos(theta * 7.0) + rng.normal(0, 0.9, n_points)
    else:
        x = x + rng.normal(0, 0.25, n_points)
        y = y + rng.normal(0, 0.25, n_points)
    xi = np.clip(np.round(x).astype(int), 0, size - 1)
    yi = np.clip(np.round(y).astype(int), 0, size - 1)
    img[yi, xi] = 0.0
    return img


def _draw_wave(size: int, rng: np.random.Generator, impaired: bool) -> np.ndarray:
    img = np.ones((size, size), dtype=np.float64)
    n_points = 1200
    t = np.linspace(0, size * 0.9, n_points)
    amplitude = size * 0.12
    frequency = 2.5 * np.pi / size
    y = size / 2 + amplitude * np.sin(frequency * t * 3.0)
    x = size * 0.05 + t
    if impaired:
        tremor = 1.8 + 0.8 * rng.random()
        y = y + tremor * np.sin(t / 3.0) + rng.normal(0, 1.0, n_points)
        x = x + rng.normal(0, 0.8, n_points)
    else:
        y = y + rng.normal(0, 0.3, n_points)
    xi = np.clip(np.round(x).astype(int), 0, size - 1)
    yi = np.clip(np.round(y).astype(int), 0, size - 1)
    img[yi, xi] = 0.0
    return img


def generate_spatial_fixtures(
    out_dir: str | Path,
    n_subjects_per_class: int = 6,
    drawings_per_subject: int = 2,
    size: int = 96,
    seed: int = SYNTHETIC_SEED,
) -> Path:
    """Write a miniature Kaggle-layout drawing tree and return its root.

    Layout: ``<out>/spiral/training/<label>/V..PD01.png`` and matching
    ``testing`` + ``wave`` directories.  Files are clearly synthetic
    renders; directory counts are printed for the dataset card.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    rng = np.random.default_rng(seed)
    for task in ("spiral", "wave"):
        for label in ("parkinson", "healthy"):
            for subject in range(1, n_subjects_per_class + 1):
                for drawing in range(1, drawings_per_subject + 1):
                    impaired = label == "parkinson"
                    img = _draw_wave(size, rng, impaired) if task == "wave" else _draw_spiral(size, rng, impaired)
                    encoded = Image.fromarray((img * 255).astype(np.uint8))
                    # One rendered drawing is written to BOTH vendor splits,
                    # mirroring how the real Kaggle tree repeats participants
                    # across training/ and testing/ (the audited leakage).
                    for split in ("training", "testing"):
                        subdir = out_dir / task / split / label
                        subdir.mkdir(parents=True, exist_ok=True)
                        group_code = "PD" if impaired else "HE"
                        name = f"V{subject:02d}{group_code}{drawing:02d}.png"
                        encoded.save(subdir / name)
    return out_dir


def generate_temporal_fixtures(
    out_dir: str | Path,
    n_subjects_per_class: int = 6,
    n_points: int = 1200,
    seed: int = SYNTHETIC_SEED,
) -> Path:
    """Write synthetic tablet recordings in the UCI column format and
    return the directory."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for filename_prefix, class_label in (("PWP", "parkinson"), ("Healthy", "healthy")):
        impaired = class_label == "parkinson"
        for subject in range(1, n_subjects_per_class + 1):
            t = np.arange(n_points) * (5.0 if impaired else 4.0)
            speed = 0.9 if impaired else 1.4  # slower synthetic stroke
            theta = np.linspace(0, 4 * np.pi, n_points)
            x = 200 + np.cumsum(speed + 0.3 * np.cos(theta) + rng.normal(0, 0.05, n_points))
            y = 200 + np.cumsum(0.4 * np.sin(theta) + rng.normal(0, 0.05, n_points))
            if impaired:  # 5 Hz tremor
                x = x + 6.0 * np.sin(2 * np.pi * 5.0 * np.arange(n_points) / 50.0)
                y = y + 5.0 * np.sin(2 * np.pi * 5.0 * np.arange(n_points) / 50.0 + 0.7)
            z = rng.normal(0, 0.5, n_points)
            pressure = (
                np.clip(0.45 + 0.1 * rng.random() + rng.normal(0, 0.03, n_points), 0.05, 1.0)
                * (1440 if impaired else 900)
            )
            grip = rng.normal(30 if impaired else 60, 4.0, n_points)
            lines = [
                f"{x[i]:.0f};{y[i]:.0f};{z[i]:.2f};{pressure[i]:.0f};{grip[i]:.1f};{t[i]:.0f};0"
                for i in range(n_points)
            ]
            code = f"{subject}"
            (out_dir / f"{filename_prefix} ({code}).txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir


def generate_paired_fixtures(
    out_dir: str | Path,
    n_subjects_per_class: int = 6,
    seed: int = SYNTHETIC_SEED,
) -> dict:
    """Generate *paired* synthetic modalities for fusion testing.

    For every synthetic participant the function produces BOTH a tablet
    recording (UCI format) and a spiral drawing (PNG) under the same
    class-aware subject id — the pairing that the two real datasets lack
    and that any genuine fusion analysis requires.

    The class signal is split across modalities: roughly half of the
    impaired participants carry the distinguishing signal only in the
    recording (strong tremor, normal drawing) and half only in the
    drawing (tremulous stroke, normal recording).  A single modality
    therefore cannot separate all participants, so a correct fusion
    implementation measurably outperforms both single-modality models.
    This makes the fixture a real test of fusion behaviour, not a
    rubber stamp.

    Returns ``{"temporal_dir": ..., "image_dir": ...}``.
    """
    from PIL import Image

    out = Path(out_dir)
    temporal_dir = out / "temporal"
    image_dir = out / "images" / "spiral" / "alldata"
    temporal_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    size = 96
    n_points = 900

    for label_prefix, class_name in (("PWP", "parkinson"), ("Healthy", "healthy")):
        impaired = class_name == "parkinson"
        for subject in range(1, n_subjects_per_class + 1):
            subject_code = f"{class_name}:V{subject:02d}"
            # Deterministic per-subject assignment of which modality carries signal.
            signal_in_recording = impaired and (int(subject) % 2 == 0)
            signal_in_drawing = impaired and not signal_in_recording

            # --- recording (tremor ONLY if signal_in_recording; every other
            # property is drawn from class-independent distributions so the
            # recording genuinely carries no other class signal) ---
            t = np.arange(n_points) * 4.0
            theta = np.linspace(0, 4 * np.pi, n_points)
            speed = 1.2 + 0.1 * rng.random()
            x = 200 + np.cumsum(speed + 0.3 * np.cos(theta) + rng.normal(0, 0.05, n_points))
            y = 200 + np.cumsum(0.4 * np.sin(theta) + rng.normal(0, 0.05, n_points))
            tremor_amp = 6.0 if signal_in_recording else 0.0
            x = x + tremor_amp * np.sin(2 * np.pi * 5.0 * np.arange(n_points) / 50.0)
            y = y + tremor_amp * np.sin(2 * np.pi * 5.0 * np.arange(n_points) / 50.0 + 0.7)
            z = rng.normal(0, 0.5, n_points)
            pressure = np.clip(
                0.35 + 0.2 * rng.random() + rng.normal(0, 0.03, n_points), 0.05, 1.0
            ) * 1000
            grip = rng.normal(50, 5.0, n_points)
            lines = [
                f"{x[i]:.0f};{y[i]:.0f};{z[i]:.2f};{pressure[i]:.0f};{grip[i]:.1f};{t[i]:.0f};0"
                for i in range(n_points)
            ]
            (temporal_dir / f"{label_prefix} ({subject}).txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

            # --- drawing (tremulous stroke only if signal_in_drawing) ---
            img = _draw_spiral(size, rng, impaired=signal_in_drawing)
            group_code = "PD" if impaired else "HE"
            Image.fromarray((img * 255).astype(np.uint8)).save(
                image_dir / f"V{subject:02d}{group_code}01.png"
            )
            _ = subject_code  # pairing documented by shared filename numbering
    return {"temporal_dir": temporal_dir, "image_dir": image_dir.parent.parent}


def two_regime_participants(
    n_per_regime: int = 12,
    seed: int = SYNTHETIC_SEED,
):
    """Matrix fixture for mixture-of-experts evaluation: per-participant
    rows drawn from two latent regimes with different class rules.

    * Regime A rows are separable by a diagonal linear rule in block 1
    * Regime B rows are separable by a band rule in block 2
    * Two "signature" dimensions encode the regime (strong separation),
      so a gate can in principle localise experts perfectly

    Returns ``(X, y, regimes, subjects)`` where ``subjects`` are unique
    participant ids and ``y`` is the rule output (the classification
    target).  Deliberately small by default: the interesting question is
    whether EM mixtures help at small n, and the answer is measured, not
    assumed.
    """
    rng = np.random.default_rng(seed)
    rows, targets, regime_labels = [], [], []
    for i in range(n_per_regime * 2):
        regime = "A" if i < n_per_regime else "B"
        b1 = rng.normal(size=2)
        b2 = rng.normal(size=2)
        signature = rng.normal(size=2, scale=0.1) + (
            np.array([6.0, -6.0]) if regime == "A" else np.array([-6.0, 6.0])
        )
        target = int(b1[0] + b1[1] > 0.8) if regime == "A" else int(abs(b2[0]) < 0.8)
        rows.append(np.concatenate([b1, b2, signature]))
        targets.append(target)
        regime_labels.append(regime)
    X = np.vstack(rows)
    y = np.array(targets)
    regimes = np.array(regime_labels)
    subjects = np.array([f"subject:{i:03d}" for i in range(len(y))])
    return X, y, regimes, subjects
