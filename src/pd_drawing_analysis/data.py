"""Dataset inventory, loading and validation.

Two sources are supported:

Spatial (Kaggle "Parkinson's Drawings", CC BY-NC-ND 4.0)
    Static PNG drawings, two tasks (spiral, wave).  Filenames encode the
    participant, so a subject-level inventory can be built from names.

Temporal (UCI "Parkinson Disease Spiral Drawings Using Digitized
Graphics Tablet", CC BY 4.0)
    Semicolon-separated tablet recordings, one file per recording.
    Columns: X;Y;Z;Pressure;GripAngle;Timestamp;TestID.

Neither source's supplied train/test directories are used as an
independent evaluation set; see ``docs/data_audit.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .subject_ids import KaggleImageName, parse_kaggle_filename

__all__ = [
    "SpatialInventory",
    "TemporalInventory",
    "TEMPORAL_COLUMNS",
    "inventory_spatial",
    "inventory_temporal",
    "load_image_gray",
    "load_temporal_recording",
    "supplied_split_leakage_table",
]

TEMPORAL_COLUMNS = ["x", "y", "z", "pressure", "grip_angle", "timestamp", "test_id"]
TEMPORAL_MIN_COLUMNS = 6  # test_id may be absent in some exports


# --------------------------------------------------------------------------
# Spatial (Kaggle images)
# --------------------------------------------------------------------------


@dataclass
class SpatialInventory:
    """Subject-level inventory of one or both drawing tasks."""

    frame: pd.DataFrame  # columns: path, task, vendor_split, label, subject_id
    duplicates: list = field(default_factory=list)

    @property
    def subjects(self) -> np.ndarray:
        return self.frame["subject_id"].to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self.frame["label"].to_numpy()

    def summary(self) -> dict:
        df = self.frame
        per_task = {}
        for task, sub in df.groupby("task"):
            counts = sub.groupby("label")["subject_id"].nunique().to_dict()
            per_task[task] = {
                "n_images": int(len(sub)),
                "n_images_by_label": {k: int(v) for k, v in sub["label"].value_counts().items()},
                "n_subjects": int(sub["subject_id"].nunique()),
                "n_subjects_by_label": {k: int(v) for k, v in counts.items()},
                "drawings_per_subject": {
                    str(k): int(v)
                    for k, v in sub.groupby("subject_id").size().value_counts().sort_index().items()
                },
            }
        tasks = list(per_task)
        shared = 0
        if len(tasks) == 2:
            a = set(df[df.task == tasks[0]]["subject_id"])
            b = set(df[df.task == tasks[1]]["subject_id"])
            shared = len(a & b)
        return {"per_task": per_task, "shared_subjects_across_tasks": shared}


def inventory_spatial(root: str | Path, tasks: tuple = ("spiral", "wave")) -> SpatialInventory:
    """Walk a Kaggle-layout drawing tree and build a subject inventory.

    The expected layout (as shipped by the Kaggle dataset) is::

        <root>/spiral/training/<healthy|parkinson>/V01HE01.png ...
        <root>/spiral/testing/...
        <root>/wave/training/...
        <root>/wave/testing/...

    ``alldata``-style directories are also accepted; their images are
    assigned ``vendor_split='alldata'``.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Spatial data root not found: {root}")

    records: list[dict] = []
    seen_hashes: dict[str, str] = {}
    duplicates: list[str] = []

    for task in tasks:
        task_dir = root / task
        if not task_dir.is_dir():
            alt = root / "drawings" / task
            if alt.is_dir():
                task_dir = alt
            else:
                raise FileNotFoundError(f"No task directory for {task!r} under {root}")
        for png in sorted(task_dir.rglob("*.png")):
            rel_parts = [p.lower() for p in png.relative_to(task_dir).parts]
            if "healthy" in rel_parts:
                dir_label = "healthy"
            elif "parkinson" in rel_parts:
                dir_label = "parkinson"
            else:
                raise ValueError(f"Cannot infer class from directory layout for {png}")
            parsed: KaggleImageName = parse_kaggle_filename(png.name)
            if parsed.label != dir_label:
                raise ValueError(
                    f"Label conflict for {png}: directory says {dir_label!r} "
                    f"but filename says {parsed.label!r}"
                )
            vendor_split = "training" if "training" in rel_parts else (
                "testing" if "testing" in rel_parts else "alldata"
            )
            digest = _file_md5(png)
            if digest in seen_hashes:
                duplicates.append(str(png))
            else:
                seen_hashes[digest] = str(png)
            records.append(
                {
                    "path": str(png),
                    "task": task,
                    "vendor_split": vendor_split,
                    "label": parsed.label,
                    "subject_id": parsed.subject_id,
                }
            )

    if not records:
        raise FileNotFoundError(f"No PNG drawings found under {root}")
    frame = pd.DataFrame.from_records(records)
    frame["label"] = frame["label"].map({"healthy": 0, "parkinson": 1}).astype(int)
    return SpatialInventory(frame=frame, duplicates=duplicates)


def supplied_split_leakage_table(inv: SpatialInventory) -> pd.DataFrame:
    """Quantify subject overlap in the vendor's training/testing directories.

    A row is returned per task with counts of subjects shared between the
    supplied training and testing directories.
    """
    rows = []
    for task, sub in inv.frame.groupby("task"):
        train_subjects = set(sub[sub.vendor_split == "training"]["subject_id"])
        test_subjects = set(sub[sub.vendor_split == "testing"]["subject_id"])
        overlap = train_subjects & test_subjects
        rows.append(
            {
                "task": task,
                "train_images": int((sub.vendor_split == "training").sum()),
                "test_images": int((sub.vendor_split == "testing").sum()),
                "train_subjects": len(train_subjects),
                "test_subjects": len(test_subjects),
                "shared_subjects": len(overlap),
                "test_subjects_leaked": f"{100 * len(overlap) / max(len(test_subjects), 1):.0f}%",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Temporal (UCI tablet recordings)
# --------------------------------------------------------------------------


@dataclass
class TemporalInventory:
    """Inventory of UCI tablet recordings (one file per participant)."""

    frame: pd.DataFrame  # columns: path, label, subject_id

    @property
    def subjects(self) -> np.ndarray:
        return self.frame["subject_id"].to_numpy()

    @property
    def labels(self) -> np.ndarray:
        return self.frame["label"].to_numpy()

    def summary(self) -> dict:
        df = self.frame
        return {
            "n_recordings": int(len(df)),
            "n_subjects": int(df["subject_id"].nunique()),
            "n_subjects_by_label": {
                k: int(v) for k, v in df.groupby("label")["subject_id"].nunique().items()
            },
            "n_images_by_label": {k: int(v) for k, v in df["label"].value_counts().items()},
        }


def inventory_temporal(root: str | Path) -> TemporalInventory:
    """Inventory a directory of UCI recordings named ``PWP (N).txt`` /
    ``Healthy (N).txt`` (or the original ``*.txt`` exports).

    The file number is used as a pseudonymous participant code.  Original
    UCI filenames were lost when the files were renamed (see
    ``docs/data_audit.md``), so one-file-one-participant is a documented
    assumption, not a verified fact.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Temporal data root not found: {root}")
    records = []
    for txt in sorted(root.glob("*.txt")):
        name = txt.stem.strip().lower()
        if name.startswith("pwp"):
            label, code = "parkinson", name.replace("pwp", "").strip(" ()-")
        elif name.startswith("healthy"):
            label, code = "healthy", name.replace("healthy", "").strip(" ()-")
        else:
            raise ValueError(
                f"Cannot infer class from temporal filename {txt.name!r}; "
                "expected 'PWP (N).txt' or 'Healthy (N).txt'."
            )
        if not code:
            raise ValueError(f"Missing participant code in temporal filename {txt.name!r}")
        records.append(
            {
                "path": str(txt),
                "label": 1 if label == "parkinson" else 0,
                "subject_id": f"{label}:S{code}",
            }
        )
    if not records:
        raise FileNotFoundError(f"No .txt recordings found under {root}")
    return TemporalInventory(frame=pd.DataFrame.from_records(records))


def load_temporal_recording(path: str | Path) -> pd.DataFrame:
    """Load and validate one tablet recording.

    Raises
    ------
    ValueError
        If the file is empty, unparseable, has fewer than the required
        columns, or contains non-finite values.  No silent substitution
        of zeros or row dropping beyond documented rules.
    """
    path = Path(path)
    df = pd.read_csv(path, sep=";", header=None, comment="#")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < TEMPORAL_MIN_COLUMNS:
        raise ValueError(
            f"{path.name}: expected >= {TEMPORAL_MIN_COLUMNS} columns "
            f"(X;Y;Z;Pressure;GripAngle;Timestamp[;TestID]), got {df.shape[1]}"
        )
    df = df.iloc[:, :TEMPORAL_MIN_COLUMNS]
    df.columns = TEMPORAL_COLUMNS[:TEMPORAL_MIN_COLUMNS]
    if "test_id" not in df.columns:
        df["test_id"] = 0
    df = df.apply(pd.to_numeric, errors="raise")
    if df.empty:
        raise ValueError(f"{path.name}: recording is empty")
    if df[["x", "y", "pressure", "timestamp"]].isna().any().any():
        raise ValueError(f"{path.name}: non-finite values in required columns")
    return df


def load_image_gray(path: str | Path, size: tuple[int, int] = (128, 128)):
    """Load an image as a float grayscale array resized to ``size``.

    Raises
    ------
    ValueError
        If the image cannot be read or is degenerate (all one value).
        Callers must not receive silently zeroed features.
    """
    from skimage import io as _io

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        img = _io.imread(path)
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ValueError(f"Could not read image {path}: {exc}") from exc
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    if img.size == 0:
        raise ValueError(f"{path.name}: image is empty")
    img = img.astype(np.float64) / 255.0
    from skimage.transform import resize as _resize

    img = _resize(img, size, order=1, anti_aliasing=True, preserve_range=True)
    if float(img.max() - img.min()) == 0.0:
        raise ValueError(f"{path.name}: degenerate image (constant intensity)")
    return img


def _file_md5(path: Path) -> str:
    import hashlib

    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
