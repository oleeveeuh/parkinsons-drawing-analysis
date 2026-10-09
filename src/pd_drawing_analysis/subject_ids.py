"""Subject identity parsing and leakage detection.

The Kaggle drawing filenames encode participant IDs, e.g. ``V01PE02.png``
is the 2nd spiral drawing of Parkinson's participant ``V01`` and
``V01HE02.png`` is the 2nd drawing of healthy participant ``V01``.  The
numeric code alone is NOT a participant identity: participant ``V01`` in
the Parkinson's group is a different person from participant ``V01`` in
the healthy group.  All grouping in this project therefore uses
*class-aware* subject identifiers such as ``parkinson:V01``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "KaggleImageName",
    "parse_kaggle_filename",
    "class_aware_subject_id",
    "subject_overlap",
    "assert_subject_disjoint",
]

# V<participant><group><drawing>.png
# Groups observed in the dataset: PO/PE (Parkinson), HO/HE (Healthy).
_KAGGLE_NAME_RE = re.compile(
    r"^V(?P<participant>\d+)(?P<group>P[ODE]|H[ODE])(?P<drawing>\d+)\.(?:png|PNG)$"
)

_GROUP_CLASS = {
    "PO": "parkinson",
    "PD": "parkinson",
    "PE": "parkinson",
    "HO": "healthy",
    "HD": "healthy",
    "HE": "healthy",
}


@dataclass(frozen=True)
class KaggleImageName:
    """Parsed components of a Kaggle drawing filename."""

    participant: str
    group_code: str
    drawing: str
    label: str  # 'parkinson' or 'healthy'
    subject_id: str  # class-aware subject id, e.g. 'parkinson:V01'


def parse_kaggle_filename(name: str) -> KaggleImageName:
    """Parse a Kaggle drawing filename into its components.

    Raises
    ------
    ValueError
        If the filename does not match the expected pattern.  Malformed
        names must fail loudly: silently dropping or mislabeling an image
        would corrupt the subject inventory.
    """
    match = _KAGGLE_NAME_RE.match(Path(name).name)
    if match is None:
        raise ValueError(
            f"Unrecognized drawing filename: {name!r}. Expected pattern "
            "'V<participant><group><drawing>.png', e.g. 'V01PE02.png'."
        )
    participant = match.group("participant")
    group_code = match.group("group")
    label = _GROUP_CLASS[group_code]
    return KaggleImageName(
        participant=participant,
        group_code=group_code,
        drawing=match.group("drawing"),
        label=label,
        subject_id=f"{label}:V{participant}",
    )


def class_aware_subject_id(label: str, participant: str) -> str:
    """Build a class-aware subject id, e.g. ``('parkinson', '01') -> 'parkinson:V01'``.

    Namespacing by class prevents accidental merging of unrelated
    participants that share a numeric code across the Parkinson's and
    healthy groups.
    """
    label = label.strip().lower()
    if label not in {"parkinson", "healthy"}:
        raise ValueError(f"Unknown label {label!r}; expected 'parkinson' or 'healthy'.")
    return f"{label}:V{str(participant).strip()}"


def subject_overlap(train_subjects, eval_subjects) -> set:
    """Return the set of subjects appearing in both an (apparent) training
    and evaluation collection.

    Used both to audit the dataset vendors' supplied splits and as a guard
    inside cross-validation.
    """
    return set(train_subjects) & set(eval_subjects)


def assert_subject_disjoint(train_subjects, eval_subjects, context: str = "") -> None:
    """Raise if any subject appears on both sides of a split.

    This is the single most important invariant of the project: no
    participant may contribute drawings to both the training and the
    evaluation side of any split, at any nesting level.
    """
    overlap = subject_overlap(train_subjects, eval_subjects)
    if overlap:
        raise AssertionError(
            f"Subject leakage detected{' in ' + context if context else ''}: "
            f"{len(overlap)} subject(s) appear on both sides of the split: "
            f"{sorted(overlap)[:10]}{'...' if len(overlap) > 10 else ''}"
        )
