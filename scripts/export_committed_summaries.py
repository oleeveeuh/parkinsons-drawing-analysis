#!/usr/bin/env python3
"""Write numbers-only summaries of the frozen result artifacts to
``results/summaries/`` — the committed receipts cited by the README and
the portfolio entry. The full artifacts (with row-level predictions)
stay git-ignored under ``results/``.

Stripped from every summary:
* ``models.*.oof`` and ``models.*.participant_table`` — row-level and
  participant-level predictions (nothing per-drawing or per-person).
* ``data_audit.json`` per-file listings (``images`` / ``recordings`` and
  any per-file path lists) — replaced with the aggregate counts the
  docs quote (file counts, subjects by label, drawings-per-subject
  histogram, supplied-split exposure).

Kept in every summary: dataset counts, protocol, per-model metrics,
bootstrap CIs, confusion matrices, fold configs, gate weights,
environment versions, seeds, and aggregate derived numbers — including
``data_audit.json``'s existing ``summary`` and ``vendor_split_leakage``
blocks, which carry the counts and split-exposure numbers the docs cite.

Usage:
    python scripts/export_committed_summaries.py [--results-dir results]

Deterministic output (sorted keys, fixed floats) so re-runs diff cleanly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Row-level prediction blocks, wherever they appear; plus machine-local
# data paths (``root``), which no metric depends on.
STRIP_KEYS = {"oof", "participant_table", "root"}
# Per-file listings and machine-local paths, stripped from data_audit.json.
AUDIT_STRIP_KEYS = {"images", "recordings", "root", "duplicate_image_paths"}

SOURCES = [
    "data_audit.json",
    "impact_numbers.json",
    "real/spatial_real_cnn_hog_shape.json",
    "real/spatial_real_hog_shape.json",
    "real/temporal_real.json",
    "real/temporal_real_with_z.json",
    "fusion/fusion_feature_ablation_real.json",
    "fusion/fusion_paired_synthetic.json",
    "moe/moe_modality_experts_synthetic.json",
    "moe/moe_synthetic.json",
    "moe/moe_task_experts_real.json",
    "moe/moe_validate_large.json",
]

PROVENANCE_NOTE = (
    "Numbers-only committed summary of results/{rel} (git-ignored). "
    "Row-level predictions and per-file listings are stripped; every "
    "retained value is copied unchanged. Regenerate the full artifact "
    "with the README Setup commands, then rerun "
    "scripts/export_committed_summaries.py."
)


def _strip(obj, rel: str):
    if isinstance(obj, dict):
        return {
            k: _strip(v, rel)
            for k, v in sorted(obj.items())
            if k not in STRIP_KEYS
        }
    if isinstance(obj, list):
        return [_strip(v, rel) for v in obj]
    return obj


def _summarize_listing(entries: list[dict]) -> dict:
    """Aggregate a per-file audit listing into countable facts."""
    subjects: Counter = Counter()
    labels: Counter = Counter()
    per_subject: Counter = Counter()
    for e in entries:
        sid = e["subject_id"]
        subjects[sid] = e["label"]
        labels[e["label"]] += 1
        per_subject[sid] += 1
    by_label_n = Counter(subjects.values())
    return {
        "n_files": len(entries),
        "n_images_by_label": {str(k): v for k, v in sorted(labels.items())},
        "n_unique_subjects": len(subjects),
        "subjects_by_label": {str(k): v for k, v in sorted(by_label_n.items())},
        "drawings_per_subject_histogram": {
            str(k): v for k, v in sorted(Counter(per_subject.values()).items())
        },
    }


def _transform(rel: str, doc):
    doc = _strip(doc, rel)
    if rel == "data_audit.json":
        # The artifact already carries the aggregate counts the docs cite
        # (``summary`` per arm, ``vendor_split_leakage``) — keep those,
        # drop only the per-file listings and local paths.
        n_dup = None
        for arm in ("spatial", "temporal"):
            if arm in doc and isinstance(doc[arm], dict):
                if "duplicate_image_paths" in doc[arm]:
                    n_dup = len(doc[arm].pop("duplicate_image_paths"))
                if "root" in doc[arm]:
                    del doc[arm]["root"]
                for key in AUDIT_STRIP_KEYS - {"root", "duplicate_image_paths"}:
                    if key in doc[arm]:
                        del doc[arm][key]
                doc[arm]["listing_note"] = (
                    "per-file listing and local root stripped; aggregate "
                    "counts in 'summary', supplied-split audit in "
                    "'vendor_split_leakage'"
                )
                if arm == "spatial" and n_dup is not None:
                    doc[arm]["n_duplicate_image_paths"] = n_dup
    doc = {"_provenance": PROVENANCE_NOTE.format(rel=rel), **doc}
    return doc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    out_root = args.results_dir / "summaries"
    out_root.mkdir(parents=True, exist_ok=True)

    for rel in SOURCES:
        src = args.results_dir / rel
        if not src.exists():
            print(f"MISSING (skipped): {src}")
            continue
        summary = _transform(rel, json.loads(src.read_text()))
        dst = out_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"written: {dst} ({dst.stat().st_size} bytes, "
              f"full artifact {src.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
