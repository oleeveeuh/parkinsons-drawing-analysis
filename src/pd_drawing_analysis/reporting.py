"""Reproducible result artifacts.

Every evaluation writes one JSON artifact containing the run environment,
the exact fold-to-subject assignment, and the out-of-fold predictions
keyed by original sample index, plus a human-readable markdown summary.
README numbers must be copied from these artifacts, never typed by hand.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["write_json_artifact", "write_markdown_summary", "environment_info"]


def environment_info() -> dict:
    """Versions of Python and every package the pipelines depend on."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "packages": {},
    }
    for module_name in (
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "skimage",
        "yaml",
        "xgboost",
        "torch",
        "torchvision",
    ):
        try:
            module = __import__(module_name)
            info["packages"][module_name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - e.g. xgboost raises a non-ImportError when libomp is missing
            info["packages"][module_name] = None
    return info


def _to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj


def write_json_artifact(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"environment": environment_info(), **_to_serializable(payload)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


def write_markdown_summary(title: str, payload: dict, output_path: str | Path) -> Path:
    """Write a markdown summary with metrics tables and provenance.

    ``payload`` is the same structure passed to :func:`write_json_artifact`;
    recognized keys (``dataset``, ``models``/``results``) are rendered as
    tables, everything else is preserved in the JSON artifact.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    for key in ("dataset", "protocol"):
        if key in payload:
            lines.append(f"## {key.capitalize()}")
            lines.append("```json")
            lines.append(json.dumps(_to_serializable(payload[key]), indent=2, sort_keys=True))
            lines.append("```")
            lines.append("")
    models = payload.get("models") or payload.get("results") or {}
    if models:
        lines.append("## Participant-level results (out-of-fold)")
        lines.append("")
        lines.append(
            "| Model | Accuracy | 95% CI (subject bootstrap) | Bal. acc. | F1 | ROC-AUC | Sens. | Spec. |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        for name, result in models.items():
            if "participant_metrics" not in result:
                continue
            m = result["participant_metrics"]
            ci = m.get("subject_bootstrap_ci", {}).get("accuracy")
            auc = f"{m['roc_auc']:.3f}" if m.get("roc_auc") is not None else "n/a"
            ci_text = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "n/a"
            lines.append(
                f"| {name} | {m['accuracy']:.3f} | {ci_text} | {m['balanced_accuracy']:.3f} "
                f"| {m['f1']:.3f} | {auc} | {m['sensitivity']:.3f} | {m['specificity']:.3f} |"
            )
        lines.append("")
        lines.append(
            "Accuracy intervals are deterministic subject-level bootstrap "
            "percentile intervals (participants resampled with replacement, "
            "class counts preserved). Fold-to-fold standard deviation is "
            "reported separately in the JSON artifact and is *not* a "
            "confidence interval."
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path
