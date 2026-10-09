"""pd_drawing_analysis

Exploratory, research-only benchmark for subject-aware evaluation of
Parkinson's disease drawing features.  NOT for diagnosis, screening, or
any clinical use.
"""

from . import data, evaluation, features, metrics, reporting, subject_ids, synthetic

__version__ = "1.0.0"

__all__ = [
    "data",
    "evaluation",
    "features",
    "metrics",
    "reporting",
    "subject_ids",
    "synthetic",
    "__version__",
]


def print_environment() -> None:
    """Print Python and package versions for reproducibility records."""
    import json

    print(json.dumps(reporting.environment_info(), indent=2))
