#!/usr/bin/env python3
"""Print Python/package versions and configuration for reproducibility.

    python scripts/env_info.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pd_drawing_analysis import __version__
from pd_drawing_analysis.evaluation import SEED
from pd_drawing_analysis.reporting import environment_info


def main() -> int:
    info = environment_info()
    info["package_version"] = __version__
    info["default_random_seed"] = SEED
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
