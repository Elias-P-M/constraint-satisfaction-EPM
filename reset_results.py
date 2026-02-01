#!/usr/bin/env python3
"""
Delete generated results/plots for a clean rerun.

Targets (if present):
- results_*
- plots_*
- plots_seed_* (legacy)
- plots_metrics*
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _delete_path(path: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would delete {path}")
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    print(f"deleted {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Remove generated results/plots.")
    p.add_argument("--dry-run", action="store_true", help="List what would be deleted")
    args = p.parse_args()

    base = Path(__file__).resolve().parent

    # Directories with glob patterns
    patterns = [
        "results_*",
        "plots_*",
        "plots_seed_*",
        "plots_metrics*",
    ]
    for pat in patterns:
        for path in sorted(base.glob(pat)):
            if path.exists():
                _delete_path(path, args.dry_run)

if __name__ == "__main__":
    main()
