#!/usr/bin/env python3
"""CLI entry point for Step 09 – hydrologic baseline run."""
from __future__ import annotations

import argparse
from pathlib import Path

from hydrosis.pipeline import run_step09_hydrologic_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step 09 of the Upper Truckee ten-step pipeline: hydrologic baseline simulation.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/upper_truckee_project.yml"),
        help="Path to the unified project configuration YAML file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_step09_hydrologic_run(args.config)
    for label, path in sorted(outputs.items()):
        print(f"{label}: {Path(path)}")


if __name__ == "__main__":
    main()
