#!/usr/bin/env python3
"""CLI entry point for Step 08 – areal precipitation interpolation."""
from __future__ import annotations

import argparse
from pathlib import Path

from hydrosis.pipeline import run_step08_areal_precipitation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step 08 of the Upper Truckee ten-step pipeline: areal precipitation interpolation.",
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
    outputs = run_step08_areal_precipitation(args.config)
    for label, path in sorted(outputs.items()):
        print(f"{label}: {Path(path)}")


if __name__ == "__main__":
    main()
