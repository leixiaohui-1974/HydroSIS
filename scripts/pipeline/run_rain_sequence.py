#!/usr/bin/env python3
"""CLI entry point for Step 06 – rain sequence generation."""
from __future__ import annotations

import argparse
from pathlib import Path

from hydrosis.pipeline import run_step06_rain_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Step 06 of the Upper Truckee ten-step pipeline: rain sequence generation.",
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
    outputs = run_step06_rain_sequence(args.config)
    for label, path in sorted(outputs.items()):
        print(f"{label}: {Path(path)}")


if __name__ == "__main__":
    main()
