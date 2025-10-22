#!/usr/bin/env python3
"""CLI entry point for the final pipeline report aggregation."""
from __future__ import annotations

import argparse
from pathlib import Path

from hydrosis.pipeline import run_final_pipeline_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate step reports into the final pipeline summary document.",
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
    outputs = run_final_pipeline_report(args.config)
    for label, path in sorted(outputs.items()):
        print(f"{label}: {Path(path)}")


if __name__ == "__main__":
    main()
