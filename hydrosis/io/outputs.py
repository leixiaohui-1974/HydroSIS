"""Utilities for persisting HydroSIS outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import csv


def write_time_series(path: Path, values: Iterable[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="") as fp:
        writer = csv.writer(fp)
        for idx, value in enumerate(values):
            writer.writerow([idx, value])


def write_simulation_results(directory: Path, results: Dict[str, List[float]]) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for sub_id, series in results.items():
        write_time_series(directory / f"{sub_id}.csv", series)


def write_markdown(path: Path, content: str) -> None:
    """Persist markdown content to disk with UTF-8 encoding."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
