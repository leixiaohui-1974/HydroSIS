"""Utility helpers for loading HydroSIS inputs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import csv


def load_time_series(path: Path) -> List[float]:
    values: List[float] = []
    with Path(path).open("r", newline="") as fp:
        reader = csv.reader(fp)
        for row in reader:
            if not row:
                continue
            try:
                values.append(float(row[-1]))
            except ValueError:
                continue
    return values


def load_forcing(directory: Path) -> Dict[str, List[float]]:
    forcing: Dict[str, List[float]] = {}
    for csv_path in Path(directory).glob("*.csv"):
        forcing[csv_path.stem] = load_time_series(csv_path)
    return forcing
