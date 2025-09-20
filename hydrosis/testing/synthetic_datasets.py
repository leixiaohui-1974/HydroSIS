"""Utility helpers providing deterministic synthetic datasets for tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

__all__ = ["write_synthetic_delineation_inputs", "SYNTHETIC_DEM_GRID", "SYNTHETIC_POUR_POINTS"]


# A gently sloping DEM matching the lightweight JSON format consumed by the
# ``simple_grid`` delineator.  Each cell represents a 1 km² area so cell counts
# are directly comparable with subbasin areas.
SYNTHETIC_DEM_GRID: Dict[str, object] = {
    "transform": {
        "x_origin": 500_000.0,
        "y_origin": 3_500_000.0,
        "pixel_width": 1_000.0,
        "pixel_height": 1_000.0,
    },
    "crs": "EPSG:3857",
    "grid": [
        [120.0, 115.0, 110.0, 110.0, 115.0, 120.0],
        [118.0, 112.0, 108.0, 108.0, 112.0, 118.0],
        [116.0, 108.0, 100.0, 100.0, 108.0, 116.0],
        [114.0, 106.0, 96.0, 96.0, 106.0, 114.0],
        [112.0, 104.0, 92.0, 92.0, 104.0, 112.0],
        [110.0, 102.0, 88.0, 88.0, 102.0, 110.0],
    ],
}


# Three pour points positioned to create a simple headwater -> mid catchment ->
# outlet hierarchy when combined with ``SYNTHETIC_DEM_GRID``.
SYNTHETIC_POUR_POINTS: Dict[str, object] = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [501_500.0, 3_498_500.0]},
            "properties": {"id": "S1", "name": "Headwater gauge"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [503_500.0, 3_496_500.0]},
            "properties": {"id": "S2", "name": "Mid catchment reservoir"},
        },
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [503_500.0, 3_494_500.0]},
            "properties": {"id": "S3", "name": "Outlet station"},
        },
    ],
}


def write_synthetic_delineation_inputs(directory: Path) -> Tuple[Path, Path]:
    """Persist the synthetic DEM and pour points to ``directory``.

    The helper returns the paths to the generated JSON files so callers can
    instantiate :class:`~hydrosis.delineation.dem_delineator.DelineationConfig`
    directly without needing to embed the fixtures inline.
    """

    directory.mkdir(parents=True, exist_ok=True)

    dem_path = directory / "synthetic_dem.json"
    pour_points_path = directory / "synthetic_pour_points.geojson"

    dem_path.write_text(
        json.dumps(SYNTHETIC_DEM_GRID, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pour_points_path.write_text(
        json.dumps(SYNTHETIC_POUR_POINTS, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return dem_path, pour_points_path

