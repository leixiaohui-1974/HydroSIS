"""Pour point data structures and I/O operations."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
import rasterio.transform


@dataclass
class PourPoint:
    """Discrete outlet used to delineate upstream contributing areas."""

    id: str
    row: int
    col: int
    x: float
    y: float
    accumulation: float
    attributes: Dict[str, object] = field(default_factory=dict)


def read_pour_points_geojson(path: Path) -> List[PourPoint]:
    """Load pour point definitions from a GeoJSON file.

    The expected schema is a FeatureCollection with Point geometries.
    Each feature should have ``id``, ``row``, ``col``, and optionally
    ``accumulation`` properties. Any additional properties are preserved
    inside :attr:`PourPoint.attributes`.
    """

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    features = data.get("features", [])
    pour_points: List[PourPoint] = []

    for feature in features:
        geometry = feature.get("geometry") or {}
        props = feature.get("properties", {}) or {}
        coords = geometry.get("coordinates", [])

        point_id = str(props.get("id", len(pour_points) + 1))
        row = props.get("row")
        col = props.get("col")
        if row is None or col is None:
            continue

        x_val: float
        y_val: float
        if coords and len(coords) >= 2 and all(coord is not None for coord in coords[:2]):
            x_val = float(coords[0])
            y_val = float(coords[1])
        else:
            x_val = float(props.get("x", 0.0))
            y_val = float(props.get("y", 0.0))

        attributes = {
            key: value
            for key, value in props.items()
            if key not in {"id", "row", "col", "accumulation", "x", "y"}
        }

        pour_points.append(
            PourPoint(
                id=point_id,
                row=int(row),
                col=int(col),
                x=x_val,
                y=y_val,
                accumulation=float(props.get("accumulation", 0.0)),
                attributes=attributes,
            )
        )
    return pour_points


D8_OFFSETS: Dict[int, Tuple[int, int]] = {
    1: (0, 1),
    2: (-1, 1),
    3: (-1, 0),
    4: (-1, -1),
    5: (0, -1),
    6: (1, -1),
    7: (1, 0),
    8: (1, 1),
}


def ensure_inputs(
    dem_path: Path,
    flow_acc_path: Path,
    flow_dir_path: Path,
    output_dir: Path,
) -> None:
    if not dem_path.exists():
        raise FileNotFoundError(f"DEM not found: {dem_path}")
    if not flow_acc_path.exists():
        raise FileNotFoundError(f"Flow accumulation grid not found: {flow_acc_path}")
    if not flow_dir_path.exists():
        raise FileNotFoundError(f"Flow direction grid not found: {flow_dir_path}")
    output_dir.mkdir(parents=True, exist_ok=True)


# More functions will be moved here in subsequent steps


def derive_pour_points(
    flow_acc_path: Path,
    pour_point_count: int,
    accumulation_threshold: float,
    min_spacing: float,
) -> List[PourPoint]:
    """Select pour points from flow accumulation maxima."""


    with rasterio.open(flow_acc_path) as acc_ds:
        acc = acc_ds.read(1)
        acc = np.where(np.isfinite(acc), acc, 0.0)
        transform = acc_ds.transform

    rows, cols = acc.shape
    row_edges = np.linspace(0, rows, pour_point_count + 1, dtype=int)
    points: List[PourPoint] = []

    for i in range(pour_point_count, 0, -1):
        r0, r1 = row_edges[i - 1], row_edges[i]
        if r1 <= r0:
            continue
        window = acc[r0:r1, :]
        flat_idx = np.argmax(window)
        value = float(window.ravel()[flat_idx])
        if value < accumulation_threshold:
            continue
        local_row, local_col = divmod(flat_idx, window.shape[1])
        row = r0 + local_row
        col = local_col
        x, y = rasterio.transform.xy(transform, row, col, offset="center")
        if any(math.hypot(x - pp.x, y - pp.y) < min_spacing for pp in points):
            continue
        points.append(PourPoint(id=f"P{len(points)+1}", row=int(row), col=int(col), x=float(x), y=float(y), accumulation=value))

    if not points:
        raise RuntimeError("No pour points satisfied the accumulation threshold.")

    return points


def write_pour_points_geojson(pour_points: Sequence[PourPoint], path: Path) -> None:
    """Persist pour points to GeoJSON."""

    features = []
    for point in pour_points:
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [point.x, point.y],
            },
            "properties": {
                "id": point.id,
                "row": point.row,
                "col": point.col,
                "accumulation": point.accumulation,
                **(point.attributes or {}),
            },
        }
        features.append(feature)
    payload = {"type": "FeatureCollection", "features": features}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


