"""Lightweight DEM delineation fallback without external dependencies.

This module provides a very small grid based implementation that can be used
when ``rasterio``/``richdem`` are not available.  It operates on a JSON based
DEM description containing a regular grid of elevations together with the grid
spacing.  The goal is not to replace the full featured delineation pipeline but
to offer a deterministic, dependency free backend that is sufficient for the
unit tests and demonstration scripts shipped with the repository.

The algorithms implemented here intentionally favour readability over
performance.  They operate on plain Python lists so that the module can be used
on minimal environments where ``numpy`` is not present.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

GridLocation = Tuple[int, int]


@dataclass
class GridTransform:
    """Affine transform describing the position of the grid."""

    x_origin: float
    y_origin: float
    pixel_width: float
    pixel_height: float

    def cell_bounds(self, row: int, col: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the bounds of a grid cell as ``((xmin, ymin), (xmax, ymax))``."""

        xmin = self.x_origin + col * self.pixel_width
        xmax = xmin + self.pixel_width
        ymax = self.y_origin - row * self.pixel_height
        ymin = ymax - self.pixel_height
        return (xmin, ymin), (xmax, ymax)

    def cell_centre(self, row: int, col: int) -> Tuple[float, float]:
        """Return the centre coordinate of a cell."""

        (xmin, ymin), (xmax, ymax) = self.cell_bounds(row, col)
        return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0

    def to_grid_location(self, x: float, y: float) -> GridLocation:
        """Convert a coordinate to the corresponding grid location."""

        col = int((x - self.x_origin) / self.pixel_width)
        row = int((self.y_origin - y) / self.pixel_height)
        return row, col


@dataclass
class SimpleGridDEM:
    """Container holding DEM information for the lightweight delineator."""

    elevations: List[List[float]]
    transform: GridTransform
    crs: Optional[str] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return len(self.elevations), len(self.elevations[0]) if self.elevations else 0

    def iter_cells(self) -> Iterator[Tuple[int, int, float]]:
        for row, row_values in enumerate(self.elevations):
            for col, value in enumerate(row_values):
                yield row, col, value


def load_dem(path: Path) -> SimpleGridDEM:
    """Load a DEM from the JSON representation used by the examples."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    transform = GridTransform(
        x_origin=float(data["transform"]["x_origin"]),
        y_origin=float(data["transform"]["y_origin"]),
        pixel_width=float(data["transform"]["pixel_width"]),
        pixel_height=float(data["transform"]["pixel_height"]),
    )
    rows: List[List[float]] = []
    for row in data["grid"]:
        rows.append([float(value) for value in row])
    return SimpleGridDEM(rows, transform, data.get("crs"))


def load_pour_points(path: Path, transform: GridTransform) -> List[Tuple[str, GridLocation]]:
    """Load pour points from the GeoJSON helper file."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    features = data.get("features", [])
    pour_points: List[Tuple[str, GridLocation]] = []
    for feature in features:
        geometry = feature.get("geometry", {}) or {}
        coords = geometry.get("coordinates")
        if not coords:
            continue
        point_id = str(feature.get("properties", {}).get("id", len(pour_points) + 1))
        row, col = transform.to_grid_location(float(coords[0]), float(coords[1]))
        pour_points.append((point_id, (row, col)))
    return pour_points


def _flow_neighbour_offsets() -> Mapping[int, Tuple[int, int]]:
    return {
        0: (-1, 0),
        1: (-1, 1),
        2: (0, 1),
        3: (1, 1),
        4: (1, 0),
        5: (1, -1),
        6: (0, -1),
        7: (-1, -1),
    }


def compute_flow_directions(dem: SimpleGridDEM) -> Dict[GridLocation, Optional[GridLocation]]:
    """Return the downstream neighbour for each grid cell using a D8 scheme."""

    flow: Dict[GridLocation, Optional[GridLocation]] = {}
    rows, cols = dem.shape
    offsets = list(_flow_neighbour_offsets().values())

    for row, col, elevation in dem.iter_cells():
        best_location: Optional[GridLocation] = None
        best_drop = 0.0
        for drow, dcol in offsets:
            nrow, ncol = row + drow, col + dcol
            if not (0 <= nrow < rows and 0 <= ncol < cols):
                continue
            neighbour = dem.elevations[nrow][ncol]
            drop = elevation - neighbour
            if drop > best_drop:
                best_drop = drop
                best_location = (nrow, ncol)
        flow[(row, col)] = best_location
    return flow


def compute_flow_accumulation(
    dem: SimpleGridDEM, flow: Mapping[GridLocation, Optional[GridLocation]]
) -> Dict[GridLocation, float]:
    """Compute flow accumulation counts for each grid cell."""

    flat_cells = sorted(((elev, row, col) for row, col, elev in dem.iter_cells()), reverse=True)
    accumulation: Dict[GridLocation, float] = { (row, col): 1.0 for _, row, col in flat_cells }

    for _, row, col in flat_cells:
        downstream = flow[(row, col)]
        if downstream is not None:
            accumulation[downstream] += accumulation[(row, col)]
    return accumulation


def delineate_watersheds(
    flow: Mapping[GridLocation, Optional[GridLocation]],
    pour_points: Sequence[Tuple[str, GridLocation]],
) -> Dict[str, List[GridLocation]]:
    """Assign each grid cell to the pour point it ultimately drains to."""

    mapping: Dict[str, List[GridLocation]] = {pid: [] for pid, _ in pour_points}
    target_lookup: Dict[GridLocation, str] = {location: pid for pid, location in pour_points}

    for location in flow:
        visited: List[GridLocation] = []
        current = location
        while current is not None and current not in target_lookup and current not in visited:
            visited.append(current)
            current = flow[current]
        if current in target_lookup:
            basin_id = target_lookup[current]
            mapping[basin_id].append(location)
    for basin_id, outlet in pour_points:
        mapping.setdefault(basin_id, []).append(outlet)
    return mapping


def downstream_relations(
    flow: Mapping[GridLocation, Optional[GridLocation]],
    pour_points: Sequence[Tuple[str, GridLocation]],
) -> Dict[str, Optional[str]]:
    """Infer downstream pour point relationships by following flow directions."""

    reverse_lookup = {location: pid for pid, location in pour_points}
    relations: Dict[str, Optional[str]] = {pid: None for pid, _ in pour_points}

    for pid, location in pour_points:
        downstream = flow.get(location)
        while downstream is not None:
            if downstream in reverse_lookup:
                relations[pid] = reverse_lookup[downstream]
                break
            downstream = flow.get(downstream)
    return relations


def build_subbasin_polygons(
    dem: SimpleGridDEM,
    watersheds: Mapping[str, Sequence[GridLocation]],
) -> Dict[str, List[List[Tuple[float, float]]]]:
    """Return simple polygon outlines for each watershed.

    The polygons are generated as bounding boxes covering the contributing
    cells.  While simplistic, the approach keeps the representation light weight
    and sufficient for producing schematic GIS figures in documentation.
    """

    polygons: Dict[str, List[List[Tuple[float, float]]]] = {}
    for basin_id, cells in watersheds.items():
        if not cells:
            continue
        min_row = min(row for row, _ in cells)
        max_row = max(row for row, _ in cells)
        min_col = min(col for _, col in cells)
        max_col = max(col for _, col in cells)
        (xmin, ymin), _ = dem.transform.cell_bounds(min_row, min_col)
        _, (xmax, ymax) = dem.transform.cell_bounds(max_row, max_col)
        polygon = [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
            (xmin, ymin),
        ]
        polygons[basin_id] = [polygon]
    return polygons


def delineate_from_json(
    dem_path: Path,
    pour_points_path: Path,
    accumulation_threshold: float,
) -> Tuple[
    SimpleGridDEM,
    Dict[str, List[GridLocation]],
    Dict[str, Optional[str]],
    Dict[GridLocation, float],
]:
    """Entry point used by :class:`~hydrosis.delineation.dem_delineator.DelineationConfig`."""

    dem = load_dem(dem_path)
    pour_points = load_pour_points(pour_points_path, dem.transform)
    if not pour_points:
        raise RuntimeError("No pour points defined for JSON DEM delineation")

    flow = compute_flow_directions(dem)
    accumulation = compute_flow_accumulation(dem, flow)

    watersheds = delineate_watersheds(flow, pour_points)
    if accumulation_threshold > 0:
        filtered: Dict[str, List[GridLocation]] = {}
        for basin_id, cells in watersheds.items():
            filtered[basin_id] = [
                cell
                for cell in cells
                if accumulation[cell] >= accumulation_threshold
            ]
        watersheds = filtered

    relations = downstream_relations(flow, pour_points)
    return dem, watersheds, relations, accumulation


def watershed_area(square_km_per_cell: float, cells: Sequence[GridLocation]) -> float:
    return square_km_per_cell * float(len(cells))


def grid_to_geojson_features(
    dem: SimpleGridDEM,
    values: Mapping[GridLocation, float],
    property_name: str,
) -> List[Mapping[str, object]]:
    """Convert cell wise values to GeoJSON polygon features."""

    features: List[Mapping[str, object]] = []
    for (row, col), value in values.items():
        (xmin, ymin), (xmax, ymax) = dem.transform.cell_bounds(row, col)
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [xmin, ymin],
                            [xmax, ymin],
                            [xmax, ymax],
                            [xmin, ymax],
                            [xmin, ymin],
                        ]
                    ],
                },
                "properties": {property_name: value, "row": row, "col": col},
            }
        )
    return features


__all__ = [
    "SimpleGridDEM",
    "GridTransform",
    "delineate_from_json",
    "build_subbasin_polygons",
    "grid_to_geojson_features",
    "watershed_area",
]

