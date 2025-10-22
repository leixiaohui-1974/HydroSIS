"""Utilities for generating perturbed rain gauge series and Thiessen polygons."""
from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.errors import GEOSException


def generate_station_ids(count: int) -> list[str]:
    if count <= 0:
        raise ValueError("Station count must be positive.")
    return [f"station_{idx:02d}" for idx in range(1, count + 1)]


def perturb_precipitation_series(
    base_series: pd.Series,
    station_ids: Sequence[str],
    perturbation: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    if base_series.empty:
        raise ValueError("Base precipitation series must contain data.")
    if not 0.0 <= perturbation <= 1.0:
        raise ValueError("Perturbation must lie between 0 and 1 (fractional amplitude).")
    if rng is None:
        rng = np.random.default_rng()

    base_values = base_series.to_numpy(dtype=float)
    station_matrix = np.empty((base_values.size, len(station_ids)), dtype=float)
    for index, station_id in enumerate(station_ids):
        noise = rng.uniform(-perturbation, perturbation, size=base_values.size)
        station_matrix[:, index] = np.clip(base_values * (1.0 + noise), a_min=0.0, a_max=None)
    return pd.DataFrame(station_matrix, index=base_series.index, columns=list(station_ids))


def _ensure_polygon(geometry: BaseGeometry) -> BaseGeometry:
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if geometry.geom_type == "GeometryCollection":
        polygons = [geom for geom in geometry.geoms if geom.geom_type in {"Polygon", "MultiPolygon"}]
        if not polygons:
            raise ValueError("Boundary geometry does not contain polygon components.")
        merged = polygons[0]
        for geom in polygons[1:]:
            merged = merged.union(geom)
        return merged
    raise TypeError("Boundary geometry must be a Polygon or MultiPolygon.")


def sample_station_positions(
    boundary: BaseGeometry,
    station_ids: Sequence[str],
    rng: Optional[np.random.Generator] = None,
    max_attempts: int = 10000,
) -> Dict[str, Point]:
    if rng is None:
        rng = np.random.default_rng()
    polygon = _ensure_polygon(boundary).buffer(0)
    minx, miny, maxx, maxy = polygon.bounds
    extent_x = maxx - minx
    extent_y = maxy - miny
    if extent_x <= 0 or extent_y <= 0:
        raise ValueError("Boundary geometry must have positive extent.")
    bounding_area = extent_x * extent_y
    coverage_ratio = polygon.area / bounding_area if bounding_area > 0 else 0.0
    per_station_attempts = max_attempts
    if coverage_ratio > 0:
        estimated_attempts = int(math.ceil(5.0 / coverage_ratio))
        per_station_attempts = max(max_attempts, estimated_attempts)
    per_station_attempts = min(per_station_attempts, 1_000_000)
    points: Dict[str, Point] = {}
    for station_id in station_ids:
        station_attempts = 0
        while station_attempts < per_station_attempts:
            station_attempts += 1
            candidate = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
            if polygon.covers(candidate):
                points[station_id] = candidate
                break
        else:
            raise RuntimeError("Unable to sample station locations inside the boundary.")
    return points


def _voronoi_finite_polygons_2d(vor: Voronoi, radius: Optional[float] = None) -> Tuple[list[list[int]], np.ndarray]:
    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2-dimensional.")
    if radius is None:
        spread = np.ptp(vor.points, axis=0)
        radius = float(spread.max() * 2.0) if np.any(spread) else 1.0

    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    all_ridges: Dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for point_index, region_index in enumerate(vor.point_region):
        region_vertices = vor.regions[region_index]
        if -1 not in region_vertices:
            new_regions.append(region_vertices)
            continue

        ridge_vertices: list[int] = [v for v in region_vertices if v >= 0]
        for neighbour, v1, v2 in all_ridges.get(point_index, []):
            if v1 >= 0 and v2 >= 0:
                continue

            tangent = vor.points[neighbour] - vor.points[point_index]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])

            midpoint = (vor.points[point_index] + vor.points[neighbour]) / 2.0
            direction = np.sign(np.dot(midpoint - center, normal)) * normal

            base_vertex_index = v1 if v1 >= 0 else v2
            if base_vertex_index >= 0:
                base_point = vor.vertices[base_vertex_index]
            else:
                base_point = midpoint

            new_point = base_point + direction * radius
            new_vertices.append(new_point.tolist())
            ridge_vertices.append(len(new_vertices) - 1)

        polygon = np.asarray([new_vertices[v] for v in ridge_vertices])
        centroid = polygon.mean(axis=0)
        angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
        ordered = np.array(ridge_vertices)[np.argsort(angles)]
        new_regions.append(ordered.tolist())

    return new_regions, np.asarray(new_vertices)


def thiessen_polygons_for_stations(
    positions: Mapping[str, Point],
    boundary: BaseGeometry,
    radius: Optional[float] = None,
) -> Dict[str, BaseGeometry]:
    if len(positions) < 2:
        raise ValueError("At least two stations are required to build Thiessen polygons.")
    polygon = _ensure_polygon(boundary)

    station_ids = list(positions.keys())
    coords = np.array([[positions[sid].x, positions[sid].y] for sid in station_ids])
    vor = Voronoi(coords)
    regions, vertices = _voronoi_finite_polygons_2d(vor, radius=radius)

    polygons: Dict[str, BaseGeometry] = {}
    for index, region in enumerate(regions):
        sid = station_ids[index]
        poly = Polygon(vertices[region]).intersection(polygon)
        if poly.is_empty:
            continue
        polygons[sid] = poly
    return polygons


def compute_subbasin_station_weights(
    subbasins: Mapping[str, BaseGeometry],
    thiessen_polygons: Mapping[str, BaseGeometry],
    tolerance: float = 1e-6,
) -> Dict[str, Dict[str, float]]:
    if not subbasins:
        raise ValueError("No subbasin geometries provided.")
    if not thiessen_polygons:
        raise ValueError("No Thiessen polygons provided.")

    weights: Dict[str, Dict[str, float]] = {}
    for sub_id, geom in subbasins.items():
        polygon = _ensure_polygon(geom)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        area = polygon.area
        if area <= 0.0:
            raise ValueError(f"Subbasin '{sub_id}' has non-positive area.")

        station_weights: MutableMapping[str, float] = {}
        for sid, region in thiessen_polygons.items():
            region_geom = _ensure_polygon(region)
            if not region_geom.is_valid:
                region_geom = region_geom.buffer(0)
            if region_geom.is_empty:
                continue
            try:
                overlap_area = polygon.intersection(region_geom).area
            except GEOSException:
                overlap_area = polygon.buffer(0).intersection(region_geom.buffer(0)).area
            if overlap_area > 0.0:
                station_weights[sid] = overlap_area / area

        total = sum(station_weights.values())
        if total <= tolerance:
            uniform = 1.0 / len(thiessen_polygons)
            station_weights = {sid: uniform for sid in thiessen_polygons.keys()}
        else:
            station_weights = {sid: val / total for sid, val in station_weights.items()}
        weights[sub_id] = dict(station_weights)
    return weights


def interpolate_station_series(
    station_series: pd.DataFrame,
    weights: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    if station_series.empty:
        raise ValueError("Station series DataFrame is empty.")
    missing: set[str] = set()
    for weight_map in weights.values():
        missing.update(weight_map.keys())
    missing -= set(station_series.columns)
    if missing:
        raise KeyError(f"Station series missing columns for stations: {sorted(missing)}")

    subbasin_data = {}
    station_values = station_series.to_numpy(dtype=float)
    column_index = {col: idx for idx, col in enumerate(station_series.columns)}

    for sub_id, weight_map in weights.items():
        if not weight_map:
            subbasin_data[sub_id] = np.zeros(station_values.shape[0], dtype=float)
            continue
        series = np.zeros(station_values.shape[0], dtype=float)
        for station_id, weight in weight_map.items():
            series += station_values[:, column_index[station_id]] * float(weight)
        subbasin_data[sub_id] = series

    return pd.DataFrame(subbasin_data, index=station_series.index)


__all__ = [
    "generate_station_ids",
    "perturb_precipitation_series",
    "sample_station_positions",
    "thiessen_polygons_for_stations",
    "compute_subbasin_station_weights",
    "interpolate_station_series",
]
