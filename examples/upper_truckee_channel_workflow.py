"""Upper Truckee end-to-end example demonstrating channel-aware routing."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Polygon as MplPolygon

from hydrosis.config import (
    DelineationConfig,
    IOConfig,
    ModelConfig,
    ModelStructureConfig,
    OutputArtifactsConfig,
    ParameterPartitionConfig,
    RoutingModelConfig,
    RunoffModelConfig,
)
from hydrosis.delineation import utils as dutils
from hydrosis.delineation.channel_analysis import (
    compute_channel_mask,
    segments_to_feature_collection,
    trace_channel_segments,
)
from hydrosis.workflow.stages import (
    aggregate_parameter_precipitation,
    generate_precipitation_for_parameters,
    run_channel_diagnostics,
    run_delineation_stage,
    snap_pour_points_to_flow_cells,
    suggest_accumulation_threshold,
)
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from hydrosis.parameters.partition import partition_parameter_zones
from hydrosis.model import ChannelNetwork
from hydrosis.workflow import run_workflow

from examples.multi_model_storm_comparison import (
    PRECIP_COLUMN,
    ROUTING_LIBRARY,
    RUNOFF_LIBRARY,
    _build_parameter_zones,
    _channel_routing_details,
    _terminal_subbasin_id,
    generate_report,
    generate_storm_forcing,
    plot_comparison,
    prepare_metrics,
)


def _write_geojson(features: Dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(features, indent=2), encoding="utf-8")


def _build_pour_point_collection(pour_points: List[dutils.PourPoint]) -> Dict[str, object]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [pp.x, pp.y]},
                "properties": {
                    "id": pp.id,
                    "row": pp.row,
                    "col": pp.col,
                    "accumulation": pp.accumulation,
                },
            }
            for pp in pour_points
        ],
    }


def _load_time_indexed_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{label} is empty: {path}")
    time_column = df.columns[0]
    try:
        df[time_column] = pd.to_datetime(df[time_column])
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unable to parse timestamps in {path}") from exc
    df.set_index(time_column, inplace=True)
    df.index.name = "Timestamp"
    return df


def _load_subbasin_geometries(subbasin_path: Path) -> Dict[str, object]:
    if not subbasin_path.exists():
        raise FileNotFoundError(f"Subbasin geometry file not found: {subbasin_path}")
    data = json.loads(subbasin_path.read_text(encoding="utf-8"))
    geometries: Dict[str, object] = {}
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        subbasin_id = str(properties.get("id", "")).strip()
        if not subbasin_id:
            continue
        geometries[subbasin_id] = shape(feature.get("geometry"))
    if not geometries:
        raise ValueError(f"No subbasin geometries could be loaded from {subbasin_path}")
    return geometries


def _load_parameter_subbasins(path: Path) -> Tuple[Dict[str, object], Dict[str, str], Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"Parameter subbasin geometry file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    geometries: Dict[str, object] = {}
    parent_map: Dict[str, str] = {}
    area_map: Dict[str, float] = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        subzone_id = str(props.get("subzone_id") or props.get("id") or "")
        if not subzone_id:
            continue
        zone_id = str(props.get("zone_id") or props.get("parent_id") or subzone_id)
        geom = shape(feature.get("geometry"))
        area_km2 = float(props.get("area_km2")) if props.get("area_km2") is not None else float(geom.area / 1_000_000.0)
        geometries[subzone_id] = geom
        parent_map[subzone_id] = zone_id
        area_map[subzone_id] = area_km2
    if not geometries:
        raise ValueError(f"No parameter subbasin geometries could be loaded from {path}")
    return geometries, parent_map, area_map


def _load_thiessen_polygons(path: Path) -> Dict[str, BaseGeometry]:
    if not path.exists():
        raise FileNotFoundError(f"Thiessen polygon file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    polygons: Dict[str, BaseGeometry] = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        station_id = str(props.get("station_id") or props.get("id") or "").strip()
        if not station_id:
            continue
        polygons[station_id] = shape(feature.get("geometry"))
    if not polygons:
        raise ValueError(f"No Thiessen polygons found in {path}")
    return polygons


def _compute_parameter_station_weights(
    parameter_geometries: Mapping[str, BaseGeometry],
    thiessen_polygons: Mapping[str, BaseGeometry],
    tolerance: float = 1e-9,
) -> Dict[str, Dict[str, float]]:
    if not thiessen_polygons:
        raise ValueError("Thiessen polygons must be provided to compute station weights.")
    weights: Dict[str, Dict[str, float]] = {}
    for sub_id, geom in parameter_geometries.items():
        polygon = geom
        area = polygon.area
        if area <= tolerance:
            raise ValueError(f"Parameter subbasin '{sub_id}' has non-positive area.")
        station_weights: Dict[str, float] = {}
        for station_id, thiessen in thiessen_polygons.items():
            overlap_area = polygon.intersection(thiessen).area
            if overlap_area > tolerance:
                station_weights[station_id] = overlap_area / area
        total = sum(station_weights.values())
        if total <= tolerance:
            uniform = 1.0 / len(thiessen_polygons)
            station_weights = {station_id: uniform for station_id in thiessen_polygons.keys()}
        else:
            station_weights = {station_id: value / total for station_id, value in station_weights.items()}
        weights[sub_id] = station_weights
    return weights


def _interpolate_parameter_precipitation(
    station_series: pd.DataFrame,
    weights: Mapping[str, Mapping[str, float]],
) -> pd.DataFrame:
    parameter_precip = pd.DataFrame(index=station_series.index)
    for sub_id, weight_map in weights.items():
        if not weight_map:
            parameter_precip[sub_id] = 0.0
            continue
        missing = [sid for sid in weight_map if sid not in station_series.columns]
        if missing:
            raise KeyError(f"Rain gauge series missing stations for subbasin '{sub_id}': {missing}")
        series = sum(station_series[sid] * float(weight) for sid, weight in weight_map.items())
        parameter_precip[sub_id] = series
    parameter_precip.index.name = station_series.index.name
    return parameter_precip


def _build_pour_point_collection(pour_points: List[dutils.PourPoint]) -> Dict[str, object]:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [pp.x, pp.y]},
                "properties": {
                    "id": pp.id,
                    "row": pp.row,
                    "col": pp.col,
                    "accumulation": pp.accumulation,
                },
            }
            for pp in pour_points
        ],
    }


def _snap_pour_points_to_flow_cells(
    pour_points: List[dutils.PourPoint],
    flow_dir_path: Path,
    flow_acc_path: Path,
    max_radius: int = 30,
) -> List[dutils.PourPoint]:
    """Snap automatically derived pour points onto valid flow direction cells."""

    flowdir, upstream, shape = dutils.build_flow_network(flow_dir_path)
    rows, cols = shape
    with rasterio.open(flow_acc_path) as acc_ds:
        accumulation = acc_ds.read(1)
        transform = acc_ds.transform

    def drains_to_target(row: int, col: int, target: Tuple[int, int]) -> bool:
        d8 = dutils.D8_OFFSETS
        visited: Set[Tuple[int, int]] = set()
        r, c = row, col
        while (r, c) != target:
            if (r, c) in visited:
                return False
            visited.add((r, c))
            code = int(flowdir[r, c])
            offset = d8.get(code)
            if offset is None:
                return False
            r += offset[0]
            c += offset[1]
            if not (0 <= r < rows and 0 <= c < cols):
                return False
        return True

    valid_directions = set(DelineationConfig._direction_mapping().keys())
    adjusted: List[dutils.PourPoint] = []

    for point in pour_points:
        base_row, base_col = point.row, point.col
        best_row, best_col = base_row, base_col
        base_acc = float(accumulation[base_row, base_col]) if np.isfinite(accumulation[base_row, base_col]) else 0.0
        best_acc = -np.inf
        best_dist_sq = float("inf")

        if int(flowdir[base_row, base_col]) not in valid_directions:
            for radius in range(1, max_radius + 1):
                rmin = max(0, base_row - radius)
                rmax = min(rows, base_row + radius + 1)
                cmin = max(0, base_col - radius)
                cmax = min(cols, base_col + radius + 1)
                for r in range(rmin, rmax):
                    for c in range(cmin, cmax):
                        code = int(flowdir[r, c])
                        if code not in valid_directions:
                            continue
                        acc_val = float(accumulation[r, c])
                        if not np.isfinite(acc_val):
                            continue
                        dist_sq = (r - base_row) ** 2 + (c - base_col) ** 2
                        if (
                            acc_val > best_acc
                            or (math.isclose(acc_val, best_acc) and dist_sq < best_dist_sq)
                        ):
                            best_row, best_col = r, c
                            best_acc = acc_val
                            best_dist_sq = dist_sq
            if not np.isfinite(best_acc) or best_acc == -np.inf:
                warnings.warn(
                    f"Unable to snap pour point '{point.id}' to a valid flow direction cell; using original location.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                best_row, best_col = base_row, base_col
                best_acc = base_acc
        else:
            best_acc = base_acc
            best_dist_sq = 0.0

        world_x, world_y = rasterio.transform.xy(transform, best_row, best_col, offset="center")
        adjusted.append(
            dutils.PourPoint(
                id=point.id,
                row=int(best_row),
                col=int(best_col),
                x=float(world_x),
                y=float(world_y),
                accumulation=float(best_acc),
            )
        )

    if not adjusted:
        return adjusted

    outlet_point = max(adjusted, key=lambda p: p.accumulation)
    outlet_coord = (outlet_point.row, outlet_point.col)
    outlet_mask = dutils.delineate_watershed(outlet_point, upstream, shape)
    used_cells = {(pp.row, pp.col) for pp in adjusted if drains_to_target(pp.row, pp.col, outlet_coord)}

    for idx, point in enumerate(adjusted):
        if drains_to_target(point.row, point.col, outlet_coord):
            continue

        best_candidate: Optional[Tuple[float, float, int, int]] = None  # (acc, dist_sq, row, col)
        max_search = max(rows, cols)
        required_acc = max(5000.0, float(point.accumulation))
        for radius in range(10, max_search, 10):
            r_min = max(0, point.row - radius)
            r_max = min(rows, point.row + radius + 1)
            c_min = max(0, point.col - radius)
            c_max = min(cols, point.col + radius + 1)
            for r in range(r_min, r_max):
                row_mask = outlet_mask[r]
                if not row_mask.any():
                    continue
                for c in range(c_min, c_max):
                    if not row_mask[c]:
                        continue
                    if (r, c) in used_cells:
                        continue
                    acc_val = float(accumulation[r, c])
                    if acc_val <= 0.0:
                        continue
                    if not drains_to_target(r, c, outlet_coord):
                        continue
                    dist_sq = (r - point.row) ** 2 + (c - point.col) ** 2
                    if best_candidate is None or acc_val > best_candidate[0] or (
                        acc_val == best_candidate[0] and dist_sq < best_candidate[1]
                    ):
                        best_candidate = (acc_val, dist_sq, r, c)
            if best_candidate is not None and best_candidate[0] >= required_acc:
                break

        if best_candidate is None:
            # Fall back to highest accumulation cell in outlet watershed that is not already used.
            flat_indices = np.argwhere(outlet_mask)
            fallback = None
            for r, c in flat_indices:
                if (int(r), int(c)) in used_cells:
                    continue
                if not drains_to_target(int(r), int(c), outlet_coord):
                    continue
                acc_val = float(accumulation[int(r), int(c)])
                if fallback is None or acc_val > fallback[0]:
                    fallback = (acc_val, int(r), int(c))
            if fallback is None:
                continue
            acc_val, row_sel, col_sel = fallback
        else:
            acc_val, _, row_sel, col_sel = best_candidate

        world_x, world_y = rasterio.transform.xy(transform, row_sel, col_sel, offset="center")
        adjusted[idx] = dutils.PourPoint(
            id=point.id,
            row=int(row_sel),
            col=int(col_sel),
            x=float(world_x),
            y=float(world_y),
            accumulation=float(acc_val),
        )
        used_cells.add((int(row_sel), int(col_sel)))

    return adjusted


def _auto_accumulation_threshold(pour_points: List[dutils.PourPoint], default: float) -> float:
    """Compute a conservative accumulation threshold based on pour point characteristics."""

    finite_values = [pp.accumulation for pp in pour_points if np.isfinite(pp.accumulation) and pp.accumulation > 0]
    if not finite_values:
        return 0.0
    min_acc = min(finite_values)
    candidate = min(default, min_acc * 0.5)
    if candidate < 1.0:
        return 0.0
    return candidate


@dataclass
class ZoneNode:
    id: str
    pour_point: dutils.PourPoint
    downstream_id: Optional[str]
    runoff_method: str = "HBV"
    routing_method: str = "Muskingum"


def _plot_feature_collection(
    collection: Dict[str, object],
    output_path: Path,
    color_key: str,
    title: str,
    line_width: float = 1.5,
    channel_collection: Optional[Dict[str, object]] = None,
) -> None:
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    patches: List[MplPolygon] = []
    colors: List[float] = []
    color_map: Dict[str, int] = {}
    current_index = 0
    xs_all: List[float] = []
    ys_all: List[float] = []

    def _get_color_index(value: Optional[str]) -> int:
        nonlocal current_index
        key = value or "None"
        if key not in color_map:
            color_map[key] = current_index
            current_index += 1
        return color_map[key]

    for feature in collection.get("features", []):
        geom = feature.get("geometry", {})
        props = feature.get("properties", {}) or {}
        color_value = props.get(color_key)
        color_index = _get_color_index(color_value)
        geom_type = geom.get("type")
        coords = geom.get("coordinates", [])
        if geom_type == "MultiPolygon":
            for polygon in coords:
                if not polygon:
                    continue
                ring = polygon[0]
                patches.append(MplPolygon(ring, closed=True))
                colors.append(color_index)
                xs_all.extend(x for x, _ in ring)
                ys_all.extend(y for _, y in ring)
        elif geom_type == "Polygon":
            if coords:
                patches.append(MplPolygon(coords[0], closed=True))
                colors.append(color_index)
                xs_all.extend(x for x, _ in coords[0])
                ys_all.extend(y for _, y in coords[0])
        elif geom_type == "LineString":
            if coords:
                xs, ys = zip(*coords)
                ax.plot(xs, ys, linewidth=line_width, color=plt.cm.tab20(color_index % 20))
                xs_all.extend(xs)
                ys_all.extend(ys)

    if patches:
        collection_patch = PatchCollection(patches, cmap=plt.cm.tab20, alpha=0.6, edgecolor="black", linewidth=0.6)
        collection_patch.set_array(np.array(colors))
        ax.add_collection(collection_patch)
        ax.autoscale_view()

    if channel_collection and channel_collection.get("features"):
        for feature in channel_collection["features"]:
            geom = feature.get("geometry", {})
            if geom.get("type") == "LineString":
                coords = geom.get("coordinates", [])
                if coords:
                    xs, ys = zip(*coords)
                    ax.plot(xs, ys, color="black", linewidth=1.2, alpha=0.8)
                    xs_all.extend(xs)
                    ys_all.extend(ys)

    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if xs_all and ys_all:
        ax.set_xlim(min(xs_all), max(xs_all))
        ax.set_ylim(min(ys_all), max(ys_all))

    legend_labels = [label for label in color_map.keys() if label != "None"]
    if legend_labels:
        legend_handles = [
            Patch(facecolor=plt.cm.tab20(color_map[label] % 20), edgecolor="black", alpha=0.6)
            for label in legend_labels
        ]
        ax.legend(legend_handles, legend_labels, loc="best", fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def _compute_downstream_map(
    pour_points: Dict[str, dutils.PourPoint],
    flowdir: np.ndarray,
) -> Tuple[Dict[str, Optional[str]], Dict[str, List[str]]]:
    coord_lookup = {(pp.row, pp.col): pid for pid, pp in pour_points.items()}
    downstream_map: Dict[str, Optional[str]] = {}
    child_map: Dict[str, List[str]] = {pid: [] for pid in pour_points}
    rows, cols = flowdir.shape
    d8 = dutils.D8_OFFSETS

    for pid, pp in pour_points.items():
        r, c = pp.row, pp.col
        visited: Set[Tuple[int, int]] = set()
        downstream: Optional[str] = None
        while True:
            code = int(flowdir[r, c])
            offset = d8.get(code)
            if offset is None:
                break
            nr, nc = r + offset[0], c + offset[1]
            if not (0 <= nr < rows and 0 <= nc < cols):
                break
            if (nr, nc) in visited:
                break
            visited.add((nr, nc))
            if (nr, nc) in coord_lookup:
                downstream = coord_lookup[(nr, nc)]
                break
            r, c = nr, nc
        downstream_map[pid] = downstream
        if downstream:
            child_map.setdefault(downstream, []).append(pid)

    return downstream_map, child_map


def _find_branch_zones(
    zone_id: str,
    zone_mask: np.ndarray,
    upstream_point: dutils.PourPoint,
    downstream_point: Optional[dutils.PourPoint],
    flowdir: np.ndarray,
    channel_mask: np.ndarray,
    accumulation: np.ndarray,
    upstream: List[List[Tuple[int, int]]],
    transform: rasterio.Affine,
    min_branch_area_cells: int,
    max_branches: int,
    existing_ids: Set[str],
) -> List[Tuple[str, dutils.PourPoint, np.ndarray]]:
    rows, cols = flowdir.shape
    target = (downstream_point.row, downstream_point.col) if downstream_point is not None else None
    main_path = _trace_flow_path(flowdir, (upstream_point.row, upstream_point.col), target, zone_mask)
    main_path_set = set(main_path)

    candidates: List[Tuple[float, Tuple[int, int], np.ndarray]] = []
    evaluated: Set[Tuple[int, int]] = set()
    for cell in main_path:
        neighbors = _upstream_channel_neighbors(flowdir, channel_mask, cell)
        for nr, nc in neighbors:
            if (nr, nc) in main_path_set or (nr, nc) in evaluated:
                continue
            evaluated.add((nr, nc))
            candidate_pp = dutils.PourPoint(id=f"{zone_id}_branch", row=nr, col=nc, x=0.0, y=0.0, accumulation=0.0)
            mask = dutils.delineate_watershed(candidate_pp, upstream, flowdir.shape)
            mask = np.logical_and(mask, zone_mask)
            area_cells = int(mask.sum())
            if area_cells < min_branch_area_cells:
                continue
            candidates.append((float(area_cells), (nr, nc), mask))

    candidates.sort(reverse=True)
    branches: List[Tuple[str, dutils.PourPoint, np.ndarray]] = []
    assigned = np.zeros_like(zone_mask, dtype=bool)
    suffix_index = 0

    for _, (row, col), mask in candidates:
        mask = np.logical_and(mask, np.logical_not(assigned))
        area_cells = int(mask.sum())
        if area_cells < min_branch_area_cells:
            continue
        # assign unique id
        while True:
            candidate_id = f"{zone_id}{chr(ord('a') + suffix_index)}"
            suffix_index += 1
            if candidate_id not in existing_ids:
                break
        existing_ids.add(candidate_id)
        assigned = np.logical_or(assigned, mask)
        x, y = rasterio.transform.xy(transform, row, col, offset="center")
        branch_pp = dutils.PourPoint(
            id=candidate_id,
            row=int(row),
            col=int(col),
            x=float(x),
            y=float(y),
            accumulation=float(accumulation[row, col]),
        )
        branches.append((candidate_id, branch_pp, mask))
        if len(branches) >= max_branches:
            break

    return branches


def _find_downstream_subzone(
    flowdir: np.ndarray,
    subzone_index_grid: np.ndarray,
    subzone_ids: List[str],
    current_index: int,
    start_row: int,
    start_col: int,
    max_steps: int = 20000,
) -> Optional[str]:
    rows, cols = flowdir.shape
    r, c = start_row, start_col
    d8 = dutils.D8_OFFSETS
    steps = 0
    visited: Set[Tuple[int, int]] = set()

    while steps < max_steps:
        code = int(flowdir[r, c])
        offset = d8.get(code)
        if offset is None:
            return None
        r += offset[0]
        c += offset[1]
        if not (0 <= r < rows and 0 <= c < cols):
            return None
        if (r, c) in visited:
            return None
        visited.add((r, c))
        next_index = subzone_index_grid[r, c]
        if next_index >= 0 and next_index != current_index:
            return subzone_ids[next_index]
        steps += 1
    return None


def _export_intermediate_delineation_artifacts(
    dem_path: Path,
    flow_acc_path: Path,
    flow_dir_path: Path,
    pour_points: List[dutils.PourPoint],
    delineation_config: DelineationConfig,
    partition_cfg: ParameterPartitionConfig,
    channel_threshold: float,
    output_dir: Path,
) -> None:
    """Generate watershed delineation, parameter zone, subbasin, and channel artifacts."""
    try:
        _export_intermediate_delineation_artifacts_impl(
            dem_path,
            flow_acc_path,
            flow_dir_path,
            pour_points,
            delineation_config,
            partition_cfg,
            channel_threshold,
            output_dir,
        )
    except Exception as exc:
        print(f"[parameterization] error: {exc}", file=sys.stderr)
        raise


def _export_intermediate_delineation_artifacts_impl(
    dem_path: Path,
    flow_acc_path: Path,
    flow_dir_path: Path,
    pour_points: List[dutils.PourPoint],
    delineation_config: DelineationConfig,
    partition_cfg: ParameterPartitionConfig,
    channel_threshold: float,
    output_dir: Path,
) -> None:
    """Implementation helper for exporting artifacts."""

    # --- Base raster/material preparation -----------------------------------------
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    parameter_dir = output_dir / "parameters"
    parameter_dir.mkdir(parents=True, exist_ok=True)

    flowdir, upstream, shape = dutils.build_flow_network(flow_dir_path)
    rows, cols = shape

    with rasterio.open(dem_path) as dem_ds:
        transform = dem_ds.transform
        dem_data = dem_ds.read(1)
        dem_data = np.where(np.isfinite(dem_data), dem_data, np.nan)
        dem_crs = dem_ds.crs

    with rasterio.open(flow_acc_path) as acc_ds:
        accumulation = acc_ds.read(1)
    accumulation = np.where(np.isfinite(accumulation), accumulation, 0.0)

    masks = {pp.id: dutils.delineate_watershed(pp, upstream, shape) for pp in pour_points}
    polygons = dutils.masks_to_polygons(masks, transform)

    subbasin_features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": [coords]},
                "properties": {"id": basin_id},
            }
            for basin_id, coords in polygons.items()
            if coords
        ],
    }
    (intermediate_dir / "subbasins.geojson").write_text(json.dumps(subbasin_features, indent=2), encoding="utf-8")

    dutils.plot_overview_map(
        dem_data,
        transform,
        polygons,
        pour_points,
        intermediate_dir / "overview_map.png",
        dem_crs,
    )

    accumulation_plot = np.where(accumulation > 0, accumulation, np.nan)
    dutils.plot_raster(
        np.log1p(accumulation_plot),
        "Log1p flow accumulation",
        intermediate_dir / "flow_accumulation.png",
        "inferno",
    )
    dutils.plot_flow_direction(flowdir, intermediate_dir / "flow_direction.png")
    dutils.plot_masks(masks, pour_points, intermediate_dir / "watershed_masks.png")

    channel_mask = compute_channel_mask(accumulation, channel_threshold)
    segments = trace_channel_segments(flowdir, upstream, channel_mask, transform, dem_data)
    network = ChannelNetwork()
    for segment in segments:
        network.add_segment(segment)

    channel_features = segments_to_feature_collection(segments, transform)
    (intermediate_dir / "channel_network.geojson").write_text(json.dumps(channel_features, indent=2), encoding="utf-8")

    segment_records = [
        {
            "id": seg.id,
            "length_m": float(seg.length_m),
            "drop_m": float(seg.drop_m) if seg.drop_m is not None else None,
            "slope": float(seg.slope) if seg.slope is not None else None,
            "downstream_id": seg.downstream,
            "upstream_ids": seg.upstream_ids,
            "start_elevation": float(seg.start_elevation) if seg.start_elevation is not None else None,
            "end_elevation": float(seg.end_elevation) if seg.end_elevation is not None else None,
            "cells": seg.cells,
        }
        for seg in segments
    ]
    (intermediate_dir / "channel_segments.json").write_text(json.dumps(segment_records, indent=2), encoding="utf-8")

    channel_mask_plot = np.where(channel_mask, 1.0, np.nan)
    dutils.plot_raster(
        channel_mask_plot,
        f"Channel mask (threshold={channel_threshold:g})",
        intermediate_dir / "channel_mask.png",
        "Blues",
    )

    statistics = dutils.compute_statistics(pour_points, masks, dem_data, accumulation, transform)
    dutils.write_summary(statistics, intermediate_dir)
    dutils.write_attributes_csv(statistics, intermediate_dir)

    # --- Parameter zones and hierarchical splits ----------------------------------
    outputs_cfg = OutputArtifactsConfig()

    delineation_config.flow_direction_path = flow_dir_path
    delineation_config.flow_accumulation_path = flow_acc_path
    delineation_config.intermediate_directory = intermediate_dir
    delineation_config.parameter_directory = parameter_dir

    model_structure = ModelStructureConfig(
        default_runoff_model="HBV",
        default_routing_model="Muskingum",
    )

    partition_outputs = partition_parameter_zones(
        delineation_config,
        partition_cfg,
        model_structure,
        outputs_cfg,
    )

    channel_collection = partition_outputs.channel_features
    zone_collection = partition_outputs.zone_features
    subzone_collection = partition_outputs.subzone_features
    network = partition_outputs.channel_network
    diagnostics = {
        "dem_path": str(dem_path),
        "flow_acc_path": str(flow_acc_path),
        "flow_dir_path": str(flow_dir_path),
        "pour_points": [pp.__dict__ for pp in pour_points],
        "accumulation_threshold": channel_threshold,
        "channel_summary": network.summary(),
        "statistics": statistics,
    }
    (intermediate_dir / "delineation_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )


def _run_channel_experiments(
    subbasins,
    delineation_config: DelineationConfig,
    forcing_df: pd.DataFrame,
    subbasin_forcing: Dict[str, List[float]],
    dt_seconds: float,
    dt_hours: float,
    rainfall_volume_m3: float,
    storm_output: Path,
    dynamic_wave_threshold: float,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    """Run a Muskingum baseline and channel-aware scenario."""
    results: Dict[str, object] = {}
    model_meta: Dict[str, Dict[str, object]] = {}
    terminal_subbasin = _terminal_subbasin_id(subbasins)
    basin_area_km2 = sum(sub.area_km2 for sub in subbasins)

    combos = [
        ("hbv", "muskingum"),
        ("hbv", "channel_aware"),
    ]

    for runoff_slug, routing_slug in combos:
        runoff_info = RUNOFF_LIBRARY[runoff_slug]
        routing_info = ROUTING_LIBRARY[routing_slug]

        runoff_config = RunoffModelConfig(
            id=f"{runoff_slug}_{routing_slug}_runoff",
            model_type=runoff_slug,
            parameters=dict(runoff_info.get("parameters", {})),
        )

        routing_parameters = dict(routing_info.get("parameters", {}))
        routing_configs: List[RoutingModelConfig] = []
        id_to_slug: Dict[str, str] = {}
        routing_param_map: Dict[str, Dict[str, float]] = {}

        if routing_slug == "channel_aware":
            dynamic_params = dict(routing_parameters.get("dynamic_wave", {}))
            dynamic_params["time_step"] = dt_hours
            muskingum_params = dict(routing_parameters.get("muskingum", {}))
            muskingum_params["time_step"] = dt_hours

            dynamic_config = RoutingModelConfig(
                id=f"{runoff_slug}_dynamic_reach",
                model_type="dynamic_wave",
                parameters=dynamic_params,
            )
            muskingum_config = RoutingModelConfig(
                id=f"{runoff_slug}_muskingum_reach",
                model_type="muskingum",
                parameters=muskingum_params,
            )
            routing_configs = [dynamic_config, muskingum_config]
            id_to_slug = {
                dynamic_config.id: "dynamic_wave",
                muskingum_config.id: "muskingum",
            }
            routing_param_map = {
                dynamic_config.id: dynamic_params,
                muskingum_config.id: muskingum_params,
            }

            def select_routing(sub):
                length = float(getattr(sub, "channel_length_m", 0.0) or 0.0)
                return dynamic_config.id if length >= dynamic_wave_threshold else muskingum_config.id

        else:
            if routing_slug in {"muskingum", "dynamic_wave"}:
                routing_parameters["time_step"] = dt_hours
            routing_config = RoutingModelConfig(
                id=f"{runoff_slug}_{routing_slug}",
                model_type=routing_slug,
                parameters=routing_parameters,
            )
            routing_configs = [routing_config]
            id_to_slug = {routing_config.id: routing_slug}
            routing_param_map = {routing_config.id: routing_parameters}

            def select_routing(sub, routing_id=routing_config.id):
                return routing_id

        parameter_zones, assignment_map = _build_parameter_zones(
            subbasins,
            runoff_config.id,
            select_routing,
        )
        assignment_labels = {
            sub_id: id_to_slug.get(routing_id, routing_id)
            for sub_id, routing_id in assignment_map.items()
        }
        channel_details = []
        if routing_slug == "channel_aware":
            channel_details = _channel_routing_details(subbasins, assignment_map, id_to_slug, routing_param_map)

        config = ModelConfig(
            delineation=delineation_config,
            runoff_models=[runoff_config],
            routing_models=routing_configs,
            parameter_zones=parameter_zones,
            io=IOConfig(precipitation=storm_output),
            scenarios=[],
            evaluation=None,
        )

        forcing_data = {sub.id: list(subbasin_forcing.get(sub.id, [])) for sub in subbasins}
        result = run_workflow(config, forcing_data)
        label = f"{runoff_info['label']}-{routing_info['label']}"
        results[label] = result
        model_meta[label] = {
            "unit": runoff_info["unit"],
            "runoff_slug": runoff_slug,
            "routing_slug": routing_slug,
            "label_cn": f"{runoff_info['label_cn']}-{routing_info['label_cn']}",
            "label": label,
            "stability": None,
            "channel_assignment": assignment_labels,
            "channel_details": channel_details,
            "target_subbasin": terminal_subbasin,
        }

    metrics = prepare_metrics(
        forcing_df,
        results,
        dt_seconds,
        rainfall_volume_m3,
        model_meta,
        terminal_subbasin,
        basin_area_km2,
    )

    return results, model_meta, metrics, basin_area_km2, terminal_subbasin


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Upper Truckee routing workflow with channel-aware switching.",
    )
    parser.add_argument(
        "--dem",
        type=Path,
        default=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"),
        help="Input DEM GeoTIFF for the Upper Truckee case study.",
    )
    parser.add_argument(
        "--flow-accum",
        type=Path,
        default=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/flowaccum.tif"),
        help="Optional flow accumulation raster used to seed pour-point selection.",
    )
    parser.add_argument(
        "--flow-dir",
        type=Path,
        default=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/flowdir.tif"),
        help="Optional flow direction raster (used for diagnostics/plots).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/upper_truckee_channel_demo"),
        help="Directory where delineation artefacts and reports are written.",
    )
    parser.add_argument(
        "--channel-threshold",
        type=float,
        default=15000.0,
        help="Accumulation threshold for channel extraction (cells).",
    )
    parser.add_argument(
        "--target-subzone-area",
        type=float,
        default=25.0,
        help="Approximate target area (km^2) for parameter subzones.",
    )
    parser.add_argument(
        "--min-subzone-area",
        type=float,
        default=5.0,
        help="Minimum subzone area (km^2) allowed during partitioning.",
    )
    parser.add_argument(
        "--max-subzones",
        type=int,
        default=6,
        help="Maximum subzones generated per parameter zone.",
    )
    parser.add_argument(
        "--subzone-balance-tolerance",
        type=float,
        default=0.35,
        help="Allowable fractional deviation from the target subzone area.",
    )
    parser.add_argument(
        "--rain-inputs-dir",
        type=Path,
        default=None,
        help="Directory containing repaired rain gauge inputs (CSV/GeoJSON/JSON).",
    )
    parser.add_argument(
        "--dynamic-wave-length",
        type=float,
        default=2000.0,
        help="Reaches longer than this (m) use the dynamic-wave routing.",
    )
    parser.add_argument(
        "--pour-count",
        type=int,
        default=6,
        help="Number of pour points to select automatically from flow accumulation.",
    )
    parser.add_argument(
        "--use-tree-pour-points",
        action="store_true",
        help="Generate hierarchical pour points from accumulation instead of simple maxima.",
    )
    parser.add_argument(
        "--tree-acc-threshold",
        type=float,
        default=0.15,
        help="Accumulation threshold (absolute or quantile when between 0 and 1) for tree pour point selection.",
    )
    parser.add_argument(
        "--tree-min-distance",
        type=int,
        default=45,
        help="Minimum spacing in raster cells between automatically generated pour points.",
    )
    parser.add_argument(
        "--tree-max-children",
        type=int,
        default=3,
        help="Maximum upstream branches explored per node when generating tree pour points.",
    )
    parser.add_argument(
        "--tree-copy-to",
        type=Path,
        default=None,
        help="Optional directory or file path receiving a copy of the generated pour-point GeoJSON.",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    dutils.ensure_inputs(args.dem, args.flow_accum, args.flow_dir, output_dir)

    pour_points_path = output_dir / "pour_points.geojson"

    if args.use_tree_pour_points:
        auto_copy_target = args.tree_copy_to or (output_dir / "inputs")
        tree_plot = output_dir / "intermediate" / "tree_flow_accumulation.png"
        pour_points = dutils.generate_tree_pour_points(
            args.flow_accum,
            args.flow_dir,
            count=args.pour_count,
            accumulation_threshold=args.tree_acc_threshold,
            min_distance_cells=args.tree_min_distance,
            max_children=args.tree_max_children,
            output_geojson=pour_points_path,
            accumulation_plot=tree_plot,
            copy_to=auto_copy_target,
        )
    else:
        pour_points = dutils.derive_pour_points(
            args.flow_accum,
            args.pour_count,
            accumulation_threshold=5000.0,
            min_spacing=2500.0,
        )
        pour_points = snap_pour_points_to_flow_cells(
            pour_points,
            args.flow_dir,
            args.flow_accum,
        )
        pour_geojson = _build_pour_point_collection(pour_points)
        _write_geojson(pour_geojson, pour_points_path)

    default_threshold = float(args.channel_threshold or 5000.0)
    accumulation_threshold = suggest_accumulation_threshold(pour_points, default=default_threshold)
    if args.channel_threshold is None and accumulation_threshold != default_threshold:
        print(f"Auto accumulation threshold selected: {accumulation_threshold:.1f}")

    delineation_config = DelineationConfig(
        dem_path=args.dem,
        pour_points_path=pour_points_path,
        accumulation_threshold=accumulation_threshold,
        channel_threshold=args.channel_threshold,
        flow_direction_path=args.flow_dir,
        flow_accumulation_path=args.flow_accum,
    )
    partition_cfg = ParameterPartitionConfig(
        pour_points_path=pour_points_path,
        target_subzone_area_km2=args.target_subzone_area,
        min_subzone_area_km2=args.min_subzone_area,
        max_subzones_per_zone=args.max_subzones,
        area_balance_tolerance=args.subzone_balance_tolerance,
        subzone_accumulation_threshold=1000.0,
    )

    try:
        subbasins = delineation_config.to_subbasins()
    except AttributeError:
        print(
            "The installed richdem build lacks D8 flow support (FlowProportions/FlowAccumulation). "
            "Install the conda-forge richdem package for full functionality."
        )
        sys.exit(1)

    _export_intermediate_delineation_artifacts(
        args.dem,
        args.flow_accum,
        args.flow_dir,
        pour_points,
        delineation_config,
        partition_cfg,
        float(delineation_config.channel_threshold or accumulation_threshold),
        output_dir,
    )
    intermediate_dir = output_dir / "intermediate"

    subbasin_path = intermediate_dir / "subbasins.geojson"
    subbasin_geometries = _load_subbasin_geometries(subbasin_path)
    parameter_subbasin_path = output_dir / "parameters" / "parameter_subbasins.geojson"
    parameter_geometries, param_parent_map, param_area_map = _load_parameter_subbasins(parameter_subbasin_path)
    ordered_columns = [sub.id for sub in subbasins]
    hydrologic_precip_path = intermediate_dir / "subbasin_areal_precipitation.csv"

    parameter_precip: pd.DataFrame
    subbasin_precip: pd.DataFrame
    outputs: Dict[str, Path]

    if args.rain_inputs_dir:
        rain_inputs_dir = Path(args.rain_inputs_dir).expanduser().resolve()
        station_series = _load_time_indexed_csv(
            rain_inputs_dir / "rain_gauge_forcing.csv",
            "Rain gauge forcing",
        )
        thiessen_polygons = _load_thiessen_polygons(
            rain_inputs_dir / "rain_gauge_thiessen_polygons.geojson"
        )
        parameter_precip = station_series.reindex(columns=sorted(parameter_geometries.keys())).copy()
        subbasin_precip = aggregate_parameter_precipitation(
            parameter_precip,
            param_parent_map,
            param_area_map,
            subbasins,
        )
        precipitation_stage = None

        outputs = {
            "gauges": intermediate_dir / "rain_gauge_forcing.csv",
            "subbasin": intermediate_dir / "parameter_subbasin_areal_precipitation.csv",
            "stations": intermediate_dir / "rain_gauge_locations.geojson",
            "thiessen": intermediate_dir / "rain_gauge_thiessen_polygons.geojson",
            "weights": intermediate_dir / "rain_gauge_weights.json",
        }
        station_series.to_csv(outputs["gauges"])
        parameter_precip.to_csv(outputs["subbasin"])
        subbasin_precip.to_csv(hydrologic_precip_path)

        for filename in ("rain_gauge_locations.geojson", "rain_gauge_thiessen_polygons.geojson"):
            source = rain_inputs_dir / filename
            if source.exists():
                shutil.copy2(source, intermediate_dir / filename)
        weights_path = rain_inputs_dir / "rain_gauge_weights.json"
        if weights_path.exists():
            shutil.copy2(weights_path, intermediate_dir / "rain_gauge_weights.json")
        else:
            (intermediate_dir / "rain_gauge_weights.json").write_text(
                json.dumps({}, indent=2),
                encoding="utf-8",
            )
    else:
        synthetic_forcing = generate_storm_forcing(
            total_hours=1440,      # 扩展到60天（1440小时）
            storm_hours=48,        # 2天暴雨期（更真实的暴雨持续时间）
            lead_hours=336,        # 14天前置期（让土壤达到稳定状态）
            tail_hours=1056,       # 44天退水期（充分观察基流衰退）
            time_step_minutes=60,  # 保持1小时时间步长
        )
        base_series = synthetic_forcing[PRECIP_COLUMN]

        precipitation_stage = generate_precipitation_for_parameters(
            base_precipitation=base_series,
            parameter_geometries=parameter_geometries,
            parameter_to_zone=param_parent_map,
            parameter_areas=param_area_map,
            subbasins=subbasins,
            rainfall_options={
                "station_count": 10,
                "rng_seed": 42,
                "heterogeneity_strength": 0.8,
                "min_burst_events": 3,
                "max_burst_events": 5,
            },
        )

        parameter_precip = precipitation_stage.parameter_series
        subbasin_precip = precipitation_stage.subbasin_series

        if precipitation_stage.rain_inputs is not None:
            outputs = precipitation_stage.rain_inputs.write(
                intermediate_dir,
                gauges_filename="rain_gauge_forcing.csv",
                subbasin_filename="parameter_subbasin_areal_precipitation.csv",
                stations_geojson="rain_gauge_locations.geojson",
                thiessen_geojson="rain_gauge_thiessen_polygons.geojson",
                weights_json="rain_gauge_weights.json",
            )
        else:
            outputs = {
                "gauges": intermediate_dir / "rain_gauge_forcing.csv",
                "subbasin": intermediate_dir / "parameter_subbasin_areal_precipitation.csv",
                "weights": intermediate_dir / "rain_gauge_weights.json",
            }
            precipitation_stage.station_series.to_csv(outputs["gauges"])
            parameter_precip.to_csv(outputs["subbasin"])
            (intermediate_dir / "rain_gauge_weights.json").write_text(
                json.dumps(precipitation_stage.station_weights, indent=2),
                encoding="utf-8",
            )
    subbasin_precip.to_csv(hydrologic_precip_path)

    area_lookup = {sub.id: float(sub.area_km2) for sub in subbasins}
    total_area = sum(area_lookup.values())
    if total_area <= 0.0:
        raise ValueError("Total basin area must be positive.")
    weighted_series = sum(subbasin_precip[sub.id] * area_lookup[sub.id] for sub in subbasins) / total_area
    storm_forcing = pd.DataFrame({PRECIP_COLUMN: weighted_series}, index=subbasin_precip.index)

    subbasin_forcing = {sub.id: subbasin_precip[sub.id].tolist() for sub in subbasins}

    print(f"Rain gauge forcing written to {outputs['gauges']}")
    if "subbasin" in outputs:
        print(f"Parameter subbasin precipitation written to {outputs['subbasin']}")
    print(f"Hydrologic subbasin precipitation written to {hydrologic_precip_path}")
    for key, label in (("stations", "Rain gauge locations"), ("thiessen", "Thiessen polygons"), ("weights", "Rain gauge weights")):
        if key in outputs:
            print(f"{label} written to {outputs[key]}")

    storm_output = output_dir / "storm_forcing.csv"
    storm_forcing.to_csv(storm_output)

    dt_seconds = (storm_forcing.index[1] - storm_forcing.index[0]).total_seconds()
    dt_hours = dt_seconds / 3600.0
    rainfall_depth_mm = (storm_forcing[PRECIP_COLUMN] * dt_hours).sum()
    basin_area_km2 = sum(sub.area_km2 for sub in subbasins)
    rainfall_volume_m3 = rainfall_depth_mm / 1000.0 * basin_area_km2 * 1_000_000.0

    results, model_meta, metrics_summary, basin_area_km2, terminal_subbasin = _run_channel_experiments(
        subbasins,
        delineation_config,
        storm_forcing,
        subbasin_forcing,
        dt_seconds,
        dt_hours,
        rainfall_volume_m3,
        storm_output,
        args.dynamic_wave_length,
    )

    plot_path = output_dir / "upper_truckee_hydrograph.png"
    plot_comparison(
        storm_forcing,
        results,
        str(plot_path),
        dt_seconds,
        model_meta,
        terminal_subbasin,
        basin_area_km2,
    )
    generate_report(
        str(plot_path),
        str(output_dir),
        rainfall_depth_mm,
        rainfall_volume_m3,
        basin_area_km2,
        metrics_summary,
    )

    try:
        parameter_dir = output_dir / "parameters"
        baseline_dir = output_dir / "hydro_project" / "baseline"
        local_dir = output_dir / "hydro_project" / "baseline_local"
        diagnostics = run_channel_diagnostics(
            parameter_dir=parameter_dir,
            baseline_dir=baseline_dir,
            intermediate_dir=intermediate_dir,
            local_dir=local_dir,
            precipitation_path=storm_output,
        )
        print(f"Channel flow timeseries written to {diagnostics.timeseries_path}")
        print(f"Channel flow comparison figure written to {diagnostics.comparison_path}")
        print(f"Zone runoff coefficients written to {diagnostics.runoff_coefficients_path}")
    except Exception as exc:  # pragma: no cover - best effort post-processing
        warnings.warn(f"Channel diagnostics failed: {exc}")

    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    main()






