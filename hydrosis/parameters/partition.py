"""Parameter partitioning utilities bridging delineation outputs and model config.

The helpers in this module convert delineation artefacts (pour points, flow
direction grids, accumulation rasters) into parameter zones, subzones, and
channel segment collections.  The implementation is derived from the bespoke
logic that previously lived in the ``upper_truckee_channel_workflow`` example,
generalised so that production workflows can re-use it without hard-coded
behaviour.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Set

import numpy as np
import rasterio
from affine import Affine

from ..delineation import utils as dutils
from ..delineation.dem_delineator import DelineationConfig
from ..model import ChannelNetwork, ChannelSegment, Subbasin
from ..parameters.zone import ParameterZoneConfig
from ..config import ParameterPartitionConfig, OutputArtifactsConfig, ModelStructureConfig
from .zone import ParameterZoneBuilder

GridPath = Tuple[int, int]


@dataclass
class ZoneSummary:
    """Metadata describing a derived parameter zone."""

    id: str
    downstream_id: Optional[str]
    area_km2: float
    runoff_method: str
    routing_method: str


@dataclass
class SubzoneSummary:
    """Metadata describing a derived parameter subzone."""

    zone_id: str
    subzone_id: str
    area_km2: float
    downstream_subzone_id: Optional[str]
    mean_elevation: Optional[float]
    max_accumulation: Optional[float]
    pour_row: int
    pour_col: int


@dataclass
class ChannelSummary:
    """Metadata describing a derived channel segment."""

    segment_id: str
    zone_id: str
    subzone_id: str
    downstream_id: Optional[str]
    length_m: float
    slope: float
    drop_m: float


@dataclass
class PartitionOutputs:
    """Collection of derived artefacts for parameter partitioning."""

    parameter_zones: List[ParameterZoneConfig]
    zone_definitions: Dict[str, Dict[str, object]]
    zone_features: Dict[str, object]
    subzone_features: Dict[str, object]
    channel_features: Dict[str, object]
    pour_point_features: Dict[str, object]
    zone_table: List[Dict[str, object]]
    subzone_table: List[Dict[str, object]]
    channel_table: List[Dict[str, object]]
    zone_summaries: List[ZoneSummary]
    subzone_summaries: List[SubzoneSummary]
    channel_summaries: List[ChannelSummary]
    channel_network: ChannelNetwork


def _load_core_grids(
    delineation_cfg: DelineationConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Affine]:
    """Read the elevation, flow direction, and accumulation grids."""

    if not delineation_cfg.flow_direction_path or not delineation_cfg.flow_accumulation_path:
        raise ValueError(
            "DelineationConfig must provide 'flow_direction_path' and 'flow_accumulation_path' "
            "to build parameter partitions."
        )
    with rasterio.open(delineation_cfg.dem_path) as dem_ds:
        dem_data = dem_ds.read(1, masked=True).filled(np.nan)
        dem_transform = dem_ds.transform

    with rasterio.open(delineation_cfg.flow_direction_path) as dir_ds:
        flowdir = dir_ds.read(1)
        direction_transform = dir_ds.transform

    with rasterio.open(delineation_cfg.flow_accumulation_path) as acc_ds:
        accumulation = acc_ds.read(1).astype(float)
        accumulation_transform = acc_ds.transform
        accumulation = np.where(np.isfinite(accumulation), accumulation, 0.0)

    if not np.array_equal(direction_transform, accumulation_transform):
        raise ValueError("Flow direction and accumulation rasters must share the same transform.")

    if not np.array_equal(direction_transform, dem_transform):
        raise ValueError("DEM, flow direction, and flow accumulation rasters must align.")

    return dem_data, flowdir, accumulation, direction_transform


def _ensure_intermediate_directories(delineation_cfg: DelineationConfig) -> Tuple[Path, Path]:
    intermediate_dir = (
        delineation_cfg.intermediate_directory
        if delineation_cfg.intermediate_directory is not None
        else delineation_cfg.dem_path.parent / "derived"
    )
    parameter_dir = (
        delineation_cfg.parameter_directory
        if delineation_cfg.parameter_directory is not None
        else intermediate_dir.parent / "parameters"
    )
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    parameter_dir.mkdir(parents=True, exist_ok=True)
    return intermediate_dir, parameter_dir


@dataclass
class ZoneNode:
    """Internal representation used while deriving parameter partitions."""

    id: str
    pour_point: dutils.PourPoint
    downstream_id: Optional[str]
    runoff_method: str = "default_runoff"
    routing_method: str = "default_routing"


def _compute_depth_map(downstream_map: Mapping[str, Optional[str]]) -> Dict[str, int]:
    """Return a depth score representing distance to outlet for each identifier."""

    depth_cache: Dict[str, int] = {}

    def _depth(node: str) -> int:
        if node in depth_cache:
            return depth_cache[node]
        downstream = downstream_map.get(node)
        if downstream is None or downstream == node:
            depth_cache[node] = 0
        else:
            depth_cache[node] = _depth(downstream) + 1
        return depth_cache[node]

    for key in downstream_map:
        _depth(key)
    return depth_cache


def _trace_flow_path(
    flowdir: np.ndarray,
    start: Tuple[int, int],
    target: Optional[Tuple[int, int]],
    mask: Optional[np.ndarray] = None,
    max_steps: int = 20000,
) -> List[Tuple[int, int]]:
    rows, cols = flowdir.shape
    path: List[Tuple[int, int]] = []
    r, c = start
    steps = 0
    d8 = dutils.D8_OFFSETS
    visited: Set[Tuple[int, int]] = set()

    while True:
        path.append((r, c))
        if target is not None and (r, c) == target:
            break
        if steps >= max_steps:
            break
        code = int(flowdir[r, c])
        offset = d8.get(code)
        if offset is None:
            break
        nr, nc = r + offset[0], c + offset[1]
        if not (0 <= nr < rows and 0 <= nc < cols):
            break
        if mask is not None and not mask[nr, nc]:
            path.append((nr, nc))
            break
        if (nr, nc) in visited:
            break
        visited.add((nr, nc))
        r, c = nr, nc
        steps += 1
    return path


def _upstream_channel_neighbors(
    flowdir: np.ndarray,
    channel_mask: np.ndarray,
    cell: Tuple[int, int],
) -> List[Tuple[int, int]]:
    r, c = cell
    neighbors: List[Tuple[int, int]] = []
    rows, cols = flowdir.shape
    for code, (dr, dc) in dutils.D8_OFFSETS.items():
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if not channel_mask[nr, nc]:
            continue
        neighbor_code = int(flowdir[nr, nc])
        offset = dutils.D8_OFFSETS.get(neighbor_code)
        if offset is None:
            continue
        if nr + offset[0] == r and nc + offset[1] == c:
            neighbors.append((nr, nc))
    return neighbors


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


def _move_pour_point_downstream(
    point: dutils.PourPoint,
    accumulation: np.ndarray,
    flowdir: np.ndarray,
    transform: Affine,
    target_cells: float,
    stop: Optional[Tuple[int, int]],
    occupied: Set[Tuple[int, int]],
    allowed_mask: Optional[np.ndarray],
) -> Optional[dutils.PourPoint]:
    """Shift a pour point downstream until its upstream area approaches the target."""

    rows, cols = flowdir.shape
    d8 = dutils.D8_OFFSETS
    start_row, start_col = int(point.row), int(point.col)
    best_row, best_col = start_row, start_col
    best_acc = float(accumulation[start_row, start_col])
    visited: Set[Tuple[int, int]] = {(start_row, start_col)}
    r, c = start_row, start_col

    def _is_available(row: int, col: int) -> bool:
        return (row, col) not in occupied

    while True:
        if stop and (r, c) == stop:
            break
        code = int(flowdir[r, c])
        offset = d8.get(code)
        if offset is None:
            break
        nr, nc = r + offset[0], c + offset[1]
        if not (0 <= nr < rows and 0 <= nc < cols):
            break
        if (nr, nc) in visited:
            break
        if stop and (nr, nc) == stop:
            break
        if allowed_mask is not None and not allowed_mask[nr, nc]:
            break
        visited.add((nr, nc))
        r, c = nr, nc

        acc_val = float(accumulation[r, c])
        if _is_available(r, c) and acc_val > best_acc:
            best_row, best_col = r, c
            best_acc = acc_val
        if _is_available(r, c) and acc_val >= target_cells:
            best_row, best_col = r, c
            best_acc = acc_val
            break

    if (best_row, best_col) == (start_row, start_col):
        return None

    world_x, world_y = rasterio.transform.xy(transform, best_row, best_col, offset="center")
    return replace(
        point,
        row=int(best_row),
        col=int(best_col),
        x=float(world_x),
        y=float(world_y),
        accumulation=float(best_acc),
    )


def _rebalance_pour_points(
    pour_points: Sequence[dutils.PourPoint],
    flowdir: np.ndarray,
    upstream: Sequence[Sequence[Tuple[int, int]]],
    shape: Tuple[int, int],
    accumulation: np.ndarray,
    transform: Affine,
    *,
    min_fraction: float = 0.5,
    max_iterations: int = 3,
) -> List[dutils.PourPoint]:
    """Move undersized pour points downstream to balance zone areas."""

    if not pour_points:
        return []

    order = [pp.id for pp in pour_points]
    pour_point_map: Dict[str, dutils.PourPoint] = {pp.id: pp for pp in pour_points}

    for _ in range(max_iterations):
        downstream_map, children_map = _compute_downstream_map(pour_point_map, flowdir)
        base_masks = {
            pid: dutils.delineate_watershed(pp, upstream, shape) for pid, pp in pour_point_map.items()
        }
        zone_masks: Dict[str, np.ndarray] = {}
        for pid in pour_point_map.keys():
            mask = base_masks[pid].copy()
            for child in children_map.get(pid, []):
                mask = np.logical_and(mask, np.logical_not(base_masks[child]))
            if not mask.any() and base_masks[pid].any():
                mask = base_masks[pid].copy()
            zone_masks[pid] = mask

        area_lookup = {pid: int(mask.sum()) for pid, mask in zone_masks.items()}
        positive_areas = [area for area in area_lookup.values() if area > 0]
        if not positive_areas:
            break

        target_cells = float(np.median(positive_areas))
        if target_cells <= 0:
            break

        changed = False
        root_ids = {pid for pid, downstream in downstream_map.items() if downstream is None}
        for pid in sorted(order, key=lambda zid: area_lookup.get(zid, 0)):
            area_cells = area_lookup.get(pid, 0)
            if area_cells <= 0 or area_cells >= target_cells * min_fraction:
                continue
            if pid in root_ids:
                continue
            if children_map.get(pid):
                continue

            downstream_id = downstream_map.get(pid)
            stop_coord = None
            if downstream_id and downstream_id in pour_point_map:
                downstream_pp = pour_point_map[downstream_id]
                stop_coord = (int(downstream_pp.row), int(downstream_pp.col))
            parent_mask = zone_masks.get(downstream_id) if downstream_id else None
            allowed_mask = zone_masks[pid]
            if parent_mask is not None:
                allowed_mask = np.logical_or(allowed_mask, parent_mask)

            occupied = {(int(pp.row), int(pp.col)) for zid, pp in pour_point_map.items() if zid != pid}
            updated_point = _move_pour_point_downstream(
                pour_point_map[pid],
                accumulation,
                flowdir,
                transform,
                target_cells,
                stop_coord,
                occupied,
                allowed_mask,
            )
            if updated_point is not None:
                pour_point_map[pid] = updated_point
                changed = True

        if not changed:
            break

    return [pour_point_map[pid] for pid in order if pid in pour_point_map]


def _find_downstream_subzone(
    flowdir: np.ndarray,
    subzone_index_grid: np.ndarray,
    subzone_ids: Sequence[str],
    start_index: int,
    row: int,
    col: int,
) -> Optional[str]:
    rows, cols = flowdir.shape
    d8 = dutils.D8_OFFSETS
    visited: Set[Tuple[int, int]] = set()
    r, c = row, col
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
        idx = int(subzone_index_grid[nr, nc])
        if idx >= 0 and idx != start_index:
            return subzone_ids[idx]
        r, c = nr, nc
    return None


def partition_parameter_zones(
    delineation_cfg: DelineationConfig,
    partition_cfg: ParameterPartitionConfig,
    model_structure: ModelStructureConfig,
    outputs_cfg: OutputArtifactsConfig,
) -> PartitionOutputs:
    """Derive parameter zones, subzones, and channel segments for a project.

    The workflow assumes that the DEM delineation step has already produced
    flow-direction and flow-accumulation rasters, and that the parameter
    partition input pour points map onto delineated subbasin identifiers.
    """

    dem_data, flowdir_array, accumulation, transform = _load_core_grids(delineation_cfg)
    flowdir, upstream, shape = dutils.build_flow_network(delineation_cfg.flow_direction_path)
    if not np.array_equal(flowdir, flowdir_array):
        flowdir = flowdir_array
    pour_points = dutils.read_pour_points_geojson(partition_cfg.pour_points_path)
    if not pour_points:
        raise RuntimeError("No pour points supplied for parameter partitioning")

    pour_points = _rebalance_pour_points(
        pour_points,
        flowdir,
        upstream,
        shape,
        accumulation,
        transform,
    )

    pour_point_map = {pp.id: pp for pp in pour_points}
    downstream_base, children_map = _compute_downstream_map(pour_point_map, flowdir)

    base_masks = {pid: dutils.delineate_watershed(pp, upstream, shape) for pid, pp in pour_point_map.items()}
    zone_masks: Dict[str, np.ndarray] = {}
    for pid in pour_point_map:
        mask = base_masks[pid].copy()
        for child in children_map.get(pid, []):
            mask = np.logical_and(mask, np.logical_not(base_masks[child]))
        if not mask.any() and base_masks[pid].any():
            mask = base_masks[pid].copy()
        zone_masks[pid] = mask

    cell_area_km2 = abs(transform.a * transform.e) / 1_000_000.0
    subbasins = delineation_cfg.to_subbasins()
    subbasin_lookup = {sub.id: sub for sub in subbasins}

    zones: Dict[str, ZoneNode] = {}
    for pid, pp in pour_point_map.items():
        zones[pid] = ZoneNode(
            id=pid,
            pour_point=pp,
            downstream_id=downstream_base.get(pid),
            runoff_method=model_structure.default_runoff_model or "default_runoff",
            routing_method=model_structure.default_routing_model or "default_routing",
        )

    intermediate_dir, parameter_dir = _ensure_intermediate_directories(delineation_cfg)

    zone_stats_rows: List[Dict[str, object]] = []
    zone_stats_lookup: Dict[str, Dict[str, object]] = {}
    zone_definitions: Dict[str, Dict[str, object]] = {}

    for zone_id, node in zones.items():
        mask = zone_masks.get(zone_id)
        if mask is None or int(mask.sum()) == 0:
            continue
        area_cells = int(mask.sum())
        area_km2 = area_cells * cell_area_km2
        controlled_subbasins = (
            ParameterZoneBuilder._get_upstream_catchment(zone_id, subbasin_lookup)
            if zone_id in subbasin_lookup
            else set()
        )
        row = {
            "zone_id": zone_id,
            "downstream_id": node.downstream_id or "",
            "area_cells": area_cells,
            "area_km2": area_km2,
            "runoff_model": "",
            "routing_model": "",
        }
        zone_stats_rows.append(row)
        zone_stats_lookup[zone_id] = row
        zone_definitions[zone_id] = {
            "control_points": [zone_id],
            "downstream_id": node.downstream_id,
            "area_km2": area_km2,
            "subbasins": sorted(controlled_subbasins),
            "mask": mask.copy(),
            "area_cells": area_cells,
        }

    default_runoff = model_structure.default_runoff_model or "default_runoff"
    default_routing = model_structure.default_routing_model or "default_routing"

    for zone_id, definition in zone_definitions.items():
        zones[zone_id].runoff_method = default_runoff
        zones[zone_id].routing_method = default_routing
        sub_ids = definition.get("subbasins", []) or []
        extra_parameters: Dict[str, object] = {}
        for assignment in model_structure.subbasin_assignments:
            if any(assignment.matches(sub_id) for sub_id in sub_ids):
                if assignment.runoff_model:
                    zones[zone_id].runoff_method = assignment.runoff_model
                if assignment.routing_model:
                    zones[zone_id].routing_method = assignment.routing_model
                if getattr(assignment, "parameters", None):
                    extra_parameters.update(assignment.parameters)
        definition["parameters"] = {
            "runoff_model": zones[zone_id].runoff_method,
            "routing_model": zones[zone_id].routing_method,
        }
        if extra_parameters:
            definition["parameters"].update(extra_parameters)

    pour_point_features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [pp.x, pp.y]},
                "properties": {
                    "id": pp.id,
                    "zone_id": pp.id,
                    "row": pp.row,
                    "col": pp.col,
                    "accumulation": pp.accumulation,
                },
            }
            for pp in pour_points
        ],
    }
    (parameter_dir / "pour_point_zones.geojson").write_text(
        json.dumps(pour_point_features, indent=2), encoding="utf-8"
    )

    # Subzone derivation with balanced areas
    subzone_features: List[Dict[str, object]] = []
    subzone_rows: List[Dict[str, object]] = []
    subzone_masks: Dict[str, np.ndarray] = {}
    subzone_ids: List[str] = []
    subzone_to_zone: Dict[str, str] = {}
    subzone_index_grid = np.full(shape, -1, dtype=np.int32)
    zone_polygons: Dict[str, List[Sequence[Tuple[float, float]]]] = defaultdict(list)
    zone_area_accumulator: Dict[str, float] = defaultdict(float)
    zone_cell_accumulator: Dict[str, int] = defaultdict(int)

    def _drains_to_target(row: int, col: int, target: Tuple[int, int]) -> bool:
        d8 = dutils.D8_OFFSETS
        visited: Set[Tuple[int, int]] = set()
        r, c = row, col
        while (r, c) != target:
            code = int(flowdir[r, c])
            offset = d8.get(code)
            if offset is None:
                return False
            r += offset[0]
            c += offset[1]
            if not (0 <= r < shape[0] and 0 <= c < shape[1]):
                return False
            if (r, c) in visited:
                return False
            visited.add((r, c))
        return True

    accum_threshold_raw = partition_cfg.subzone_accumulation_threshold
    accum_threshold_map = getattr(partition_cfg, "subzone_accumulation_thresholds", {})

    for row in zone_stats_rows:
        zid = row["zone_id"]
        requested_threshold = accum_threshold_map.get(zid, accum_threshold_raw)
        row["requested_accum_threshold"] = (
            float(requested_threshold) if requested_threshold is not None else None
        )
        row["effective_accum_threshold"] = None
        row["threshold_relaxed"] = False
        row["subzone_count"] = 0

    def _masked_watershed(seed_row: int, seed_col: int, available_mask: np.ndarray) -> np.ndarray:
        rows, cols = available_mask.shape
        result = np.zeros_like(available_mask, dtype=bool)
        stack = [(seed_row, seed_col)]
        while stack:
            r, c = stack.pop()
            if not (0 <= r < rows and 0 <= c < cols):
                continue
            if not available_mask[r, c] or result[r, c]:
                continue
            result[r, c] = True
            idx = r * cols + c
            for nr, nc in upstream[idx]:
                if available_mask[nr, nc] and not result[nr, nc]:
                    stack.append((nr, nc))
        return result

    def _generate_subzones_for_zone(zone_id: str, zone_mask: np.ndarray) -> List[Dict[str, object]]:
        """Split a zone mask into subzone definitions using accumulation thresholds."""
        node = zones[zone_id]
        total_cells = int(zone_mask.sum())
        if total_cells <= 0:
            return []

        zone_threshold = accum_threshold_map.get(zone_id, accum_threshold_raw)
        if zone_threshold is not None and zone_threshold > 0:
            target_cells = float(zone_threshold)
        else:
            target_area = partition_cfg.target_subzone_area_km2 or (total_cells * cell_area_km2)
            if target_area <= 0:
                target_area = total_cells * cell_area_km2
            target_cells = max(1.0, target_area / cell_area_km2)

        balance_tol = partition_cfg.area_balance_tolerance or 0.0
        min_area_km2 = partition_cfg.min_subzone_area_km2
        if (min_area_km2 is None or min_area_km2 <= 0) and target_cells > 0:
            min_area_km2 = max(target_cells * cell_area_km2 * (1.0 - balance_tol), 0.0)
        if min_area_km2 is None or min_area_km2 <= 0:
            min_area_km2 = cell_area_km2

        min_area_cells = max(1, int(round(min_area_km2 / cell_area_km2)))
        max_area_cells = max(min_area_cells, int(round(target_cells * (1.0 + balance_tol))))
        max_subzones = partition_cfg.max_subzones_per_zone if partition_cfg.max_subzones_per_zone else None

        visited = np.full(zone_mask.shape, False, dtype=bool)
        sub_defs: List[Dict[str, object]] = []
        current_cells: List[Tuple[int, int]] = []
        stack: List[Tuple[int, int]] = [(int(node.pour_point.row), int(node.pour_point.col))]
        target_cells = max(target_cells, float(min_area_cells))

        def _build_entry(cells: List[Tuple[int, int]]) -> Dict[str, object]:
            mask = np.zeros_like(zone_mask, dtype=bool)
            rows_idx, cols_idx = zip(*cells)
            mask[rows_idx, cols_idx] = True
            acc_vals = accumulation[rows_idx, cols_idx]
            max_idx = int(np.argmax(acc_vals))
            return {
                "mask": mask,
                "pour_row": int(rows_idx[max_idx]),
                "pour_col": int(cols_idx[max_idx]),
                "seed_accumulation": float(acc_vals[max_idx]),
            }

        while stack:
            r, c = stack.pop()
            if not zone_mask[r, c] or visited[r, c]:
                continue
            visited[r, c] = True
            current_cells.append((int(r), int(c)))

            idx = r * flowdir.shape[1] + c
            for nr, nc in upstream[idx]:
                if zone_mask[nr, nc] and not visited[nr, nc]:
                    stack.append((nr, nc))

            if len(current_cells) >= target_cells:
                sub_defs.append(_build_entry(current_cells))
                current_cells = []

        if current_cells:
            sub_defs.append(_build_entry(current_cells))

        sub_defs.sort(key=lambda entry: accumulation[entry["pour_row"], entry["pour_col"]], reverse=True)

        merged_defs: List[Dict[str, object]] = []
        for entry in sub_defs:
            area_cells = int(entry["mask"].sum())
            if area_cells < min_area_cells and merged_defs:
                merged_defs[-1]["mask"] = np.logical_or(merged_defs[-1]["mask"], entry["mask"])
            else:
                merged_defs.append(entry)

        final_defs: List[Dict[str, object]] = []
        for entry in merged_defs:
            cells = [tuple(coord) for coord in np.argwhere(entry["mask"])]
            cells.sort(key=lambda rc: accumulation[rc[0], rc[1]], reverse=True)
            area_cells = len(cells)
            if area_cells <= max_area_cells or area_cells <= min_area_cells * 2:
                final_defs.append(entry)
                continue
            start = 0
            while start < area_cells:
                chunk = cells[start:start + max_area_cells]
                if not chunk:
                    break
                mask = np.zeros_like(zone_mask, dtype=bool)
                rows_idx, cols_idx = zip(*chunk)
                mask[rows_idx, cols_idx] = True
                acc_vals = accumulation[rows_idx, cols_idx]
                max_idx = int(np.argmax(acc_vals))
                final_defs.append(
                    {
                        "mask": mask,
                        "pour_row": int(rows_idx[max_idx]),
                        "pour_col": int(cols_idx[max_idx]),
                        "seed_accumulation": float(acc_vals[max_idx]),
                    }
                )
                start += max_area_cells

        return final_defs
    zone_subzone_definitions: Dict[str, List[Dict[str, object]]] = {}
    for zone_id, node in zones.items():
        mask = zone_masks.get(zone_id)
        if mask is None or int(mask.sum()) == 0:
            continue
        zone_subzone_definitions[zone_id] = _generate_subzones_for_zone(zone_id, mask)

    # Remap zone_subzone_definitions to use new zone IDs
    new_zone_subzone_definitions: Dict[str, List[Dict[str, object]]] = {}
    for old_zone_id, sub_defs in zone_subzone_definitions.items():
        new_zone_id = old_to_new_zone_id.get(old_zone_id)
        if new_zone_id:
            new_zone_subzone_definitions[new_zone_id] = sub_defs
    zone_subzone_definitions = new_zone_subzone_definitions

    for zone_id, node in zones.items():
        mask = zone_masks.get(zone_id)
        if mask is None or int(mask.sum()) == 0:
            continue

        zone_row = zone_stats_lookup[zone_id]
        requested_threshold = zone_row.get("requested_accum_threshold")
        sub_definitions = zone_subzone_definitions.get(zone_id) or []
        if not sub_definitions:
            continue

        zone_row["subzone_count"] = len(sub_definitions)
        zone_row["effective_accum_threshold"] = requested_threshold
        zone_row["threshold_relaxed"] = False

        # zone_id is now already sequential (1, 2, 3, ...), use it directly
        zone_index = int(zone_id)

        for idx, definition in enumerate(sub_definitions, start=1):
            # New encoding scheme: zone_id * 100 + subzone_index
            sub_id = str(zone_index * 100 + idx)
            sub_mask = definition["mask"]
            area_cells = int(sub_mask.sum())
            if area_cells == 0:
                continue
            area_km2 = area_cells * cell_area_km2
            pour_row = int(definition["pour_row"])
            pour_col = int(definition["pour_col"])
            seed_acc = float(definition.get("seed_accumulation", 0.0))
            mean_elev = float(np.nanmean(dem_data[sub_mask])) if sub_mask.any() else None
            max_acc = float(np.nanmax(accumulation[sub_mask])) if sub_mask.any() else None

            subzone_masks[sub_id] = sub_mask
            subzone_to_zone[sub_id] = zone_id
            zone_area_accumulator[zone_id] += area_km2
            zone_cell_accumulator[zone_id] += area_cells

            polygons = dutils.masks_to_polygons({sub_id: sub_mask}, transform).get(sub_id) or []
            if polygons:
                subzone_features.append(
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": [[ring] for ring in polygons],
                        },
                        "properties": {
                            "zone_id": zone_id,
                            "subzone_id": sub_id,
                            "area_km2": area_km2,
                            "mean_elevation": mean_elev,
                            "max_accumulation": max_acc,
                            "pour_row": pour_row,
                            "pour_col": pour_col,
                            "seed_accumulation": seed_acc,
                            "seed_threshold": requested_threshold,
                        },
                    }
                )

            subzone_rows.append(
                {
                    "zone_id": zone_id,
                    "subzone_id": sub_id,
                    "area_cells": area_cells,
                    "area_km2": area_km2,
                    "mean_elevation": mean_elev if mean_elev is not None else 0.0,
                    "max_accumulation": max_acc if max_acc is not None else 0.0,
                    "pour_row": pour_row,
                    "pour_col": pour_col,
                    "seed_accumulation": seed_acc,
                    "seed_threshold": requested_threshold if requested_threshold is not None else "",
                    "downstream_subzone_id": "",
                }
            )
            current_index = len(subzone_ids)
            subzone_ids.append(sub_id)
            subzone_index_grid[sub_mask] = current_index

    subzone_id_to_index = {sid: idx for idx, sid in enumerate(subzone_ids)}
    for row in subzone_rows:
        sub_id = row["subzone_id"]
        idx = subzone_id_to_index[sub_id]
        downstream_sub = _find_downstream_subzone(
            flowdir,
            subzone_index_grid,
            subzone_ids,
            idx,
            int(row["pour_row"]),
            int(row["pour_col"]),
        )
        if downstream_sub:
            row["downstream_subzone_id"] = downstream_sub

    # Aggregate subzone geometry back into zone-level summaries
    for zone_id in zones.keys():
        zone_polygons[zone_id] = []
        zone_polygon = dutils.masks_to_polygons({zone_id: zone_masks[zone_id]}, transform).get(zone_id) or []
        for ring in zone_polygon:
            zone_polygons[zone_id].append(ring)

    for zone_id in zones.keys():
        if zone_polygons[zone_id]:
            continue
        fallback_mask = zone_definitions[zone_id]["mask"]
        if fallback_mask is None:
            continue
        area_cells = int(fallback_mask.sum())
        zone_cell_accumulator[zone_id] = max(zone_cell_accumulator[zone_id], area_cells)
        if zone_area_accumulator[zone_id] <= 0.0:
            zone_area_accumulator[zone_id] = area_cells * cell_area_km2
        fallback_polygons = dutils.masks_to_polygons({zone_id: fallback_mask}, transform).get(zone_id) or []
        for ring in fallback_polygons:
            zone_polygons[zone_id].append(ring)

    # Update zone summaries using aggregated subzone areas
    for zone_id, definition in zone_definitions.items():
        area_cells = max(zone_cell_accumulator[zone_id], definition.get("area_cells", 0))
        area_km2 = max(zone_area_accumulator[zone_id], definition.get("area_km2", 0.0))
        definition["area_cells"] = area_cells
        definition["area_km2"] = area_km2
        if zone_id in zone_stats_lookup:
            zone_stats_lookup[zone_id]["area_cells"] = area_cells
            zone_stats_lookup[zone_id]["area_km2"] = area_km2

    # Compute depth and create old_id -> new_id mapping
    zone_downstream_map = {zone_id: node.downstream_id for zone_id, node in zones.items()}
    zone_depth = _compute_depth_map(zone_downstream_map)
    sorted_zone_ids = sorted(zones.keys(), key=lambda zid: zone_depth.get(zid, 0), reverse=True)

    # Create mapping from old zone_id to new sequential zone_id
    old_to_new_zone_id: Dict[str, str] = {}
    new_to_old_zone_id: Dict[str, str] = {}
    for idx, old_zone_id in enumerate(sorted_zone_ids, start=1):
        new_zone_id = str(idx)
        old_to_new_zone_id[old_zone_id] = new_zone_id
        new_to_old_zone_id[new_zone_id] = old_zone_id

    # Remap all zone references to use new sequential IDs
    new_zones: Dict[str, ZoneNode] = {}
    for old_zone_id, node in zones.items():
        new_zone_id = old_to_new_zone_id[old_zone_id]
        old_downstream = node.downstream_id
        new_downstream = old_to_new_zone_id.get(old_downstream) if old_downstream else None
        new_zones[new_zone_id] = ZoneNode(
            id=new_zone_id,
            pour_point=node.pour_point,
            downstream_id=new_downstream,
            runoff_method=node.runoff_method,
            routing_method=node.routing_method,
        )
    zones = new_zones

    # Remap zone_masks, zone_definitions, zone_stats_lookup
    new_zone_masks: Dict[str, np.ndarray] = {}
    for old_zone_id, mask in zone_masks.items():
        new_zone_id = old_to_new_zone_id[old_zone_id]
        new_zone_masks[new_zone_id] = mask
    zone_masks = new_zone_masks

    new_zone_definitions: Dict[str, Dict[str, object]] = {}
    for old_zone_id, definition in zone_definitions.items():
        new_zone_id = old_to_new_zone_id[old_zone_id]
        new_definition = dict(definition)
        old_downstream = definition.get("downstream_id")
        new_definition["downstream_id"] = old_to_new_zone_id.get(old_downstream) if old_downstream else None
        new_zone_definitions[new_zone_id] = new_definition
    zone_definitions = new_zone_definitions

    new_zone_stats_lookup: Dict[str, Dict[str, object]] = {}
    new_zone_stats_rows: List[Dict[str, object]] = []
    for row in zone_stats_rows:
        old_zone_id = row["zone_id"]
        new_zone_id = old_to_new_zone_id[old_zone_id]
        new_row = dict(row)
        new_row["zone_id"] = new_zone_id
        old_downstream = row.get("downstream_id", "")
        new_row["downstream_id"] = old_to_new_zone_id.get(old_downstream) if old_downstream else ""
        new_zone_stats_lookup[new_zone_id] = new_row
        new_zone_stats_rows.append(new_row)
    zone_stats_lookup = new_zone_stats_lookup
    zone_stats_rows = new_zone_stats_rows

    # Update zone_downstream_map and zone_depth with new IDs
    zone_downstream_map = {zone_id: node.downstream_id for zone_id, node in zones.items()}
    zone_depth = _compute_depth_map(zone_downstream_map)
    sorted_zone_ids = sorted(zones.keys(), key=lambda zid: zone_depth.get(zid, 0), reverse=True)

    zone_features: List[Dict[str, object]] = []
    feature_lookup: Dict[str, Dict[str, object]] = {}
    for zone_id in sorted_zone_ids:
        rings = zone_polygons.get(zone_id, [])
        if not rings:
            continue
        coordinates = [[ring] for ring in rings]
        feature = {
            "type": "Feature",
            "geometry": {"type": "MultiPolygon", "coordinates": coordinates},
            "properties": {
                "zone_id": zone_id,
                "downstream_id": zones[zone_id].downstream_id,
                "area_km2": zone_definitions[zone_id]["area_km2"],
                "runoff_method": zones[zone_id].runoff_method,
                "routing_method": zones[zone_id].routing_method,
            },
        }
        zone_features.append(feature)
        feature_lookup[zone_id] = feature

    for row in zone_stats_rows:
        zid = row["zone_id"]
        row["runoff_model"] = zones[zid].runoff_method
        row["routing_model"] = zones[zid].routing_method
        feature = feature_lookup.get(zid)
        if feature:
            props = feature.setdefault("properties", {})
            props["runoff_method"] = zones[zid].runoff_method
            props["routing_method"] = zones[zid].routing_method

    zone_stats_rows.sort(key=lambda row: zone_depth.get(row["zone_id"], 0), reverse=True)
    zone_collection = {"type": "FeatureCollection", "features": zone_features}

    (parameter_dir / "parameter_zones.geojson").write_text(json.dumps(zone_collection, indent=2), encoding="utf-8")
    with (parameter_dir / "parameter_zones.csv").open("w", encoding="utf-8") as handle:
        handle.write(
            "zone_id,downstream_id,area_cells,area_km2,runoff_model,routing_model,subzone_count,requested_accum_threshold,effective_accum_threshold,threshold_relaxed\n"
        )
        for row in zone_stats_rows:
            requested = row.get("requested_accum_threshold")
            effective = row.get("effective_accum_threshold")
            handle.write(
                "{zone_id},{downstream_id},{area_cells},{area_km2:.6f},{runoff_model},{routing_model},{subzone_count},{requested},{effective},{threshold_relaxed}\n".format(
                    zone_id=row["zone_id"],
                    downstream_id=row.get("downstream_id", ""),
                    area_cells=row.get("area_cells", 0),
                    area_km2=row.get("area_km2", 0.0),
                    runoff_model=row.get("runoff_model", ""),
                    routing_model=row.get("routing_model", ""),
                    subzone_count=row.get("subzone_count", 0),
                    requested="" if requested is None else f"{requested:.3f}",
                    effective="" if effective is None else f"{float(effective):.3f}",
                    threshold_relaxed=row.get("threshold_relaxed", False),
                )
            )

    channel_features: List[Dict[str, object]] = []
    channel_rows: List[Dict[str, object]] = []
    subzone_lookup = {row["subzone_id"]: row for row in subzone_rows}

    for row in subzone_rows:
        subzone_id = row["subzone_id"]
        zone_id = row["zone_id"]
        start_row = int(row["pour_row"])
        start_col = int(row["pour_col"])

        if row["downstream_subzone_id"]:
            downstream_row = int(subzone_lookup[row["downstream_subzone_id"]]["pour_row"])
            downstream_col = int(subzone_lookup[row["downstream_subzone_id"]]["pour_col"])
            target = (downstream_row, downstream_col)
            target_mask = None
        else:
            downstream_zone = zones[zone_id].downstream_id
            if downstream_zone and downstream_zone in zones:
                target = (
                    zones[downstream_zone].pour_point.row,
                    zones[downstream_zone].pour_point.col,
                )
                target_mask = None
            else:
                target = None
                target_mask = zone_masks[zone_id]

        path = _trace_flow_path(flowdir, (start_row, start_col), target, mask=target_mask)
        if len(path) < 2:
            continue

        coords: List[Tuple[float, float]] = []
        for r, c in path:
            x, y = rasterio.transform.xy(transform, r, c, offset="center")
            coords.append((x, y))
        length_m = 0.0
        for (x0, y0), (x1, y1) in zip(coords[:-1], coords[1:]):
            length_m += math.hypot(x1 - x0, y1 - y0)
        start_elev = float(dem_data[path[0][0], path[0][1]])
        end_elev = float(dem_data[path[-1][0], path[-1][1]])
        drop_m = start_elev - end_elev
        slope = drop_m / length_m if length_m > 0 else 0.0
        downstream_id = row["downstream_subzone_id"]

        channel_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "segment_id": subzone_id,
                    "zone_id": zone_id,
                    "subzone_id": subzone_id,
                    "length_m": length_m,
                    "slope": slope,
                    "drop_m": drop_m,
                    "downstream_id": downstream_id,
                    "upstream_ids": [],
                },
            }
        )
        channel_rows.append(
            {
                "segment_id": subzone_id,
                "zone_id": zone_id,
                "subzone_id": subzone_id,
                "length_m": length_m,
                "slope": slope,
                "drop_m": drop_m,
                "downstream_id": downstream_id or "",
                "upstream_ids": "",
            }
        )

    downstream_to_upstream: Dict[str, List[str]] = {}
    channel_lookup: Dict[str, Dict[str, object]] = {}
    for row in channel_rows:
        channel_lookup[row["segment_id"]] = row
        downstream_id = row.get("downstream_id")
        if downstream_id:
            downstream_to_upstream.setdefault(downstream_id, []).append(row["segment_id"])

    segment_depth_cache: Dict[str, int] = {}

    def _segment_depth(segment_id: str) -> int:
        if segment_id in segment_depth_cache:
            return segment_depth_cache[segment_id]
        row = channel_lookup.get(segment_id)
        if row is None:
            segment_depth_cache[segment_id] = 0
            return 0
        downstream_id = row.get("downstream_id") or ""
        if downstream_id and downstream_id in channel_lookup:
            depth = _segment_depth(downstream_id) + 1
        else:
            zone_id = row.get("zone_id")
            depth = zone_depth.get(zone_id, 0)
        segment_depth_cache[segment_id] = depth
        return depth

    for upstream_list in downstream_to_upstream.values():
        upstream_list.sort(key=_segment_depth, reverse=True)

    for row in channel_rows:
        downstream_id = row.get("downstream_id")
        upstream_list = downstream_to_upstream.get(row["segment_id"], [])
        row["upstream_ids"] = ";".join(upstream_list)

    for feature in channel_features:
        props = feature.get("properties", {}) or {}
        seg_id = str(props.get("segment_id"))
        props["upstream_ids"] = downstream_to_upstream.get(seg_id, [])

    subzone_rows.sort(key=lambda row: _segment_depth(row["subzone_id"]), reverse=True)
    subzone_features_sorted = sorted(
        subzone_features,
        key=lambda feature: _segment_depth(str(feature.get("properties", {}).get("subzone_id", ""))),
        reverse=True,
    )

    channel_rows.sort(key=lambda row: _segment_depth(row["segment_id"]), reverse=True)
    channel_features_sorted = sorted(
        channel_features,
        key=lambda feature: _segment_depth(str(feature.get("properties", {}).get("segment_id", ""))),
        reverse=True,
    )

    subzone_collection = {"type": "FeatureCollection", "features": subzone_features_sorted}
    (parameter_dir / "parameter_subbasins.geojson").write_text(json.dumps(subzone_collection, indent=2), encoding="utf-8")
    with (parameter_dir / "parameter_subbasins.csv").open("w", encoding="utf-8") as handle:
        handle.write(
            "zone_id,subzone_id,area_cells,area_km2,mean_elevation,max_accumulation,pour_row,pour_col,seed_accumulation,seed_threshold,downstream_subzone_id\n"
        )
        for row in subzone_rows:
            seed_threshold_str = row.get("seed_threshold")
            if seed_threshold_str == "":
                seed_threshold_str = ""
            elif seed_threshold_str is None:
                seed_threshold_str = ""
            else:
                seed_threshold_str = f"{float(seed_threshold_str):.3f}"
            handle.write(
                "{zone_id},{subzone_id},{area_cells},{area_km2:.6f},{mean_elevation:.2f},{max_accumulation:.2f},{pour_row},{pour_col},{seed_accumulation:.2f},{seed_threshold},{downstream_subzone_id}\n".format(
                    zone_id=row["zone_id"],
                    subzone_id=row["subzone_id"],
                    area_cells=row["area_cells"],
                    area_km2=row["area_km2"],
                    mean_elevation=row["mean_elevation"],
                    max_accumulation=row["max_accumulation"],
                    pour_row=row["pour_row"],
                    pour_col=row["pour_col"],
                    seed_accumulation=row.get("seed_accumulation", 0.0),
                    seed_threshold=seed_threshold_str,
                    downstream_subzone_id=row.get("downstream_subzone_id", ""),
                )
            )

    channel_collection = {"type": "FeatureCollection", "features": channel_features_sorted}
    (parameter_dir / "parameter_channels.geojson").write_text(json.dumps(channel_collection, indent=2), encoding="utf-8")
    with (parameter_dir / "parameter_channels.csv").open("w", encoding="utf-8") as handle:
        handle.write("segment_id,zone_id,subzone_id,length_m,slope,drop_m,downstream_id,upstream_ids\n")
        for row in channel_rows:
            handle.write(
                "{segment_id},{zone_id},{subzone_id},{length_m:.2f},{slope:.6f},{drop_m:.2f},{downstream_id},{upstream_ids}\n".format(
                    segment_id=row["segment_id"],
                    zone_id=row["zone_id"],
                    subzone_id=row["subzone_id"],
                    length_m=row["length_m"],
                    slope=row["slope"],
                    drop_m=row["drop_m"],
                    downstream_id=row["downstream_id"],
                    upstream_ids=row["upstream_ids"],
                )
            )

    if outputs_cfg.enable_figures:
        dutils.plot_masks({pid: zone_masks[pid] for pid in zone_masks}, pour_points, parameter_dir / "subzone_masks.png")
        from matplotlib import pyplot as plt
        from matplotlib import patheffects as PathEffects
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon as MplPolygon
        from shapely.geometry import shape

        def _plot_feature_collection(
            collection: Dict[str, object],
            output_path: Path,
            color_key: str,
            title: str,
            channel_layer: Optional[Dict[str, object]] = None,
            pour_points_overlay: Optional[Sequence[dutils.PourPoint]] = None,
            label_key: Optional[str] = None,
            label_formatter: Optional[Callable[[str], str]] = None,
            label_kwargs: Optional[Dict[str, Any]] = None,
        ) -> None:
            plt.figure(figsize=(8, 8))
            ax = plt.gca()
            patches: List[MplPolygon] = []
            colors: List[float] = []
            color_map: Dict[str, int] = {}
            xs_all: List[float] = []
            ys_all: List[float] = []
            current_index = 0
            label_kwargs = label_kwargs or {}
            label_points: List[Tuple[float, float, str]] = []

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
                elif geom_type == "Polygon" and coords:
                    patches.append(MplPolygon(coords[0], closed=True))
                    colors.append(color_index)
                    xs_all.extend(x for x, _ in coords[0])
                    ys_all.extend(y for _, y in coords[0])
                elif geom_type == "LineString" and coords:
                    xs, ys = zip(*coords)
                    ax.plot(xs, ys, linewidth=1.5, color=plt.cm.tab20(color_index % 20))
                    xs_all.extend(xs)
                    ys_all.extend(ys)

                if label_key:
                    raw_label = props.get(label_key)
                    if raw_label not in (None, ""):
                        text_value = label_formatter(str(raw_label)) if label_formatter else str(raw_label)
                        try:
                            geom_obj = shape(feature.get("geometry", {}))
                            if not geom_obj.is_empty:
                                rep_point = geom_obj.representative_point()
                                label_points.append((rep_point.x, rep_point.y, text_value))
                        except Exception:
                            # Skip this label on topology error
                            continue

            if patches:
                collection_patch = PatchCollection(patches, cmap=plt.cm.tab20, alpha=0.6, edgecolor="black", linewidth=0.6)
                collection_patch.set_array(np.array(colors))
                ax.add_collection(collection_patch)
                ax.autoscale_view()

            if channel_layer:
                for feature in channel_layer.get("features", []):
                    geom = feature.get("geometry", {})
                    if geom.get("type") == "LineString":
                        coords = geom.get("coordinates", [])
                        if coords:
                            xs, ys = zip(*coords)
                            ax.plot(xs, ys, color="black", linewidth=1.2, alpha=0.8)
                            xs_all.extend(xs)
                            ys_all.extend(ys)

            if pour_points_overlay:
                xs_pp = [pp.x for pp in pour_points_overlay]
                ys_pp = [pp.y for pp in pour_points_overlay]
                if xs_pp and ys_pp:
                    ax.scatter(xs_pp, ys_pp, c="red", edgecolors="white", s=20, zorder=5, linewidths=0.6)
                    xs_all.extend(xs_pp)
                    ys_all.extend(ys_pp)

            if label_points:
                fontsize = label_kwargs.get("fontsize", 9)
                color = label_kwargs.get("color", "black")
                fontweight = label_kwargs.get("fontweight", "bold")
                outline = label_kwargs.get("outline", True)
                outline_color = label_kwargs.get("outline_color", "white")
                outline_width = label_kwargs.get("outline_width", 2.4)
                for x_coord, y_coord, text_value in label_points:
                    text_artist = ax.text(
                        x_coord,
                        y_coord,
                        text_value,
                        ha="center",
                        va="center",
                        fontsize=fontsize,
                        color=color,
                        fontweight=fontweight,
                        zorder=6,
                    )
                    if outline:
                        text_artist.set_path_effects(
                            [PathEffects.withStroke(linewidth=outline_width, foreground=outline_color)]
                        )

            ax.set_title(title)
            ax.set_aspect("equal", adjustable="box")
            ax.axis("off")
            if xs_all and ys_all:
                ax.set_xlim(min(xs_all), max(xs_all))
                ax.set_ylim(min(ys_all), max(ys_all))
            plt.tight_layout()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=200)
            plt.close()

        def _format_subzone_label(subzone_id: str) -> str:
            # Display full subzone ID (e.g., "101", "201") which encodes zone and subzone info
            return str(subzone_id)

        subzone_label_style = {
            "fontsize": 8,
            "fontweight": "bold",
            "outline": True,
            "outline_width": 2.2,
            "outline_color": "white",
        }

        _plot_feature_collection(
            zone_collection,
            parameter_dir / "parameter_zones_map.png",
            "zone_id",
            "Parameter Zones",
            channel_layer=channel_collection,
            pour_points_overlay=pour_points,
            label_key="zone_id",
            label_kwargs={
                "fontsize": 10,
                "fontweight": "bold",
                "outline": True,
                "outline_width": 3.0,
                "outline_color": "white",
            },
        )
        _plot_feature_collection(
            subzone_collection,
            parameter_dir / "parameter_subbasins_map.png",
            "zone_id",
            "Parameter Subbasins",
            channel_layer=channel_collection,
            pour_points_overlay=pour_points,
            label_key="subzone_id",
            label_formatter=_format_subzone_label,
            label_kwargs={**subzone_label_style, "fontsize": 7},
        )
        _plot_feature_collection(
            channel_collection,
            parameter_dir / "parameter_channels_map.png",
            "segment_id",
            "Parameter Channels",
            channel_layer=None,
            pour_points_overlay=pour_points,
        )

        zone_subzone_features: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for feature in subzone_collection.get("features", []):
            props = feature.get("properties", {}) or {}
            zone_id = str(props.get("zone_id") or "").strip()
            if zone_id:
                zone_subzone_features[zone_id].append(feature)

        per_zone_dir = parameter_dir / "subbasin_maps"
        per_zone_dir.mkdir(parents=True, exist_ok=True)
        for zone_id, features in zone_subzone_features.items():
            if not features:
                continue
            collection_subset = {"type": "FeatureCollection", "features": features}
            output_path = per_zone_dir / f"{zone_id}_subbasins.png"
            _plot_feature_collection(
                collection_subset,
                output_path,
                "subzone_id",
                f"{zone_id} Subbasins",
                channel_layer=channel_collection,
                pour_points_overlay=None,
                label_key="subzone_id",
                label_formatter=_format_subzone_label,
                label_kwargs=dict(subzone_label_style),
            )

    parameter_zone_configs: List[ParameterZoneConfig] = []
    sorted_zone_ids = sorted(zone_definitions.keys(), key=lambda zid: zone_depth.get(zid, 0), reverse=True)
    for zone_id in sorted_zone_ids:
        definition = zone_definitions[zone_id]
        parameters = dict(definition.get("parameters", {}))
        sub_ids = list(definition.get("subbasins") or [])
        parameter_zone_configs.append(
            ParameterZoneConfig(
                id=zone_id,
                description=f"Parameter zone {zone_id}",
                control_points=list(definition.get("control_points", [])),
                parameters=parameters,
                explicit_subbasins=sub_ids if sub_ids else None,
            )
        )

    zone_summaries = [
        ZoneSummary(
            id=row["zone_id"],
            downstream_id=row.get("downstream_id") or None,
            area_km2=float(row.get("area_km2", 0.0)),
            runoff_method=zones[row["zone_id"]].runoff_method,
            routing_method=zones[row["zone_id"]].routing_method,
        )
        for row in zone_stats_rows
    ]
    subzone_summaries = [
        SubzoneSummary(
            zone_id=row["zone_id"],
            subzone_id=row["subzone_id"],
            area_km2=float(row["area_km2"]),
            downstream_subzone_id=row.get("downstream_subzone_id") or None,
            mean_elevation=float(row.get("mean_elevation", 0.0)),
            max_accumulation=float(row.get("max_accumulation", 0.0)),
            pour_row=int(row["pour_row"]),
            pour_col=int(row["pour_col"]),
        )
        for row in subzone_rows
    ]
    channel_summaries = [
        ChannelSummary(
            segment_id=row["segment_id"],
            zone_id=row["zone_id"],
            subzone_id=row["subzone_id"],
            downstream_id=row.get("downstream_id") or None,
            length_m=float(row["length_m"]),
            slope=float(row["slope"]),
            drop_m=float(row["drop_m"]),
        )
        for row in channel_rows
    ]

    network = ChannelNetwork()
    for feature in channel_collection.get("features", []):
        props = feature.get("properties", {}) or {}
        network.add_segment(
            ChannelSegment(
                id=str(props.get("segment_id")),
                downstream=str(props.get("downstream_id")) if props.get("downstream_id") else None,
                upstream_ids=list(props.get("upstream_ids", [])),
                length_m=float(props.get("length_m", 0.0)),
                slope=float(props.get("slope", 0.0)),
                drop_m=float(props.get("drop_m", 0.0)),
            )
        )

    return PartitionOutputs(
        parameter_zones=parameter_zone_configs,
        zone_definitions=zone_definitions,
        zone_features=zone_collection,
        subzone_features=subzone_collection,
        channel_features=channel_collection,
        pour_point_features=pour_point_features,
        zone_table=zone_stats_rows,
        subzone_table=subzone_rows,
        channel_table=channel_rows,
        zone_summaries=zone_summaries,
        subzone_summaries=subzone_summaries,
        channel_summaries=channel_summaries,
        channel_network=network,
    )

