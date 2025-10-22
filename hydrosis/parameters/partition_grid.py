"""Parameter partitioning utilities - Grid Operations

Grid loading, flow path tracing, and geometric computations.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Set

import numpy as np


try:
    import rasterio
    from affine import Affine
except ImportError:
    rasterio = None
    Affine = None

from ..delineation.dem_delineator import DelineationConfig
from .partition_models import ZoneSummary

GridPath = Tuple[int, int]


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


