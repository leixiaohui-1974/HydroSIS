"""Parameter partitioning utilities - Zone Rebalancing

Pour point adjustment and zone rebalancing logic.
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

from .partition_models import SubzoneSummary
from .partition_grid import GridPath, _trace_flow_path

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


