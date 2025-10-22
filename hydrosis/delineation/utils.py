"""Reusable delineation utilities."""
from __future__ import annotations

import csv
import json
import math
import shutil
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rasterio.features
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon as MplPolygon
from pyproj import Transformer
from rasterio.transform import array_bounds

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


def _generate_tree_pour_points_legacy(
    flow_acc_path: Path,
    flow_dir_path: Path,
    *,
    count: int = 6,
    accumulation_threshold: float = 0.0,
    min_distance_cells: int = 30,
    max_children: Optional[int] = 3,
    output_geojson: Optional[Path] = None,
    accumulation_plot: Optional[Path] = None,
    copy_to: Optional[Path] = None,
    main_stem_fraction: float = 0.5,
    area_balance_tolerance: float = 0.35,
) -> List[PourPoint]:
    """Legacy pour point generator constrained to a single watershed.

    The algorithm starts from the cell with maximum flow accumulation (treated as
    the outlet) and recursively walks upstream following the most dominant
    tributaries.  Additional pour points are selected along the main stem and
    major branches, enforcing an approximate spacing (``min_distance_cells``) so
    that the resulting set exhibits a tree-like arrangement.

    Parameters
    ----------
    flow_acc_path:
        Flow accumulation raster (GeoTIFF) aligned with the flow-direction grid.
    flow_dir_path:
        Flow direction raster encoded using the D8 convention.
    count:
        Target number of pour points to select.
    accumulation_threshold:
        Minimum accumulation required for candidate cells.  When ``0 < value < 1``,
        the value is interpreted as a quantile of the accumulation distribution.
    min_distance_cells:
        Minimum Euclidean spacing (in raster cells) enforced between selected
        pour points.  Set to 0 to disable spacing constraints.
    max_children:
        Maximum number of upstream branches explored at each node.  ``None``
        preserves all branches.
    output_geojson:
        Optional file path where the generated points will be written.
    accumulation_plot:
        Optional file path for storing a log-scaled accumulation plot (PNG).
    copy_to:
        Optional destination path (file or directory) that receives a copy of
        the GeoJSON output.  The ``output_geojson`` parameter must be provided
        when ``copy_to`` is specified.
    main_stem_fraction:
        Fraction of pour points forced onto the main stem.  Defaults to 0.5 so
        that roughly half of the requested points sit on the trunk with the
        remainder distributed across tributaries.  The value is clamped between
        0.25 and 0.75 to avoid pathological splits.
    area_balance_tolerance:
        Allowable deviation from the mean watershed area (expressed as a
        fraction) when selecting tributary pour points.  Smaller tributaries are
        automatically replaced with higher-order branches whenever possible so
        that subcatchment areas remain broadly comparable.
    """

    with rasterio.open(flow_acc_path) as acc_ds:
        accumulation = acc_ds.read(1)
        accumulation = np.where(np.isfinite(accumulation), accumulation, 0.0)
        transform = acc_ds.transform

    with rasterio.open(flow_dir_path) as dir_ds:
        flowdir = dir_ds.read(1)

    rows, cols = accumulation.shape
    if flowdir.shape != accumulation.shape:
        raise ValueError("Flow accumulation and flow direction grids must align.")

    # Resolve accumulation threshold (absolute or quantile).
    acc_values = accumulation[np.isfinite(accumulation)]
    if 0.0 < accumulation_threshold < 1.0 and acc_values.size:
        effective_threshold = float(np.quantile(acc_values, accumulation_threshold))
    else:
        effective_threshold = float(accumulation_threshold)

    flat_idx = int(np.argmax(accumulation))
    outlet_row, outlet_col = divmod(flat_idx, cols)

    flowdir_array = np.where(np.isfinite(flowdir), flowdir, 0).astype(int)

    def build_upstream_index_local(array: np.ndarray) -> List[List[Tuple[int, int]]]:
        up: List[List[Tuple[int, int]]] = [[] for _ in range(array.shape[0] * array.shape[1])]
        for r in range(array.shape[0]):
            for c in range(array.shape[1]):
                code = int(array[r, c])
                offset = D8_OFFSETS.get(code)
                if offset is None:
                    continue
                nr, nc = r + offset[0], c + offset[1]
                if 0 <= nr < array.shape[0] and 0 <= nc < array.shape[1]:
                    up[nr * array.shape[1] + nc].append((r, c))
        return up

    upstream_index = build_upstream_index_local(flowdir_array)

    selected_cells: List[Tuple[int, int]] = []
    main_selected: Set[Tuple[int, int]] = set()

    cell_area_km2 = abs(transform.a * transform.e) / 1_000_000.0
    outlet_area_cells = float(accumulation[outlet_row, outlet_col] + 1.0)
    total_watershed_area_km2 = outlet_area_cells * cell_area_km2
    target_area_km2 = total_watershed_area_km2 / max(1, float(count))
    min_area_km2 = target_area_km2 * max(0.0, 1.0 - area_balance_tolerance)

    def distance_ok(rc: Tuple[int, int]) -> bool:
        if min_distance_cells <= 0:
            return True
        r, c = rc
        for sr, sc in selected_cells:
            if math.hypot(r - sr, c - sc) < min_distance_cells:
                return False
        return True

    def trace_path(start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        current = start
        seen: Set[Tuple[int, int]] = set()
        while True:
            path.append(current)
            idx = current[0] * cols + current[1]
            upstream_cells = upstream_index[idx]
            if not upstream_cells:
                break
            next_cell = max(upstream_cells, key=lambda rc: accumulation[rc[0], rc[1]])
            if next_cell in seen:
                break
            seen.add(next_cell)
            current = next_cell
        return path

    main_path = trace_path((outlet_row, outlet_col))

    def add_cell(cell: Tuple[int, int], *, is_main: bool = False) -> None:
        if accumulation[cell[0], cell[1]] < effective_threshold:
            return
        if cell in selected_cells:
            return
        if distance_ok(cell):
            selected_cells.append(cell)
            if is_main:
                main_selected.add(cell)

    if main_path:
        clamped_fraction = min(0.75, max(0.25, main_stem_fraction))
        main_target = min(max(1, int(round(count * clamped_fraction))), len(main_path))
        main_target = min(main_target, count)
        total_cells = float(outlet_area_cells)
        if main_target == 1:
            fractions = [0.5]
        else:
            fractions = [
                (main_target - i + 0.5) / max(1, main_target)
                for i in range(1, main_target + 1)
            ]
        fractions = [min(0.95, max(0.1, frac)) for frac in fractions]
        search_index = 0
        for frac in fractions:
            target_cells = total_cells * frac
            selected_index: Optional[int] = None
            for idx in range(search_index, len(main_path)):
                cell = main_path[idx]
                if accumulation[cell[0], cell[1]] <= target_cells:
                    selected_index = idx
                    break
            if selected_index is None:
                selected_index = len(main_path) - 1
            candidate_idx = selected_index
            candidate_cell = main_path[candidate_idx]
            area_km2 = (accumulation[candidate_cell[0], candidate_cell[1]] + 1.0) * cell_area_km2
            if (area_km2 < min_area_km2) or (candidate_cell in selected_cells):
                fallback_idx = candidate_idx
                found = False
                while fallback_idx > 0:
                    fallback_idx -= 1
                    candidate = main_path[fallback_idx]
                    candidate_area = (accumulation[candidate[0], candidate[1]] + 1.0) * cell_area_km2
                    if candidate_area >= min_area_km2 and candidate not in selected_cells:
                        candidate_idx = fallback_idx
                        candidate_cell = candidate
                        found = True
                        break
                if not found and candidate_cell in selected_cells:
                    # advance upstream to find unused cell
                    ahead_idx = candidate_idx + 1
                    while ahead_idx < len(main_path):
                        candidate = main_path[ahead_idx]
                        candidate_area = (accumulation[candidate[0], candidate[1]] + 1.0) * cell_area_km2
                        if candidate not in selected_cells and candidate_area >= min_area_km2:
                            candidate_idx = ahead_idx
                            candidate_cell = candidate
                            found = True
                            break
                        ahead_idx += 1
            add_cell(candidate_cell, is_main=True)
            search_index = min(len(main_path) - 1, candidate_idx + 1)
        if len(selected_cells) < main_target:
            for cell in main_path:
                if len(selected_cells) >= main_target:
                    break
                if cell in selected_cells:
                    continue
                area_km2 = (accumulation[cell[0], cell[1]] + 1.0) * cell_area_km2
                if area_km2 < min_area_km2:
                    continue
                add_cell(cell, is_main=True)

    # Collect branch candidates
    branch_candidates_info: List[Dict[str, object]] = []
    for i, cell in enumerate(main_path):
        idx = cell[0] * cols + cell[1]
        upstream_cells = upstream_index[idx]
        main_next = main_path[i + 1] if i + 1 < len(main_path) else None
        for upstream_cell in upstream_cells:
            if upstream_cell == main_next:
                continue
            branch_path = trace_path(upstream_cell)
            if not branch_path:
                continue
            rep_index = max(0, min(len(branch_path) - 1, len(branch_path) // 2))
            rep_cell = branch_path[rep_index]
            branch_score = float(accumulation[upstream_cell[0], upstream_cell[1]])
            branch_area_km2 = (accumulation[rep_cell[0], rep_cell[1]] + 1.0) * cell_area_km2
            branch_candidates_info.append(
                {"cell": rep_cell, "score": branch_score, "area": branch_area_km2}
            )

    remaining = count - len(selected_cells)
    if remaining > 0:
        branch_candidates_info.sort(key=lambda item: item["score"], reverse=True)
        min_area_dynamic = min_area_km2
        branch_pool_iter = branch_candidates_info.copy()
        while len(selected_cells) < count and branch_pool_iter:
            added = False
            for idx, candidate in enumerate(branch_pool_iter):
                cell = candidate["cell"]  # type: ignore[index]
                area_km2 = candidate["area"]  # type: ignore[index]
                if cell in main_selected or cell in selected_cells:
                    continue
                if area_km2 < min_area_dynamic:
                    continue
                if not distance_ok(cell):
                    continue
                add_cell(cell)
                branch_pool_iter.pop(idx)
                added = True
                break
            if not added:
                if min_area_dynamic <= 0:
                    break
                min_area_dynamic *= 0.7
                if min_area_dynamic < 0:
                    min_area_dynamic = 0.0
        if len(selected_cells) < count:
            for cell in main_path:
                if len(selected_cells) >= count:
                    break
                if cell in main_selected:
                    continue
                area_km2 = (accumulation[cell[0], cell[1]] + 1.0) * cell_area_km2
                if area_km2 < min_area_km2:
                    continue
                add_cell(cell, is_main=True)

    if len(selected_cells) < count:
        # Fallback: scan main path then whole watershed for additional distinct cells
        for cell in main_path:
            if len(selected_cells) >= count:
                break
            area_km2 = (accumulation[cell[0], cell[1]] + 1.0) * cell_area_km2
            if area_km2 < min_area_km2:
                continue
            add_cell(cell, is_main=True)
        if len(selected_cells) < count:
            watershed_mask = np.zeros_like(accumulation, dtype=bool)
            stack = [(outlet_row, outlet_col)]
            while stack:
                r, c = stack.pop()
                if watershed_mask[r, c]:
                    continue
                watershed_mask[r, c] = True
                idx = r * cols + c
                for nr, nc in upstream_index[idx]:
                    if not watershed_mask[nr, nc]:
                        stack.append((nr, nc))
            candidate_indices = np.argwhere(watershed_mask)
            candidate_indices = candidate_indices[
                accumulation[candidate_indices[:, 0], candidate_indices[:, 1]] >= effective_threshold
            ]
            sorted_candidates = sorted(
                [tuple(idx) for idx in candidate_indices],
                key=lambda rc: accumulation[rc[0], rc[1]],
                reverse=True,
            )
            min_area_dynamic = min_area_km2
            for rc in sorted_candidates:
                if len(selected_cells) >= count:
                    break
                area_km2 = (accumulation[rc[0], rc[1]] + 1.0) * cell_area_km2
                if area_km2 < min_area_dynamic:
                    continue
                add_cell(rc)
            while len(selected_cells) < count and min_area_dynamic > 0:
                min_area_dynamic *= 0.7
                for rc in sorted_candidates:
                    if len(selected_cells) >= count:
                        break
                    area_km2 = (accumulation[rc[0], rc[1]] + 1.0) * cell_area_km2
                    if area_km2 < min_area_dynamic:
                        continue
                    add_cell(rc)

    selected_cells = selected_cells[:count]

    def balance_branch_areas() -> None:
        if not selected_cells:
            return
        branch_pool_iter = [
            candidate for candidate in branch_candidates_info
            if candidate["cell"] not in selected_cells and candidate["cell"] not in main_selected  # type: ignore[index]
        ]
        min_allowed = min_area_km2
        max_iters = len(selected_cells) * 3
        iter_count = 0
        while iter_count < max_iters:
            iter_count += 1
            replaced = False
            for cell in list(selected_cells):
                if cell in main_selected:
                    continue
                area_km2 = (accumulation[cell[0], cell[1]] + 1.0) * cell_area_km2
                if area_km2 >= min_allowed:
                    continue
                replacement: Optional[Tuple[int, int]] = None
                for candidate in list(branch_pool_iter):
                    cand_cell = candidate["cell"]  # type: ignore[index]
                    cand_area = candidate["area"]  # type: ignore[index]
                    if cand_area >= min_allowed and distance_ok(cand_cell):
                        replacement = cand_cell
                        branch_pool_iter.remove(candidate)
                        break
                if replacement is None:
                    continue
                selected_cells.remove(cell)
                selected_cells.append(replacement)
                replaced = True
                break
            if not replaced:
                break

    balance_branch_areas()

    def downstream_cell(cell: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        r, c = cell
        code = int(flowdir_array[r, c]) if 0 <= r < rows and 0 <= c < cols else 0
        offset = D8_OFFSETS.get(code)
        if offset is None:
            return None
        nr, nc = r + offset[0], c + offset[1]
        if 0 <= nr < rows and 0 <= nc < cols:
            return (nr, nc)
        return None

    adjusted_cells: List[Tuple[int, int]] = []
    seen_after_adjust: Set[Tuple[int, int]] = set()
    for cell in selected_cells:
        current = cell
        while (
            accumulation[current[0], current[1]] < effective_threshold
            and current != (outlet_row, outlet_col)
        ):
            next_cell = downstream_cell(current)
            if next_cell is None or next_cell == current:
                break
            current = next_cell
        if current not in seen_after_adjust:
            adjusted_cells.append(current)
            seen_after_adjust.add(current)

    selected_cells = adjusted_cells

    pour_points: List[PourPoint] = []
    for idx, (row, col) in enumerate(selected_cells, start=1):
        x, y = rasterio.transform.xy(transform, row, col, offset="center")
        pp = PourPoint(
            id=f"P{idx}",
            row=int(row),
            col=int(col),
            x=float(x),
            y=float(y),
            accumulation=float(accumulation[row, col]),
            attributes={},
        )
        pour_points.append(pp)

    if accumulation_plot:
        accumulation_plot.parent.mkdir(parents=True, exist_ok=True)
        plot_raster(
            np.log1p(accumulation),
            "Log1p flow accumulation",
            accumulation_plot,
            "inferno",
        )

    if output_geojson:
        write_pour_points_geojson(pour_points, output_geojson)
        if copy_to:
            copy_to = Path(copy_to)
            if copy_to.is_dir():
                target = copy_to / output_geojson.name
            else:
                target = copy_to
                target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_geojson, target)

    return pour_points


def generate_tree_pour_points(
    flow_acc_path: Path,
    flow_dir_path: Path,
    *,
    count: int = 6,
    accumulation_threshold: float = 0.0,
    min_distance_cells: int = 30,
    max_children: Optional[int] = 3,
    output_geojson: Optional[Path] = None,
    accumulation_plot: Optional[Path] = None,
    copy_to: Optional[Path] = None,
    main_stem_fraction: float = 0.5,
    area_balance_tolerance: float = 0.35,
) -> List[PourPoint]:
    """Generate pour points with explicit channel-network awareness."""

    if count <= 0:
        return []

    with rasterio.open(flow_acc_path) as acc_ds:
        accumulation = acc_ds.read(1)
        accumulation = np.where(np.isfinite(accumulation), accumulation, 0.0)
        transform = acc_ds.transform

    with rasterio.open(flow_dir_path) as dir_ds:
        flowdir = dir_ds.read(1)

    rows, cols = accumulation.shape
    if flowdir.shape != accumulation.shape:
        raise ValueError("Flow accumulation and flow direction grids must align.")

    acc_values = accumulation[np.isfinite(accumulation)]
    if 0.0 < accumulation_threshold < 1.0 and acc_values.size:
        effective_threshold = float(np.quantile(acc_values, accumulation_threshold))
    else:
        effective_threshold = float(accumulation_threshold)

    flat_idx = int(np.argmax(accumulation))
    outlet_row, outlet_col = divmod(flat_idx, cols)

    flowdir_array = np.where(np.isfinite(flowdir), flowdir, 0).astype(int)

    pixel_width = abs(transform.a)
    pixel_height = abs(transform.e)

    def step_length(dr: int, dc: int) -> float:
        return math.hypot(dc * pixel_width, dr * pixel_height)

    step_length_lookup: Dict[int, float] = {
        code: step_length(offset[0], offset[1])
        for code, offset in D8_OFFSETS.items()
    }

    def build_upstream_index() -> List[List[Tuple[int, int]]]:
        up: List[List[Tuple[int, int]]] = [[] for _ in range(rows * cols)]
        for r in range(rows):
            for c in range(cols):
                code = int(flowdir_array[r, c])
                offset = D8_OFFSETS.get(code)
                if offset is None:
                    continue
                nr, nc = r + offset[0], c + offset[1]
                if 0 <= nr < rows and 0 <= nc < cols:
                    up[nr * cols + nc].append((r, c))
        return up

    upstream_index = build_upstream_index()

    @lru_cache(maxsize=None)
    def longest_path_length(row: int, col: int) -> Tuple[float, Optional[Tuple[int, int]]]:
        idx = row * cols + col
        upstream_cells = upstream_index[idx]
        if not upstream_cells:
            return 0.0, None

        best_length = 0.0
        best_cell: Optional[Tuple[int, int]] = None
        for ur, uc in upstream_cells:
            code = int(flowdir_array[ur, uc])
            offset = D8_OFFSETS.get(code)
            if offset is None:
                continue
            step = step_length_lookup.get(code, step_length(offset[0], offset[1]))
            length, _ = longest_path_length(ur, uc)
            total = length + step
            if total > best_length:
                best_length = total
                best_cell = (ur, uc)
        return best_length, best_cell

    def extract_main_path() -> Tuple[List[Tuple[int, int]], List[float]]:
        path: List[Tuple[int, int]] = []
        cumulative: List[float] = []
        current = (outlet_row, outlet_col)
        total = 0.0
        while current is not None:
            path.append(current)
            cumulative.append(total)
            _, next_cell = longest_path_length(current[0], current[1])
            if next_cell is None:
                break
            code = int(flowdir_array[next_cell[0], next_cell[1]])
            offset = D8_OFFSETS.get(code)
            if offset is None:
                break
            step = step_length_lookup.get(code, step_length(offset[0], offset[1]))
            total += step
            current = next_cell
        return path, cumulative

    main_path, main_lengths = extract_main_path()
    main_index_lookup: Dict[Tuple[int, int], int] = {
        cell: idx for idx, cell in enumerate(main_path)
    }
    main_areas = [(accumulation[r, c] + 1.0) * abs(transform.a * transform.e) / 1_000_000.0 for r, c in main_path]

    selected_cells: List[Tuple[int, int]] = []
    main_selected: Set[Tuple[int, int]] = set()

    cell_area_km2 = abs(transform.a * transform.e) / 1_000_000.0
    outlet_area_cells = float(accumulation[outlet_row, outlet_col] + 1.0)
    total_watershed_area_km2 = outlet_area_cells * cell_area_km2
    target_area_km2 = total_watershed_area_km2 / max(1, float(count))
    min_area_km2 = target_area_km2 * max(0.0, 1.0 - area_balance_tolerance)

    def distance_ok(rc: Tuple[int, int]) -> bool:
        if min_distance_cells <= 0:
            return True
        r, c = rc
        for sr, sc in selected_cells:
            if math.hypot(r - sr, c - sc) < min_distance_cells:
                return False
        return True

    def select_main_points() -> Tuple[List[Tuple[int, int]], Set[int]]:
        if not main_path:
            return [], set()

        total_length = main_lengths[-1] if main_lengths else 0.0
        fractions = [0.25, 0.5, 0.75] if total_length > 0 else [0.3, 0.6, 0.9]
        clamped_fraction = min(0.75, max(0.25, main_stem_fraction))
        main_target = min(max(1, int(round(count * clamped_fraction))), len(main_path))
        used_indices: Set[int] = set()
        chosen: List[Tuple[int, int]] = []

        for frac in fractions[:main_target]:
            target_length = total_length * frac if total_length > 0 else 0.0
            idx_candidate = len(main_lengths) - 1
            for idx, length in enumerate(main_lengths):
                if length >= target_length:
                    idx_candidate = idx
                    break

            # Move downstream if necessary to satisfy area constraints.
            downstream_idx = idx_candidate
            while downstream_idx < len(main_path) and (
                main_areas[downstream_idx] < min_area_km2 or downstream_idx in used_indices
            ):
                downstream_idx += 1
            if downstream_idx < len(main_path):
                idx_candidate = downstream_idx
            else:
                upstream_idx = idx_candidate
                while upstream_idx >= 0 and (
                    main_areas[upstream_idx] < min_area_km2 or upstream_idx in used_indices
                ):
                    upstream_idx -= 1
                if upstream_idx >= 0:
                    idx_candidate = upstream_idx

            used_indices.add(idx_candidate)
            chosen.append(main_path[idx_candidate])

        return chosen, used_indices

    main_points, used_main_indices = select_main_points()
    selected_cells.extend(main_points)
    main_selected.update(main_points)

    def trace_branch(start: Tuple[int, int]) -> List[Tuple[int, int]]:
        path: List[Tuple[int, int]] = []
        current = start
        visited: Set[Tuple[int, int]] = set()
        while True:
            path.append(current)
            idx = current[0] * cols + current[1]
            upstream_cells = upstream_index[idx]
            if not upstream_cells:
                break
            next_cell = max(upstream_cells, key=lambda rc: accumulation[rc[0], rc[1]])
            if next_cell in visited:
                break
            visited.add(next_cell)
            current = next_cell
        return path

    branch_candidates: List[Dict[str, object]] = []
    for idx, cell in enumerate(main_path):
        upstream_cells = upstream_index[cell[0] * cols + cell[1]]
        main_next = main_path[idx + 1] if idx + 1 < len(main_path) else None
        children_seen = 0
        for upstream_cell in upstream_cells:
            if main_next is not None and upstream_cell == main_next:
                continue
            if max_children is not None and children_seen >= max_children:
                break
            children_seen += 1
            branch_path = trace_branch(upstream_cell)
            if not branch_path:
                continue

            branch_length = 0.0
            branch_lengths = [0.0]
            for j in range(1, len(branch_path)):
                prev = branch_path[j - 1]
                curr = branch_path[j]
                code = int(flowdir_array[curr[0], curr[1]])
                offset = D8_OFFSETS.get(code)
                if offset is None:
                    continue
                step = step_length_lookup.get(code, step_length(offset[0], offset[1]))
                branch_length += step
                branch_lengths.append(branch_length)

            repr_idx = 0
            if branch_lengths[-1] > 0:
                target = branch_lengths[-1] * 0.5
                for j, length in enumerate(branch_lengths):
                    if length >= target:
                        repr_idx = j
                        break
            repr_cell = branch_path[repr_idx]
            branch_area = (accumulation[repr_cell[0], repr_cell[1]] + 1.0) * cell_area_km2
            branch_candidates.append(
                {
                    "cell": repr_cell,
                    "area": branch_area,
                    "score": branch_area,
                    "parent_index": idx,
                }
            )

    branch_candidates.sort(key=lambda item: item["score"], reverse=True)

    branch_points: List[Tuple[int, int]] = []
    used_parent_indices: Set[int] = set()
    branch_target = max(0, count - len(main_points))
    for candidate in branch_candidates:
        if len(branch_points) >= branch_target:
            break
        cell = candidate["cell"]  # type: ignore[index]
        area = candidate["area"]  # type: ignore[index]
        parent_index = candidate["parent_index"]  # type: ignore[index]
        if cell in main_selected or cell in selected_cells:
            continue
        if not distance_ok(cell):
            continue
        if area < min_area_km2:
            continue
        if parent_index in used_parent_indices and len(branch_points) < branch_target - 1:
            continue
        branch_points.append(cell)
        used_parent_indices.add(parent_index)
        selected_cells.append(cell)

    if len(selected_cells) < count:
        for candidate in branch_candidates:
            if len(selected_cells) >= count:
                break
            cell = candidate["cell"]  # type: ignore[index]
            if cell in main_selected or cell in selected_cells:
                continue
            if not distance_ok(cell):
                continue
            selected_cells.append(cell)

    if len(selected_cells) > count:
        selected_cells = selected_cells[:count]

    area_cache: Dict[Tuple[int, int], float] = {}

    def subbasin_area(cell: Tuple[int, int]) -> float:
        if cell in area_cache:
            return area_cache[cell]
        pp = PourPoint(
            id="tmp",
            row=cell[0],
            col=cell[1],
            x=0.0,
            y=0.0,
            accumulation=float(accumulation[cell[0], cell[1]]),
        )
        mask = delineate_watershed(pp, upstream_index, (rows, cols))
        value = float(mask.sum()) * cell_area_km2
        area_cache[cell] = value
        return value

    def enforce_area_balance() -> None:
        # selected_cells is a list, so we can modify it directly without nonlocal
        branch_pool = [
            candidate
            for candidate in branch_candidates
            if candidate["cell"] not in selected_cells and candidate["cell"] not in main_selected  # type: ignore[index]
        ]
        branch_pool.sort(key=lambda item: item["area"], reverse=True)

        current_threshold = min_area_km2
        for _ in range(5):
            adjustments = False
            for idx, cell in enumerate(list(selected_cells)):
                area = subbasin_area(cell)
                if area >= current_threshold:
                    continue
                adjustments = True
                if cell in main_selected:
                    main_idx = main_index_lookup.get(cell, None)
                    if main_idx is not None:
                        search_idx = max(0, main_idx - 1)
                        while search_idx >= 0:
                            candidate_cell = main_path[search_idx]
                            if candidate_cell in selected_cells or candidate_cell in main_selected:
                                search_idx -= 1
                                continue
                            if subbasin_area(candidate_cell) >= current_threshold and distance_ok(candidate_cell):
                                selected_cells[idx] = candidate_cell
                                main_selected.remove(cell)
                                main_selected.add(candidate_cell)
                                break
                            search_idx -= 1
                else:
                    replacement = None
                    for candidate in list(branch_pool):
                        candidate_cell = candidate["cell"]  # type: ignore[index]
                        candidate_area = candidate["area"]  # type: ignore[index]
                        if candidate_cell in selected_cells or candidate_cell in main_selected:
                            branch_pool.remove(candidate)
                            continue
                        if candidate_area >= current_threshold and distance_ok(candidate_cell):
                            replacement = candidate_cell
                            branch_pool.remove(candidate)
                            break
                    if replacement is not None:
                        selected_cells[idx] = replacement
            if not adjustments:
                break
            current_threshold *= 0.85

    enforce_area_balance()

    unique_cells: List[Tuple[int, int]] = []
    for cell in selected_cells:
        if cell not in unique_cells:
            unique_cells.append(cell)
    selected_cells = unique_cells[:count]

    def downstream_cell(cell: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        r, c = cell
        code = int(flowdir_array[r, c]) if 0 <= r < rows and 0 <= c < cols else 0
        offset = D8_OFFSETS.get(code)
        if offset is None:
            return None
        nr, nc = r + offset[0], c + offset[1]
        if 0 <= nr < rows and 0 <= nc < cols:
            return (nr, nc)
        return None

    adjusted_cells: List[Tuple[int, int]] = []
    seen_after_adjust: Set[Tuple[int, int]] = set()
    for cell in selected_cells:
        current = cell
        while (
            accumulation[current[0], current[1]] < effective_threshold
            and current != (outlet_row, outlet_col)
        ):
            next_cell = downstream_cell(current)
            if next_cell is None or next_cell == current:
                break
            current = next_cell
        if current not in seen_after_adjust:
            adjusted_cells.append(current)
            seen_after_adjust.add(current)

    selected_cells = adjusted_cells[:count]

    def sort_key(cell: Tuple[int, int]) -> Tuple[int, float]:
        if cell in main_selected:
            return (0, float(main_index_lookup.get(cell, 0)))
        return (1, -subbasin_area(cell))

    selected_cells.sort(key=sort_key)

    pour_points: List[PourPoint] = []
    for idx, (row, col) in enumerate(selected_cells, start=1):
        x, y = rasterio.transform.xy(transform, row, col, offset="center")
        pp = PourPoint(
            id=f"P{idx}",
            row=int(row),
            col=int(col),
            x=float(x),
            y=float(y),
            accumulation=float(accumulation[row, col]),
            attributes={},
        )
        pour_points.append(pp)

    if accumulation_plot:
        accumulation_plot.parent.mkdir(parents=True, exist_ok=True)
        plot_raster(
            np.log1p(accumulation),
            "Log1p flow accumulation",
            accumulation_plot,
            "inferno",
        )

    if output_geojson:
        write_pour_points_geojson(pour_points, output_geojson)
        if copy_to:
            copy_to = Path(copy_to)
            if copy_to.is_dir():
                target = copy_to / output_geojson.name
            else:
                target = copy_to
                target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output_geojson, target)

    return pour_points


def build_flow_network(flow_dir_path: Path, offsets: Dict[int, Tuple[int, int]] = D8_OFFSETS) -> Tuple[np.ndarray, List[List[Tuple[int, int]]], Tuple[int, int]]:
    with rasterio.open(flow_dir_path) as dir_ds:
        flowdir = dir_ds.read(1)
    rows, cols = flowdir.shape

    upstream: List[List[Tuple[int, int]]] = [[] for _ in range(rows * cols)]
    for row in range(rows):
        for col in range(cols):
            code = int(flowdir[row, col])
            if code not in offsets:
                continue
            dr, dc = offsets[code]
            nr, nc = row + dr, col + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                upstream[nr * cols + nc].append((row, col))
    return flowdir, upstream, (rows, cols)


def delineate_watershed(
    pour_point: PourPoint,
    upstream: Sequence[List[Tuple[int, int]]],
    shape: Tuple[int, int],
) -> np.ndarray:
    rows, cols = shape
    mask = np.zeros((rows, cols), dtype=bool)
    stack = [(pour_point.row, pour_point.col)]
    while stack:
        r, c = stack.pop()
        if not (0 <= r < rows and 0 <= c < cols):
            continue
        if mask[r, c]:
            continue
        mask[r, c] = True
        idx = r * cols + c
        for nr, nc in upstream[idx]:
            if not mask[nr, nc]:
                stack.append((nr, nc))
    return mask


def masks_to_polygons(mask_map: Dict[str, np.ndarray], transform: rasterio.Affine) -> Dict[str, List[Sequence[Tuple[float, float]]]]:
    polygons: Dict[str, List[Sequence[Tuple[float, float]]]] = {}
    for basin_id, mask in mask_map.items():
        coords: List[Sequence[Tuple[float, float]]] = []
        for geom, value in rasterio.features.shapes(mask.astype(np.uint8), transform=transform):
            if value != 1:
                continue
            coords.append(geom["coordinates"][0])
        polygons[basin_id] = coords
    return polygons

FLOWDIR_COLORS = [
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#999999",
]


def compute_statistics(
    pour_points: Sequence[PourPoint],
    masks: Dict[str, np.ndarray],
    dem: np.ndarray,
    accumulation: np.ndarray,
    transform: rasterio.Affine,
) -> List[Dict[str, float]]:
    stats: List[Dict[str, float]] = []
    cell_area = cell_area_sqkm(transform)
    for pp in pour_points:
        mask = masks[pp.id]
        if not mask.any():
            continue
        area_km2 = mask.sum() * cell_area
        elevations = dem[mask]
        accum_values = accumulation[mask]
        rows, cols = np.nonzero(mask)
        xs = transform.c + cols * transform.a + rows * transform.b
        ys = transform.f + cols * transform.d + rows * transform.e
        stats.append(
            {
                "id": pp.id,
                "area_km2": float(area_km2),
                "accumulation": float(pp.accumulation),
                "mean_elevation": float(np.nanmean(elevations)),
                "max_accumulation": float(np.nanmax(accum_values)),
                "min_elevation": float(np.nanmin(elevations)),
                "max_elevation": float(np.nanmax(elevations)),
                "centroid_x": float(np.nanmean(xs)),
                "centroid_y": float(np.nanmean(ys)),
            }
        )
    return stats


def write_summary(statistics: Sequence[Dict[str, float]], output_dir: Path) -> None:
    lines = [
        "# Watershed summary",
        "| ID | Area (km^2) | Mean Elev. | Max Elev. | Min Elev. | Max Accumulation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in statistics:
        lines.append(
            f"| {item['id']} | {item['area_km2']:.3f} | {item['mean_elevation']:.1f} | "
            f"{item['max_elevation']:.1f} | {item['min_elevation']:.1f} | {item['max_accumulation']:.0f} |"
        )
    (output_dir / "delineation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_attributes_csv(statistics: Sequence[Dict[str, float]], output_dir: Path) -> None:
    fieldnames = [
        "id",
        "area_km2",
        "accumulation",
        "mean_elevation",
        "min_elevation",
        "max_elevation",
        "max_accumulation",
        "centroid_x",
        "centroid_y",
    ]
    csv_path = output_dir / "delineation_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in statistics:
            writer.writerow(item)


def plot_raster(data: np.ndarray, title: str, output: Path, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(data, origin="upper", cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_xlabel("column index")
    ax.set_ylabel("row index")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_flow_direction(data: np.ndarray, output: Path) -> None:
    direction = np.where(np.isin(data, list(D8_OFFSETS.keys())), data, np.nan)
    cmap = ListedColormap(FLOWDIR_COLORS, name="d8_flow")
    bounds = np.arange(0.5, 8.6, 1.0)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(direction, origin="upper", cmap=cmap, norm=norm)
    ax.set_title("Flow direction (D8 codes)")
    ax.set_xlabel("column index")
    ax.set_ylabel("row index")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, ticks=np.arange(1, 9))
    cbar.ax.set_yticklabels(["East", "North-East", "North", "North-West", "West", "South-West", "South", "South-East"])
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_masks(masks: Dict[str, np.ndarray], pour_points: Sequence[PourPoint], output: Path) -> None:
    sample = next(iter(masks.values()))
    canvas = np.zeros(sample.shape, dtype=float)
    for idx, (pp_id, mask) in enumerate(masks.items(), start=1):
        canvas[mask] = idx

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(canvas, origin="upper", cmap="tab20")
    ax.set_title("Watershed masks")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Subbasin index")
    for pp in pour_points:
        ax.scatter(pp.col, pp.row, c="black", marker="x", s=40)
        ax.text(pp.col + 4, pp.row + 4, pp.id, color="white")
    ax.set_xlabel("column index")
    ax.set_ylabel("row index")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def cell_area_sqkm(transform: rasterio.Affine) -> float:
    width = abs(transform.a)
    height = abs(transform.e)
    sqft = width * height
    return sqft * (0.304800609601219 ** 2) / 1_000_000.0


def plot_overview_map(
    dem: np.ndarray,
    transform: rasterio.Affine,
    polygons: Dict[str, List[Sequence[Tuple[float, float]]]],
    pour_points: Sequence[PourPoint],
    output: Path,
    dem_crs,
) -> None:
    dem_display = np.where(np.isfinite(dem), dem, np.nan)
    rows, cols = dem_display.shape
    left, bottom, right, top = array_bounds(rows, cols, transform)

    cell_height = abs(transform.e)
    cell_width = abs(transform.a)
    gy, gx = np.gradient(dem_display, cell_height, cell_width, edge_order=2)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    hillshade = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope) * np.cos(azimuth - aspect)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(hillshade, origin="upper", cmap="gray", extent=(left, right, bottom, top))
    im = ax.imshow(dem_display, origin="upper", cmap="terrain", extent=(left, right, bottom, top), alpha=0.6)
    ax.set_title("DEM with delineated watersheds")
    fig.colorbar(im, ax=ax, shrink=0.75, label="Elevation")

    for coords_list in polygons.values():
        for coords in coords_list:
            polygon = MplPolygon(coords, fill=False, edgecolor="white", linewidth=1.3)
            ax.add_patch(polygon)

    transformer = Transformer.from_crs(dem_crs, "EPSG:4326", always_xy=True)
    for pp in pour_points:
        ax.scatter(pp.x, pp.y, c="black", marker="x", s=60)
        lon, lat = transformer.transform(pp.x, pp.y)
        ax.text(pp.x + 150, pp.y + 150, f"{pp.id}\n({lat:.3f}, {lon:.3f})", color="black", fontsize=9, weight="bold")

    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
