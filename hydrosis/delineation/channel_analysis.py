"""Channel network extraction utilities built on top of delineation outputs."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from affine import Affine
from rasterio.transform import xy

from hydrosis.model import ChannelSegment

from .utils import D8_OFFSETS

GridIndex = Tuple[int, int]


def compute_channel_mask(accumulation: np.ndarray, threshold: float) -> np.ndarray:
    """Return boolean mask of cells that exceed the accumulation threshold."""

    if threshold <= 0:
        raise ValueError("Accumulation threshold must be positive.")
    if accumulation.ndim != 2:
        raise ValueError("Accumulation array must be 2-dimensional.")
    mask = np.asarray(accumulation >= threshold, dtype=bool)
    return mask


def _downstream_cell(
    row: int,
    col: int,
    flowdir: np.ndarray,
    rows: int,
    cols: int,
) -> Optional[GridIndex]:
    code = int(flowdir[row, col])
    offset = D8_OFFSETS.get(code)
    if offset is None:
        return None
    dr, dc = offset
    nr, nc = row + dr, col + dc
    if 0 <= nr < rows and 0 <= nc < cols:
        return nr, nc
    return None


def identify_channel_starts(
    channel_mask: np.ndarray,
    flowdir: np.ndarray,
    upstream: Sequence[List[GridIndex]],
) -> List[GridIndex]:
    """Locate starting cells for channel segments (sources and junction heads)."""

    rows, cols = channel_mask.shape
    starts: List[GridIndex] = []
    for row in range(rows):
        for col in range(cols):
            if not channel_mask[row, col]:
                continue
            idx = row * cols + col
            upstream_cells = [cell for cell in upstream[idx] if channel_mask[cell[0], cell[1]]]
            if len(upstream_cells) != 1:
                # channel source (0) or junction head (>1)
                starts.append((row, col))
    return starts


def trace_channel_segments(
    flowdir: np.ndarray,
    upstream: Sequence[List[GridIndex]],
    channel_mask: np.ndarray,
    transform: Affine,
    dem: Optional[np.ndarray] = None,
) -> List[ChannelSegment]:
    """Trace channel segments and compute basic geometric attributes."""

    if channel_mask.shape != flowdir.shape:
        raise ValueError("Channel mask and flow direction grids must share shape.")
    if dem is not None and dem.shape != channel_mask.shape:
        raise ValueError("DEM grid must match flow direction shape.")

    rows, cols = channel_mask.shape
    starts = identify_channel_starts(channel_mask, flowdir, upstream)
    visited: Set[GridIndex] = set()
    processed_starts: Set[GridIndex] = set()
    segments: List[ChannelSegment] = []
    cell_to_segment: Dict[GridIndex, str] = {}
    def step_length(current: GridIndex, nxt: GridIndex) -> float:
        x0, y0 = xy(transform, current[0], current[1], offset="center")
        x1, y1 = xy(transform, nxt[0], nxt[1], offset="center")
        return math.hypot(x1 - x0, y1 - y0)

    seg_counter = 1
    for start in starts:
        if start in processed_starts:
            continue
        current = start
        cells: List[GridIndex] = [current]
        visited.add(current)
        idx = current[0] * cols + current[1]
        while True:
            downstream = _downstream_cell(current[0], current[1], flowdir, rows, cols)
            if (
                downstream is None
                or not channel_mask[downstream[0], downstream[1]]
            ):
                break
            upstream_cells = [
                cell
                for cell in upstream[downstream[0] * cols + downstream[1]]
                if channel_mask[cell[0], cell[1]]
            ]
            cells.append(downstream)
            current = downstream
            if downstream in visited:
                break
            visited.add(downstream)
            # stop at next junction (including pour point)
            if len(upstream_cells) != 1:
                break
        seg_id = f"S{seg_counter}"
        seg_counter += 1
        cell_to_segment.update({cell: seg_id for cell in cells})

        length = 0.0
        for i in range(1, len(cells)):
            length += step_length(cells[i - 1], cells[i])
        if length <= 0.0:
            pixel_size = math.hypot(transform.a, transform.e)
            if pixel_size <= 0.0:
                pixel_size = max(abs(transform.a), abs(transform.e), 1.0)
            length = float(pixel_size)

        if dem is not None:
            start_elev = float(dem[cells[0][0], cells[0][1]])
            end_elev = float(dem[cells[-1][0], cells[-1][1]])
            if np.isnan(start_elev):
                start_elev = None
            if np.isnan(end_elev):
                end_elev = None
        else:
            start_elev = None
            end_elev = None
        drop = None
        slope = None
        if start_elev is not None and end_elev is not None:
            drop = start_elev - end_elev
            slope = drop / length if length > 0 else None

        segment = ChannelSegment(
            id=seg_id,
            cells=cells,
            length_m=length,
            drop_m=drop,
            slope=slope,
            start_elevation=start_elev,
            end_elevation=end_elev,
        )
        segments.append(segment)
        processed_starts.add(start)

    # Establish connectivity
    for segment in segments:
        tail = segment.cells[-1]
        downstream = _downstream_cell(tail[0], tail[1], flowdir, rows, cols)
        if downstream is None:
            continue
        sibling_id = cell_to_segment.get(downstream)
        if sibling_id is None:
            continue
        segment.downstream = sibling_id

    downstream_to_upstream: Dict[str, List[str]] = {}
    for segment in segments:
        if segment.downstream:
            downstream_to_upstream.setdefault(segment.downstream, []).append(segment.id)

    for segment in segments:
        segment.upstream_ids = downstream_to_upstream.get(segment.id, [])

    return segments


def segments_to_feature_collection(
    segments: Iterable[ChannelSegment],
    transform: Affine,
) -> Dict[str, object]:
    """Convert traced segments into a GeoJSON FeatureCollection."""

    features: List[Dict[str, object]] = []
    for segment in segments:
        coordinates: List[Tuple[float, float]] = []
        for row, col in segment.cells:
            x_coord, y_coord = xy(transform, row, col, offset="center")
            coordinates.append((x_coord, y_coord))
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
            "properties": {
                "id": segment.id,
                "length_m": segment.length_m,
                "slope": segment.slope,
                "drop_m": segment.drop_m,
                "downstream_id": segment.downstream,
                "upstream_ids": segment.upstream_ids,
            },
        }
        features.append(feature)
    return {"type": "FeatureCollection", "features": features}
