from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import json
import math

import numpy as np

try:
    import rasterio
    from rasterio.windows import from_bounds
    from rasterio.transform import Affine
    RASTERIO_AVAILABLE = True
except Exception:  # pragma: no cover
    RASTERIO_AVAILABLE = False


@dataclass
class DEMProcessingResult:
    flow_direction_tif: Optional[Path]
    flow_accumulation_tif: Optional[Path]
    flow_accumulation_geojson: Optional[Path]
    stream_network_geojson: Optional[Path]
    parameter_zones_tif: Optional[Path]
    parameter_zones_geojson: Optional[Path]
    stats: dict


def crop_dem_to_bbox(dem_path: Path, bbox: Sequence[float], out_path: Path) -> Path:
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio 未安装，无法裁剪 DEM")
    minx, miny, maxx, maxy = bbox
    with rasterio.open(dem_path) as src:
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)
        profile = src.profile.copy()
        data = src.read(1, window=window)
        transform = rasterio.windows.transform(window, src.transform)
        profile.update({"height": data.shape[0], "width": data.shape[1], "transform": transform})
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
    return out_path


def compute_d8_flow_direction(dem: np.ndarray, *, transform: Optional["Affine"] = None) -> np.ndarray:
    """Compute D8 flow direction (ESRI style) codes.
    Codes: 1(E), 2(SE), 4(S), 8(SW), 16(W), 32(NW), 64(N), 128(NE)
    """
    h, w = dem.shape
    directions = np.zeros_like(dem, dtype=np.uint8)

    dx = 1.0
    dy = 1.0
    if transform is not None:
        try:
            dx = abs(float(transform.a))
            dy = abs(float(transform.e))
        except Exception:
            pass

    diag = math.hypot(dx, dy)

    # neighbor offsets and corresponding codes
    neighbors = [
        (0, 1, 1, dx),    # E
        (1, 1, 2, diag),  # SE
        (1, 0, 4, dy),    # S
        (1, -1, 8, diag), # SW
        (0, -1, 16, dx),  # W
        (-1, -1, 32, diag), # NW
        (-1, 0, 64, dy),  # N
        (-1, 1, 128, diag) # NE
    ]

    # pad with nan around to avoid bounds checks
    pad = np.pad(dem, 1, mode='edge')

    for i in range(h):
        for j in range(w):
            ci, cj = i + 1, j + 1
            z = pad[ci, cj]
            if not np.isfinite(z):
                continue
            max_slope = -np.inf
            code = 0
            for di, dj, c, distance in neighbors:
                nz = pad[ci + di, cj + dj]
                if not np.isfinite(nz):
                    continue
                drop = z - nz
                slope = drop / distance if distance > 0 else drop
                if slope > max_slope and drop > 0:
                    max_slope = slope
                    code = c
            directions[i, j] = code if max_slope > 0 else 0
    return directions


def compute_flow_accumulation(directions: np.ndarray) -> np.ndarray:
    """Compute flow accumulation using a linear-time topological propagation.

    Each cell contributes 1 to itself and flows to its single D8 neighbor.
    We build indegree of each cell from upstream neighbors, then process cells
    in a queue when their upstream contributions are complete.
    """
    h, w = directions.shape
    acc = np.ones((h, w), dtype=np.int64)

    dir_map = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1),
        32: (-1, -1), 64: (-1, 0), 128: (-1, 1)
    }

    indeg = np.zeros((h, w), dtype=np.int32)

    # Compute indegree: for each cell, increment downstream cell's indegree
    for i in range(h):
        for j in range(w):
            code = int(directions[i, j])
            if code == 0:
                continue
            di, dj = dir_map.get(code, (0, 0))
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w:
                indeg[ni, nj] += 1

    from collections import deque
    q: deque[tuple[int, int]] = deque()
    for i in range(h):
        for j in range(w):
            if indeg[i, j] == 0:
                q.append((i, j))

    while q:
        i, j = q.popleft()
        code = int(directions[i, j])
        if code == 0:
            continue
        di, dj = dir_map.get(code, (0, 0))
        ni, nj = i + di, j + dj
        if 0 <= ni < h and 0 <= nj < w:
            acc[ni, nj] += acc[i, j]
            indeg[ni, nj] -= 1
            if indeg[ni, nj] == 0:
                q.append((ni, nj))

    return acc


def save_array_as_tif(array: np.ndarray, ref_profile: dict, transform: Affine, out_path: Path, dtype: str = "uint32") -> Path:
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("rasterio 未安装，无法写出 TIF")
    profile = ref_profile.copy()
    profile.update({"dtype": dtype, "count": 1, "height": array.shape[0], "width": array.shape[1], "transform": transform})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(array.astype(profile["dtype"]), 1)
    return out_path


def export_accumulation_geojson(acc: np.ndarray, transform: Affine, threshold: int, out_path: Path) -> Path:
    """Export accumulation cells above threshold as Point GeoJSON (cell centers)."""
    features = []
    h, w = acc.shape
    for i in range(h):
        for j in range(w):
            if acc[i, j] >= threshold:
                x, y = transform * (j + 0.5, i + 0.5)
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [x, y]},
                    "properties": {"acc": int(acc[i, j])}
                })
    fc = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc))
    return out_path


def export_parameter_zones(acc: np.ndarray, transform: Affine, ref_profile: dict, out_tif: Path, out_geojson: Path) -> Tuple[Path, Path]:
    """Classify accumulation into 3 parameter zones and export as TIF + GeoJSON grid.

    Zones:
      1 = 低流量区 (low)
      2 = 中等流量区 (mid)
      3 = 高流量区 (high)
    """
    vals = acc.flatten()
    vals = vals[vals > 0]
    if vals.size == 0:
        zones = np.zeros_like(acc, dtype=np.uint8)
        q1 = q2 = 0
    else:
        q1 = int(np.percentile(vals, 33))
        q2 = int(np.percentile(vals, 66))
        zones = np.zeros_like(acc, dtype=np.uint8)
        zones[(acc > 0) & (acc <= q1)] = 1
        zones[(acc > q1) & (acc <= q2)] = 2
        zones[acc > q2] = 3

    # Save zones raster
    save_array_as_tif(zones, ref_profile, transform, out_tif, dtype="uint8")

    # Export grid polygons with zone attribute
    features = []
    h, w = zones.shape
    for i in range(h):
        for j in range(w):
            z = int(zones[i, j])
            if z == 0:
                continue
            x0, y0 = transform * (j, i)
            x1, y1 = transform * (j + 1, i + 1)
            xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
            ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
            features.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [[[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax], [xmin, ymin]]]},
                "properties": {"zone": {1: "low", 2: "mid", 3: "high"}.get(z, "unknown"), "zone_code": z}
            })
    out_geojson.parent.mkdir(parents=True, exist_ok=True)
    out_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return out_tif, out_geojson


def _dir_offset(code: int) -> Tuple[int, int]:
    mapping = {
        1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1),
        32: (-1, -1), 64: (-1, 0), 128: (-1, 1)
    }
    return mapping.get(code, (0, 0))


def export_stream_network_geojson(acc: np.ndarray, directions: np.ndarray, transform: Affine, threshold: int, out_path: Path) -> Path:
    """Trace simple stream centerlines from accumulation mask and export as LineString GeoJSON.

    Algorithm: identify source cells (above threshold with no upstream neighbors above threshold),
    then follow D8 direction through contiguous threshold cells to build polylines.
    """
    h, w = acc.shape
    mask = acc >= threshold

    # Precompute upstream neighbor presence for each cell
    has_upstream = np.zeros_like(mask, dtype=bool)
    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                continue
            # check 8 neighbors if any flows into (i,j)
            for code, (di, dj) in {
                1: (0, -1), 2: (-1, -1), 4: (-1, 0), 8: (-1, 1), 16: (0, 1), 32: (1, 1), 64: (1, 0), 128: (1, -1)
            }.items():
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and mask[ni, nj]:
                    # neighbor flows to (i,j) if its direction points to our cell
                    odi, odj = _dir_offset(int(directions[ni, nj]))
                    if ni + odi == i and nj + odj == j:
                        has_upstream[i, j] = True
                        break

    sources = [(i, j) for i in range(h) for j in range(w) if mask[i, j] and not has_upstream[i, j]]

    features = []
    visited = np.zeros_like(mask, dtype=bool)
    for si, sj in sources:
        coords = []
        i, j = si, sj
        while 0 <= i < h and 0 <= j < w and mask[i, j] and not visited[i, j]:
            visited[i, j] = True
            x, y = transform * (j + 0.5, i + 0.5)
            coords.append([x, y])
            di, dj = _dir_offset(int(directions[i, j]))
            if di == 0 and dj == 0:
                break
            i, j = i + di, j + dj
        if len(coords) >= 2:
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"threshold": int(threshold), "length_vertices": len(coords)}
            })

    fc = {"type": "FeatureCollection", "features": features}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fc))
    return out_path


def run_dem_flow_pipeline(
    dem_path: Path,
    bbox: Sequence[float],
    out_dir: Path,
    accumulation_threshold: int = 100,
    max_pixels_no_decimate: int = 256_000_000,
    max_cells_d8: int = 16_000_000,
) -> DEMProcessingResult:
    if not RASTERIO_AVAILABLE:
        raise RuntimeError("请先安装 rasterio：pip install rasterio")

    out_dir.mkdir(parents=True, exist_ok=True)
    cropped = out_dir / "dem_cropped.tif"
    crop_dem_to_bbox(dem_path, bbox, cropped)

    with rasterio.open(cropped) as src:
        dem = src.read(1)
        transform = src.transform
        profile = src.profile

    # 对大范围数据进行自动降采样以提升速度（提高阈值以保留更多细节）
    rows, cols = dem.shape
    total = rows * cols
    decimate = 1
    import math
    # 规则1：尽量不降采样（除非超过可视阈值）
    if total > max_pixels_no_decimate:
        decimate = int(math.ceil(math.sqrt(total / max_pixels_no_decimate)))
    # 规则2：确保D8计算像素不超过上限，避免长时间卡顿
    d8_needed = int(math.ceil(math.sqrt(total / max_cells_d8)))
    decimate = max(decimate, d8_needed)
    if decimate > 1:
        dem = dem[::decimate, ::decimate]
        transform = transform * Affine.scale(decimate, decimate)

    directions = compute_d8_flow_direction(dem, transform=transform)
    accumulation = compute_flow_accumulation(directions)

    fd_tif = out_dir / "flow_direction.tif"
    fa_tif = out_dir / "flow_accumulation.tif"
    save_array_as_tif(directions, profile, transform, fd_tif, dtype="uint8")
    save_array_as_tif(accumulation, profile, transform, fa_tif, dtype="uint32")

    fa_geojson = out_dir / "flow_accumulation.geojson"
    export_accumulation_geojson(accumulation, transform, accumulation_threshold, fa_geojson)
    streams_geojson = out_dir / "stream_network.geojson"
    export_stream_network_geojson(accumulation, directions, transform, accumulation_threshold, streams_geojson)

    # Parameter zoning derived from accumulation quantiles
    zones_tif = out_dir / "parameter_zones.tif"
    zones_geojson = out_dir / "parameter_zones.geojson"
    export_parameter_zones(accumulation, transform, profile, zones_tif, zones_geojson)

    stats = {
        "dem_min": float(np.nanmin(dem)),
        "dem_max": float(np.nanmax(dem)),
        "acc_max": int(np.max(accumulation)),
        "points_exported": int(len(json.loads(fa_geojson.read_text())["features"]))
    }

    return DEMProcessingResult(
        flow_direction_tif=fd_tif,
        flow_accumulation_tif=fa_tif,
        flow_accumulation_geojson=fa_geojson,
        stream_network_geojson=streams_geojson,
        parameter_zones_tif=zones_tif,
        parameter_zones_geojson=zones_geojson,
        stats=stats,
    )
