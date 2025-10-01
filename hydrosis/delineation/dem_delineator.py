"""DEM based delineation logic for HydroSIS.

The original lightweight implementation only supported injecting
pre-computed subbasin definitions.  The module now integrates
``rasterio`` and ``richdem`` so full raster based delineation can be
performed automatically whenever those optional dependencies are
available.  The fallback code path that consumes explicit definitions is
still kept for environments that prefer to pre-process the catchment
offline.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency, imported lazily
    import rasterio
except Exception:  # pragma: no cover - keep working without rasterio
    rasterio = None  # type: ignore

try:  # pragma: no cover - optional dependency, imported lazily
    import richdem as rd
except Exception:  # pragma: no cover - keep working without richdem
    rd = None  # type: ignore

try:  # pragma: no cover - optional dependency, imported lazily
    import numpy as np
except Exception:  # pragma: no cover - keep working without numpy
    np = None  # type: ignore

from ..model import Subbasin
from .simple_grid import (
    build_subbasin_polygons,
    delineate_from_json,
    grid_to_geojson_features,
    watershed_area,
)

Coordinates = Tuple[float, float]
GridLocation = Tuple[int, int]


@dataclass
class DelineationConfig:
    dem_path: Path
    pour_points_path: Path
    accumulation_threshold: float = 1000.0
    burn_streams_path: Optional[Path] = None
    precomputed_subbasins: Optional[List[Mapping[str, object]]] = None
    # Optional external subbasin boundary layer (GeoJSON). When provided,
    # report helpers can prefer this geometry over derived outlines.
    boundaries_path: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "DelineationConfig":
        return cls(
            dem_path=Path(data["dem_path"]),
            pour_points_path=Path(data["pour_points_path"]),
            accumulation_threshold=float(data.get("accumulation_threshold", 1000.0)),
            burn_streams_path=Path(data["burn_streams_path"]) if data.get("burn_streams_path") else None,
            precomputed_subbasins=list(data.get("precomputed_subbasins", [])) or None,
            boundaries_path=Path(data["boundaries_path"]) if data.get("boundaries_path") else None,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "dem_path": str(self.dem_path),
            "pour_points_path": str(self.pour_points_path),
            "accumulation_threshold": self.accumulation_threshold,
            "burn_streams_path": str(self.burn_streams_path) if self.burn_streams_path else None,
            "precomputed_subbasins": list(self.precomputed_subbasins) if self.precomputed_subbasins else None,
            "boundaries_path": str(self.boundaries_path) if self.boundaries_path else None,
        }

    def to_subbasins(self) -> List[Subbasin]:
        """Convert either pre-computed delineations or generate them from the DEM."""

        if self.precomputed_subbasins is not None:
            return self._from_precomputed(self.precomputed_subbasins)

        if (
            (rasterio is None or rd is None or np is None)
            and self.dem_path.suffix.lower() == ".json"
        ):
            return self._delineate_from_json()

        if rasterio is None or rd is None or np is None:
            raise RuntimeError(
                "Automatic DEM delineation requires the optional rasterio, richdem"
                " and numpy"
                " dependencies.  Install them or provide `precomputed_subbasins` in"
                " the configuration.  The demonstration configuration ships with"
                " a JSON DEM that exercises the lightweight fallback delineator."
            )

        return self._delineate_from_dem()

    def _delineate_from_json(self) -> List[Subbasin]:
        dem, watersheds, downstream_map, accumulation = delineate_from_json(
            self.dem_path,
            self.pour_points_path,
            float(self.accumulation_threshold),
        )

        cell_area_km2 = abs(
            dem.transform.pixel_width * dem.transform.pixel_height
        ) / 1_000_000.0
        subbasins: List[Subbasin] = []
        for basin_id, cells in watersheds.items():
            area = watershed_area(cell_area_km2, cells)
            subbasins.append(
                Subbasin(
                    id=basin_id,
                    area_km2=area,
                    downstream=downstream_map.get(basin_id),
                    parameters={},
                )
            )

        # Store intermediate artefacts for the GIS report helper.  The
        # lightweight delineator saves them next to the DEM to avoid changing
        # the broader configuration interface.
        artifacts_dir = self.dem_path.parent / "derived"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        geojson = {
            "type": "FeatureCollection",
            "features": grid_to_geojson_features(dem, accumulation, "accumulation"),
        }
        (artifacts_dir / "flow_accumulation.geojson").write_text(
            json.dumps(geojson, indent=2),
            encoding="utf-8",
        )

        polygons = build_subbasin_polygons(dem, watersheds)
        features = []
        for basin_id, polygon in polygons.items():
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiPolygon", "coordinates": [polygon]},
                    "properties": {"id": basin_id},
                }
            )
        (artifacts_dir / "subbasins.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
            encoding="utf-8",
        )

        return subbasins

    @staticmethod
    def _from_precomputed(entries: Iterable[Mapping[str, object]]) -> List[Subbasin]:
        subbasins: List[Subbasin] = []
        for entry in entries:
            subbasins.append(
                Subbasin(
                    id=str(entry["id"]),
                    area_km2=float(entry.get("area_km2", 0.0)),
                    downstream=entry.get("downstream"),
                    parameters=dict(entry.get("parameters", {})),
                )
            )
        return subbasins

    # -- Automated delineation utilities -------------------------------------------------

    def _delineate_from_dem(self) -> List[Subbasin]:
        assert rasterio is not None and rd is not None and np is not None

        with rasterio.open(self.dem_path) as dataset:
            dem = dataset.read(1, masked=True)
            transform = dataset.transform
            cell_area_km2 = abs(transform.a * transform.e) / 1_000_000.0
            dem_array = rd.rdarray(
                dem.filled(dataset.nodata if dataset.nodata is not None else np.nan),
                no_data=dataset.nodata if dataset.nodata is not None else np.nan,
            )
            dem_array.projection = dataset.crs.to_wkt() if dataset.crs else None
            dem_array.geotransform = (
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            )

        # richdem processing chain
        filled = rd.FillDepressions(dem_array, in_place=False)
        flowdir = rd.FlowDirD8(filled)
        accumulation = rd.FlowAccumulation(flowdir, method="D8")
        accumulation_array = np.array(accumulation, dtype=float)

        pour_points = list(self._load_pour_points(self.pour_points_path))
        if not pour_points:
            raise RuntimeError("No pour points found for automatic delineation")

        id_to_mask: Dict[str, np.ndarray] = {}
        for point_id, location in pour_points:
            mask = self._watershed_mask(flowdir, location)
            if self.accumulation_threshold > 0:
                mask = np.logical_and(
                    mask,
                    accumulation_array >= float(self.accumulation_threshold),
                )
            id_to_mask[point_id] = mask

        subbasins: List[Subbasin] = []
        downstream_map = self._infer_downstream_relationships(pour_points, flowdir)
        for basin_id, mask in id_to_mask.items():
            area_cells = int(mask.sum())
            subbasins.append(
                Subbasin(
                    id=basin_id,
                    area_km2=area_cells * cell_area_km2,
                    downstream=downstream_map.get(basin_id),
                    parameters={},
                )
            )
        # Export derived artefacts for GIS reporting (parity with JSON delineation)
        try:
            derived_dir = Path(self.dem_path).parent / "derived"
            derived_dir.mkdir(parents=True, exist_ok=True)

            # --- Flow accumulation as GeoJSON grid ---
            def _cell_bounds(transform, row: int, col: int):
                a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
                x0 = c + a * col + b * row
                y0 = f + d * col + e * row
                x1 = c + a * (col + 1) + b * (row + 1)
                y1 = f + d * (col + 1) + e * (row + 1)
                xmin, xmax = (x0, x1) if x0 <= x1 else (x1, x0)
                ymin, ymax = (y0, y1) if y0 <= y1 else (y1, y0)
                return (xmin, ymin), (xmax, ymax)

            acc_features: List[Dict[str, object]] = []
            rows, cols = accumulation_array.shape
            for r in range(rows):
                for cidx in range(cols):
                    val = float(accumulation_array[r, cidx])
                    if np.isfinite(val):
                        (xmin, ymin), (xmax, ymax) = _cell_bounds(transform, r, cidx)
                        acc_features.append(
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
                                "properties": {"accumulation": val, "row": r, "col": cidx},
                            }
                        )
            (derived_dir / "flow_accumulation.geojson").write_text(
                json.dumps({"type": "FeatureCollection", "features": acc_features}, indent=2),
                encoding="utf-8",
            )

            # --- Subbasin polygons constructed from masks by cancelling shared edges ---
            def _mask_to_rings(mask: np.ndarray) -> List[List[Tuple[float, float]]]:
                edge_count: Dict[frozenset, int] = {}
                rows_m, cols_m = mask.shape
                for r in range(rows_m):
                    for cidx in range(cols_m):
                        if not bool(mask[r, cidx]):
                            continue
                        (xmin, ymin), (xmax, ymax) = _cell_bounds(transform, r, cidx)
                        edges = [
                            ((xmin, ymin), (xmax, ymin)),
                            ((xmax, ymin), (xmax, ymax)),
                            ((xmax, ymax), (xmin, ymax)),
                            ((xmin, ymax), (xmin, ymin)),
                        ]
                        for a, b in edges:
                            key = frozenset((a, b))
                            edge_count[key] = edge_count.get(key, 0) + 1

                boundary_edges: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
                for key, count in edge_count.items():
                    if count == 1:
                        a, b = tuple(key)
                        if a <= b:
                            boundary_edges.append((a, b))
                        else:
                            boundary_edges.append((b, a))

                from collections import defaultdict
                adjacency: Dict[Tuple[float, float], List[Tuple[float, float]]] = defaultdict(list)
                for a, b in boundary_edges:
                    adjacency[a].append(b)
                    adjacency[b].append(a)

                rings: List[List[Tuple[float, float]]] = []
                visited: set[Tuple[Tuple[float, float], Tuple[float, float]]] = set()
                for a, b in boundary_edges:
                    if (a, b) in visited or (b, a) in visited:
                        continue
                    ring: List[Tuple[float, float]] = [a, b]
                    prev, curr = a, b
                    visited.add((a, b))
                    while True:
                        neighbours = adjacency[curr]
                        next_pt = None
                        for n in neighbours:
                            if n != prev and ((curr, n) not in visited):
                                next_pt = n
                                break
                        if next_pt is None:
                            if curr != ring[0]:
                                ring.append(ring[0])
                            break
                        ring.append(next_pt)
                        visited.add((curr, next_pt))
                        prev, curr = curr, next_pt
                        if curr == ring[0]:
                            break
                    if ring[-1] != ring[0]:
                        ring.append(ring[0])
                    if len(ring) >= 4:
                        rings.append(ring)
                return rings

            sb_features: List[Dict[str, object]] = []
            for basin_id, mask in id_to_mask.items():
                rings = _mask_to_rings(mask)
                if not rings:
                    continue
                sb_features.append(
                    {
                        "type": "Feature",
                        "geometry": {"type": "MultiPolygon", "coordinates": [rings]},
                        "properties": {"id": basin_id},
                    }
                )
            (derived_dir / "subbasins.geojson").write_text(
                json.dumps({"type": "FeatureCollection", "features": sb_features}, indent=2),
                encoding="utf-8",
            )
        except Exception:
            # Optional export; ignore failures
            pass

        return subbasins

    def _load_pour_points(self, path: Path) -> Iterator[Tuple[str, GridLocation]]:
        if rasterio is None:
            return iter(())

        with rasterio.open(self.dem_path) as dataset:
            transform = dataset.transform
            suffix = path.suffix.lower()
            if suffix == ".csv":
                with open(path, "r", encoding="utf-8") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        try:
                            x = float(row["x"])
                            y = float(row["y"])
                        except (KeyError, ValueError):
                            raise RuntimeError(
                                "Pour point CSV must contain numeric 'x' and 'y' columns"
                            ) from None
                        row_idx, col_idx = rasterio.transform.rowcol(transform, x, y)
                        yield str(row.get("id", f"PP_{row_idx}_{col_idx}")), (row_idx, col_idx)
            elif suffix in {".json", ".geojson"}:
                data = json.loads(path.read_text(encoding="utf-8"))
                features = data.get("features", [])
                for feature in features:
                    geometry = feature.get("geometry", {}) or {}
                    coords = geometry.get("coordinates")
                    if not coords:
                        continue
                    x, y = coords[:2]
                    properties = feature.get("properties", {}) or {}
                    row_idx, col_idx = rasterio.transform.rowcol(transform, x, y)
                    yield str(properties.get("id", f"PP_{row_idx}_{col_idx}")), (row_idx, col_idx)
            else:
                raise RuntimeError(
                    "Unsupported pour points format. Provide CSV or GeoJSON inputs."
                )

    @staticmethod
    def _watershed_mask(flowdir: "rd.rdarray", location: GridLocation) -> np.ndarray:
        assert rd is not None and np is not None
        mask = rd.Watershed(flowdir, outlet=location)
        if isinstance(mask, rd.rdarray):  # pragma: no cover - depends on richdem internals
            return np.array(mask, dtype=bool)
        return np.array(mask, dtype=bool)

    def _infer_downstream_relationships(
        self,
        pour_points: Sequence[Tuple[str, GridLocation]],
        flowdir: "rd.rdarray",
    ) -> Dict[str, Optional[str]]:
        assert rd is not None and np is not None

        direction_map = self._direction_mapping()
        downstream: Dict[str, Optional[str]] = {pid: None for pid, _ in pour_points}
        lookup = {loc: pid for pid, loc in pour_points}
        flow_array = np.array(flowdir)
        rows, cols = flow_array.shape

        for pid, (row_idx, col_idx) in pour_points:
            direction = int(flow_array[row_idx, col_idx])
            delta = direction_map.get(direction)
            if delta is None:
                continue
            nrow, ncol = row_idx + delta[0], col_idx + delta[1]
            if 0 <= nrow < rows and 0 <= ncol < cols:
                downstream[pid] = lookup.get((nrow, ncol))
        return downstream

    @staticmethod
    def _direction_mapping() -> Dict[int, Tuple[int, int]]:
        # D8 codes used by richdem's FlowDirD8 implementation.
        return {
            1: (0, 1),  # East
            2: (-1, 1),  # North-East
            4: (-1, 0),  # North
            8: (-1, -1),  # North-West
            16: (0, -1),  # West
            32: (1, -1),  # South-West
            64: (1, 0),  # South
            128: (1, 1),  # South-East
        }


__all__ = ["DelineationConfig"]
