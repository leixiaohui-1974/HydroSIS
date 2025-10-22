"""Sample cross-sections orthogonal to channel centre lines."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from rasterio import transform as rio_transform
from shapely.geometry import LineString, shape


def _unit_normal(p0: Tuple[float, float], p1: Tuple[float, float]) -> np.ndarray:
    vec = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=float)
    length = np.linalg.norm(vec)
    if length == 0:
        return np.array([0.0, 0.0], dtype=float)
    tangent = vec / length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    return normal / np.linalg.norm(normal)


def _sample_dem_along_line(
    dem: rasterio.io.DatasetReader,
    start: Tuple[float, float],
    end: Tuple[float, float],
    n_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(start[0], end[0], n_points)
    ys = np.linspace(start[1], end[1], n_points)
    elevations = []
    for x, y in zip(xs, ys):
        sample = next(dem.sample([(x, y)]), [math.nan])
        elevations.append(float(sample[0]))
    distances = np.linspace(-0.5, 0.5, n_points)
    return xs, ys, np.asarray(elevations), distances


def extract_cross_sections(
    geojson_path: Path,
    dem_path: Path,
    *,
    zone_filter: Iterable[str],
    spacing_m: float,
    half_width_m: float,
    n_points: int,
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with geojson_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    with rasterio.open(dem_path) as dem:
        crs = dem.crs
        transform = dem.transform
        outputs: List[Path] = []
        for feature in data.get("features", []):
            props = feature.get("properties") or {}
            zone_id = props.get("zone_id")
            if zone_filter and zone_id not in zone_filter:
                continue
            segment_id = props.get("segment_id") or props.get("subzone_id")
            if segment_id is None:
                continue

            geom = shape(feature.get("geometry"))
            if not isinstance(geom, LineString):
                continue
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            cumulative = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
            )
            if cumulative[-1] == 0:
                continue
            n_sections = max(1, int(cumulative[-1] // spacing_m))
            distances = np.linspace(0.0, cumulative[-1], n_sections)
            sections: List[pd.DataFrame] = []
            for dist in distances:
                idx = np.searchsorted(cumulative, dist, side="right") - 1
                idx = max(0, min(idx, len(coords) - 2))
                local_start = np.array(coords[idx])
                local_end = np.array(coords[idx + 1])
                local_frac = (
                    (dist - cumulative[idx])
                    / max(cumulative[idx + 1] - cumulative[idx], 1e-6)
                )
                centre = local_start + (local_end - local_start) * local_frac
                normal = _unit_normal(local_start, local_end)
                start = centre - normal * half_width_m
                end = centre + normal * half_width_m
                xs, ys, elev, rel = _sample_dem_along_line(dem, start, end, n_points)
                df = pd.DataFrame(
                    {
                        "x": xs,
                        "y": ys,
                        "distance_from_center_m": rel * 2 * half_width_m,
                        "elevation_m": elev,
                        "station_m": dist,
                        "segment_id": segment_id,
                        "zone_id": zone_id,
                        "subzone_id": props.get("subzone_id"),
                    }
                )
                sections.append(df)

            if sections:
                profile_df = pd.concat(sections, ignore_index=True)
                profile_df["dem_crs"] = str(crs)
                profile_df["dem_transform"] = str(transform)
                output_path = output_dir / f"{segment_id}_cross_sections.csv"
                profile_df.to_csv(output_path, index=False)
                outputs.append(output_path)
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract channel cross-sections from DEM along parameter channels."
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        default=Path("results/upper_truckee_channel_demo/parameters/parameter_channels.geojson"),
        help="Parameter channel GeoJSON.",
    )
    parser.add_argument(
        "--dem",
        type=Path,
        default=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"),
        help="DEM used for elevation sampling.",
    )
    parser.add_argument(
        "--zones",
        nargs="*",
        default=["P3", "P4"],
        help="Zone identifiers to process.",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=500.0,
        help="Spacing (m) between consecutive cross-sections along the channel.",
    )
    parser.add_argument(
        "--half-width",
        type=float,
        default=150.0,
        help="Half-width (m) to sample on either side of the channel centre line.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=41,
        help="Number of sample points per cross-section.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/upper_truckee_channel_demo/intermediate/channel_cross_sections"),
        help="Directory to write cross-section CSV files.",
    )
    args = parser.parse_args()
    created = extract_cross_sections(
        args.geojson,
        args.dem,
        zone_filter=args.zones,
        spacing_m=args.spacing,
        half_width_m=args.half_width,
        n_points=args.points,
        output_dir=args.output_dir,
    )
    if created:
        print("Generated cross-section files:")
        for path in created:
            print("  ", path)
    else:
        print("No cross-sections were generated for the requested zones.")


if __name__ == "__main__":
    main()
