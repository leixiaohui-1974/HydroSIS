"""Extract longitudinal channel profiles for selected zones."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
import rasterio
from shapely.geometry import LineString, shape


def _cumulative_distances(coords: Iterable[Tuple[float, float]]) -> np.ndarray:
    coords_array = np.asarray(list(coords), dtype=float)
    if coords_array.shape[0] == 0:
        return np.zeros(0, dtype=float)
    deltas = np.linalg.norm(np.diff(coords_array, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(deltas)))


def extract_profiles(
    geojson_path: Path,
    dem_path: Path,
    *,
    zone_filter: Iterable[str],
    output_dir: Path,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with geojson_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    with rasterio.open(dem_path) as dem:
        profile_paths: List[Path] = []
        for feature in data.get("features", []):
            properties = feature.get("properties") or {}
            zone_id = properties.get("zone_id")
            if zone_filter and zone_id not in zone_filter:
                continue
            segment_id = properties.get("segment_id") or properties.get("subzone_id")
            if segment_id is None:
                continue

            geom = shape(feature.get("geometry"))
            if not isinstance(geom, LineString):
                continue
            coords = list(geom.coords)
            if not coords:
                continue

            distances = _cumulative_distances(coords)
            elevations: List[float] = []
            for x, y in coords:
                sample = next(dem.sample([(x, y)]), [np.nan])
                elevations.append(float(sample[0]))

            profile_df = pd.DataFrame(
                {
                    "distance_m": distances,
                    "x": [c[0] for c in coords],
                    "y": [c[1] for c in coords],
                    "elevation_m": elevations,
                }
            )
            profile_df["segment_length_m"] = float(properties.get("length_m", np.nan))
            profile_df["segment_slope"] = float(properties.get("slope", np.nan))
            profile_df["segment_drop_m"] = float(properties.get("drop_m", np.nan))
            profile_df["zone_id"] = zone_id
            profile_df["subzone_id"] = properties.get("subzone_id")

            output_path = output_dir / f"{segment_id}_profile.csv"
            profile_df.to_csv(output_path, index=False)
            profile_paths.append(output_path)
        return profile_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract longitudinal channel profiles from parameter channel geometries."
    )
    parser.add_argument(
        "--geojson",
        type=Path,
        default=Path("results/upper_truckee_channel_demo/parameters/parameter_channels.geojson"),
        help="Path to the parameter channel GeoJSON file.",
    )
    parser.add_argument(
        "--dem",
        type=Path,
        default=Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"),
        help="DEM used to sample elevations along the channel.",
    )
    parser.add_argument(
        "--zones",
        type=str,
        nargs="*",
        default=["P3", "P4"],
        help="Zone identifiers whose channels should be processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/upper_truckee_channel_demo/intermediate/channel_profiles"),
        help="Directory where profile CSVs will be written.",
    )
    args = parser.parse_args()

    created = extract_profiles(
        args.geojson,
        args.dem,
        zone_filter=args.zones,
        output_dir=args.output_dir,
    )
    if created:
        print("Generated profiles:")
        for path in created:
            print("  ", path)
    else:
        print("No matching channel segments were found for the requested zones.")


if __name__ == "__main__":
    main()
