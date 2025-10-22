"""Standalone utility for generating synthetic rain gauge inputs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from hydrosis.precipitation import generate_rain_gauge_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic rain gauge forcing for testing.")
    parser.add_argument("--base-precip", type=Path, required=True, help="CSV file containing the base precipitation time series.")
    parser.add_argument("--column", type=str, default=None, help="Column name holding the base precipitation (defaults to the first numeric column).")
    parser.add_argument("--subbasins", type=Path, required=True, help="GeoJSON file with subbasin polygons (must include 'id' property).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory used to write generated artefacts.")
    parser.add_argument("--station-count", type=int, default=10, help="Number of synthetic rain gauges to create.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--heterogeneity", type=float, default=0.5, help="Strength of spatial heterogeneity (0-1).")
    parser.add_argument("--min-bursts", type=int, default=2, help="Minimum number of localised burst events per station.")
    parser.add_argument("--max-bursts", type=int, default=4, help="Maximum number of localised burst events per station.")
    parser.add_argument("--gauges-filename", type=str, default="rain_gauge_forcing.csv", help="Filename for rain gauge time series output.")
    parser.add_argument("--subbasin-filename", type=str, default="subbasin_areal_precipitation.csv", help="Filename for subbasin precipitation output.")
    parser.add_argument("--stations-geojson", type=str, default="rain_gauge_locations.geojson", help="Filename for station GeoJSON output.")
    parser.add_argument("--thiessen-geojson", type=str, default="rain_gauge_thiessen_polygons.geojson", help="Filename for Thiessen polygon GeoJSON output.")
    parser.add_argument("--weights-json", type=str, default="rain_gauge_weights.json", help="Filename for station weight JSON output.")
    return parser.parse_args()


def _load_base_series(csv_path: Path, column: str | None) -> pd.Series:
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if column is None:
        # Choose the first numeric column automatically
        numeric_columns: List[str] = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_columns:
            raise ValueError(f"No numeric columns found in {csv_path}.")
        column = numeric_columns[0]
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in {csv_path}.")
    series = pd.to_numeric(df[column], errors="coerce")
    if series.isna().all():
        raise ValueError(f"Column '{column}' in {csv_path} does not contain valid numeric values.")
    series.name = column
    return series


def _load_subbasin_geometries(path: Path):
    import json
    from shapely.geometry import shape

    data = json.loads(path.read_text(encoding="utf-8"))
    geometries = {}
    for feature in data.get("features", []):
        properties = feature.get("properties", {}) or {}
        sub_id = str(properties.get("id") or feature.get("id") or "").strip()
        if not sub_id:
            continue
        geometries[sub_id] = shape(feature.get("geometry"))
    if not geometries:
        raise ValueError(f"No subbasin geometries found in {path}")
    return geometries


def main() -> None:
    args = parse_args()

    base_series = _load_base_series(args.base_precip, args.column)
    subbasin_geometries = _load_subbasin_geometries(args.subbasins)

    rain_inputs = generate_rain_gauge_inputs(
        base_series,
        subbasin_geometries,
        station_count=args.station_count,
        rng_seed=args.seed,
        heterogeneity_strength=args.heterogeneity,
        min_burst_events=args.min_bursts,
        max_burst_events=args.max_bursts,
    )

    outputs = rain_inputs.write(
        args.output_dir,
        gauges_filename=args.gauges_filename,
        subbasin_filename=args.subbasin_filename,
        stations_geojson=args.stations_geojson,
        thiessen_geojson=args.thiessen_geojson,
        weights_json=args.weights_json,
    )

    for label, path in outputs.items():
        print(f"{label.capitalize()} written to {path}")


if __name__ == "__main__":
    main()
