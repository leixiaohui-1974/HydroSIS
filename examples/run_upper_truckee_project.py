"""End-to-end Upper Truckee project runner built on reusable workflow stages."""
# TODO(stage-migration): Port scenario comparisons, zone diagnostics, and rainfall visualisations
# from examples/upper_truckee_channel_workflow.py into dedicated workflow stages.
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from hydrosis.config import HydroProjectConfig
from hydrosis.delineation import utils as dutils
from hydrosis.model import Subbasin
from hydrosis.parameters.partition import ChannelSummary, PartitionOutputs
from hydrosis.workflow import run_workflow
from hydrosis.workflow.stages import (
    generate_precipitation_for_parameters,
    run_channel_diagnostics,
    run_delineation_stage,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Upper Truckee HydroSIS project using staged helpers.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/upper_truckee_project.yml"),
        help="Path to the HydroProject configuration YAML file.",
    )
    parser.add_argument(
        "--base-precipitation",
        type=Path,
        help="Optional CSV supplying a basin-average precipitation series used to seed station generation.",
    )
    parser.add_argument(
        "--station-series",
        type=Path,
        help="Observed rain-gauge CSV to interpolate instead of generating synthetic gauges.",
    )
    parser.add_argument(
        "--thiessen-polygons",
        type=Path,
        help="GeoJSON with Thiessen polygons for the provided station series.",
    )
    parser.add_argument(
        "--station-count",
        type=int,
        help="Number of synthetic rain gauges to generate when no station series is supplied.",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        help="Random seed forwarded to the precipitation generation routine.",
    )
    parser.add_argument(
        "--disable-auto-threshold",
        action="store_true",
        help="Disable automatic channel-threshold suggestion during delineation.",
    )
    parser.add_argument(
        "--channel-threshold",
        type=float,
        help="Override channel-accumulation threshold when auto-selection is disabled.",
    )
    parser.add_argument(
        "--persist-report",
        action="store_true",
        help="Generate the evaluation report after the workflow completes.",
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip channel diagnostics step (automatically skipped when no baseline output is available).",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        dest="scenarios",
        help="Scenario identifier to run in addition to the baseline. Can be provided multiple times.",
    )
    return parser.parse_args()


def _load_base_precipitation(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Base precipitation file {path} is empty.")
    time_column = df.columns[0]
    timestamps = pd.to_datetime(df[time_column])
    candidate_cols = [col for col in df.columns if col != time_column]
    if not candidate_cols:
        candidate_cols = list(df.columns)
    numeric_values = df[candidate_cols].apply(pd.to_numeric, errors="coerce").dropna(
        how="all",
        axis=1,
    )
    if numeric_values.empty:
        raise ValueError(f"Failed to identify numeric precipitation columns in {path}.")
    values = numeric_values.astype(float)
    values.index = timestamps
    base_series = values.mean(axis=1)
    base_series.name = values.columns[0]
    return base_series


def _load_station_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Station series file {path} is empty.")
    time_column = df.columns[0]
    timestamps = pd.to_datetime(df[time_column])
    numeric_values = (
        df.drop(columns=[time_column], errors="ignore")
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all", axis=1)
    )
    if numeric_values.empty:
        raise ValueError(f"No numeric station columns detected in {path}.")
    station_df = numeric_values.astype(float)
    station_df.index = timestamps
    station_df.index.name = "Timestamp"
    return station_df


def _load_thiessen_polygons(path: Path) -> Dict[str, BaseGeometry]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    features = data.get("features", [])
    polygons: Dict[str, BaseGeometry] = {}
    for feature in features:
        geometry = feature.get("geometry")
        props = feature.get("properties", {}) or {}
        station_id = props.get("station_id") or props.get("id")
        if geometry and station_id:
            polygons[str(station_id)] = shape(geometry)
    return polygons


def _extract_parameter_mappings(
    partition_outputs: PartitionOutputs,
) -> Tuple[Dict[str, BaseGeometry], Dict[str, str], Dict[str, float]]:
    features = partition_outputs.subzone_features.get("features", [])
    geometries: Dict[str, BaseGeometry] = {}
    for feature in features:
        geometry = feature.get("geometry")
        props = feature.get("properties", {}) or {}
        subzone_id = props.get("id") or props.get("subzone_id")
        if geometry and subzone_id:
            geometries[str(subzone_id)] = shape(geometry)
    parameter_to_zone: Dict[str, str] = {}
    parameter_areas: Dict[str, float] = {}
    for row in partition_outputs.subzone_table:
        subzone_id = str(row["subzone_id"])
        parameter_to_zone[subzone_id] = subzone_id
        parameter_areas[subzone_id] = float(row.get("area_km2", 0.0))
    return geometries, parameter_to_zone, parameter_areas


def _apply_precomputed_subbasins(
    project: HydroProjectConfig,
    outputs: PartitionOutputs,
) -> HydroProjectConfig:
    if not outputs.subzone_summaries:
        return project

    channel_by_subzone: Dict[str, ChannelSummary] = {
        summary.subzone_id: summary for summary in outputs.channel_summaries
    }
    zone_to_subzones: Dict[str, List[str]] = defaultdict(list)
    zone_outlets: Dict[str, str] = {}
    precomputed: List[Dict[str, object]] = []

    for summary in outputs.subzone_summaries:
        entry: Dict[str, object] = {
            "id": summary.subzone_id,
            "area_km2": summary.area_km2,
            "downstream": summary.downstream_subzone_id,
            "parameters": {},
        }
        channel = channel_by_subzone.get(summary.subzone_id)
        if channel is not None:
            entry.update(
                {
                    "channel_id": channel.segment_id,
                    "channel_length_m": channel.length_m,
                    "channel_slope": channel.slope,
                    "channel_drop_m": channel.drop_m,
                }
            )
        zone_to_subzones[summary.zone_id].append(summary.subzone_id)
        precomputed.append(entry)

    for summary in outputs.subzone_summaries:
        downstream = summary.downstream_subzone_id
        if (
            downstream is None
            or not downstream.startswith(summary.zone_id)
            or downstream not in zone_to_subzones[summary.zone_id]
        ):
            zone_outlets.setdefault(summary.zone_id, summary.subzone_id)

    for zone_id, subzones in zone_to_subzones.items():
        if zone_id not in zone_outlets and subzones:
            zone_outlets[zone_id] = subzones[0]

    delineation_cfg = replace(
        project.delineation,
        precomputed_subbasins=precomputed,
    )
    delineation_cfg._channel_network = outputs.channel_network  # type: ignore[attr-defined]

    updated_zones = []
    for cfg in outputs.parameter_zones:
        members = zone_to_subzones.get(cfg.id, list(cfg.explicit_subbasins or []))
        control_point = zone_outlets.get(cfg.id)
        control_points = [control_point] if control_point else list(cfg.control_points)
        updated_zones.append(
            type(cfg)(
                id=cfg.id,
                description=cfg.description,
                control_points=control_points,
                parameters=dict(cfg.parameters),
                explicit_subbasins=members or None,
            )
        )
    outputs.parameter_zones = updated_zones
    for zone_id, definition in outputs.zone_definitions.items():
        if zone_id in zone_to_subzones:
            definition["subbasins"] = sorted(zone_to_subzones[zone_id])
        if zone_id in zone_outlets:
            definition["control_points"] = [zone_outlets[zone_id]]

    return replace(project, delineation=delineation_cfg)


def _write_precipitation_outputs(
    station_series: pd.DataFrame,
    parameter_series: pd.DataFrame,
    subbasin_series: pd.DataFrame,
    station_weights: Mapping[str, Mapping[str, float]],
    intermediate_dir: Path,
) -> Dict[str, Path]:
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    gauges_path = intermediate_dir / "rain_gauge_forcing.csv"
    parameter_path = intermediate_dir / "parameter_subbasin_areal_precipitation.csv"
    hydro_path = intermediate_dir / "subbasin_areal_precipitation.csv"
    weights_path = intermediate_dir / "rain_gauge_weights.json"

    station_series.to_csv(gauges_path)
    parameter_series.to_csv(parameter_path)
    subbasin_series.to_csv(hydro_path)
    weights_payload = {
        sub_id: {sid: float(weight) for sid, weight in mapping.items()}
        for sub_id, mapping in station_weights.items()
    }
    weights_path.write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")
    return {
        "gauges": gauges_path,
        "subbasin": parameter_path,
        "hydrologic": hydro_path,
        "weights": weights_path,
    }


def _basin_average_precipitation(
    subbasin_series: pd.DataFrame,
    subbasins: Sequence[Subbasin],
) -> pd.Series:
    area_lookup = {sub.id: float(sub.area_km2) for sub in subbasins}
    total_area = sum(area_lookup.values())
    if total_area <= 0.0:
        raise ValueError("Total basin area must be positive to compute weighted rainfall.")
    accumulator = None
    for sub_id, area in area_lookup.items():
        if sub_id not in subbasin_series.columns:
            raise KeyError(f"Missing precipitation series for subbasin {sub_id}.")
        contribution = subbasin_series[sub_id] * area
        accumulator = contribution if accumulator is None else accumulator + contribution
    weighted = accumulator / total_area
    weighted.name = weighted.name or "precipitation_mm_per_hr"
    return weighted


def _build_forcing(
    subbasin_series: pd.DataFrame,
    subbasins: Sequence[Subbasin],
) -> Dict[str, List[float]]:
    forcing: Dict[str, List[float]] = {}
    missing: List[str] = []
    for sub in subbasins:
        if sub.id not in subbasin_series.columns:
            missing.append(sub.id)
            continue
        forcing[sub.id] = subbasin_series[sub.id].astype(float).tolist()
    if missing:
        raise KeyError(f"Missing precipitation series for hydrologic subbasins: {', '.join(missing)}")
    return forcing


def main() -> None:
    args = _parse_args()
    project = HydroProjectConfig.from_yaml(args.config)

    pour_points = dutils.read_pour_points_geojson(project.delineation.pour_points_path)

    intermediate_root = (
        project.delineation.intermediate_directory.parent
        if project.delineation.intermediate_directory is not None
        else project.io.results_directory.parent
    )

    delineation_stage = run_delineation_stage(
        project.delineation,
        project.partition,
        pour_points,
        output_dir=intermediate_root,
        auto_threshold=not args.disable_auto_threshold,
        channel_threshold=args.channel_threshold,
        model_structure=project.model,
        outputs_cfg=project.outputs,
    )

    if delineation_stage.partition_outputs is None:
        raise RuntimeError("Partition outputs were not generated by the delineation stage.")

    partition_outputs = delineation_stage.partition_outputs
    delineation_cfg = replace(
        project.delineation,
        pour_points_path=delineation_stage.pour_points_path,
        accumulation_threshold=delineation_stage.accumulation_threshold,
    )
    partition_cfg = replace(
        project.partition,
        pour_points_path=delineation_stage.pour_points_path,
    )
    project = replace(project, delineation=delineation_cfg, partition=partition_cfg)
    project = _apply_precomputed_subbasins(project, partition_outputs)

    parameter_geometries, parameter_to_zone, parameter_areas = _extract_parameter_mappings(partition_outputs)

    base_precip_path = args.base_precipitation or project.io.precipitation
    base_precip_series = _load_base_precipitation(base_precip_path)

    station_series: Optional[pd.DataFrame] = None
    thiessen_polygons: Optional[Mapping[str, BaseGeometry]] = None
    if args.station_series:
        station_series = _load_station_series(args.station_series)
        if not args.thiessen_polygons:
            raise ValueError("--thiessen-polygons must be supplied alongside --station-series.")
        thiessen_polygons = _load_thiessen_polygons(args.thiessen_polygons)

    rainfall_options: Dict[str, object] = {}
    if args.station_count is not None:
        rainfall_options["station_count"] = args.station_count
    if args.rng_seed is not None:
        rainfall_options["rng_seed"] = args.rng_seed

    hydrologic_subbasins = project.delineation.to_subbasins()

    precipitation_stage = generate_precipitation_for_parameters(
        base_precipitation=base_precip_series,
        parameter_geometries=parameter_geometries,
        parameter_to_zone=parameter_to_zone,
        parameter_areas=parameter_areas,
        subbasins=hydrologic_subbasins,
        rainfall_options=rainfall_options or None,
        station_series=station_series,
        thiessen_polygons=thiessen_polygons,
    )

    intermediate_dir = delineation_stage.intermediate_dir
    if precipitation_stage.rain_inputs is not None:
        precipitation_stage.rain_inputs.write(
            intermediate_dir,
            gauges_filename="rain_gauge_forcing.csv",
            subbasin_filename="parameter_subbasin_areal_precipitation.csv",
            stations_geojson="rain_gauge_locations.geojson",
            thiessen_geojson="rain_gauge_thiessen_polygons.geojson",
            weights_json="rain_gauge_weights.json",
        )
        precipitation_stage.subbasin_series.to_csv(
            intermediate_dir / "subbasin_areal_precipitation.csv",
        )
    else:
        _write_precipitation_outputs(
            precipitation_stage.station_series,
            precipitation_stage.parameter_series,
            precipitation_stage.subbasin_series,
            precipitation_stage.station_weights,
            intermediate_dir,
        )

    basin_series = _basin_average_precipitation(
        precipitation_stage.subbasin_series,
        hydrologic_subbasins,
    )
    basin_precip_path = intermediate_dir / "storm_forcing.csv"
    basin_series.to_frame(name=basin_series.name or "precipitation_mm_per_hr").to_csv(basin_precip_path)

    forcing = _build_forcing(
        precipitation_stage.subbasin_series,
        hydrologic_subbasins,
    )
    if forcing:
        sample_id = next(iter(forcing))
        print(f"Forcing sample '{sample_id}' length: {len(forcing[sample_id])}")
    else:
        print("Warning: forcing dictionary is empty.")

    model_config = project.build_model_config(partition_outputs.parameter_zones)
    scenario_ids = args.scenarios
    workflow_result = run_workflow(
        model_config,
        forcing,
        observations=None,
        scenario_ids=scenario_ids,
        persist_outputs=True,
        generate_report=args.persist_report,
    )

    parameter_dir = project.delineation.parameter_directory or (intermediate_dir.parent / "parameters")

    aggregated_series = workflow_result.baseline.aggregated
    first_series = next(iter(aggregated_series.values()), [])
    timestep_count = len(first_series)
    zone_count = len(workflow_result.baseline.zone_discharge)

    print(f"Delineation threshold: {delineation_stage.accumulation_threshold:.0f}")
    print(f"Generated {len(partition_outputs.parameter_zones)} parameter zones.")
    print(f"Simulated {len(aggregated_series)} hydrologic subbasins with {timestep_count} timesteps.")
    print(f"Parameter zone controllers simulated: {zone_count}")

    diagnostics = None
    should_run_diagnostics = (
        not args.skip_diagnostics
        and aggregated_series
        and any(len(series) > 0 for series in aggregated_series.values())
    )
    if should_run_diagnostics:
        diagnostics = run_channel_diagnostics(
            parameter_dir=parameter_dir,
            baseline_dir=project.io.results_directory / "baseline",
            intermediate_dir=intermediate_dir,
            local_dir=project.io.results_directory / "baseline_local",
            precipitation_path=basin_precip_path,
        )
        print("Diagnostic artefacts:")
        print(f"  Flow timeseries: {diagnostics.timeseries_path}")
        print(f"  Flow comparison: {diagnostics.comparison_path}")
        print(f"  Runoff coefficients: {diagnostics.runoff_coefficients_path}")
    else:
        reason = "baseline discharge is empty" if not aggregated_series else "diagnostics explicitly skipped"
        print(f"Channel diagnostics skipped ({reason}).")

    print(f"Results stored in {project.io.results_directory}")


if __name__ == "__main__":
    main()
