"""Reusable workflow stages for delineation and precipitation processing."""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import rasterio
from shapely.geometry.base import BaseGeometry

from hydrosis.config import (
    DelineationConfig,
    ModelStructureConfig,
    OutputArtifactsConfig,
    ParameterPartitionConfig,
)
from hydrosis.delineation import utils as dutils
from hydrosis.parameters.partition import PartitionOutputs, partition_parameter_zones
from hydrosis.precipitation import (
    RainGaugeInputs,
    compute_subbasin_station_weights,
    generate_rain_gauge_inputs,
    interpolate_station_series,
)


@dataclass
class DelineationStageResult:
    """Outputs from the delineation and partition workflow."""

    subbasins: Sequence[dutils.Subbasin]
    partition_outputs: Optional[PartitionOutputs]
    accumulation_threshold: float
    pour_points_path: Path
    intermediate_dir: Path


def snap_pour_points_to_flow_cells(
    pour_points: Sequence[dutils.PourPoint],
    flow_dir_path: Path,
    flow_acc_path: Path,
    max_radius: int = 30,
    valid_directions: Optional[Iterable[int]] = None,
) -> List[dutils.PourPoint]:
    """Snap automatically derived pour points onto valid flow direction cells."""

    flowdir, upstream, shape_dims = dutils.build_flow_network(flow_dir_path)
    rows, cols = shape_dims
    with rasterio.open(flow_acc_path) as acc_ds:
        accumulation = acc_ds.read(1)
        transform = acc_ds.transform

    directions = set(valid_directions or DelineationConfig._direction_mapping().keys())

    def drains_to_target(row: int, col: int, target: Tuple[int, int]) -> bool:
        d8 = dutils.D8_OFFSETS
        visited: set[Tuple[int, int]] = set()
        r, c = row, col
        while (r, c) != target:
            if (r, c) in visited:
                return False
            visited.add((r, c))
            code = int(flowdir[r, c])
            offset = d8.get(code)
            if offset is None:
                return False
            r += offset[0]
            c += offset[1]
            if not (0 <= r < rows and 0 <= c < cols):
                return False
        return True

    adjusted: List[dutils.PourPoint] = []
    for point in pour_points:
        base_row, base_col = point.row, point.col
        best_row, best_col = base_row, base_col
        base_acc = (
            float(accumulation[base_row, base_col])
            if np.isfinite(accumulation[base_row, base_col])
            else 0.0
        )
        best_acc = -np.inf
        best_dist_sq = float("inf")

        if int(flowdir[base_row, base_col]) not in directions:
            for radius in range(1, max_radius + 1):
                rmin = max(0, base_row - radius)
                rmax = min(rows, base_row + radius + 1)
                cmin = max(0, base_col - radius)
                cmax = min(cols, base_col + radius + 1)
                for r in range(rmin, rmax):
                    for c in range(cmin, cmax):
                        code = int(flowdir[r, c])
                        if code not in directions:
                            continue
                        acc_val = float(accumulation[r, c])
                        if not np.isfinite(acc_val):
                            continue
                        dist_sq = (r - base_row) ** 2 + (c - base_col) ** 2
                        if acc_val > best_acc or (
                            np.isclose(acc_val, best_acc) and dist_sq < best_dist_sq
                        ):
                            best_row, best_col = r, c
                            best_acc = acc_val
                            best_dist_sq = dist_sq
            if not np.isfinite(best_acc) or best_acc == -np.inf:
                best_row, best_col = base_row, base_col
                best_acc = base_acc
        else:
            best_acc = base_acc
            best_dist_sq = 0.0

        world_x, world_y = rasterio.transform.xy(transform, best_row, best_col, offset="center")
        adjusted.append(
            dutils.PourPoint(
                id=point.id,
                row=best_row,
                col=best_col,
                x=world_x,
                y=world_y,
                accumulation=best_acc,
            )
        )
    return adjusted


def generate_hierarchical_pour_points(
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
) -> List[dutils.PourPoint]:
    """Generate pour points that form a hierarchical drainage tree within one basin."""

    return dutils.generate_tree_pour_points(
        flow_acc_path,
        flow_dir_path,
        count=count,
        accumulation_threshold=accumulation_threshold,
        min_distance_cells=min_distance_cells,
        max_children=max_children,
        output_geojson=output_geojson,
        accumulation_plot=accumulation_plot,
        copy_to=copy_to,
    )


def suggest_accumulation_threshold(
    pour_points: Sequence[dutils.PourPoint],
    default: float = 5000.0,
    ratio: float = 0.5,
) -> float:
    """Suggest a channel accumulation threshold based on pour point accumulations."""

    if not pour_points:
        return default
    accumulations = [float(point.accumulation or 0.0) for point in pour_points]
    if not accumulations:
        return default
    candidate = float(np.quantile(accumulations, ratio))
    return candidate if candidate > 0 else default


def run_delineation_stage(
    delineation_config: DelineationConfig,
    partition_cfg: ParameterPartitionConfig,
    pour_points: Sequence[dutils.PourPoint],
    output_dir: Path,
    *,
    auto_threshold: bool = True,
    channel_threshold: Optional[float] = None,
    model_structure: Optional[ModelStructureConfig] = None,
    outputs_cfg: Optional[OutputArtifactsConfig] = None,
) -> DelineationStageResult:
    """Execute delineation and parameter partitioning."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if auto_threshold:
        threshold = suggest_accumulation_threshold(pour_points, default=float(delineation_config.channel_threshold or 0.0) or 5000.0)
    else:
        threshold = float(channel_threshold or delineation_config.channel_threshold or 5000.0)

    pour_points_path = output_dir / "pour_points.geojson"
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [p.x, p.y]},
                "properties": {
                    "id": p.id,
                    "row": p.row,
                    "col": p.col,
                    "accumulation": p.accumulation,
                },
            }
            for p in pour_points
        ],
    }
    pour_points_path.write_text(json.dumps(feature_collection, indent=2), encoding="utf-8")

    delineation_config = replace(
        delineation_config,
        pour_points_path=pour_points_path,
        accumulation_threshold=threshold,
    )
    partition_cfg = replace(partition_cfg, pour_points_path=pour_points_path)

    subbasins = delineation_config.to_subbasins()
    partition_outputs: Optional[PartitionOutputs] = None
    if model_structure is not None and outputs_cfg is not None:
        partition_outputs = partition_parameter_zones(
            delineation_cfg=delineation_config,
            partition_cfg=partition_cfg,
            model_structure=model_structure,
            outputs_cfg=outputs_cfg,
        )

    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    return DelineationStageResult(
        subbasins=subbasins,
        partition_outputs=partition_outputs,
        accumulation_threshold=threshold,
        pour_points_path=pour_points_path,
        intermediate_dir=intermediate_dir,
    )


@dataclass
class PrecipitationStageResult:
    """Outputs from the precipitation generation/interpolation stage."""

    station_series: pd.DataFrame
    parameter_series: pd.DataFrame
    subbasin_series: pd.DataFrame
    station_weights: Dict[str, Dict[str, float]]
    rain_inputs: Optional[RainGaugeInputs]


def generate_precipitation_for_parameters(
    base_precipitation: pd.Series,
    parameter_geometries: Mapping[str, BaseGeometry],
    parameter_to_zone: Mapping[str, str],
    parameter_areas: Mapping[str, float],
    subbasins: Sequence[dutils.Subbasin],
    rainfall_options: Optional[Mapping[str, object]] = None,
    *,
    station_series: Optional[pd.DataFrame] = None,
    thiessen_polygons: Optional[Mapping[str, BaseGeometry]] = None,
) -> PrecipitationStageResult:
    """Generate or interpolate precipitation time-series for parameter subbasins."""

    rainfall_options = dict(rainfall_options or {})
    station_count = int(rainfall_options.pop("station_count", 10))
    rng_seed = rainfall_options.pop("rng_seed", None)

    rain_inputs: Optional[RainGaugeInputs]
    if station_series is not None:
        # Build RainGaugeInputs clone to reuse downstream utilities
        if thiessen_polygons is None:
            raise ValueError("Thiessen polygons must be provided when station_series is supplied.")
        station_weights = compute_subbasin_station_weights(parameter_geometries, thiessen_polygons)
        parameter_series = interpolate_station_series(station_series, station_weights)
        rain_inputs = None
    else:
        rain_inputs = generate_rain_gauge_inputs(
            base_precipitation,
            parameter_geometries,
            station_count=station_count,
            rng_seed=rng_seed,
            **rainfall_options,
        )
        station_weights = rain_inputs.station_weights
        station_series = rain_inputs.station_series
        parameter_series = rain_inputs.subbasin_series

    parameter_series = parameter_series.reindex(columns=sorted(parameter_geometries.keys()))
    parameter_series.index.name = "Timestamp"

    subbasin_precip = aggregate_parameter_precipitation(
        parameter_series,
        parameter_to_zone,
        parameter_areas,
        subbasins,
    )

    return PrecipitationStageResult(
        station_series=station_series,
        parameter_series=parameter_series,
        subbasin_series=subbasin_precip,
        station_weights={sub_id: {sid: float(w) for sid, w in mapping.items()} for sub_id, mapping in station_weights.items()},
        rain_inputs=rain_inputs,
    )


def aggregate_parameter_precipitation(
    parameter_series: pd.DataFrame,
    parameter_to_zone: Mapping[str, str],
    parameter_areas: Mapping[str, float],
    subbasins: Sequence[dutils.Subbasin],
) -> pd.DataFrame:
    """Aggregate parameter (subzone-level) precipitation to hydrologic subbasins."""

    aggregated_columns: Dict[str, np.ndarray] = {}
    base_index = parameter_series.index
    for sub in subbasins:
        parent_id = sub.id
        member_ids = [pid for pid, zone in parameter_to_zone.items() if zone == parent_id]
        available_ids = [pid for pid in member_ids if pid in parameter_series.columns]
        if not available_ids:
            raise KeyError(f"No parameter subbasins with data mapped to hydrologic subbasin {parent_id}")
        areas = np.asarray([parameter_areas.get(pid, 0.0) for pid in available_ids], dtype=float)
        if not np.any(areas > 0):
            raise ValueError(f"No positive area information available for subbasin {parent_id}")
        weights = areas / areas.sum()
        subframe = parameter_series[available_ids].to_numpy(dtype=float, copy=False)
        aggregated_columns[parent_id] = subframe @ weights
    subbasin_precip = pd.DataFrame(aggregated_columns, index=base_index)
    subbasin_precip.index.name = parameter_series.index.name
    return subbasin_precip


@dataclass
class ChannelDiagnosticsResult:
    """Paths written by the channel diagnostics stage."""

    timeseries_path: Path
    comparison_path: Path
    runoff_coefficients_path: Path


def run_channel_diagnostics(
    parameter_dir: Path,
    baseline_dir: Path,
    intermediate_dir: Path,
    *,
    local_dir: Optional[Path],
    precipitation_path: Path,
) -> ChannelDiagnosticsResult:
    """Execute channel diagnostics helpers provided by analysis modules."""

    from hydrosis.analysis.channel_flow import compare_channel_flows, compute_channel_flows
    from hydrosis.analysis.runoff_coefficients import compute_zone_runoff_coefficients

    flow_output = compute_channel_flows(
        parameter_dir=parameter_dir,
        baseline_dir=baseline_dir,
        intermediate_dir=intermediate_dir,
        local_dir=local_dir,
    )
    comparison_path = compare_channel_flows(
        parameter_dir=parameter_dir,
        baseline_dir=baseline_dir,
        aggregated_dir=baseline_dir,
        intermediate_dir=intermediate_dir,
        local_dir=local_dir,
    )
    coeff_path = compute_zone_runoff_coefficients(
        parameter_dir=parameter_dir,
        aggregated_dir=baseline_dir,
        local_dir=local_dir,
        precipitation_path=precipitation_path,
        output_path=intermediate_dir / "zone_runoff_coefficients.csv",
    )
    return ChannelDiagnosticsResult(
        timeseries_path=flow_output,
        comparison_path=comparison_path,
        runoff_coefficients_path=coeff_path,
    )


__all__ = [
    "ChannelDiagnosticsResult",
    "DelineationStageResult",
    "PrecipitationStageResult",
    "aggregate_parameter_precipitation",
    "generate_precipitation_for_parameters",
    "run_channel_diagnostics",
    "run_delineation_stage",
    "snap_pour_points_to_flow_cells",
    "suggest_accumulation_threshold",
]
