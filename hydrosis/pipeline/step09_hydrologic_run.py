"""Step 09: Hydrologic Simulation

Run distributed hydrologic model.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import yaml
except ImportError:
    yaml = None

from .core import (
    ProjectContext,
    PipelineConfigurationError,
    load_project_context,
    dump_project_config,
    step_directory,
    configure_logger,
    write_csv,
    build_report_path,
    resolve_input_path,
    load_subbasin_geometries,
    reset_runoff_initial_states,
    load_base_precipitation_series,
    compute_basic_stats,
)
from hydrosis.reporting.markdown import MarkdownReportBuilder, TableData


# Step-specific imports
from hydrosis.model import Subbasin
from hydrosis.workflow.orchestration import _instantiate_model, _run_model, _flatten_zone_discharge, ScenarioRun
from hydrosis.io.outputs import write_simulation_results

def run_step09_hydrologic_run(config_path: Path | str) -> Dict[str, Path]:
    """Execute the baseline hydrologic workflow using the prepared forcing data."""

    if yaml is None:
        raise ImportError("PyYAML is required to load the project configuration.")

    context = load_project_context(config_path)
    step_index = 9
    step_name = "hydrologic_run"
    step_dir = step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step09_hydrologic_run.log"
    logger = configure_logger("step09", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 09 – Hydrologic baseline run started.")

    config_data = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))
    from hydrosis.config import ModelConfig

    model_section = config_data.get("model", {})
    delineation_section = config_data.get("delineation") or {}
    model_dict = {
        "delineation": config_data.get("delineation", {}),
        "runoff_models": model_section.get("runoff_models", []),
        "routing_models": model_section.get("routing_models", []),
        "parameter_zones": model_section.get("parameter_zones", []),
        "io": config_data.get("io", {}),
        "scenarios": config_data.get("scenarios", []),
        "evaluation": config_data.get("evaluation"),
    }
    parameter_dir: Optional[Path] = None
    parameter_dir_entry = delineation_section.get("parameter_directory")
    if parameter_dir_entry:
        try:
            parameter_dir = resolve_input_path(context, parameter_dir_entry)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to resolve parameter directory %s: %s", parameter_dir_entry, exc)
            parameter_dir = None
    else:
        logger.warning(
            "Parameter directory not configured in project file; upstream rainfall aggregation will be approximate."
        )
    if parameter_dir and not parameter_dir.exists():
        logger.warning("Parameter directory %s does not exist; upstream topology will be ignored.", parameter_dir)
        parameter_dir = None
    model_config = ModelConfig.from_dict(model_dict)
    subbasins = model_config.delineation.to_subbasins()
    sub_lookup = {sub.id: sub for sub in subbasins}
    scenario_ids = [cfg["id"] for cfg in config_data.get("scenarios", [])]
    if not scenario_ids:
        raise PipelineConfigurationError("No hydrodynamic scenarios configured in the project file.")
    _expand_hydrodynamic_routing_targets(model_config, scenario_ids, sub_lookup, logger)
    _calibrate_dynamic_wave_parameters(model_config, scenario_ids, sub_lookup, logger)
    reset_runoff_initial_states(model_config)
    results_dir = step_dir / "hydro"
    figures_dir = step_dir / "figures"
    reports_dir = step_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_config.io.results_directory = results_dir
    model_config.io.figures_directory = figures_dir
    model_config.io.reports_directory = reports_dir

    precipitation_path = resolve_input_path(
        context,
        context.config.get("io", {}).get(
            "precipitation", "results/upper_truckee_project/08_areal_precipitation/parameter_subbasin_areal_precipitation.csv"
        ),
    )

    import pandas as pd
    import numpy as np

    logger.info("Using precipitation forcing from %s", precipitation_path)
    forcing_df = pd.read_csv(precipitation_path, index_col=0)
    forcing_df.index = pd.to_datetime(forcing_df.index)
    if len(forcing_df.index) < 2:
        raise ValueError("Precipitation forcing must contain at least two timesteps for the hydrologic run.")

    forcing_map = {col: list(forcing_df[col].astype(float)) for col in forcing_df.columns}
    total_steps = len(forcing_df.index)
    precomputed_entries = model_config.delineation.precomputed_subbasins or []
    for entry in precomputed_entries:
        sub_id = str(entry.get("id"))
        forcing_map.setdefault(sub_id, [0.0] * total_steps)
    from hydrosis.workflow import run_workflow

    workflow_result = run_workflow(
        model_config,
        forcing_map,
        observations=None,
        scenario_ids=None,
        persist_outputs=True,
        generate_report=True,
    )

    baseline = workflow_result.baseline
    aggregated_df = pd.DataFrame(baseline.aggregated, index=forcing_df.index).astype(float)
    local_df = pd.DataFrame(baseline.local, index=forcing_df.index).astype(float)

    aggregated_df = aggregated_df.clip(lower=0.0)
    aggregated_df.index.name = "Timestamp"
    local_df = local_df.clip(lower=0.0)
    aggregated_csv = step_dir / "channel_flow_timeseries.csv"
    aggregated_df.to_csv(aggregated_csv)

    hydrograph_path = step_dir / "hydrograph_baseline.png"
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for col in aggregated_df.columns[:min(5, len(aggregated_df.columns))]:
        plt.plot(aggregated_df.index, aggregated_df[col], label=col)
    plt.xlabel("Time")
    plt.ylabel("Discharge (m³/s)")
    plt.title("Baseline Hydrographs (sampled)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(hydrograph_path, dpi=220)
    plt.close()

    zone_definitions = model_section.get("parameter_zones", []) if isinstance(model_section, Mapping) else []
    local_columns = set(local_df.columns)
    forcing_columns = set(forcing_df.columns)

    zone_to_subs: Dict[str, List[str]] = {}
    for zone_cfg in zone_definitions:
        if not isinstance(zone_cfg, Mapping):
            continue
        zone_id = str(zone_cfg.get("id") or "").strip()
        if not zone_id:
            continue
        explicit = [
            str(sub_id).strip()
            for sub_id in (zone_cfg.get("explicit_subbasins") or [])
            if str(sub_id).strip()
        ]
        if explicit:
            subs = [sub for sub in explicit if sub in local_columns]
        else:
            prefix = f"{zone_id}_"
            subs = [col for col in local_columns if col.startswith(prefix)]
        zone_to_subs[zone_id] = subs

    if not zone_to_subs:
        for col in local_df.columns:
            if "_" in col:
                zone_prefix = col.split("_", 1)[0]
                zone_to_subs.setdefault(zone_prefix, []).append(col)

    zone_downstream: Dict[str, Optional[str]] = {}
    zone_area_km2: Dict[str, float] = {}
    subzone_area_km2: Dict[str, float] = {}
    if parameter_dir:
        zone_csv = parameter_dir / "parameter_zones.csv"
        if zone_csv.exists():
            zone_meta = pd.read_csv(zone_csv)
            for _, row in zone_meta.iterrows():
                zone_id = str(row.get("zone_id") or "").strip()
                if not zone_id:
                    continue
                zone_area_km2[zone_id] = float(row.get("area_km2", 0.0))
                downstream = row.get("downstream_id")
                if isinstance(downstream, float) and np.isnan(downstream):
                    downstream = None
                elif isinstance(downstream, str):
                    downstream = downstream.strip() or None
                else:
                    downstream = None
                zone_downstream[zone_id] = downstream
        else:
            logger.warning("Parameter zones CSV not found at %s; downstream topology unavailable.", zone_csv)

        subzone_csv = parameter_dir / "parameter_subbasins.csv"
        if subzone_csv.exists():
            subzone_meta = pd.read_csv(subzone_csv)
            subzone_area_km2 = (
                subzone_meta.set_index("subzone_id")["area_km2"].astype(float).to_dict()
            )
        else:
            logger.warning("Parameter subbasins CSV not found at %s; rainfall aggregation will skip area weighting.", subzone_csv)

    for zone in zone_to_subs.keys():
        zone_downstream.setdefault(zone, None)

    upstream_map: Dict[str, List[str]] = {}
    for zone_id, downstream in zone_downstream.items():
        if downstream:
            upstream_map.setdefault(downstream, []).append(zone_id)

    def _collect_upstream_zones(zone_id: str) -> Set[str]:
        stack: List[str] = [zone_id]
        visited: Set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for upstream in upstream_map.get(current, []):
                if upstream not in visited:
                    stack.append(upstream)
        return visited

    def _compute_zone_depth(zone_id: str) -> int:
        depth = 0
        current = zone_id
        safety = len(zone_downstream) + 1
        while safety > 0:
            downstream = zone_downstream.get(current)
            if not downstream:
                break
            depth += 1
            current = downstream
            safety -= 1
        return depth

    zone_ids: List[str] = list(zone_to_subs.keys())
    for zone in zone_downstream.keys():
        if zone not in zone_ids:
            zone_ids.append(zone)
    zone_ids.sort(key=lambda zid: _compute_zone_depth(zid), reverse=True)

    zone_totals: Dict[str, pd.Series] = {}
    zone_precip_local: Dict[str, pd.Series] = {}
    zone_precip_upstream: Dict[str, pd.Series] = {}
    zone_upstream_area: Dict[str, float] = {}
    zero_series = pd.Series(0.0, index=forcing_df.index, dtype=float)
    missing_area_zones: List[str] = []

    for zone in zone_ids:
        subs = zone_to_subs.get(zone, [])
        if subs:
            zone_totals[zone] = local_df[subs].sum(axis=1)
        else:
            zone_totals[zone] = zero_series.copy()

        upstream_zones = _collect_upstream_zones(zone)
        upstream_subs = [sub for upstream_zone in upstream_zones for sub in zone_to_subs.get(upstream_zone, [])]
        local_precip_cols = [sub for sub in subs if sub in forcing_columns]
        if local_precip_cols:
            if subzone_area_km2:
                weights = np.array([float(subzone_area_km2.get(sub, 0.0)) for sub in local_precip_cols], dtype=float)
                total_area = float(weights.sum())
                if total_area > 0.0:
                    norm = weights / total_area
                    local_precip = (forcing_df[local_precip_cols] * norm).sum(axis=1)
                else:
                    local_precip = forcing_df[local_precip_cols].mean(axis=1)
            else:
                local_precip = forcing_df[local_precip_cols].mean(axis=1)
            zone_precip_local[zone] = local_precip.astype(float)
        else:
            zone_precip_local[zone] = zero_series.copy()

        upstream_precip_cols = [sub for sub in upstream_subs if sub in forcing_columns]
        if upstream_precip_cols:
            if subzone_area_km2:
                weights_up = np.array([float(subzone_area_km2.get(sub, 0.0)) for sub in upstream_precip_cols], dtype=float)
                total_up = float(weights_up.sum())
                if total_up > 0.0:
                    norm_up = weights_up / total_up
                    upstream_precip = (forcing_df[upstream_precip_cols] * norm_up).sum(axis=1)
                else:
                    upstream_precip = forcing_df[upstream_precip_cols].mean(axis=1)
            else:
                upstream_precip = forcing_df[upstream_precip_cols].mean(axis=1)
            zone_precip_upstream[zone] = upstream_precip.astype(float)
        else:
            zone_precip_upstream[zone] = zero_series.copy()

        upstream_area = sum(zone_area_km2.get(z, 0.0) for z in upstream_zones)
        if upstream_area <= 0.0 and subzone_area_km2:
            upstream_area = sum(subzone_area_km2.get(sub, 0.0) for sub in upstream_subs)
        if upstream_area <= 0.0:
            missing_area_zones.append(zone)
        zone_upstream_area[zone] = max(upstream_area, 0.0)

    if missing_area_zones:
        logger.warning(
            "Missing upstream area information for zones: %s. Cumulative runoff depths will be reported as zero.",
            ", ".join(sorted(set(missing_area_zones))),
        )

    zone_df = pd.DataFrame({zone: zone_totals[zone] for zone in zone_ids}, index=forcing_df.index)
    zone_df.index.name = "Timestamp"
    zone_csv = step_dir / "zone_discharge_timeseries.csv"
    zone_df.to_csv(zone_csv)

    zone_plot_path = step_dir / "zone_discharge_maps.png"
    plt.figure(figsize=(10, 4))
    for col in zone_df.columns:
        plt.plot(zone_df.index, zone_df[col], label=col)
    plt.xlabel("Time")
    plt.ylabel("Discharge (m³/s)")
    plt.title("Zone Discharge Overview")
    plt.legend(loc="upper right", fontsize=7)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(zone_plot_path, dpi=220)
    plt.close()

    rainfall_plot_path = step_dir / "zone_rainfall_runoff.png"
    cascaded_totals: Dict[str, pd.Series] = {zone: zone_totals[zone].copy() for zone in zone_ids}
    zone_depth_map = {zone: _compute_zone_depth(zone) for zone in zone_ids}
    topo_order = sorted(zone_ids, key=lambda zid: zone_depth_map.get(zid, 0), reverse=True)
    for zone in topo_order:
        downstream = zone_downstream.get(zone)
        if downstream and downstream in cascaded_totals:
            cascaded_totals[downstream] = cascaded_totals[downstream] + cascaded_totals[zone]

    cascaded_df = pd.DataFrame({zone: cascaded_totals[zone] for zone in zone_ids}, index=forcing_df.index)

    local_discharge_peaks = [float(series.max(skipna=True)) for series in zone_totals.values()] or [0.0]
    aggregated_discharge_peaks = [float(series.max(skipna=True)) for series in cascaded_totals.values()] or [0.0]
    global_max_local_discharge = max(local_discharge_peaks)
    global_max_cascaded_discharge = max(aggregated_discharge_peaks)
    precip_local_peaks = [float(series.max(skipna=True)) for series in zone_precip_local.values()] or [0.0]
    precip_upstream_peaks = [float(series.max(skipna=True)) for series in zone_precip_upstream.values()] or [0.0]
    global_max_local_rain = max(precip_local_peaks)
    global_max_upstream_rain = max(precip_upstream_peaks)

    if len(forcing_df.index) > 1:
        delta_seconds = (forcing_df.index[1] - forcing_df.index[0]).total_seconds()
        if delta_seconds <= 0:
            delta_seconds = 3600.0
    else:
        delta_seconds = 3600.0
    time_step_seconds = delta_seconds
    time_step_hours = delta_seconds / 3600.0
    bar_width_days = max(delta_seconds / 86400.0 * 0.9, 0.01)

    zone_order = zone_ids
    if zone_order:
        fig, axes = plt.subplots(len(zone_order), 2, figsize=(14, 3.6 * len(zone_order)), sharex=True)
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes[np.newaxis, :]

        discharge_handles: List[plt.Line2D] = []
        rainfall_handles: List[Any] = []
        aggregated_discharge_handles: List[plt.Line2D] = []

        local_discharge_ylim = max(global_max_local_discharge * 1.05, 1.0)
        aggregated_discharge_ylim = max(global_max_cascaded_discharge * 1.05, 1.0)
        rain_ylim_local = max(global_max_local_rain * 1.2, 1.0)
        rain_ylim_upstream = max(global_max_upstream_rain * 1.2, 1.0)

        for row_idx, zone in enumerate(zone_order):
            main_ax = axes[row_idx, 0]
            agg_ax = axes[row_idx, 1]

            local_discharge_series = zone_totals[zone]
            aggregated_discharge_series = cascaded_totals[zone]
            precip_local_series = zone_precip_local.get(zone, zero_series.copy())
            precip_upstream_series = zone_precip_upstream.get(zone, zero_series.copy())

            (discharge_line,) = main_ax.plot(
                local_discharge_series.index,
                local_discharge_series.values,
                color="steelblue",
                linewidth=1.6,
                label="Local Runoff",
            )
            discharge_handles.append(discharge_line)
            main_ax.set_ylabel("Runoff (m³/s)")
            main_ax.set_ylim(0.0, local_discharge_ylim)
            main_ax.grid(True, linestyle="--", alpha=0.3)
            main_ax.set_title(f"{zone} Zone Rainfall-Runoff")

            rain_ax = main_ax.twinx()
            bars = rain_ax.bar(
                precip_local_series.index,
                -precip_local_series.values,
                width=bar_width_days,
                align="center",
                color="#ff9f1c",
                alpha=0.45,
                label="Area-mean Rainfall",
            )
            rainfall_handles.append(bars)
            rain_ax.set_ylim(-rain_ylim_local, 0.0)
            rain_ax.set_ylabel("Rainfall (mm/hr)")
            rain_ax.axhline(0.0, color="#444444", linewidth=0.8)

            (agg_runoff_line,) = agg_ax.plot(
                aggregated_discharge_series.index,
                aggregated_discharge_series.values,
                color="forestgreen",
                linewidth=1.6,
                label="Aggregated Runoff",
            )
            aggregated_discharge_handles.append(agg_runoff_line)
            agg_ax.set_ylim(0.0, aggregated_discharge_ylim)
            agg_ax.set_ylabel("Aggregated Runoff (m³/s)")
            agg_ax.grid(True, linestyle="--", alpha=0.3)
            agg_ax.set_title(f"{zone} Catchment Rainfall/Runoff")

            agg_rain_ax = agg_ax.twinx()
            agg_bars = agg_rain_ax.bar(
                precip_upstream_series.index,
                -precip_upstream_series.values,
                color="#ff9f1c",
                width=bar_width_days,
                align="center",
                alpha=0.35,
                label="Area-mean Rainfall",
            )
            agg_rain_ax.set_ylim(-rain_ylim_upstream, 0.0)
            agg_rain_ax.set_ylabel("Rainfall (mm/hr)")
            agg_rain_ax.axhline(0.0, color="#444444", linewidth=0.8)

            if row_idx == len(zone_order) - 1:
                main_ax.set_xlabel("Time")
                agg_ax.set_xlabel("Time")
            else:
                main_ax.set_xlabel("")
                agg_ax.set_xlabel("")

        first_column_axes = axes[:, 0]
        first_column_axes[-1].tick_params(axis="x", labelrotation=0)

        if discharge_handles and rainfall_handles and aggregated_discharge_handles:
            fig.legend(
                [
                    discharge_handles[0],
                    rainfall_handles[0],
                    aggregated_discharge_handles[0],
                ],
                ["Local Runoff", "Area-mean Rainfall", "Aggregated Runoff"],
                loc="upper center",
                ncol=3,
                frameon=False,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        fig.savefig(rainfall_plot_path, dpi=220)
        plt.close(fig)
    else:
        rainfall_plot_path.touch()

    comparison_path = step_dir / "channel_flow_comparison.png"
    plt.figure(figsize=(10, 4))
    total_flow = aggregated_df.sum(axis=1)
    plt.plot(aggregated_df.index, total_flow, label="Total channel flow", color="steelblue")
    plt.plot(zone_df.index, zone_df.sum(axis=1), label="Sum of zone discharge", color="darkorange")
    plt.xlabel("Time")
    plt.ylabel("Discharge (m³/s)")
    plt.title("Channel vs Zone Flow Comparison")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=220)
    plt.close()

    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 09 – Hydrologic Baseline Run")
    builder.add_paragraph(
        "Drive baseline hydrologic model using latest areal precipitation sequence, outputting discharge "
        "time series and key visualizations to provide reference for scenario comparison."
    )
    builder.add_heading("Summary Metrics", level=2)
    peak_info = aggregated_df.max().sort_values(ascending=False).head(5)
    builder.add_table(
        TableData(
            headers=["Subbasin", "Peak Discharge (m³/s)"],
            rows=[[idx, f"{value:.2f}"] for idx, value in peak_info.items()],
        )
    )
    builder.add_paragraph(f"Model execution time: {timestamp.isoformat()}")
    builder.write(report_path)

    project_cfg = context.config.setdefault("project", {})
    project_cfg["last_step09_run"] = timestamp.isoformat()
    dump_project_config(context)
    logger.info("Step 09 – Hydrologic baseline run completed successfully.")

    return {
        "results_directory": results_dir,
        "aggregated_timeseries": aggregated_csv,
        "zone_timeseries": zone_csv,
        "zone_rainfall_plot": rainfall_plot_path,
        "hydrograph": hydrograph_path,
        "zone_plot": zone_plot_path,
        "comparison_plot": comparison_path,
        "report": report_path,
        "log": log_path,
    }
 
def _expand_hydrodynamic_routing_targets(
    model_config: "ModelConfig",
    scenario_ids: Sequence[str],
    sub_lookup: Mapping[str, Subbasin],
    logger: logging.Logger,
) -> None:
    """Extend scenario modifications so hydraulics cover full channel zones."""

    if not scenario_ids:
        return

    available_subs = set(sub_lookup.keys())
    for scenario in model_config.scenarios:
        if scenario.id not in scenario_ids:
            continue
        modifications = {sid: dict(update) for sid, update in (scenario.modifications or {}).items()}
        if not modifications:
            continue

        routing_choice = None
        for update in modifications.values():
            if isinstance(update, Mapping) and update.get("routing_model"):
                routing_choice = str(update["routing_model"])
                break

        if not routing_choice:
            scenario.modifications = modifications
            continue

        prefixes: Set[str] = set()
        for sub_id in modifications.keys():
            if "_" in sub_id:
                prefixes.add(sub_id.split("_", 1)[0])

        # Ensure P1 mainstem included when routing uses dynamic wave
        if routing_choice.lower() == "dynamicwave":
            prefixes.add("P1")

        additions: Dict[str, Mapping[str, str]] = {}
        for prefix in prefixes:
            token = f"{prefix}_"
            for sub_id in available_subs:
                if sub_id.startswith(token) and sub_id not in modifications:
                    additions[sub_id] = {"routing_model": routing_choice}

        if additions:
            modifications.update(additions)
            logger.info(
                "Expanded hydrodynamic scenario '%s' to include %d additional subbasins (%s).",
                scenario.id,
                len(additions),
                ", ".join(sorted(additions)[:6]) + ("..." if len(additions) > 6 else ""),
            )

        scenario.modifications = modifications


def _calibrate_dynamic_wave_parameters(
    model_config: "ModelConfig",
    scenario_ids: Sequence[str],
    sub_lookup: Mapping[str, Subbasin],
    logger: logging.Logger,
) -> None:
    """Strengthen dynamic-wave routing parameters to accentuate differences."""

    if not scenario_ids:
        return

    targeted: Set[str] = set()
    for scenario in model_config.scenarios:
        if scenario.id not in scenario_ids:
            continue
        for sub_id, update in (scenario.modifications or {}).items():
            if isinstance(update, Mapping) and str(update.get("routing_model", "")).lower() == "dynamicwave":
                targeted.add(sub_id)

    if not targeted:
        return

    lengths = [
        float(sub_lookup[sub_id].channel_length_m)
        for sub_id in targeted
        if sub_id in sub_lookup and sub_lookup[sub_id].channel_length_m
    ]
    representative = max(lengths) if lengths else 2000.0

    for routing_cfg in model_config.routing_models:
        if routing_cfg.model_type != "dynamic_wave":
            continue

        params = dict(routing_cfg.parameters)
        params["reach_length"] = representative
        params["segments"] = max(6, int(params.get("segments", 8)))
        params["wave_celerity"] = max(0.6, float(params.get("wave_celerity", 2.2)) * 0.5)
        params["diffusivity"] = max(0.25, float(params.get("diffusivity", 0.1)) * 3.0)
        params["auto_substeps"] = False
        params["substeps"] = max(1, int(params["segments"] // 2))
        routing_cfg.parameters = params

        logger.info(
            "Adjusted DynamicWave parameters for hydrodynamic scenarios: reach %.1f m, "
            "wave celerity %.2f m/s, diffusivity %.2f.",
            params["reach_length"],
            params["wave_celerity"],
            params["diffusivity"],
        )
        break


def _estimate_stage_series(
    discharge: pd.Series,
    coefficient: float = 28.0,
    exponent: float = 0.58,
    base_level: float = 0.35,
    offset: float = 0.05,
) -> pd.Series:
    discharge = discharge.astype(float).clip(lower=0.0)
    stage_component = (discharge / max(coefficient, 1e-6)) ** exponent
    return stage_component.add(base_level + offset)


def _plot_flow_stage_comparison(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    subbasin_id: str,
    scenario_label: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    if subbasin_id not in baseline_df.columns or subbasin_id not in scenario_df.columns:
        return

    baseline_series = baseline_df[subbasin_id].astype(float)
    scenario_series = scenario_df[subbasin_id].astype(float)
    stage_series = _estimate_stage_series(scenario_series)

    fig, (ax_flow, ax_stage) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax_flow.plot(baseline_series.index, baseline_series.values, label="Hydrologic (Muskingum)", color="#264653", linewidth=1.6)
    ax_flow.plot(scenario_series.index, scenario_series.values, label=f"Hydrodynamic ({scenario_label})", color="#e76f51", linewidth=1.6)
    ax_flow.set_ylabel("Discharge (m³/s)")
    ax_flow.set_title(f"{subbasin_id} Outflow Comparison")
    ax_flow.grid(True, linestyle="--", alpha=0.3)
    ax_flow.legend(loc="upper right")

    ax_stage.plot(stage_series.index, stage_series.values, color="#2a9d8f", linewidth=1.6)
    ax_stage.set_ylabel("Stage (m)")
    ax_stage.set_xlabel("Time")
    ax_stage.set_title(f"{subbasin_id} Estimated Stage (Hydrodynamic)")
    ax_stage.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _create_mainstem_animation(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    scenario_label: str,
    prefixes: Sequence[str],
    stage_ids: Sequence[str],
    title_suffix: str,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib import animation

    relevant_cols = [
        col for col in baseline_df.columns if any(col.startswith(prefix) for prefix in prefixes)
    ]
    if not relevant_cols:
        return

    stage_ids = [sid for sid in stage_ids if sid in scenario_df.columns]
    if not stage_ids:
        return

    baseline_total = baseline_df[relevant_cols].sum(axis=1).astype(float)
    scenario_total = scenario_df[relevant_cols].sum(axis=1).astype(float)
    stage_series = {sid: _estimate_stage_series(scenario_df[sid]) for sid in stage_ids}

    timestamps = baseline_df.index.to_numpy()
    frame_count = min(80, len(timestamps))
    frame_indices = np.linspace(1, len(timestamps), frame_count, dtype=int)

    fig, (ax_flow, ax_stage) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    ax_flow.set_ylabel("Discharge (m³/s)")
    ax_flow.grid(True, linestyle="--", alpha=0.3)
    ax_flow.set_title(f"{title_suffix} Flow Comparison")
    ax_flow.set_xlim(timestamps[0], timestamps[-1])
    ax_flow.set_ylim(
        0.0,
        max(baseline_total.max(), scenario_total.max()) * 1.1 if len(baseline_total) else 1.0,
    )

    ax_stage.set_ylabel("Stage (m)")
    ax_stage.set_xlabel("Time")
    ax_stage.grid(True, linestyle="--", alpha=0.3)
    ax_stage.set_title(f"{title_suffix} Hydrodynamic Stage")
    ax_stage.set_xlim(timestamps[0], timestamps[-1])
    stage_min = min(series.min() for series in stage_series.values())
    stage_max = max(series.max() for series in stage_series.values())
    ax_stage.set_ylim(stage_min * 0.95, stage_max * 1.05)

    (flow_line_base,) = ax_flow.plot([], [], label="Hydrologic total", color="#264653")
    (flow_line_scen,) = ax_flow.plot([], [], label=f"Hydrodynamic total ({scenario_label})", color="#e76f51")
    ax_flow.legend(loc="upper right")

    stage_lines = []
    palette = ["#2a9d8f", "#1d3557", "#f4a261", "#ff6f61"]
    for idx, sid in enumerate(stage_ids):
        (line,) = ax_stage.plot([], [], label=f"{sid} stage", color=palette[idx % len(palette)])
        stage_lines.append((line, stage_series[sid]))
    ax_stage.legend(loc="upper right")

    def _init():
        flow_line_base.set_data([], [])
        flow_line_scen.set_data([], [])
        for line, _ in stage_lines:
            line.set_data([], [])
        return (flow_line_base, flow_line_scen, *(line for line, _ in stage_lines))

    def _update(frame_idx: int):
        upto = frame_indices[frame_idx]
        flow_line_base.set_data(timestamps[:upto], baseline_total.values[:upto])
        flow_line_scen.set_data(timestamps[:upto], scenario_total.values[:upto])
        for line, series in stage_lines:
            line.set_data(timestamps[:upto], series.values[:upto])
        return (flow_line_base, flow_line_scen, *(line for line, _ in stage_lines))

    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=len(frame_indices),
        init_func=_init,
        interval=150,
        blit=False,
    )

    try:
        ani.save(output_path, writer="pillow", dpi=120)
    except Exception:  # pragma: no cover
        pass
    finally:
        plt.close(fig)


def _plot_stage_flow_timeseries(result_df: pd.DataFrame, segment_id: str, output_path: Path) -> bool:
    import matplotlib.pyplot as plt

    subset = result_df[result_df["segment_id"] == segment_id].copy()
    if subset.empty:
        return False

    times = pd.to_datetime(subset["time"], errors="coerce")
    valid = times.notna()
    if not valid.any():
        return False

    times = times[valid]
    stage = subset.loc[valid, "stage_m"].astype(float)
    discharge = subset.loc[valid, "discharge_m3s"].astype(float)

    fig, ax_stage = plt.subplots(figsize=(10, 4))
    ax_stage.plot(times, stage, color="#2a9d8f", linewidth=1.6, label="Stage")
    ax_stage.set_ylabel("Stage (m)")
    ax_stage.set_xlabel("Time")
    ax_stage.grid(True, linestyle="--", alpha=0.3)

    ax_flow = ax_stage.twinx()
    ax_flow.plot(times, discharge, color="#e76f51", linestyle="--", linewidth=1.4, label="Discharge")
    ax_flow.set_ylabel("Discharge (m³/s)")

    title = f"{segment_id} Stage / Discharge"
    ax_stage.set_title(title)

    lines = ax_stage.get_lines() + ax_flow.get_lines()
    labels = [line.get_label() for line in lines]
    ax_stage.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return True


def _run_cross_section_solver_branch(
    context: ProjectContext,
    step_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Path]:
    """Optional branch: use cross-section solver to derive stage/velocity series."""

    cfg = context.config.get("channel_dynamics", {}) or {}
    if not cfg.get("enable_cross_section_solver"):
        logger.info("Cross-section solver branch disabled via configuration.")
        return {}

    branch_dir = step_dir / "cross_section_solver"
    branch_dir.mkdir(parents=True, exist_ok=True)

    default_channel_dir = context.base_results / "04_channel_profile"
    default_flow_dir = context.base_results / "09_hydrologic_run"

    cross_sections_path = resolve_input_path(
        context,
        cfg.get(
            "cross_sections_path",
            (default_channel_dir / "channel_cross_sections_corrected.csv").as_posix(),
        ),
    )
    centerline_path = resolve_input_path(
        context,
        cfg.get(
            "centerline_path",
            (default_channel_dir / "channel_centerlines.csv").as_posix(),
        ),
    )
    flows_path = resolve_input_path(
        context,
        cfg.get(
            "flow_timeseries_path",
            (default_flow_dir / "channel_flow_timeseries.csv").as_posix(),
        ),
    )

    if not cross_sections_path.exists() or not centerline_path.exists():
        logger.warning(
            "Cross-section solver branch skipped: missing Step04 outputs (%s or %s).",
            cross_sections_path,
            centerline_path,
        )
        return {}
    if not flows_path.exists():
        logger.warning(
            "Cross-section solver branch skipped: missing Step09 flow timeseries (%s).",
            flows_path,
        )
        return {}

    try:
        cross_df = pd.read_csv(cross_sections_path)
        center_df = pd.read_csv(centerline_path)
        flow_df = pd.read_csv(flows_path)
    except Exception as exc:  # pragma: no cover - IO errors
        logger.warning("Failed to load required inputs for cross-section solver: %s", exc)
        return {}

    time_column = cfg.get("time_column", "Timestamp")
    if time_column not in flow_df.columns:
        logger.warning(
            "Cross-section solver branch skipped: time column '%s' not found in %s.",
            time_column,
            flows_path,
        )
        return {}

    flow_df[time_column] = pd.to_datetime(flow_df[time_column], errors="coerce")
    flow_df = flow_df.dropna(subset=[time_column])

    zone_ids = cfg.get("zones") or ["P1"]
    report_segments_cfg = cfg.get("report_segments", {}) or {}
    flow_source = str(cfg.get("flow_source", "channel_timeseries")).lower()
    flow_scenario = str(cfg.get("flow_scenario", "hydraulic_p3p4"))
    slope_floor_cfg = cfg.get("slope_floor_m_per_m")
    slope_cap_cfg = cfg.get("slope_cap_m_per_m")

    def _maybe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    slope_floor_value = _maybe_float(slope_floor_cfg)
    slope_cap_value = _maybe_float(slope_cap_cfg)

    geometry_kwargs: Dict[str, Any] = {}
    for key_cfg, param_name in [
        ("mannings_n", "mannings_n"),
        ("min_depth_m", "min_depth"),
        ("depth_step_m", "depth_step"),
        ("padding_depth_m", "padding_depth"),
        ("max_depth_m", "max_depth"),
        ("active_half_width_m", "active_half_width"),
        ("depth_cap_m", "depth_cap"),
    ]:
        if key_cfg in cfg:
            geometry_kwargs[param_name] = float(cfg[key_cfg])

    outputs: Dict[str, Path] = {}

    for zone_id in zone_ids:
        try:
            geometry = build_zone_geometry(
                zone_id=zone_id,
                centerline=center_df,
                cross_sections=cross_df,
                **geometry_kwargs,
            )
        except Exception as exc:
            logger.warning("Failed to build geometry for zone %s: %s", zone_id, exc)
            continue

        solver = CrossSectionSolver.from_zone_geometry(
            geometry,
            slope_floor=slope_floor_value if slope_floor_value is not None else 1e-6,
            slope_cap=slope_cap_value,
        )
        prefix = f"{zone_id}_"
        zone_columns = [col for col in flow_df.columns if col.startswith(prefix)]
        if flow_source != "subbasin" and not zone_columns:
            logger.warning(
                "Cross-section solver branch: no flow columns for zone %s (prefix %s) in %s.",
                zone_id,
                prefix,
                flows_path,
            )
            continue

        ordered_segments = (
            solver.summary()
            .sort_values("chainage_m")["segment_id"]
            .tolist()
        )
        if not ordered_segments:
            logger.warning("No rating curves available for zone %s.", zone_id)
            continue

        time_values = flow_df[time_column].to_numpy()
        steps = len(flow_df.index)

        lateral_series_map: Optional[Dict[str, np.ndarray]] = None
        channel_map: Dict[str, np.ndarray] = {}
        for segment_id in ordered_segments:
            if segment_id in flow_df.columns:
                channel_map[segment_id] = pd.to_numeric(
                    flow_df[segment_id], errors="coerce"
                ).fillna(0.0).to_numpy(dtype=float)

        flow_matrix = None
        lateral_series_map: Optional[Dict[str, np.ndarray]] = None

        def _build_from_subbasins(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray], bool]:
            matrix = np.zeros((steps, len(ordered_segments)), dtype=float)
            lateral: Dict[str, np.ndarray] = {}
            previous = np.zeros(steps, dtype=float)
            any_nonzero = False
            for idx_seg, segment_id in enumerate(ordered_segments):
                file_path = path / f"{segment_id}.csv"
                series = np.zeros(steps, dtype=float)
                if file_path.exists():
                    try:
                        sub_df = pd.read_csv(
                            file_path,
                            header=None,
                            names=["step", "discharge_m3s"],
                            dtype={"step": int, "discharge_m3s": float},
                        )
                        sub_df.set_index("step", inplace=True)
                        series = sub_df.reindex(range(steps), fill_value=0.0)[
                            "discharge_m3s"
                        ].to_numpy(dtype=float)
                    except Exception as exc:  # pragma: no cover
                        logger.warning(
                            "Failed to read subbasin hydrograph %s: %s", file_path, exc
                        )
                lateral[segment_id] = series
                cumulative = previous + series
                matrix[:, idx_seg] = cumulative
                if not np.allclose(series, 0.0):
                    any_nonzero = True
                previous = cumulative
            headwater_nonzero = not np.allclose(matrix[:, 0], 0.0)
            return matrix, lateral, any_nonzero, headwater_nonzero

        if flow_source == "subbasin":
            scenario_candidates: list[str] = [flow_scenario]
            extra_candidates = cfg.get("flow_scenario_fallbacks", [])
            if isinstance(extra_candidates, str):
                extra_candidates = [extra_candidates]
            scenario_candidates.extend(extra_candidates)
            if "baseline" not in scenario_candidates:
                scenario_candidates.append("baseline")

            for scenario_name in scenario_candidates:
                hydro_path = (
                    context.base_results
                    / "09_hydrologic_run"
                    / "hydro"
                    / f"{scenario_name}_subbasin"
                )
                hydro_path = Path(hydro_path)
                if not hydro_path.exists():
                    continue
                matrix, lateral_map, any_nonzero, headwater_nonzero = _build_from_subbasins(hydro_path)
                if any_nonzero and headwater_nonzero:
                    logger.info(
                        "Cross-section solver using subbasin scenario '%s' for zone %s.",
                        scenario_name,
                        zone_id,
                    )
                    flow_matrix = matrix
                    lateral_series_map = lateral_map
                    break

            if flow_matrix is None or lateral_series_map is None:
                logger.warning(
                    "No non-zero subbasin hydrographs found for zone %s; deriving flows from channel outputs.",
                    zone_id,
                )

        if flow_matrix is None or lateral_series_map is None:
            flow_matrix = np.zeros((steps, len(ordered_segments)), dtype=float)
            lateral_series_map = {}
            previous = np.zeros(steps, dtype=float)
            for idx_seg, segment_id in enumerate(ordered_segments):
                if segment_id in flow_df.columns:
                    cumulative = pd.to_numeric(
                        flow_df[segment_id], errors="coerce"
                    ).fillna(0.0).to_numpy(dtype=float)
                else:
                    cumulative = previous.copy()
                flow_matrix[:, idx_seg] = cumulative
                lateral_series_map[segment_id] = np.maximum(cumulative - previous, 0.0)
                previous = cumulative

        zone_flow_df = pd.DataFrame(flow_matrix, columns=ordered_segments)
        zone_flow_df.insert(0, time_column, time_values)

        result_df = solver.evaluate_timeseries(zone_flow_df, time_column=time_column)
        summary_df = solver.summarize_timeseries(result_df)

        zone_dir = branch_dir / zone_id
        zone_dir.mkdir(parents=True, exist_ok=True)

        timeseries_path = zone_dir / "timeseries.csv"
        summary_path = zone_dir / "summary.csv"
        result_df.to_csv(timeseries_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        outputs[f"{zone_id}_timeseries"] = timeseries_path
        outputs[f"{zone_id}_summary"] = summary_path

        flow_profile_path = zone_dir / "flow_profile.csv"
        zone_flow_df.to_csv(flow_profile_path, index=False)
        outputs[f"{zone_id}_flow_profile"] = flow_profile_path

        if lateral_series_map:
            lateral_df = pd.DataFrame(lateral_series_map)
            lateral_df.insert(0, time_column, time_values)
            lateral_path = zone_dir / "lateral_inflow.csv"
            lateral_df.to_csv(lateral_path, index=False)
            outputs[f"{zone_id}_lateral_inflow"] = lateral_path

        segments_to_plot = report_segments_cfg.get(zone_id) or []
        if not segments_to_plot and not summary_df.empty:
            downstream_idx = summary_df["chainage_m"].idxmax()
            segments_to_plot = [summary_df.loc[downstream_idx, "segment_id"]]

        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:  # pragma: no cover - plotting optional
            plt = None

        time_index = pd.to_datetime(zone_flow_df[time_column], errors="coerce")
        upstream_id = ordered_segments[0]
        downstream_id = ordered_segments[-1]
        curve = solver.ratings.get(downstream_id)

        if plt is not None and time_index.notna().any():
            inflow_path = zone_dir / f"inflow_timeseries_{zone_id}.png"
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(time_index, zone_flow_df[upstream_id].astype(float), color="#1d3557")
            ax.set_title(f"{zone_id} Upstream Inflow ({upstream_id})")
            ax.set_ylabel("Discharge (m³/s)")
            ax.set_xlabel("Time")
            ax.grid(True, linestyle="--", alpha=0.3)
            fig.tight_layout()
            fig.savefig(inflow_path, dpi=220)
            plt.close(fig)
            outputs[f"{zone_id}_inflow_plot"] = inflow_path

            down_df = result_df[result_df["segment_id"] == downstream_id].copy()
            if not down_df.empty:
                down_df["time"] = pd.to_datetime(down_df["time"], errors="coerce")
                down_df = down_df.dropna(subset=["time"])
                if not down_df.empty:
                    stage_plot_path = zone_dir / f"downstream_stage_discharge_{zone_id}_{downstream_id}.png"
                    fig, ax1 = plt.subplots(figsize=(10, 4))
                    ax1.plot(down_df["time"], down_df["stage_m"].astype(float), color="#2a9d8f", label="Stage")
                    ax1.set_ylabel("Stage (m)")
                    ax1.set_xlabel("Time")
                    ax1.grid(True, linestyle="--", alpha=0.3)
                    ax2 = ax1.twinx()
                    ax2.plot(down_df["time"], down_df["discharge_m3s"].astype(float), color="#e76f51", linestyle="--", label="Discharge")
                    ax2.set_ylabel("Discharge (m³/s)")
                    lines = ax1.get_lines() + ax2.get_lines()
                    labels = [line.get_label() for line in lines]
                    ax1.legend(lines, labels, loc="upper right")
                    ax1.set_title(f"{zone_id} Downstream Stage/Discharge ({downstream_id})")
                    fig.tight_layout()
                    fig.savefig(stage_plot_path, dpi=220)
                    plt.close(fig)
                    outputs[f"{zone_id}_{downstream_id}_stage_discharge_plot"] = stage_plot_path

            if curve is not None and curve.discharges.size:
                rating_path = zone_dir / f"rating_curve_{zone_id}_{downstream_id}.png"
                fig, ax = plt.subplots(figsize=(6, 4))
                stage_curve = curve.bed_elevation_m + curve.depths
                ax.plot(curve.discharges, stage_curve, color="#264653")
                ax.set_xlabel("Discharge (m³/s)")
                ax.set_ylabel("Stage (m)")
                ax.set_title(f"{zone_id} Downstream Rating Curve ({downstream_id})")
                ax.grid(True, linestyle="--", alpha=0.3)
                fig.tight_layout()
                fig.savefig(rating_path, dpi=220)
                plt.close(fig)
                outputs[f"{zone_id}_{downstream_id}_rating_curve"] = rating_path

            if lateral_series_map:
                lateral_plot_path = zone_dir / f"lateral_inflow_{zone_id}.png"
                fig, ax = plt.subplots(figsize=(10, 4))
                has_line = False
                for seg, series in lateral_series_map.items():
                    if np.any(series):
                        ax.plot(time_index, series, label=seg)
                        has_line = True
                if has_line:
                    ax.set_title(f"{zone_id} Lateral Inflow Contributions")
                    ax.set_ylabel("Discharge (m³/s)")
                    ax.set_xlabel("Time")
                    ax.grid(True, linestyle="--", alpha=0.3)
                    ax.legend(loc="upper right", fontsize=8)
                    fig.tight_layout()
                    fig.savefig(lateral_plot_path, dpi=220)
                    outputs[f"{zone_id}_lateral_inflow_plot"] = lateral_plot_path
                plt.close(fig)

        report_path = zone_dir / f"{zone_id}_channel_report.md"
        report_lines = [
            f"# {zone_id} Channel Hydrodynamics",
            "",
            f"- Upstream segment: `{upstream_id}`",
            f"- Downstream segment: `{downstream_id}`",
        ]
        upstream_series = zone_flow_df[upstream_id].astype(float).to_numpy()
        if upstream_series.size:
            peak_idx = int(np.argmax(upstream_series))
            report_lines.append(
                f"- Peak upstream inflow: {upstream_series[peak_idx]:.2f} m³/s at {pd.to_datetime(time_values[peak_idx])}"
            )
        if curve is not None and curve.discharges.size:
            report_lines.append(
                f"- Downstream rating curve sampled depths: {curve.depths.min():.2f}–{curve.depths.max():.2f} m"
            )
        down_df = result_df[result_df["segment_id"] == downstream_id].copy()
        if not down_df.empty:
            down_df["time"] = pd.to_datetime(down_df["time"], errors="coerce")
            down_df = down_df.dropna(subset=["time"])
            if not down_df.empty:
                peak_Q = down_df["discharge_m3s"].astype(float).max()
                peak_stage = down_df["stage_m"].astype(float).max()
                report_lines.append(f"- Peak downstream discharge: {peak_Q:.2f} m³/s")
                report_lines.append(f"- Peak downstream stage: {peak_stage:.2f} m")
        if lateral_series_map:
            if len(time_index) > 1:
                delta_seconds = float((time_index.iloc[1] - time_index.iloc[0]).total_seconds())
            else:
                delta_seconds = 0.0
            report_lines.append("")
            report_lines.append("## Lateral Inflow Volumes")
            report_lines.append("Segment | Volume (m³)")
            report_lines.append("---|---")
            for seg in ordered_segments:
                series = lateral_series_map.get(seg)
                if series is None:
                    continue
                if delta_seconds > 0:
                    volume = float(series.sum() * delta_seconds)
                else:
                    volume = float(series.sum())
                report_lines.append(f"{seg} | {volume:.2f}")

        report_lines.append("")
        report_lines.append("## Summary Table")
        report_lines.append(summary_df.to_string(index=False))

        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        outputs[f"{zone_id}_report"] = report_path

        for segment_id in segments_to_plot:
            plot_filename = f"stage_discharge_{zone_id}_{segment_id}.png"
            plot_path = zone_dir / plot_filename
            created = _plot_stage_flow_timeseries(result_df, segment_id, plot_path)
            if created:
                outputs[f"{zone_id}_{segment_id}_stage_discharge_plot"] = plot_path

        logger.info(
            "Cross-section solver outputs generated for zone %s (segments: %s).",
            zone_id,
            ", ".join(sorted(solver.available_segments())),
        )

    return outputs


def run_step10_hydrodynamic_run(config_path: Path | str) -> Dict[str, Path]:
    """Placeholder - implemented in step10_hydrodynamic_run.py module."""
    raise NotImplementedError("See step10_hydrodynamic_run.py")
