"""Step 10: Hydrodynamic Routing

Run 1D hydraulic channel routing.
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
from hydrosis.hydrodynamics import build_zone_geometry, CrossSectionSolver
from hydrosis.model import Subbasin
from hydrosis.workflow.orchestration import _instantiate_model, ScenarioRun

def run_step10_hydrodynamic_run(config_path: Path | str) -> Dict[str, Path]:
    """Execute hydrodynamic routing scenarios and compare against the baseline."""

    if yaml is None:
        raise ImportError("PyYAML is required to load the project configuration.")

    context = load_project_context(config_path)
    step_index = 10
    step_name = "hydrodynamic_run"
    step_dir = step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step10_hydrodynamic_run.log"
    logger = configure_logger("step10", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 10 – Hydrodynamic scenario run started.")

    config_data = yaml.safe_load(context.config_path.read_text(encoding="utf-8"))
    from hydrosis.config import ModelConfig

    model_section = config_data.get("model", {})
    model_dict = {
        "delineation": config_data.get("delineation", {}),
        "runoff_models": model_section.get("runoff_models", []),
        "routing_models": model_section.get("routing_models", []),
        "parameter_zones": model_section.get("parameter_zones", []),
        "io": config_data.get("io", {}),
        "scenarios": config_data.get("scenarios", []),
        "evaluation": config_data.get("evaluation"),
    }
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
        raise ValueError("Precipitation forcing must contain at least two timesteps for hydrodynamic run.")

    forcing_map = {col: list(forcing_df[col].astype(float)) for col in forcing_df.columns}
    total_steps = len(forcing_df.index)
    for entry in model_config.delineation.precomputed_subbasins or []:
        sub_id = str(entry.get("id"))
        forcing_map.setdefault(sub_id, [0.0] * total_steps)

    baseline_model = _instantiate_model(model_config)
    baseline = _run_model("baseline", baseline_model, forcing_map)

    scenario_runs: Dict[str, ScenarioRun] = {}
    for index, scenario_id in enumerate(scenario_ids, start=1):
        logger.info(
            "Running hydrodynamic scenario %s (%d/%d)",
            scenario_id,
            index,
            len(scenario_ids),
        )
        scenario_config = copy.deepcopy(model_config)
        scenario_config.scenarios = [
            scenario for scenario in scenario_config.scenarios if scenario.id == scenario_id
        ]
        scenario_model = _instantiate_model(scenario_config)
        scenario_config.apply_scenario(scenario_id, scenario_model.subbasins.values())
        scenario_runs[scenario_id] = _run_model(scenario_id, scenario_model, forcing_map)

    zone_baseline = _flatten_zone_discharge(baseline.zone_discharge)
    write_simulation_results(results_dir / "baseline", zone_baseline)
    write_simulation_results(results_dir / "baseline_local", baseline.local)
    write_simulation_results(results_dir / "baseline_subbasin", baseline.aggregated)
    for scenario_id, result in scenario_runs.items():
        zone_result = _flatten_zone_discharge(result.zone_discharge)
        write_simulation_results(results_dir / scenario_id, zone_result)
        write_simulation_results(results_dir / f"{scenario_id}_local", result.local)
        write_simulation_results(
            results_dir / f"{scenario_id}_subbasin",
            result.aggregated,
        )

    baseline_df = pd.DataFrame(baseline.aggregated, index=forcing_df.index).astype(float)
    baseline_df = baseline_df.clip(lower=0.0)
    baseline_df.index.name = "Timestamp"

    difference_rows: list[tuple[str, float, float]] = []
    scenario_charts: Dict[str, Path] = {}

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    for scenario_id, scenario in scenario_runs.items():
        scenario_df = pd.DataFrame(scenario.aggregated, index=forcing_df.index).astype(float)
        scenario_df = scenario_df.clip(lower=0.0)
        scenario_df.index.name = "Timestamp"

        diff_df = scenario_df - baseline_df
        diff_stats = diff_df.abs().max()
        for sub_id, peak_diff in diff_stats.items():
            difference_rows.append((scenario_id, sub_id, float(peak_diff)))

        scenario_plot_path = step_dir / f"hydrograph_{scenario_id}.png"
        plt.figure(figsize=(10, 5))
        for col in baseline_df.columns[:min(4, len(baseline_df.columns))]:
            plt.plot(baseline_df.index, baseline_df[col], label=f"{col} (baseline)", linestyle="--")
            if col in scenario_df.columns:
                plt.plot(
                    scenario_df.index,
                    scenario_df[col],
                    label=f"{col} ({scenario_id})",
                )
        plt.xlabel("Time")
        plt.ylabel("Discharge (m³/s)")
        plt.title(f"Hydrographs Comparison – {scenario_id}")
        plt.legend(fontsize=7)
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(scenario_plot_path, dpi=220)
        plt.close()
        scenario_charts[scenario_id] = scenario_plot_path

        outlet_alias = {
            "P1_sub1": "p1_outlet",
            "P1_sub2": "p1_midreach",
            "P3_sub1": "p3_outlet",
            "P4_sub1": "p4_outlet",
        }
        for outlet_id, label in outlet_alias.items():
            stage_plot_path = step_dir / f"{label}_flow_stage_comparison_{scenario_id}.png"
            _plot_flow_stage_comparison(
                baseline_df,
                scenario_df,
                outlet_id,
                scenario_id,
                stage_plot_path,
            )
            if stage_plot_path.exists():
                scenario_charts[f"{scenario_id}_{label}_flow_stage"] = stage_plot_path

        p1_animation_path = step_dir / f"p1_mainstem_flow_stage_animation_{scenario_id}.gif"
        _create_mainstem_animation(
            baseline_df,
            scenario_df,
            scenario_id,
            prefixes=["P1_"],
            stage_ids=["P1_sub1", "P1_sub2"],
            title_suffix="P1 Mainstem",
            output_path=p1_animation_path,
        )
        if p1_animation_path.exists():
            scenario_charts[f"{scenario_id}_p1_mainstem_animation"] = p1_animation_path

    cross_section_outputs = _run_cross_section_solver_branch(context, step_dir, logger)

    difference_df = pd.DataFrame(
        difference_rows,
        columns=["scenario_id", "subbasin_id", "peak_absolute_difference_m3s"],
    )
    difference_csv = step_dir / "hydro_difference_stats.csv"
    difference_df.to_csv(difference_csv, index=False)

    if scenario_runs:
        first_scenario = next(iter(scenario_runs.values()))
        scenario_df = pd.DataFrame(first_scenario.aggregated, index=forcing_df.index).astype(float)
        scenario_df = scenario_df.sub(scenario_df.iloc[0], axis=1).clip(lower=0.0)
        diff_heatmap = scenario_df - baseline_df
        heatmap_path = step_dir / "flow_difference_heatmap.png"
        plt.figure(figsize=(10, 4))
        plt.imshow(
            diff_heatmap.T,
            aspect="auto",
            cmap="bwr",
            origin="lower",
        )
        plt.colorbar(label="Discharge difference (m³/s)")
        plt.yticks(
            range(len(diff_heatmap.columns)),
            diff_heatmap.columns,
        )
        plt.xticks(
            range(0, len(diff_heatmap.index), max(len(diff_heatmap.index) // 8, 1)),
            [
                diff_heatmap.index[i].strftime("%m-%d %H:%M")
                for i in range(0, len(diff_heatmap.index), max(len(diff_heatmap.index) // 8, 1))
            ],
            rotation=45,
            ha="right",
        )
        plt.title(f"Flow Difference Heatmap ({next(iter(scenario_runs.keys()))})")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=220)
        plt.close()
    else:
        heatmap_path = step_dir / "flow_difference_heatmap.png"

    comparison_path = step_dir / "channel_flow_comparison.png"
    plt.figure(figsize=(10, 4))
    plt.plot(baseline_df.index, baseline_df.sum(axis=1), label="Baseline total", color="steelblue")
    for scenario_id, scenario in scenario_runs.items():
        scenario_df = pd.DataFrame(scenario.aggregated, index=forcing_df.index).astype(float)
        scenario_df = scenario_df.sub(scenario_df.iloc[0], axis=1).clip(lower=0.0)
        scenario_df = scenario_df.clip(lower=0.0)
        plt.plot(
            scenario_df.index,
            scenario_df.sum(axis=1),
            label=f"{scenario_id} total",
        )
    plt.xlabel("Time")
    plt.ylabel("Discharge (m³/s)")
    plt.title("Channel Flow Comparison – Baseline vs Scenarios")
    plt.legend(fontsize=7)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(comparison_path, dpi=220)
    plt.close()

    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 10 – Hydrodynamic Scenario Run")
    builder.add_paragraph(
        "Execute hydrodynamic scenario simulations, comparing baseline and scenario discharge differences, "
        "and outputting peak deviation statistics and comparison visualizations."
    )
    builder.add_heading("Peak Deviation Overview", level=2)
    diff_preview = difference_df.sort_values("peak_absolute_difference_m3s", ascending=False).head(10)
    builder.add_table(
        TableData(
            headers=["Scenario", "Subbasin", "Peak Deviation (m³/s)"],
            rows=[
                [
                    row["scenario_id"],
                    row["subbasin_id"],
                    f"{row['peak_absolute_difference_m3s']:.2f}",
                ]
                for _, row in diff_preview.iterrows()
            ],
        )
    )
    builder.add_paragraph(f"Scenario simulation completion time: {timestamp.isoformat()}")
    builder.write(report_path)

    project_cfg = context.config.setdefault("project", {})
    project_cfg["last_step10_run"] = timestamp.isoformat()
    dump_project_config(context)
    logger.info("Step 10 – Hydrodynamic scenario run completed successfully.")

    outputs: Dict[str, Path] = {
        "results_directory": results_dir,
        "difference_stats": difference_csv,
        "heatmap": heatmap_path,
        "comparison_plot": comparison_path,
        "report": report_path,
        "log": log_path,
    }
    outputs.update({f"hydrograph_{sid}": path for sid, path in scenario_charts.items()})
    outputs.update(cross_section_outputs)
    return outputs


def run_final_pipeline_report(config_path: Path | str) -> Dict[str, Path]:
    """Aggregate step reports into a single final pipeline summary."""

    if yaml is None:
        raise ImportError("PyYAML is required to load the project configuration.")

    context = load_project_context(config_path)
    step_index = 11
    step_name = "final_report"
    step_dir = step_directory(context, step_index, step_name)
    step_dir.mkdir(parents=True, exist_ok=True)
    log_path = context.logs_directory / "step11_final_report.log"
    logger = configure_logger("step11", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 11 – Final pipeline report generation started.")

    report_files = sorted(
        context.reports_directory.glob("step??_*.md"),
        key=lambda path: (path.name[:5], path.name),
    )
    if not report_files:
        raise FileNotFoundError(
            f"No step reports found under {context.reports_directory}. "
            "Ensure previous steps have been executed."
        )

    summary_lines: List[str] = []
    sections: List[Tuple[str, List[str]]] = []
    for report_path in report_files:
        lines = report_path.read_text(encoding="utf-8").splitlines()
        title = report_path.stem
        for line in lines:
            if line.startswith("#"):
                title = line.lstrip("# ").strip() or title
                break
        summary_lines.append(f"- [{title}]({report_path.name})")
        sections.append((title, lines))

    final_report_path = context.reports_directory / "final_pipeline_report.md"
    builder = MarkdownReportBuilder("Upper Truckee Ten-Step Pipeline Summary")
    builder.add_paragraph(
        "This report integrates Markdown content from all ten pipeline stages, "
        "facilitating review of overall inputs, outputs, and key metrics."
    )
    builder.add_heading("Table of Contents", level=2)
    builder.extend(summary_lines)
    builder.add_paragraph(f"Summary generation time: {timestamp.isoformat()}")

    for title, lines in sections:
        builder.add_heading(title, level=2)
        builder.extend(lines)

    builder.write(final_report_path)

    project_cfg = context.config.setdefault("project", {})
    project_cfg["final_report_path"] = context.to_relative(final_report_path)
    project_cfg["last_final_report"] = timestamp.isoformat()
    dump_project_config(context)
    logger.info("Final pipeline report generated at %s", final_report_path)

    return {
        "final_report": final_report_path,
        "log": log_path,
    }


__all__ = [
    "PipelineConfigurationError",
    "ProjectContext",
    "load_project_context",
    "dump_project_config",
    "run_step01_dem_preprocessing",
    "run_step02_pour_points",
    "run_step03_partitioning",
    "run_step04_channel_profile",
    "run_step05_rain_gauge_layout",
    "run_step06_rain_sequence",
    "run_step07_thiessen_weights",
    "run_step08_areal_precipitation",
    "run_step09_hydrologic_run",
    "run_step10_hydrodynamic_run",
    "run_final_pipeline_report",
]
