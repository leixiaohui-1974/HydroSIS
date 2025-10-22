"""Step 06: Rain Sequence Generation

Generate precipitation time series.
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
try:
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union
except ImportError:
    shape = Point = unary_union = None

def run_step06_rain_sequence(config_path: Path | str) -> Dict[str, Path]:
    """Generate synthetic rainfall time series for rain gauges and aggregate forcing."""

    context = load_project_context(config_path)
    step_index = 6
    step_name = "rain_sequence"
    step_dir = step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step06_rain_sequence.log"
    logger = configure_logger("step06", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 06 – Rain sequence generation started.")

    project_cfg = context.config.setdefault("project", {})
    gauge_cfg = project_cfg.setdefault("rain_gauge", {})
    rainfall_cfg = project_cfg.setdefault("rainfall", {})
    delineation_cfg = context.config.get("delineation") or {}
    parameter_dir_entry = delineation_cfg.get("parameter_directory")
    if not parameter_dir_entry:
        raise PipelineConfigurationError(
            "Parameter directory not configured. Run Step 03 before Step 06."
        )

    parameter_dir = resolve_input_path(context, parameter_dir_entry)
    subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    subbasin_geometries = load_subbasin_geometries(subbasin_geojson)

    base_series_path = rainfall_cfg.get(
        "base_series_path", "results/upper_truckee_channel_demo/storm_forcing.csv"
    )
    base_series = load_base_precipitation_series(
        resolve_input_path(context, base_series_path),
        column=rainfall_cfg.get("base_column"),
    )

    station_count = int(gauge_cfg.get("station_count", 10))
    heterogeneity = float(gauge_cfg.get("heterogeneity", 0.4))
    min_burst_events = int(gauge_cfg.get("min_burst_events", 1))
    max_burst_events = int(gauge_cfg.get("max_burst_events", 3))
    seed = int(gauge_cfg.get("seed", 42))

    rain_inputs = _generate_rain_gauge_inputs(
        base_series,
        subbasin_geometries,
        station_count=station_count,
        seed=seed,
        heterogeneity=heterogeneity,
        min_burst_events=min_burst_events,
        max_burst_events=max_burst_events,
    )

    import pandas as pd
    import numpy as np

    station_series = rain_inputs.station_series.copy()
    station_series.index.name = base_series.index.name or "Timestamp"
    station_forcing_path = step_dir / "rain_gauge_forcing.csv"
    station_series.to_csv(station_forcing_path)

    aggregated = station_series.mean(axis=1)
    aggregated_df = pd.DataFrame(
        {
            "Timestamp": station_series.index,
            "Average_Intensity": aggregated.values,
            "Max_Intensity": station_series.max(axis=1).values,
            "Min_Intensity": station_series.min(axis=1).values,
        }
    )
    storm_forcing_path = step_dir / "storm_forcing.csv"
    aggregated_df.to_csv(storm_forcing_path, index=False)

    time_step_hours = 1.0
    if len(station_series.index) > 1:
        delta = (station_series.index[1] - station_series.index[0])
        time_step_hours = max(delta.total_seconds() / 3600.0, 1e-6)

    summary_rows = []
    for station_id in station_series.columns:
        series = station_series[station_id]
        total_depth = float(series.sum() * time_step_hours)
        peak_intensity = float(series.max())
        peak_time = series.idxmax()
        summary_rows.append(
            (
                station_id,
                total_depth,
                peak_intensity,
                peak_time.isoformat() if hasattr(peak_time, "isoformat") else str(peak_time),
            )
        )
    aggregated_total_depth = float(aggregated.sum() * time_step_hours)
    aggregated_peak = float(aggregated.max())
    aggregated_peak_time = aggregated.idxmax()
    summary_rows.append(
        (
            "Average",
            aggregated_total_depth,
            aggregated_peak,
            aggregated_peak_time.isoformat() if hasattr(aggregated_peak_time, "isoformat") else str(aggregated_peak_time),
        )
    )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=["series_id", "total_depth_mm", "peak_intensity_mm_per_hr", "peak_time"],
    )
    summary_path = step_dir / "forcing_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    heatmap_path = step_dir / "rainfall_heatmap.png"
    plt.figure(figsize=(10, 4))
    plt.imshow(
        station_series.T,
        aspect="auto",
        cmap="Blues",
        origin="lower",
    )
    plt.colorbar(label="Intensity (mm/hr)")
    plt.yticks(
        range(len(station_series.columns)),
        station_series.columns,
    )
    plt.xticks(
        range(0, len(station_series.index), max(len(station_series.index) // 8, 1)),
        [
            station_series.index[i].strftime("%m-%d %H:%M")
            for i in range(0, len(station_series.index), max(len(station_series.index) // 8, 1))
        ],
        rotation=45,
        ha="right",
    )
    plt.title("Rainfall Heatmap by Station")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=220)
    plt.close()

    profile_path = step_dir / "storm_profile.png"
    plt.figure(figsize=(10, 4))
    plt.plot(station_series.index, aggregated, label="Average intensity", color="navy")
    plt.fill_between(station_series.index, aggregated, color="navy", alpha=0.3)
    plt.ylabel("Intensity (mm/hr)")
    plt.xlabel("Time")
    plt.title("Storm Profile (Average Intensity)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(profile_path, dpi=220)
    plt.close()

    timeseries_path = step_dir / "rainfall_timeseries.png"
    plt.figure(figsize=(10, 4))
    for station_id in station_series.columns:
        plt.plot(
            station_series.index,
            station_series[station_id],
            label=station_id,
            linewidth=1.1,
        )
    plt.xlabel("Time")
    plt.ylabel("Intensity (mm/hr)")
    plt.title("Rainfall Evolution by Station")
    plt.legend(loc="upper right", ncol=2, fontsize=7)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(timeseries_path, dpi=220)
    plt.close()

    animation_path = step_dir / "rainfall_animation.gif"
    fig, ax = plt.subplots(figsize=(10, 4))
    lines = []
    for station_id in station_series.columns:
        (line,) = ax.plot([], [], label=station_id, linewidth=1.1)
        lines.append(line)
    ax.set_xlim(station_series.index[0], station_series.index[-1])
    y_max = float(station_series.max().max()) if not station_series.empty else 1.0
    ax.set_ylim(0.0, max(y_max * 1.1, 1.0))
    ax.set_xlabel("Time")
    ax.set_ylabel("Intensity (mm/hr)")
    ax.set_title("Rainfall Evolution by Station")
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    ax.grid(True, linestyle="--", alpha=0.3)

    def _init_animation():
        for line in lines:
            line.set_data([], [])
        return lines

    time_values = station_series.index.to_pydatetime()
    station_arrays = {
        station_id: station_series[station_id].to_numpy(dtype=float)
        for station_id in station_series.columns
    }

    def _update(frame: int):
        current_times = time_values[: frame + 1]
        for line, station_id in zip(lines, station_series.columns):
            line.set_data(current_times, station_arrays[station_id][: frame + 1])
        ax.set_title(f"Rainfall Evolution by Station\n{time_values[frame].strftime('%Y-%m-%d %H:%M')}")
        return lines

    if len(station_series.index) > 1:
        anim = FuncAnimation(
            fig,
            _update,
            frames=len(station_series.index),
            init_func=_init_animation,
            interval=120,
            blit=False,
        )
        try:
            from matplotlib.animation import PillowWriter

            anim.save(animation_path, writer=PillowWriter(fps=8))
        except Exception:
            animation_path = animation_path.with_suffix(".mp4")
            try:
                from matplotlib.animation import FFMpegWriter

                anim.save(animation_path, writer=FFMpegWriter(fps=8))
            except Exception:
                animation_path = animation_path.with_suffix(".gif")
    plt.close(fig)

    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 06 – Rain Sequence Generation")
    builder.add_paragraph(
        "根据雨量站布设结果生成时序降雨数据，输出站点及汇总强度序列，并生成热力图、动画和暴雨过程线。"
    )
    builder.add_heading("关键信息", level=2)
    builder.add_list(
        [
            f"时间步长：{time_step_hours:.2f} 小时",
            f"平均总雨量：{aggregated_total_depth:.2f} mm",
            f"峰值强度：{aggregated_peak:.2f} mm/hr",
        ]
    )
    builder.add_heading("代表站点统计", level=2)
    builder.add_table(
        TableData(
            headers=["站点", "总雨量 (mm)", "峰值强度 (mm/hr)"],
            rows=[
                [
                    row["series_id"],
                    f"{row['total_depth_mm']:.2f}",
                    f"{row['peak_intensity_mm_per_hr']:.2f}",
                ]
                for _, row in summary_df.head(6).iterrows()
            ],
        )
    )
    builder.add_paragraph(f"序列生成时间：{timestamp.isoformat()}")
    builder.write(report_path)

    rainfall_cfg["station_series_path"] = context.to_relative(station_forcing_path)
    rainfall_cfg["storm_forcing_path"] = context.to_relative(storm_forcing_path)
    rainfall_cfg["summary_path"] = context.to_relative(summary_path)
    rainfall_cfg["timeseries_plot"] = context.to_relative(timeseries_path)
    rainfall_cfg["timeseries_animation"] = context.to_relative(animation_path)
    project_cfg["last_step06_run"] = timestamp.isoformat()

    dump_project_config(context)
    logger.info("Step 06 – Rain sequence generation completed successfully.")

    return {
        "station_forcing": station_forcing_path,
        "storm_forcing": storm_forcing_path,
        "summary": summary_path,
        "heatmap": heatmap_path,
        "profile": profile_path,
        "timeseries_plot": timeseries_path,
        "animation": animation_path,
        "report": report_path,
        "log": log_path,
    }


def run_step07_thiessen_weights(config_path: Path | str) -> Dict[str, Path]:
