"""Step 08: Areal Precipitation

Calculate spatially-averaged precipitation.
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

def run_step08_areal_precipitation(config_path: Path | str) -> Dict[str, Path]:
    """Interpolates station rainfall to parameter subbasins and generates diagnostics."""

    context = load_project_context(config_path)
    step_index = 8
    step_name = "areal_precipitation"
    step_dir = step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step08_areal_precip.log"
    logger = configure_logger("step08", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 08 – Areal precipitation interpolation started.")

    project_cfg = context.config.setdefault("project", {})
    rainfall_cfg = project_cfg.setdefault("rainfall", {})
    delineation_cfg = context.config.get("delineation") or {}
    parameter_dir_entry = delineation_cfg.get("parameter_directory")
    if not parameter_dir_entry:
        raise PipelineConfigurationError(
            "Parameter directory not configured. Run Step 03 before Step 08."
        )

    weights_src = rainfall_cfg.get("weights_json")
    station_series_src = rainfall_cfg.get("station_series_path")
    if not weights_src or not station_series_src:
        raise PipelineConfigurationError(
            "Rainfall weights or station series missing. Ensure Steps 06 and 07 have been executed."
        )

    import json
    import pandas as pd
    import numpy as np

    weights_path = resolve_input_path(context, weights_src)
    station_series_path = resolve_input_path(context, station_series_src)
    weights = json.loads(weights_path.read_text(encoding="utf-8"))

    station_df = pd.read_csv(station_series_path, index_col=0)
    station_df.index = pd.to_datetime(station_df.index)

    parameter_dir = resolve_input_path(context, parameter_dir_entry)
    subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    subbasin_geometries = load_subbasin_geometries(subbasin_geojson)

    from hydrosis.precipitation import interpolate_station_series

    subbasin_series = interpolate_station_series(station_df, weights)
    subbasin_series.index.name = station_df.index.name or "Timestamp"

    subbasin_csv = step_dir / "parameter_subbasin_areal_precipitation.csv"
    subbasin_series.to_csv(subbasin_csv)

    zone_csv = step_dir / "parameter_zone_areal_precipitation.csv"
    parameter_subbasins_csv = parameter_dir / "parameter_subbasins.csv"
    if parameter_subbasins_csv.exists():
        subbasin_meta = pd.read_csv(parameter_subbasins_csv)
        area_lookup = (
            subbasin_meta.set_index("subzone_id")["area_km2"].astype(float).to_dict()
        )
        zone_groups = subbasin_meta.groupby("zone_id")["subzone_id"].apply(list).to_dict()
        zone_frames: Dict[str, pd.Series] = {}
        for zone_id, sub_ids in zone_groups.items():
            valid_subs = [sid for sid in sub_ids if sid in subbasin_series.columns]
            if not valid_subs:
                continue
            weights = np.array([float(area_lookup.get(sid, 0.0)) for sid in valid_subs], dtype=float)
            total_area = float(weights.sum())
            if total_area <= 0.0:
                zone_series = subbasin_series[valid_subs].mean(axis=1)
            else:
                normalised = weights / total_area
                zone_series = (subbasin_series[valid_subs] * normalised).sum(axis=1)
            zone_frames[zone_id] = zone_series
        zone_df = (
            pd.DataFrame(zone_frames, index=subbasin_series.index)
            if zone_frames
            else pd.DataFrame(index=subbasin_series.index)
        )
    else:
        logger.warning(
            "Parameter subbasins CSV not found at %s; falling back to simple means for zone precipitation.",
            parameter_subbasins_csv,
        )
        zone_groups: Dict[str, List[str]] = {}
        for column in subbasin_series.columns:
            zone_id = column.split("_")[0] if "_" in column else column
            zone_groups.setdefault(zone_id, []).append(column)
        zone_df = pd.DataFrame(
            {zone: subbasin_series[cols].mean(axis=1) for zone, cols in zone_groups.items()},
            index=subbasin_series.index,
        )
    zone_df.to_csv(zone_csv)

    time_step_hours = 1.0
    if len(subbasin_series.index) > 1:
        delta = (subbasin_series.index[1] - subbasin_series.index[0]).total_seconds()
        time_step_hours = max(delta / 3600.0, 1e-6)

    summary_rows = []
    for sub_id in subbasin_series.columns:
        series = subbasin_series[sub_id]
        total_depth = float(series.sum() * time_step_hours)
        peak_intensity = float(series.max())
        peak_time = series.idxmax()
        summary_rows.append(
            (
                sub_id,
                total_depth,
                peak_intensity,
                peak_time.isoformat() if hasattr(peak_time, "isoformat") else str(peak_time),
            )
        )
    areal_summary_df = pd.DataFrame(
        summary_rows,
        columns=["subbasin_id", "total_depth_mm", "peak_intensity_mm_per_hr", "peak_time"],
    )
    areal_summary_csv = step_dir / "areal_precip_summary.csv"
    areal_summary_df.to_csv(areal_summary_csv, index=False)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib import cm, colors
    from matplotlib.patches import Polygon as MplPolygon
    from shapely.ops import unary_union

    all_timeseries_path = step_dir / "subbasin_precip_timeseries.png"
    plt.figure(figsize=(12, 6))
    for column in subbasin_series.columns:
        plt.plot(subbasin_series.index, subbasin_series[column], linewidth=1.0, label=column)
    plt.xlabel("Time")
    plt.ylabel("Intensity (mm/hr)")
    plt.title("Subbasin Areal Precipitation Time Series")
    plt.legend(loc="upper right", fontsize=6, ncol=3)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(all_timeseries_path, dpi=220)
    plt.close()

    heatmap_path = step_dir / "subbasin_precipitation_heatmap.png"
    plt.figure(figsize=(10, 4))
    plt.imshow(
        subbasin_series.T,
        aspect="auto",
        cmap="YlGnBu",
        origin="lower",
    )
    plt.colorbar(label="Intensity (mm/hr)")
    plt.yticks(
        range(len(subbasin_series.columns)),
        subbasin_series.columns,
    )
    plt.xticks(
        range(0, len(subbasin_series.index), max(len(subbasin_series.index) // 8, 1)),
        [
            subbasin_series.index[i].strftime("%m-%d %H:%M")
            for i in range(0, len(subbasin_series.index), max(len(subbasin_series.index) // 8, 1))
        ],
        rotation=45,
        ha="right",
    )
    plt.title("Areal Precipitation Heatmap")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=220)
    plt.close()

    cumulative_path = step_dir / "cumulative_precip_plot.png"
    plt.figure(figsize=(10, 4))
    cumulative_series = (subbasin_series * time_step_hours).cumsum()
    for column in cumulative_series.columns:
        plt.plot(cumulative_series.index, cumulative_series[column], label=column)
    plt.xlabel("Time")
    plt.ylabel("Cumulative depth (mm)")
    plt.title("Cumulative Areal Precipitation")
    plt.legend(loc="upper left", fontsize=7, ncol=2)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(cumulative_path, dpi=220)
    plt.close()

    animation_path = step_dir / "areal_precip_animation.mp4"
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = cm.get_cmap("YlOrRd")
    vmax = float(subbasin_series.max().max())
    if vmax <= 0:
        vmax = 1.0
    norm = colors.Normalize(vmin=0.0, vmax=vmax)

    patches: Dict[str, List[MplPolygon]] = {}
    for sub_id, geom in subbasin_geometries.items():
        if geom.is_empty:
            continue
        parts = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
        bucket: List[MplPolygon] = []
        for part in parts:
            poly = MplPolygon(
                np.asarray(part.exterior.coords),
                facecolor=cmap(norm(0.0)),
                edgecolor="#555555",
                linewidth=0.4,
                alpha=0.85,
            )
            ax.add_patch(poly)
            bucket.append(poly)
        if bucket:
            patches[sub_id] = bucket

    missing_geoms = sorted(set(subbasin_series.columns) - set(patches.keys()))
    if missing_geoms:
        logger.warning("Subbasins without geometry for animation: %s", ", ".join(missing_geoms[:8]) + ("..." if len(missing_geoms) > 8 else ""))

    try:
        extent_geom = unary_union([geom for geom in subbasin_geometries.values() if not geom.is_empty])
        minx, miny, maxx, maxy = extent_geom.bounds
    except Exception:  # pragma: no cover - fallback when union fails
        xs = [coord for geom in subbasin_geometries.values() if not geom.is_empty for coord in (geom.bounds[::2])]
        ys = [coord for geom in subbasin_geometries.values() if not geom.is_empty for coord in (geom.bounds[1::2])]
        minx = min(xs) if xs else 0.0
        maxx = max(xs) if xs else 1.0
        miny = min(ys) if ys else 0.0
        maxy = max(ys) if ys else 1.0

    span = max(maxx - minx, maxy - miny)
    pad = max(span * 0.05, 500.0)
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Areal Precipitation Evolution")
    ax.axis("off")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Intensity (mm/hr)")

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=10,
        color="#222222",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
    )

    patch_artists = [patch for bucket in patches.values() for patch in bucket]

    def _update_areal(frame: int):
        values = subbasin_series.iloc[frame]
        timestamp = subbasin_series.index[frame]
        for sub_id, bucket in patches.items():
            value = float(values.get(sub_id, 0.0))
            color = cmap(norm(value))
            for patch in bucket:
                patch.set_facecolor(color)
        time_text.set_text(timestamp.strftime("%Y-%m-%d %H:%M"))
        return patch_artists + [time_text]

    anim = FuncAnimation(
        fig,
        _update_areal,
        frames=len(subbasin_series.index),
        interval=150,
        blit=False,
    )
    try:
        from matplotlib.animation import FFMpegWriter

        writer = FFMpegWriter(fps=8)
        anim.save(animation_path, writer=writer)
    except Exception:  # pragma: no cover - fallback path when ffmpeg not available
        from matplotlib.animation import PillowWriter

        fallback_path = animation_path.with_suffix(".gif")
        anim.save(fallback_path, writer=PillowWriter(fps=8))
        animation_path = fallback_path
    finally:
        plt.close(fig)

    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 08 – Areal Precipitation")
    builder.add_paragraph(
        "使用泰森权重将雨量站时序插值到参数子流域，生成面雨量表、热力图、累积过程及动画。"
    )
    builder.add_heading("统计摘要", level=2)
    summary_preview = areal_summary_df.sort_values("total_depth_mm", ascending=False).head(10)
    builder.add_table(
        TableData(
            headers=["子流域", "总雨量 (mm)", "峰值强度 (mm/hr)"],
            rows=[
                [
                    row["subbasin_id"],
                    f"{row['total_depth_mm']:.2f}",
                    f"{row['peak_intensity_mm_per_hr']:.2f}",
                ]
                for _, row in summary_preview.iterrows()
            ],
        )
    )
    builder.add_paragraph(f"面雨量插值时间：{timestamp.isoformat()}")
    builder.write(report_path)

    rainfall_cfg["areal_precip_path"] = context.to_relative(subbasin_csv)
    context.config.setdefault("io", {}).setdefault("precipitation", context.to_relative(subbasin_csv))
    project_cfg["last_step08_run"] = timestamp.isoformat()
    dump_project_config(context)
    logger.info("Step 08 – Areal precipitation completed successfully.")

    return {
        "parameter_subbasin_precip": subbasin_csv,
        "zone_precip": zone_csv,
        "summary": areal_summary_csv,
        "heatmap": heatmap_path,
        "cumulative_plot": cumulative_path,
        "animation": animation_path,
        "report": report_path,
        "log": log_path,
        "subbasin_precip_timeseries": all_timeseries_path,
    }


def run_step09_hydrologic_run(config_path: Path | str) -> Dict[str, Path]:
