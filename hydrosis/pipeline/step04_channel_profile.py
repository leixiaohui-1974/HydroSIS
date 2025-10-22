"""Step 04: Channel Profile Analysis

Extract channel geometry and compute profiles.
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
    import rasterio
    from hydrosis.delineation import utils as dutils
except ImportError:
    rasterio = None
    dutils = None

def run_step04_channel_profile(config_path: Path | str) -> Dict[str, Path]:
    """Prepare channel profiles, cross-sections, and visualisations for key zones."""

    context = load_project_context(config_path)
    step_index = 4
    step_name = "channel_profile"
    step_dir = step_directory(context, step_index, step_name)
    cross_section_dir = step_dir / "channel_cross_sections"
    cross_section_dir.mkdir(parents=True, exist_ok=True)
    log_path = context.logs_directory / "step04_channel_profile.log"
    logger = configure_logger("step04", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 04 – Channel profile preparation started.")

    delineation_cfg = context.config.get("delineation", {})
    project_cfg = context.config.get("project", {})
    channel_cfg = project_cfg.get("channel_profile", {}) if isinstance(project_cfg, Mapping) else {}

    dem_entry = delineation_cfg.get("dem_path")
    parameter_dir_entry = delineation_cfg.get("parameter_directory")
    if not dem_entry or not parameter_dir_entry:
        raise PipelineConfigurationError(
            "DEM path and parameter directory must be available before running Step 04."
        )

    dem_path = resolve_input_path(context, dem_entry)
    parameter_dir = resolve_input_path(context, parameter_dir_entry)
    channels_geojson = parameter_dir / "parameter_channels.geojson"
    channels_csv = parameter_dir / "parameter_channels.csv"
    if not channels_geojson.exists() or not channels_csv.exists():
        raise FileNotFoundError(
            "Parameter channel artefacts are missing. Ensure Step 03 completed successfully."
        )

    spacing_m = float(channel_cfg.get("cross_section_spacing_m", 500.0))
    half_width_m = float(channel_cfg.get("cross_section_half_width_m", 150.0))
    sample_points = int(channel_cfg.get("cross_section_samples", 41))
    band_width = float(channel_cfg.get("band_width", 40.0))
    target_spacing = float(channel_cfg.get("target_spacing", 150.0))
    tolerance = float(channel_cfg.get("monotonic_tolerance", 0.0))
    min_variation = float(channel_cfg.get("min_variation", 0.1))

    import pandas as pd

    channels_df = pd.read_csv(channels_csv)
    configured_zones = channel_cfg.get("zones")
    if configured_zones:
        zones = [str(zone) for zone in configured_zones]
    else:
        zones = sorted({str(zone) for zone in channels_df["zone_id"].unique()})
    if not zones:
        raise RuntimeError("No zones available for channel profiling.")

    logger.info("Target zones for channel profiling: %s", ", ".join(zones))

    created_files, cross_sections_df = _extract_cross_sections(
        channels_geojson,
        dem_path,
        zones,
        spacing_m=spacing_m,
        half_width_m=half_width_m,
        n_points=sample_points,
        output_dir=cross_section_dir,
        logger=logger,
    )
    if cross_sections_df.empty:
        raise RuntimeError("Cross-section extraction produced no data.")

    (
        segments_path,
        aggregated_cross_path,
        segments_df,
        aggregated_cross_df,
        zone_lengths,
    ) = _summarise_main_channels(
        channels_csv,
        zones,
        cross_sections_df,
        step_dir,
    )

    from hydrosis.analysis.channel_profile import (
        ChannelProfileConfig,
        build_zone_grid,
        extract_centerline,
        generate_channel_profiles,
    )

    profile_cfg = ChannelProfileConfig(
        min_variation=min_variation,
        band_width=band_width,
        target_spacing=target_spacing,
        monotonic_tolerance=tolerance,
    )
    profile_result = generate_channel_profiles(
        aggregated_cross_df,
        segments_df,
        zones,
        config=profile_cfg,
    )

    corrected_cross_sections = profile_result.cross_sections.copy()
    corrected_path = step_dir / "channel_cross_sections_corrected.csv"
    corrected_cross_sections.to_csv(corrected_path, index=False)

    centerlines_df = []
    for zone, df in profile_result.centerlines.items():
        temp = df.copy()
        temp["zone_id"] = zone
        centerlines_df.append(temp)
    centerlines_combined = (
        pd.concat(centerlines_df, ignore_index=True) if centerlines_df else pd.DataFrame()
    )
    centerline_path = step_dir / "channel_centerlines.csv"
    centerlines_combined.to_csv(centerline_path, index=False)

    profile_summary_rows: list[Dict[str, object]] = []
    for zone in zones:
        cl = profile_result.centerlines.get(zone)
        if cl is None or cl.empty:
            continue
        cl_sorted = cl.sort_values("global_station_m")
        length = float(zone_lengths.get(zone, cl_sorted["global_station_m"].iloc[-1]))
        base_start = float(cl_sorted["base_elevation_m"].iloc[0])
        base_end = float(cl_sorted["base_elevation_m"].iloc[-1])
        corrected_start = float(cl_sorted["global_corrected_elevation_m"].iloc[0])
        corrected_end = float(cl_sorted["global_corrected_elevation_m"].iloc[-1])
        drop = corrected_start - corrected_end
        slope = drop / length if length > 0 else 0.0
        max_adjustment = float(cl_sorted["global_adjustment_m"].max())
        profile_summary_rows.append(
            {
                "zone_id": zone,
                "length_m": length,
                "base_drop_m": base_start - base_end,
                "corrected_drop_m": drop,
                "average_slope": slope,
                "max_adjustment_m": max_adjustment,
                "start_elevation_m": corrected_start,
                "end_elevation_m": corrected_end,
            }
        )

    profile_summary_df = pd.DataFrame(profile_summary_rows)
    profile_summary_path = step_dir / "profile_summary.csv"
    profile_summary_df.to_csv(profile_summary_path, index=False)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    def _make_static_plot(zone_id: str) -> Optional[Path]:
        zone_df = corrected_cross_sections[corrected_cross_sections["zone_id"] == zone_id]
        if zone_df.empty:
            return None
        xs, ys, Z = build_zone_grid(
            zone_df,
            station_field="global_station_m",
            target_spacing=target_spacing,
        )
        centerline = extract_centerline(zone_df, band_width=band_width)
        if centerline is None or centerline.empty:
            logger.warning("No centerline generated for zone %s; skipping static plot.", zone_id)
            return None
        required_cols = {"global_station_m", "base_elevation_m", "global_corrected_elevation_m"}
        missing = required_cols.difference(centerline.columns)
        if missing:
            logger.warning(
                "Centerline for zone %s missing columns %s; skipping static plot.",
                zone_id,
                ", ".join(sorted(missing)),
            )
            return None
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        mesh = axes[0].pcolormesh(xs, ys, Z, shading="auto", cmap="terrain")
        axes[0].set_title(f"{zone_id} Channel Heatmap")
        axes[0].set_xlabel("Along-channel (m)")
        axes[0].set_ylabel("Distance from Centerline (m)")
        fig.colorbar(mesh, ax=axes[0], label="Elevation (m)")

        axes[1].plot(
            centerline["global_station_m"],
            centerline["base_elevation_m"],
            label="Original Baseline",
            linestyle="--",
        )
        axes[1].plot(
            centerline["global_station_m"],
            centerline["global_corrected_elevation_m"],
            label="Corrected",
        )
        axes[1].set_xlabel("Along-channel (m)")
        axes[1].set_ylabel("Elevation (m)")
        axes[1].set_title(f"{zone_id} Centerline Longitudinal Profile")
        axes[1].legend()

        output_path = step_dir / f"{zone_id}_channel_static.png"
        plt.savefig(output_path, dpi=220)
        plt.close(fig)
        return output_path

    static_figures: Dict[str, Path] = {}
    for zone in zones:
        path = _make_static_plot(zone)
        if path:
            static_figures[zone] = path

    combined_plot_path = step_dir / "combined_centerline_profile.png"
    plt.figure(figsize=(10, 5))
    for zone in zones:
        cl = profile_result.centerlines.get(zone)
        if cl is None or cl.empty:
            continue
        cl_sorted = cl.sort_values("global_station_m")
        plt.plot(
            cl_sorted["global_station_m"],
            cl_sorted["global_corrected_elevation_m"],
            label=zone,
        )
    plt.xlabel("Along-channel (m)")
    plt.ylabel("Elevation (m)")
    plt.title("Main Channel Centerline Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(combined_plot_path, dpi=220)
    plt.close()

    html_outputs: Dict[str, Path] = {}
    try:
        import plotly.graph_objs as go

        for zone in zones:
            zone_df = corrected_cross_sections[corrected_cross_sections["zone_id"] == zone]
            if zone_df.empty:
                continue
            scatter = go.Scatter3d(
                x=zone_df["global_station_m"],
                y=zone_df["distance_from_center_m"],
                z=zone_df["elevation_m"],
                mode="markers",
                marker=dict(size=2, color=zone_df["elevation_m"], colorscale="Viridis"),
            )
            fig = go.Figure(data=[scatter])
            fig.update_layout(
                title=f"{zone} Channel 3D Point Cloud",
                scene=dict(
                    xaxis_title="Along-channel (m)",
                    yaxis_title="Distance from Centerline (m)",
                    zaxis_title="Elevation (m)",
                ),
            )
            path = step_dir / f"{zone}_channel_terrain.html"
            fig.write_html(path, include_plotlyjs="cdn")
            html_outputs[f"{zone}_terrain"] = path

            xs, ys, Z = build_zone_grid(
                zone_df,
                station_field="global_station_m",
                target_spacing=target_spacing,
            )
            surface = go.Surface(x=xs, y=ys, z=Z, colorscale="Viridis")
            surf_fig = go.Figure(data=[surface])
            surf_fig.update_layout(
                title=f"{zone} Channel Surface",
                scene=dict(
                    xaxis_title="Along-channel (m)",
                    yaxis_title="Distance from Centerline (m)",
                    zaxis_title="Elevation (m)",
                ),
            )
            surf_path = step_dir / f"{zone}_channel_surface.html"
            surf_fig.write_html(surf_path, include_plotlyjs="cdn")
            html_outputs[f"{zone}_surface"] = surf_path

        combined_fig = go.Figure()
        for zone in zones:
            zone_df = corrected_cross_sections[corrected_cross_sections["zone_id"] == zone]
            if zone_df.empty:
                continue
            combined_fig.add_trace(
                go.Scatter3d(
                    x=zone_df["global_station_m"],
                    y=zone_df["distance_from_center_m"],
                    z=zone_df["elevation_m"],
                    mode="markers",
                    name=zone,
                    marker=dict(size=2),
                )
            )
        combined_fig.update_layout(
            title="Main Channel 3D Point Cloud Comparison",
            scene=dict(
                xaxis_title="Along-channel (m)",
                yaxis_title="Distance from Centerline (m)",
                zaxis_title="Elevation (m)",
            ),
        )
        combined_html = step_dir / "combined_channel_terrain.html"
        combined_fig.write_html(combined_html, include_plotlyjs="cdn")
        html_outputs["combined_terrain"] = combined_html

        combined_surface = go.Figure()
        if not corrected_cross_sections.empty:
            xs_all, ys_all, Z_all = build_zone_grid(
                corrected_cross_sections,
                station_field="global_station_m",
                target_spacing=target_spacing,
            )
            combined_surface.add_trace(
                go.Surface(x=xs_all, y=ys_all, z=Z_all, colorscale="Viridis")
            )
        combined_surface.update_layout(
            title="Main Channel 3D Surface",
            scene=dict(
                xaxis_title="Along-channel (m)",
                yaxis_title="Distance from Centerline (m)",
                zaxis_title="Elevation (m)",
            ),
        )
        combined_surface_path = step_dir / "combined_channel_surface.html"
        combined_surface.write_html(combined_surface_path, include_plotlyjs="cdn")
        html_outputs["combined_surface"] = combined_surface_path
    except ImportError:
        logger.warning("Plotly is not installed; skipping interactive HTML generation.")

    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 04 – Channel Profile Preparation")
    builder.add_paragraph(
        "This step samples cross-sections along the main parameter channels, "
        "applies monotonic centreline corrections, and generates visual diagnostics "
        "for hydraulic model calibration."
    )
    builder.add_heading("Processing Parameters", level=2)
    builder.add_list(
        [
            f"Zones: {', '.join(zones)}",
            f"Cross-section spacing: {spacing_m:.1f} m",
            f"Sampling half-width: {half_width_m:.1f} m",
            f"Samples per section: {sample_points}",
            f"Band width: {band_width:.1f} m",
            f"Monotonic tolerance: {tolerance:.2f} m",
        ]
    )
    builder.add_heading("Key Metrics", level=2)
    table_rows = [
        [
            row["zone_id"],
            f"{row['length_m']:.1f}",
            f"{row['corrected_drop_m']:.2f}",
            f"{row['average_slope']:.5f}",
            f"{row['max_adjustment_m']:.2f}",
        ]
        for _, row in profile_summary_df.iterrows()
    ]
    if table_rows:
        builder.add_table(
            TableData(
                headers=["Zone", "Length (m)", "Drop (m)", "Avg Slope", "Max Adjustment (m)"],
                rows=table_rows,
            )
        )
    builder.add_paragraph(f"Channel profile preparation completed at {timestamp.isoformat()}.")
    builder.write(report_path)

    context.config.setdefault("project", {})["last_step04_run"] = timestamp.isoformat()
    dump_project_config(context)
    logger.info("Step 04 – Channel profile preparation completed successfully.")

    outputs: Dict[str, Path] = {
        "segments": segments_path,
        "aggregated_cross_sections": aggregated_cross_path,
        "corrected_cross_sections": corrected_path,
        "centerlines": centerline_path,
        "profile_summary": profile_summary_path,
        "combined_centerline_plot": combined_plot_path,
        "report": report_path,
        "log": log_path,
    }
    for zone, path in static_figures.items():
        outputs[f"{zone}_static"] = path
    for label, path in html_outputs.items():
        outputs[label] = path
    for idx, path in enumerate(created_files):
        outputs[f"cross_section_file_{idx:02d}"] = path
    return outputs


def run_step05_rain_gauge_layout(config_path: Path | str) -> Dict[str, Path]:
    """Placeholder - implemented in step05_rain_gauge_layout.py module."""
    raise NotImplementedError("See step05_rain_gauge_layout.py")
