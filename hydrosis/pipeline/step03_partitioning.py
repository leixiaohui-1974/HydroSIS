"""Step 03: Parameter Partitioning

Create parameter zones and subbasin delineation.
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

def run_step03_partitioning(config_path: Path | str) -> Dict[str, Path]:
    """Partition parameter zones and generate summary artefacts."""

    context = load_project_context(config_path)
    step_index = 3
    step_name = "partitioning"
    step_dir = step_directory(context, step_index, step_name)
    intermediate_dir = step_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    log_path = context.logs_directory / "step03_partitioning.log"
    logger = configure_logger("step03", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 03 – Parameter partitioning started.")
    delineation_section = context.config.get("delineation")
    partition_section = context.config.get("partition")
    model_section = context.config.get("model", {})
    outputs_section = context.config.get("outputs")
    if not isinstance(delineation_section, Mapping) or not isinstance(partition_section, Mapping):
        raise PipelineConfigurationError(
            "Configuration must include 'delineation' and 'partition' sections before running Step 03."
        )

    try:
        import numpy as np
        from hydrosis.config import (
            DelineationConfig,
            ModelStructureConfig,
            OutputArtifactsConfig,
            ParameterPartitionConfig,
        )
        from hydrosis.parameters.partition import partition_parameter_zones

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon as MplPolygon
        from shapely.geometry import shape
    except ImportError as exc:  # pragma: no cover - optional deps
        logger.error("Required dependency missing during partitioning: %s", exc)
        raise

    delineation_cfg = DelineationConfig.from_dict(delineation_section)
    partition_cfg = ParameterPartitionConfig.from_dict(partition_section)
    model_cfg = ModelStructureConfig.from_dict(model_section or {})
    outputs_cfg = OutputArtifactsConfig.from_dict(outputs_section)
    outputs_cfg.enable_figures = True

    def _resolve_inplace(path: Optional[Path]) -> Optional[Path]:
        if path is None:
            return None
        resolved = resolve_input_path(context, path)
        return resolved

    delineation_cfg.dem_path = _resolve_inplace(delineation_cfg.dem_path)  # type: ignore[assignment]
    delineation_cfg.pour_points_path = _resolve_inplace(delineation_cfg.pour_points_path)  # type: ignore[assignment]
    if delineation_cfg.flow_direction_path:
        delineation_cfg.flow_direction_path = _resolve_inplace(delineation_cfg.flow_direction_path)  # type: ignore[assignment]
    if delineation_cfg.flow_accumulation_path:
        delineation_cfg.flow_accumulation_path = _resolve_inplace(delineation_cfg.flow_accumulation_path)  # type: ignore[assignment]
    if delineation_cfg.burn_streams_path:
        delineation_cfg.burn_streams_path = _resolve_inplace(delineation_cfg.burn_streams_path)  # type: ignore[assignment]
    if delineation_cfg.boundaries_path:
        delineation_cfg.boundaries_path = _resolve_inplace(delineation_cfg.boundaries_path)  # type: ignore[assignment]

    if partition_cfg.pour_points_path:
        partition_cfg.pour_points_path = _resolve_inplace(partition_cfg.pour_points_path)  # type: ignore[assignment]

    if not delineation_cfg.flow_direction_path or not delineation_cfg.flow_accumulation_path:
        raise PipelineConfigurationError(
            "Flow direction and accumulation paths must be available before running Step 03."
        )

    parameter_dir = step_dir
    delineation_cfg.intermediate_directory = intermediate_dir
    delineation_cfg.parameter_directory = parameter_dir


    existing_source_entry = partition_section.get("source_directory")
    if existing_source_entry:
        source_dir = resolve_input_path(context, str(existing_source_entry))
        if not source_dir.exists():
            raise FileNotFoundError(f"Configured partition.source_directory not found: {source_dir}")
        source_dir_resolved = source_dir.resolve()
        parameter_dir_resolved = parameter_dir.resolve()
        if source_dir_resolved == parameter_dir_resolved:
            logger.info(
                "Configured partition.source_directory %s matches the target directory; regenerating outputs in place.",
                source_dir,
            )
        else:
            logger.info("Using existing parameter partition outputs from %s.", source_dir)
            shutil.copytree(source_dir, parameter_dir, dirs_exist_ok=True)

            import pandas as pd

            parameter_zones_geojson = parameter_dir / "parameter_zones.geojson"
            parameter_zones_csv = parameter_dir / "parameter_zones.csv"
            parameter_subbasins_geojson = parameter_dir / "parameter_subbasins.geojson"
            parameter_subbasins_csv = parameter_dir / "parameter_subbasins.csv"
            parameter_channels_geojson = parameter_dir / "parameter_channels.geojson"
            parameter_channels_csv = parameter_dir / "parameter_channels.csv"
            subzone_masks_png = parameter_dir / "subzone_masks.png"
            zones_map_png = parameter_dir / "parameter_zones_map.png"
            subbasins_map_png = parameter_dir / "parameter_subbasins_map.png"
            channels_map_png = parameter_dir / "parameter_channels_map.png"
            overview_map_path = parameter_dir / "overview_map.png"

            zones_df = pd.read_csv(parameter_zones_csv)
            subzones_df = pd.read_csv(parameter_subbasins_csv)

            total_zones = len(zones_df)
            total_subzones = len(subzones_df)
            largest_zone = zones_df.sort_values("area_km2", ascending=False).iloc[0] if not zones_df.empty else None

            report_path = build_report_path(context, step_index, step_name)
            builder = MarkdownReportBuilder("Step 03 – Parameter Partitioning")
            builder.add_paragraph("本步骤复用既有参数分区成果，对主要统计与文件结构进行验证。")
            builder.add_heading("总体概览", level=2)
            highlight_items = [
                f"Parameter zones: {total_zones}",
                f"Parameter subzones: {total_subzones}",
            ]
            if largest_zone is not None:
                highlight_items.append(
                    f"最大分区：{largest_zone['zone_id']} ({largest_zone['area_km2']:.2f} km²，{largest_zone['subzone_count']} 个子分区)"
                )
            builder.add_list(highlight_items)

            builder.add_heading("区域预览", level=2)
            preview = zones_df.head(min(6, len(zones_df)))
            if not preview.empty:
                builder.add_table(
                    TableData(
                        headers=["Zone", "Area (km²)", "Subzones", "Runoff Model", "Routing Model"],
                        rows=[
                            [
                                str(row["zone_id"]),
                                f"{row['area_km2']:.2f}",
                                str(row["subzone_count"]),
                                str(row.get("runoff_model", "-")),
                                str(row.get("routing_model", "-")),
                            ]
                            for _, row in preview.iterrows()
                        ],
                    )
                )

            builder.add_heading("成果文件", level=2)
            builder.add_list(
                [
                    f"Parameter zones GeoJSON: `{context.to_relative(parameter_zones_geojson)}`",
                    f"Parameter subbasins GeoJSON: `{context.to_relative(parameter_subbasins_geojson)}`",
                    f"Parameter channels GeoJSON: `{context.to_relative(parameter_channels_geojson)}`",
                ]
            )
            builder.add_paragraph(f"验证时间：{timestamp.isoformat()}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            builder.write(report_path)

            zone_to_subzones = subzones_df.groupby("zone_id")["subzone_id"].apply(list).to_dict()
            model_section = context.config.setdefault("model", {})
            default_runoff = model_section.get("default_runoff_model", "")
            default_routing = model_section.get("default_routing_model", "")
            model_section["parameter_zones"] = [
                {
                    "id": str(row["zone_id"]),
                    "description": f"Parameter zone {row['zone_id']}",
                    "control_points": [],
                    "parameters": {
                        "runoff_model": str(row.get("runoff_model", "") or default_runoff),
                        "routing_model": str(row.get("routing_model", "") or default_routing),
                    },
                    "explicit_subbasins": zone_to_subzones.get(row["zone_id"], []),
                }
                for _, row in zones_df.iterrows()
            ]

            zone_models = {
                row["zone_id"]: {
                    "runoff_model": str(row.get("runoff_model", "") or default_runoff),
                    "routing_model": str(row.get("routing_model", "") or default_routing),
                }
                for _, row in zones_df.iterrows()
            }
            precomputed_entries = []
            for _, row in subzones_df.iterrows():
                models = zone_models.get(row["zone_id"], {})
                downstream = row.get("downstream_subzone_id")
                if isinstance(downstream, float) and math.isnan(downstream):
                    downstream = None
                precomputed_entries.append(
                    {
                        "id": str(row["subzone_id"]),
                        "area_km2": float(row.get("area_km2", 0.0)),
                        "downstream": downstream if downstream else None,
                        "parameters": {
                            "runoff_model": models.get("runoff_model", default_runoff),
                            "routing_model": models.get("routing_model", default_routing),
                        },
                    }
                )
            delineation_section["precomputed_subbasins"] = precomputed_entries
            delineation_section["parameter_directory"] = context.to_relative(parameter_dir)
            delineation_section["intermediate_directory"] = context.to_relative(intermediate_dir)
            context.config.setdefault("project", {})["last_step03_run"] = timestamp.isoformat()
            dump_project_config(context)
            logger.info("Step 03 – Parameter partitioning (existing assets) completed successfully.")

            outputs_map: Dict[str, Path] = {
                "parameter_zones_geojson": parameter_zones_geojson,
                "parameter_zones_csv": parameter_zones_csv,
                "parameter_subbasins_geojson": parameter_subbasins_geojson,
                "parameter_subbasins_csv": parameter_subbasins_csv,
                "parameter_channels_geojson": parameter_channels_geojson,
                "parameter_channels_csv": parameter_channels_csv,
                "report": report_path,
                "log": log_path,
            }
            if subzone_masks_png.exists():
                outputs_map["subzone_masks"] = subzone_masks_png
            if zones_map_png.exists():
                outputs_map["zones_map"] = zones_map_png
            if subbasins_map_png.exists():
                outputs_map["subbasins_map"] = subbasins_map_png
            if channels_map_png.exists():
                outputs_map["channels_map"] = channels_map_png
            if overview_map_path.exists():
                outputs_map["overview_map"] = overview_map_path
            return outputs_map

    logger.info("Calling partition_parameter_zones with parameter directory %s", parameter_dir)
    outputs = partition_parameter_zones(delineation_cfg, partition_cfg, model_cfg, outputs_cfg)

    parameter_zones_geojson = parameter_dir / "parameter_zones.geojson"
    parameter_zones_csv = parameter_dir / "parameter_zones.csv"
    parameter_subbasins_geojson = parameter_dir / "parameter_subbasins.geojson"
    parameter_subbasins_csv = parameter_dir / "parameter_subbasins.csv"
    parameter_channels_geojson = parameter_dir / "parameter_channels.geojson"
    parameter_channels_csv = parameter_dir / "parameter_channels.csv"
    subzone_masks_png = parameter_dir / "subzone_masks.png"
    zones_map_png = parameter_dir / "parameter_zones_map.png"
    subbasins_map_png = parameter_dir / "parameter_subbasins_map.png"
    channels_map_png = parameter_dir / "parameter_channels_map.png"

    if not parameter_zones_geojson.exists():
        raise FileNotFoundError(
            f"Expected parameter zones GeoJSON not found at {parameter_zones_geojson}."
        )

    logger.info("Creating overview map visualisation.")
    overview_map_path = parameter_dir / "overview_map.png"
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    zone_patches: List[MplPolygon] = []
    zone_colors: List[int] = []
    xs_all: List[float] = []
    ys_all: List[float] = []

    for idx, feature in enumerate(outputs.zone_features.get("features", [])):
        geom = feature.get("geometry")
        if not geom:
            continue
        shapely_geom = shape(geom)
        if shapely_geom.is_empty:
            continue
        if shapely_geom.geom_type == "Polygon":
            zone_patches.append(MplPolygon(list(shapely_geom.exterior.coords), closed=True))
            zone_colors.append(idx)
            xs_all.extend([coord[0] for coord in shapely_geom.exterior.coords])
            ys_all.extend([coord[1] for coord in shapely_geom.exterior.coords])
        elif shapely_geom.geom_type == "MultiPolygon":
            for part in shapely_geom.geoms:
                if part.is_empty:
                    continue
                zone_patches.append(MplPolygon(list(part.exterior.coords), closed=True))
                zone_colors.append(idx)
                xs_all.extend([coord[0] for coord in part.exterior.coords])
                ys_all.extend([coord[1] for coord in part.exterior.coords])

    if zone_patches:
        patch_collection = PatchCollection(
            zone_patches, cmap=plt.cm.tab20, alpha=0.65, edgecolor="black", linewidth=0.6
        )
        patch_collection.set_array(np.array(zone_colors))
        ax.add_collection(patch_collection)

    for feature in outputs.channel_features.get("features", []):
        geom = feature.get("geometry")
        if not geom:
            continue
        shapely_geom = shape(geom)
        if shapely_geom.is_empty:
            continue
        if shapely_geom.geom_type == "LineString":
            xs, ys = shapely_geom.xy
            ax.plot(xs, ys, color="black", linewidth=1.2, alpha=0.8)
            xs_all.extend(xs)
            ys_all.extend(ys)
        elif shapely_geom.geom_type == "MultiLineString":
            for line in shapely_geom.geoms:
                xs, ys = line.xy
                ax.plot(xs, ys, color="black", linewidth=1.2, alpha=0.8)
                xs_all.extend(xs)
                ys_all.extend(ys)

    pour_points_features = outputs.pour_point_features.get("features", [])
    if pour_points_features:
        xs_pp = []
        ys_pp = []
        for feature in pour_points_features:
            geom = feature.get("geometry")
            if not geom:
                continue
            shapely_geom = shape(geom)
            if shapely_geom.is_empty:
                continue
            if shapely_geom.geom_type == "Point":
                xs_pp.append(shapely_geom.x)
                ys_pp.append(shapely_geom.y)
        if xs_pp and ys_pp:
            ax.scatter(xs_pp, ys_pp, c="red", edgecolors="white", s=35, linewidths=0.6, zorder=5)
            xs_all.extend(xs_pp)
            ys_all.extend(ys_pp)

    ax.set_title("Parameter Zones Overview")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if xs_all and ys_all:
        ax.set_xlim(min(xs_all), max(xs_all))
        ax.set_ylim(min(ys_all), max(ys_all))
    plt.tight_layout()
    plt.savefig(overview_map_path, dpi=220)
    plt.close()

    logger.info("Composing delineation summary table.")
    subzones_by_zone: Dict[str, List[Dict[str, object]]] = {}
    for row in outputs.subzone_table:
        subzones_by_zone.setdefault(row["zone_id"], []).append(row)

    delineation_summary_path = step_dir / "delineation_summary.csv"
    summary_rows = []
    for zone_row in outputs.zone_table:
        zone_id = str(zone_row["zone_id"])
        subzones = subzones_by_zone.get(zone_id, [])
        total_area = float(zone_row.get("area_km2", 0.0))
        subzone_count = len(subzones)
        mean_subzone_area = total_area / subzone_count if subzone_count else 0.0
        max_subzone_area = max((float(sub["area_km2"]) for sub in subzones), default=0.0)
        summary_rows.append(
            (
                zone_id,
                str(zone_row.get("downstream_id", "") or ""),
                subzone_count,
                total_area,
                mean_subzone_area,
                max_subzone_area,
                str(zone_row.get("runoff_model", "")),
                str(zone_row.get("routing_model", "")),
            )
        )

    write_csv(
        delineation_summary_path,
        (
            "zone_id",
            "downstream_zone",
            "subzone_count",
            "area_km2",
            "mean_subzone_area_km2",
            "max_subzone_area_km2",
            "runoff_model",
            "routing_model",
        ),
        (
            (
                zone_id,
                downstream,
                subzone_count,
                f"{area:.4f}",
                f"{mean_area:.4f}",
                f"{max_area:.4f}",
                runoff,
                routing,
            )
            for zone_id, downstream, subzone_count, area, mean_area, max_area, runoff, routing in summary_rows
        ),
    )

    total_zones = len(summary_rows)
    total_subzones = sum(count for _, _, count, *_ in summary_rows)
    largest_zone = max(summary_rows, key=lambda item: item[3]) if summary_rows else None

    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 03 – Parameter Partitioning")
    builder.add_paragraph(
        "This step converts delineated pour points into parameter zones, subzones, "
        "and channel segments, preparing the configuration for hydrologic modelling."
    )
    builder.add_heading("Highlights", level=2)
    highlight_items = [
        f"Parameter zones: {total_zones}",
        f"Parameter subzones: {total_subzones}",
    ]
    if largest_zone:
        highlight_items.append(
            f"Largest zone: {largest_zone[0]} ({largest_zone[3]:.2f} km², {largest_zone[2]} subzones)"
        )
    builder.add_list(highlight_items)

    builder.add_heading("Zone Overview", level=2)
    preview_rows = [
        [
            zone_id,
            f"{area:.2f}",
            str(subzones),
            runoff or "-",
            routing or "-",
        ]
        for zone_id, _, subzones, area, _, _, runoff, routing in summary_rows[: min(6, len(summary_rows))]
    ]
    if preview_rows:
        builder.add_table(
            TableData(
                headers=["Zone", "Area (km²)", "Subzones", "Runoff Model", "Routing Model"],
                rows=preview_rows,
            )
        )
    builder.add_heading("Artefacts", level=2)
    builder.add_list(
        [
            f"Parameter zones GeoJSON: `{context.to_relative(parameter_zones_geojson)}`",
            f"Parameter subbasins GeoJSON: `{context.to_relative(parameter_subbasins_geojson)}`",
            f"Parameter channels GeoJSON: `{context.to_relative(parameter_channels_geojson)}`",
            f"Delineation summary table: `{context.to_relative(delineation_summary_path)}`",
        ]
    )
    builder.add_paragraph(f"Partitioning executed at {timestamp.isoformat()}.")
    builder.write(report_path)

    logger.info("Updating configuration with new parameter directories.")
    delineation_section["parameter_directory"] = context.to_relative(parameter_dir)
    delineation_section["intermediate_directory"] = context.to_relative(intermediate_dir)
    context.config.setdefault("project", {})["last_step03_run"] = timestamp.isoformat()
    dump_project_config(context)
    logger.info("Step 03 – Parameter partitioning completed successfully.")

    outputs_map: Dict[str, Path] = {
        "parameter_zones_geojson": parameter_zones_geojson,
        "parameter_zones_csv": parameter_zones_csv,
        "parameter_subbasins_geojson": parameter_subbasins_geojson,
        "parameter_subbasins_csv": parameter_subbasins_csv,
        "parameter_channels_geojson": parameter_channels_geojson,
        "parameter_channels_csv": parameter_channels_csv,
        "subzone_masks": subzone_masks_png,
        "zones_map": zones_map_png,
        "subbasins_map": subbasins_map_png,
        "channels_map": channels_map_png,
        "overview_map": overview_map_path,
        "delineation_summary": delineation_summary_path,
        "report": report_path,
        "log": log_path,
    }
    return outputs_map


def run_step04_channel_profile(config_path: Path | str) -> Dict[str, Path]:
