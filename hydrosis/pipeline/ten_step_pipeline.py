"""Ten-step project pipeline utilities for the Upper Truckee workflow.

This module provides reusable helpers plus concrete implementations for the
step-by-step pipeline described in the project brief.  Each step is designed
to be callable from thin CLI wrappers that live under ``scripts/pipeline``.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import datetime as _dt
import logging
import math
import json
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from types import SimpleNamespace

try:  # pragma: no cover - optional dependency for YAML serialisation
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

from hydrosis.reporting.markdown import MarkdownReportBuilder, TableData
from hydrosis.model import Subbasin
from hydrosis.workflow.orchestration import _instantiate_model, _run_model, _flatten_zone_discharge, ScenarioRun
from hydrosis.io.outputs import write_simulation_results
from hydrosis.hydrodynamics import build_zone_geometry, CrossSectionSolver


DEFAULT_RESULTS_ROOT = Path("results/upper_truckee_project")
_LOGGER_NAMESPACE = "hydrosis.pipeline.ten_step"


class PipelineConfigurationError(RuntimeError):
    """Raised when the project configuration is missing required fields."""


@dataclass(slots=True)
class ProjectContext:
    """Runtime view of the project configuration and directory layout."""

    config_path: Path
    config: MutableMapping[str, Any]
    base_results: Path
    config_directory: Path = field(init=False)
    logs_directory: Path = field(init=False)
    reports_directory: Path = field(init=False)

    def __post_init__(self) -> None:
        self.config_directory = self.config_path.parent
        self.base_results = self._normalise_results_root(self.base_results)
        self.logs_directory = self.base_results / "logs"
        self.reports_directory = self.base_results / "reports"
        self.logs_directory.mkdir(parents=True, exist_ok=True)
        self.reports_directory.mkdir(parents=True, exist_ok=True)

    def _normalise_results_root(self, root: Path) -> Path:
        if root.is_absolute():
            return root
        return (self.config_directory / root).resolve()

    def to_relative(self, path: Path) -> str:
        """Return a configuration-friendly string relative to the YAML file."""

        path = path.resolve()
        try:
            rel = path.relative_to(self.config_directory)
            return rel.as_posix()
        except ValueError:
            return path.as_posix()


def load_project_context(config_path: Path | str) -> ProjectContext:
    """Read the YAML configuration and prepare the runtime context."""

    if yaml is None:
        raise ImportError("PyYAML must be installed to use the project pipeline.")

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Project configuration not found: {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project_section = raw_config.get("project", {}) or {}
    results_root = Path(project_section.get("results_root") or DEFAULT_RESULTS_ROOT)
    if "results_root" not in project_section:
        project_section["results_root"] = results_root.as_posix()
        raw_config["project"] = project_section

    context = ProjectContext(
        config_path=config_path,
        config=raw_config,
        base_results=results_root,
    )
    return context


def dump_project_config(context: ProjectContext) -> None:
    """Persist the (potentially modified) YAML configuration to disk."""

    if yaml is None:
        raise ImportError("PyYAML must be installed to serialise the project config.")

    serialised = yaml.safe_dump(
        context.config,
        sort_keys=False,
        allow_unicode=False,
        indent=2,
    )
    context.config_path.write_text(serialised, encoding="utf-8")


def _step_slug(index: int, name: str) -> str:
    return f"{index:02d}_{name}"


def _step_directory(context: ProjectContext, index: int, name: str) -> Path:
    directory = context.base_results / _step_slug(index, name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _configure_logger(step_id: str, log_path: Path) -> logging.Logger:
    """Set up a namespaced logger that writes into the pipeline log directory."""

    logger = logging.getLogger(f"{_LOGGER_NAMESPACE}.{step_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def _build_report_path(context: ProjectContext, index: int, slug: str) -> Path:
    filename = f"step{index:02d}_{slug}.md"
    return context.reports_directory / filename


def _resolve_input_path(context: ProjectContext, path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = context.config_directory / path
    return path.resolve()


def _load_subbasin_geometries(path: Path) -> Dict[str, "BaseGeometry"]:
    import json
    from shapely.geometry import shape
    from shapely.geometry.base import BaseGeometry

    if not path.exists():
        raise FileNotFoundError(f"Parameter subbasin geometry file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    geometries: Dict[str, BaseGeometry] = {}
    for feature in data.get("features", []):
        properties = feature.get("properties", {}) or {}
        subzone_id = str(
            properties.get("subzone_id")
            or properties.get("id")
            or properties.get("zone_id")
            or ""
        ).strip()
        if not subzone_id:
            continue
        geometries[subzone_id] = shape(feature.get("geometry"))

    if not geometries:
        raise ValueError(f"No geometries found in {path}")
    return geometries


def _reset_runoff_initial_states(model_config: "ModelConfig") -> None:
    """Ensure runoff models start from neutral storage states for each pipeline run."""

    defaults = {
        "initial_snow": 0.0,
        "initial_soil": 0.0,
        "initial_upper": 0.0,
        "initial_lower": 0.0,
    }
    for runoff_cfg in model_config.runoff_models:
        model_type = runoff_cfg.model_type.lower()
        if model_type == "hbv":
            for key, value in defaults.items():
                runoff_cfg.parameters.setdefault(key, value)


def _load_base_precipitation_series(path: Path, column: Optional[str] = None) -> "pd.Series":
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Base precipitation series not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Precipitation series file is empty: {path}")

    timestamp_col = df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    if df.empty:
        raise ValueError("Precipitation series contains no data rows.")

    target_column: Optional[str] = column
    if target_column and target_column in df.columns:
        series = df[target_column]
    else:
        numeric_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        if not numeric_columns:
            raise ValueError("No numeric precipitation columns found.")
        series = df[numeric_columns[0]]
        target_column = numeric_columns[0]
    series = series.astype(float)
    series.name = target_column
    return series


def _generate_rain_gauge_inputs(
    base_series: "pd.Series",
    subbasin_geometries: Mapping[str, "BaseGeometry"],
    *,
    station_count: int,
    seed: Optional[int],
    heterogeneity: float,
    min_burst_events: int,
    max_burst_events: int,
) -> "RainGaugeInputs":
    from hydrosis.precipitation import generate_rain_gauge_inputs

    return generate_rain_gauge_inputs(
        base_series,
        subbasin_geometries,
        station_count=station_count,
        rng_seed=seed,
        heterogeneity_strength=heterogeneity,
        min_burst_events=min_burst_events,
        max_burst_events=max_burst_events,
    )


def _compute_basic_stats(values: "np.ndarray") -> Dict[str, float]:
    import numpy as np

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"min": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
    }


def run_step01_dem_preprocessing(config_path: Path | str) -> Dict[str, Path]:
    """Generate flow products, statistics, and a markdown summary for the DEM."""

    context = load_project_context(config_path)
    step_name = "dem_processing"
    step_index = 1
    step_dir = _step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step01_dem_processing.log"
    logger = _configure_logger("step01", log_path)

    logger.info("Step 01 – DEM preprocessing started.")
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    delineation_cfg = context.config.get("delineation")
    if not isinstance(delineation_cfg, Mapping):
        raise PipelineConfigurationError("Configuration is missing the 'delineation' section.")

    dem_path = delineation_cfg.get("dem_path")
    if not dem_path:
        raise PipelineConfigurationError("The delineation.dem_path entry must be configured.")
    dem_path = _resolve_input_path(context, str(dem_path))

    try:
        import numpy as np
        import rasterio
        import richdem as rd

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
    except ImportError as exc:  # pragma: no cover - optional heavy deps
        logger.error("Required dependency missing during DEM preprocessing: %s", exc)
        raise

    figures: Dict[str, Path] = {}
    tables: Dict[str, Path] = {}
    additional_outputs: Dict[str, Path] = {}

    logger.info("Loading DEM from %s", dem_path)
    with rasterio.open(dem_path) as dataset:
        dem_data = dataset.read(1, masked=True)
        transform = dataset.transform
        dem_meta = dataset.meta.copy()
        cell_area_m2 = abs(transform.a * transform.e)
        resolution_x, resolution_y = abs(transform.a), abs(transform.e)
        crs_text = dataset.crs.to_string() if dataset.crs else "Unknown"

    dem_array = np.asarray(dem_data.filled(np.nan), dtype=float)
    valid_mask = np.isfinite(dem_array)
    total_valid_cells = int(np.count_nonzero(valid_mask))
    total_area_km2 = float(total_valid_cells * cell_area_m2 / 1_000_000.0)

    logger.info("Computing flow direction, accumulation, and slope grids.")
    rd_dem = rd.rdarray(dem_array, no_data=np.nan)
    rd_dem.geotransform = (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )
    if dem_meta.get("crs"):
        rd_dem.projection = dem_meta["crs"].to_wkt()

    filled = rd.FillDepressions(rd_dem, in_place=False)
    try:
        flow_direction = rd.FlowDirectionD8(filled, in_place=False)
        flow_accum = rd.FlowAccumulation(flow_direction, method="D8")
    except AttributeError:
        logger.warning(
            "richdem.FlowDirectionD8 unavailable; using FlowProportions-based D8 fallback."
        )
        try:
            proportions = rd.FlowProportions(filled, method="D8")
        except AttributeError as exc:  # pragma: no cover - unexpected missing API
            logger.error("richdem.FlowProportions not available: %s", exc)
            raise

        props_array = np.asarray(proportions, dtype=float)
        if props_array.ndim != 3 or props_array.shape[2] not in (8, 9):
            raise RuntimeError(
                f"Unexpected FlowProportions shape {props_array.shape}; expected (rows, cols, 8/9)."
            )

        if props_array.shape[2] == 9:
            direction_codes = np.array([0, 16, 32, 64, 128, 1, 2, 4, 8], dtype=np.uint8)
        else:
            direction_codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

        positive = np.where(props_array > 0.0, props_array, 0.0)
        max_idx = np.argmax(positive, axis=2)
        max_vals = np.take_along_axis(positive, max_idx[..., None], axis=2)[..., 0]
        flow_dir_array = np.where(max_vals > 0.0, direction_codes[max_idx], 0).astype(np.uint8)

        flow_direction = flow_dir_array
        flow_accum_rd = rd.FlowAccumFromProps(proportions)
        flow_accum = np.asarray(flow_accum_rd, dtype=float)
    else:
        flow_accum = np.asarray(flow_accum, dtype=float)
        flow_direction = np.asarray(flow_direction, dtype=np.uint8)

    slope_radians = rd.TerrainAttribute(filled, attrib="slope_radians")
    slope_degrees = np.degrees(slope_radians, dtype=float)

    logger.info("Writing derived raster products.")
    float_meta = dem_meta.copy()
    float_meta.update({"dtype": "float32", "nodata": math.nan})
    int_meta = dem_meta.copy()
    int_meta.update({"dtype": "int32", "nodata": -1})

    flow_dir_path = step_dir / "flow_direction.tif"
    flow_acc_path = step_dir / "flow_accumulation.tif"
    slope_path = step_dir / "slope_degrees.tif"

    with rasterio.open(flow_dir_path, "w", **int_meta) as dst:
        dst.write(np.asarray(flow_direction, dtype="int32"), 1)

    with rasterio.open(flow_acc_path, "w", **float_meta) as dst:
        dst.write(np.asarray(flow_accum, dtype="float32"), 1)

    with rasterio.open(slope_path, "w", **float_meta) as dst:
        dst.write(np.asarray(slope_degrees, dtype="float32"), 1)

    additional_outputs.update(
        {
            "flow_direction_raster": flow_dir_path,
            "flow_accumulation_raster": flow_acc_path,
            "slope_raster": slope_path,
        }
    )

    logger.info("Generating summary tables.")
    dem_stats = _compute_basic_stats(dem_array)
    dem_stats_rows = [
        ("valid_cells", total_valid_cells, ""),
        ("total_area_km2", total_area_km2, "km^2"),
        ("min_elevation", dem_stats["min"], "m"),
        ("max_elevation", dem_stats["max"], "m"),
        ("mean_elevation", dem_stats["mean"], "m"),
        ("std_elevation", dem_stats["std"], "m"),
        ("median_elevation", dem_stats["median"], "m"),
        ("p90_elevation", dem_stats["p90"], "m"),
        ("p99_elevation", dem_stats["p99"], "m"),
        ("resolution_x", resolution_x, "m"),
        ("resolution_y", resolution_y, "m"),
    ]
    dem_summary_path = step_dir / "dem_summary.csv"
    _write_csv(dem_summary_path, ("metric", "value", "units"), dem_stats_rows)
    tables["dem_summary"] = dem_summary_path

    flow_stats = _compute_basic_stats(np.asarray(flow_accum, dtype=float))
    flow_stat_rows = [
        ("min", flow_stats["min"], ""),
        ("max", flow_stats["max"], ""),
        ("mean", flow_stats["mean"], ""),
        ("median", flow_stats["median"], ""),
        ("p90", flow_stats["p90"], ""),
        ("p99", flow_stats["p99"], ""),
    ]
    flow_stats_path = step_dir / "flow_accumulation_stats.csv"
    _write_csv(flow_stats_path, ("statistic", "value", "units"), flow_stat_rows)
    tables["flow_accumulation"] = flow_stats_path

    logger.info("Rendering figures for DEM derivatives.")
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(
        np.where(valid_mask, dem_array, np.nan),
        vert_exag=1.0,
        dx=resolution_x,
        dy=resolution_y,
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(hillshade, cmap="gray", origin="upper")
    plt.title("DEM Hillshade")
    plt.axis("off")
    dem_hillshade_path = step_dir / "dem_hillshade.png"
    plt.tight_layout()
    plt.savefig(dem_hillshade_path, dpi=200)
    plt.close()
    figures["dem_hillshade"] = dem_hillshade_path

    plt.figure(figsize=(8, 6))
    plt.imshow(np.asarray(flow_direction, dtype=float), cmap="viridis", origin="upper")
    plt.title("Flow Direction (D8)")
    plt.colorbar(label="D8 Code")
    plt.tight_layout()
    flow_direction_fig_path = step_dir / "flow_direction.png"
    plt.savefig(flow_direction_fig_path, dpi=200)
    plt.close()
    figures["flow_direction"] = flow_direction_fig_path

    log_accum = np.log1p(np.asarray(flow_accum, dtype=float))
    plt.figure(figsize=(8, 6))
    plt.imshow(log_accum, cmap="inferno", origin="upper")
    plt.title("Log Flow Accumulation")
    plt.colorbar(label="log(1 + accumulation)")
    plt.tight_layout()
    flow_accum_fig_path = step_dir / "flow_accumulation.png"
    plt.savefig(flow_accum_fig_path, dpi=200)
    plt.close()
    figures["flow_accumulation"] = flow_accum_fig_path

    # Simple watershed mask visualisation based on accumulation quantiles.
    finite_log = log_accum[np.isfinite(log_accum)]
    if finite_log.size >= 5:
        quantile_edges = np.quantile(finite_log, [0.2, 0.4, 0.6, 0.8])
        classes = np.digitize(log_accum, quantile_edges, right=True)
    else:
        classes = np.zeros_like(log_accum, dtype=int)
    plt.figure(figsize=(8, 6))
    plt.imshow(classes, cmap="tab20", origin="upper")
    plt.title("Accumulation-based Watershed Mask")
    plt.axis("off")
    plt.tight_layout()
    watershed_mask_path = step_dir / "watershed_masks.png"
    plt.savefig(watershed_mask_path, dpi=200)
    plt.close()
    figures["watershed_masks"] = watershed_mask_path

    logger.info("Composing markdown report for Step 01.")
    report_path = _build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 01 – DEM Preprocessing")
    builder.add_paragraph(
        "This step derives primary hydrologic rasters from the project DEM and "
        "computes descriptive statistics that inform downstream delineation thresholds."
    )
    builder.add_heading("Inputs", level=2)
    builder.add_list(
        [
            f"DEM path: `{context.to_relative(dem_path)}`",
            f"Projection: `{crs_text}`",
            f"Cell size: {resolution_x:.2f} m × {resolution_y:.2f} m",
        ]
    )
    builder.add_heading("Key Metrics", level=2)
    builder.add_table(
        TableData(
            headers=["Metric", "Value", "Units"],
            rows=[[name, f"{value:.3f}", units] for name, value, units in dem_stats_rows],
        )
    )
    builder.add_heading("Derived Artefacts", level=2)
    builder.add_list(
        [
            f"Flow direction raster: `{context.to_relative(flow_dir_path)}`",
            f"Flow accumulation raster: `{context.to_relative(flow_acc_path)}`",
            f"Slope raster: `{context.to_relative(slope_path)}`",
        ]
    )
    builder.add_paragraph(f"Report generated at {timestamp.isoformat()}.")
    builder.write(report_path)

    logger.info("Updating configuration with derived raster paths.")
    delineation_cfg = context.config.setdefault("delineation", {})
    delineation_cfg["flow_direction_path"] = context.to_relative(flow_dir_path)
    delineation_cfg["flow_accumulation_path"] = context.to_relative(flow_acc_path)
    delineation_cfg.setdefault(
        "intermediate_directory",
        context.to_relative(context.base_results / "03_partitioning" / "intermediate"),
    )
    context.config.setdefault("project", {})["last_step01_run"] = timestamp.isoformat()

    dump_project_config(context)
    logger.info("Step 01 – DEM preprocessing completed successfully.")

    outputs: Dict[str, Path] = {}
    outputs.update(figures)
    outputs.update(tables)
    outputs.update(additional_outputs)
    outputs["report"] = report_path
    outputs["log"] = log_path
    return outputs


def run_step02_pour_points(config_path: Path | str) -> Dict[str, Path]:
    """Derive channel-aware pour points and accompanying diagnostics."""

    context = load_project_context(config_path)
    step_index = 2
    step_name = "pour_points"
    step_dir = _step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step02_pour_points.log"
    logger = _configure_logger("step02", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 02 – Pour point extraction started.")
    delineation_cfg = context.config.get("delineation")
    if not isinstance(delineation_cfg, MutableMapping):
        raise PipelineConfigurationError("Configuration is missing the 'delineation' section.")

    flow_acc_entry = delineation_cfg.get("flow_accumulation_path")
    flow_dir_entry = delineation_cfg.get("flow_direction_path")
    dem_entry = delineation_cfg.get("dem_path")
    if not flow_acc_entry or not flow_dir_entry:
        raise PipelineConfigurationError(
            "Run Step 01 first to populate flow accumulation and direction paths."
        )
    if not dem_entry:
        raise PipelineConfigurationError("The delineation.dem_path entry must be configured.")

    flow_acc_path = _resolve_input_path(context, str(flow_acc_entry))
    flow_dir_path = _resolve_input_path(context, str(flow_dir_entry))
    dem_path = _resolve_input_path(context, str(dem_entry))

    if not flow_acc_path.exists() or not flow_dir_path.exists():
        raise FileNotFoundError(
            "Flow accumulation or direction rasters not found. Ensure Step 01 completed successfully."
        )

    project_cfg = context.config.get("project", {})
    pour_cfg = project_cfg.get("pour_points", {})
    count = int(pour_cfg.get("count", 8))
    min_distance = int(pour_cfg.get("min_distance_cells", 30))
    main_fraction = float(pour_cfg.get("main_stem_fraction", 0.5))
    main_fraction = max(0.0, min(1.0, main_fraction))
    area_tolerance = float(pour_cfg.get("area_balance_tolerance", 0.35))
    accumulation_threshold = float(delineation_cfg.get("accumulation_threshold", 0.0))

    logger.info(
        "Generating %d pour points (min_distance=%d, main_fraction=%.2f, area_tol=%.2f).",
        count,
        min_distance,
        main_fraction,
        area_tolerance,
    )

    try:
        import numpy as np
        import rasterio

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource

        from hydrosis.delineation import utils as dutils
    except ImportError as exc:  # pragma: no cover - depends on optional libs
        logger.error("Required dependency missing during pour point generation: %s", exc)
        raise

    geojson_path = step_dir / "pour_points.geojson"
    accumulation_fig_path = step_dir / "accumulation_overview.png"
    source_entry = pour_cfg.get("source_geojson")
    use_existing = False
    pour_points: List[Any]

    # Load transform early in case we need it for coordinate conversion
    with rasterio.open(flow_acc_path) as acc_ds:
        transform = acc_ds.transform

    if source_entry:
        source_path = _resolve_input_path(context, str(source_entry))
        if source_path.exists():
            logger.info("Using existing pour point dataset from %s.", source_path)
            geojson_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            data = json.loads(source_path.read_text(encoding="utf-8"))
            pour_points = []
            for feature in data.get("features", []):
                props = feature.get("properties", {}) or {}
                coord = feature.get("geometry", {}).get("coordinates", [math.nan, math.nan])
                pid = str(
                    props.get("id")
                    or props.get("pour_id")
                    or props.get("name")
                    or f"P{len(pour_points)+1}"
                )
                row = props.get("row")
                col = props.get("col")
                if row is None or col is None:
                    x, y = coord
                    col_val, row_val = ~transform * (x, y)
                    row = int(round(row_val))
                    col = int(round(col_val))
                accumulation_value = props.get("accumulation")
                if accumulation_value is None:
                    accumulation_value = 0.0
                pour_points.append(
                    SimpleNamespace(
                        id=pid,
                        x=float(coord[0]),
                        y=float(coord[1]),
                        row=int(row),
                        col=int(col),
                        accumulation=float(accumulation_value),
                    )
                )
            count = len(pour_points)
            use_existing = True
        else:
            logger.warning(
                "Configured pour_points.source_geojson %s not found; generating automatically.",
                source_path,
            )

    if not use_existing:
        pour_points = dutils.generate_tree_pour_points(
            flow_acc_path,
            flow_dir_path,
            count=count,
            accumulation_threshold=accumulation_threshold,
            min_distance_cells=min_distance,
            main_stem_fraction=main_fraction,
            area_balance_tolerance=area_tolerance,
            output_geojson=geojson_path,
            accumulation_plot=accumulation_fig_path,
        )

    if not pour_points:
        raise RuntimeError("Pour point generation produced no outputs.")

    # Read accumulation data (transform was already loaded earlier)
    with rasterio.open(flow_acc_path) as acc_ds:
        accumulation = acc_ds.read(1)
    with rasterio.open(dem_path) as dem_ds:
        dem_array = dem_ds.read(1, masked=True).filled(np.nan)
        dem_transform = dem_ds.transform

    cell_area_km2 = abs(transform.a * transform.e) / 1_000_000.0
    accumulation = np.where(np.isfinite(accumulation), accumulation, 0.0)
    log_accum = np.log1p(accumulation)

    areas = [(pp.accumulation + 1.0) * cell_area_km2 for pp in pour_points]
    ranked_indices = sorted(range(len(pour_points)), key=lambda idx: areas[idx], reverse=True)
    main_count = max(1, int(round(len(pour_points) * main_fraction))) if pour_points else 0
    main_ids = {pour_points[idx].id for idx in ranked_indices[:main_count]}

    table_rows = []
    for pp, area in zip(pour_points, areas):
        role = "main_stem" if pp.id in main_ids else "tributary"
        table_rows.append(
            (
                pp.id,
                pp.x,
                pp.y,
                pp.row,
                pp.col,
                pp.accumulation,
                area,
                role,
            )
        )

    table_path = step_dir / "pour_points_table.csv"
    csv_rows = [
        (
            row_id,
            f"{x:.3f}",
            f"{y:.3f}",
            row,
            col,
            f"{acc:.1f}",
            f"{area:.4f}",
            role,
        )
        for row_id, x, y, row, col, acc, area, role in table_rows
    ]
    _write_csv(
        table_path,
        ("id", "x", "y", "row", "col", "accumulation", "estimated_area_km2", "channel_role"),
        csv_rows,
    )

    logger.info("Rendering pour point overview map.")
    ls = LightSource(azdeg=315, altdeg=45)
    dem_valid = np.where(np.isfinite(dem_array), dem_array, np.nan)
    if np.isfinite(dem_valid).sum() == 0:
        hillshade_input = np.zeros_like(dem_array)
    else:
        fill_value = float(np.nanmedian(dem_valid))
        hillshade_input = np.where(np.isfinite(dem_valid), dem_valid, fill_value)
    hillshade = ls.hillshade(
        hillshade_input,
        vert_exag=1.0,
        dx=abs(dem_transform.a),
        dy=abs(dem_transform.e),
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(hillshade, cmap="gray", origin="upper")
    plt.imshow(log_accum, cmap="inferno", alpha=0.4, origin="upper")

    if areas:
        min_area = min(areas)
        max_area = max(areas)
        if max_area - min_area < 1e-9:
            sizes = [120.0] * len(areas)
        else:
            sizes = np.interp(areas, (min_area, max_area), (60.0, 240.0)).tolist()
    else:
        sizes = [120.0] * len(pour_points)

    main_cols = [pp.col for pp in pour_points if pp.id in main_ids]
    main_rows = [pp.row for pp in pour_points if pp.id in main_ids]
    main_sizes = [sizes[idx] for idx, pp in enumerate(pour_points) if pp.id in main_ids]

    trib_cols = [pp.col for pp in pour_points if pp.id not in main_ids]
    trib_rows = [pp.row for pp in pour_points if pp.id not in main_ids]
    trib_sizes = [sizes[idx] for idx, pp in enumerate(pour_points) if pp.id not in main_ids]

    if main_cols:
        plt.scatter(
            main_cols,
            main_rows,
            s=main_sizes,
            c="dodgerblue",
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
            label="Main stem",
        )
    if trib_cols:
        plt.scatter(
            trib_cols,
            trib_rows,
            s=trib_sizes,
            c="goldenrod",
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            label="Tributary",
        )

    for pp in pour_points:
        plt.text(pp.col + 2, pp.row + 2, pp.id, color="white", fontsize=8, ha="left", va="bottom")

    plt.title("Generated Pour Points (Main Stem vs. Tributary)")
    plt.axis("off")
    if main_cols or trib_cols:
        plt.legend(loc="upper right")
    plt.tight_layout()
    pour_map_path = step_dir / "pour_points_map.png"
    plt.savefig(pour_map_path, dpi=200)
    plt.close()

    total_area = sum(area for _, _, _, _, _, _, area, _ in table_rows)
    main_area = sum(area for _, _, _, _, _, _, area, role in table_rows if role == "main_stem")
    tributary_area = total_area - main_area

    report_path = _build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 02 – Pour Point Extraction")
    builder.add_paragraph(
        "This step analyses the flow accumulation tree to identify main-stem and "
        "tributary pour points that meet spacing and area-balance requirements."
    )
    builder.add_heading("Inputs", level=2)
    builder.add_list(
        [
            f"Flow accumulation raster: `{context.to_relative(flow_acc_path)}`",
            f"Flow direction raster: `{context.to_relative(flow_dir_path)}`",
            f"DEM: `{context.to_relative(dem_path)}`",
        ]
    )
    builder.add_heading("Parameters", level=2)
    builder.add_list(
        [
            f"Target count: {count}",
            f"Minimum cell spacing: {min_distance}",
            f"Main stem fraction: {main_fraction:.2f}",
            f"Area balance tolerance: {area_tolerance:.2f}",
        ]
    )
    builder.add_heading("Area Summary", level=2)
    builder.add_table(
        TableData(
            headers=["Category", "Estimated Area (km²)"],
            rows=[
                ["Main stem", f"{main_area:.3f}"],
                ["Tributaries", f"{tributary_area:.3f}"],
                ["Total", f"{total_area:.3f}"],
            ],
        )
    )
    builder.add_paragraph(f"Pour points generated at {timestamp.isoformat()}.")
    builder.write(report_path)

    logger.info("Updating configuration with the new pour point layer.")
    delineation_cfg["pour_points_path"] = context.to_relative(geojson_path)
    partition_cfg = context.config.setdefault("partition", {})
    partition_cfg["pour_points_path"] = context.to_relative(geojson_path)
    context.config.setdefault("project", {})["last_step02_run"] = timestamp.isoformat()

    dump_project_config(context)
    logger.info("Step 02 – Pour point extraction completed successfully.")

    outputs: Dict[str, Path] = {
        "geojson": geojson_path,
        "table": table_path,
        "map": pour_map_path,
        "report": report_path,
        "log": log_path,
        "accumulation_overview": accumulation_fig_path,
    }
    return outputs


def _extract_cross_sections(
    geojson_path: Path,
    dem_path: Path,
    zones: Sequence[str],
    *,
    spacing_m: float,
    half_width_m: float,
    n_points: int,
    output_dir: Path,
    logger: logging.Logger,
) -> "tuple[list[Path], 'pd.DataFrame']":
    import json
    import math
    from typing import List, Tuple

    import numpy as np
    import pandas as pd
    import rasterio
    from shapely.geometry import LineString, shape

    output_dir.mkdir(parents=True, exist_ok=True)
    with geojson_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    created: List[Path] = []
    frames: List[pd.DataFrame] = []
    zone_filter = set(zones)

    logger.info(
        "Extracting cross sections from %s (zones=%s, spacing=%.1f m, half_width=%.1f m, samples=%d).",
        geojson_path,
        ", ".join(zones),
        spacing_m,
        half_width_m,
        n_points,
    )

    with rasterio.open(dem_path) as dem:
        for feature in data.get("features", []):
            props = feature.get("properties") or {}
            zone_id = str(props.get("zone_id") or props.get("ZoneID") or "").strip()
            if zones and zone_id not in zone_filter:
                continue
            segment_id = props.get("segment_id") or props.get("subzone_id")
            if not segment_id:
                continue
            segment_id = str(segment_id)

            geom = shape(feature.get("geometry"))
            if not isinstance(geom, LineString):
                continue
            coords = list(geom.coords)
            if len(coords) < 2:
                continue

            cumulative = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
            )
            total_length = cumulative[-1]
            if total_length <= 0.0:
                continue

            n_sections = max(1, int(total_length // max(spacing_m, 1.0)))
            distances = np.linspace(0.0, total_length, n_sections)
            section_frames: List[pd.DataFrame] = []
            for dist in distances:
                idx = np.searchsorted(cumulative, dist, side="right") - 1
                idx = max(0, min(idx, len(coords) - 2))
                start = np.array(coords[idx])
                end = np.array(coords[idx + 1])
                segment_length = max(cumulative[idx + 1] - cumulative[idx], 1e-6)
                fraction = (dist - cumulative[idx]) / segment_length
                centre = start + (end - start) * fraction

                direction = end - start
                norm = np.linalg.norm(direction)
                if norm == 0.0:
                    continue
                tangent = direction / norm
                normal = np.array([-tangent[1], tangent[0]])

                start_pt = centre - normal * half_width_m
                end_pt = centre + normal * half_width_m

                xs = np.linspace(start_pt[0], end_pt[0], n_points)
                ys = np.linspace(start_pt[1], end_pt[1], n_points)
                elevations = []
                for x, y in zip(xs, ys):
                    sample = next(dem.sample([(float(x), float(y))]), [math.nan])
                    elevations.append(float(sample[0]))
                offsets = np.linspace(-half_width_m, half_width_m, n_points)

                df = pd.DataFrame(
                    {
                        "zone_id": zone_id,
                        "segment_id": segment_id,
                        "subzone_id": props.get("subzone_id"),
                        "station_m": dist,
                        "distance_from_center_m": offsets,
                        "elevation_m": elevations,
                        "x": xs,
                        "y": ys,
                    }
                )
                section_frames.append(df)

            if section_frames:
                profile_df = pd.concat(section_frames, ignore_index=True)
                file_path = output_dir / f"{segment_id}_cross_sections.csv"
                profile_df.to_csv(file_path, index=False)
                created.append(file_path)
                frames.append(profile_df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return created, combined


def _resolve_main_path(zone_df: "pd.DataFrame") -> "list[str]":
    import pandas as pd

    def _parse_upstream(value: object, valid_segments: set[str]) -> list[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return []
        text = str(value)
        if not text:
            return []
        candidates = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]
        return [item for item in candidates if item in valid_segments]

    segments = set(zone_df["segment_id"])
    length_map = {row["segment_id"]: float(row["length_m"]) for _, row in zone_df.iterrows()}
    downstream_map = {
        row["segment_id"]: row["downstream_id"] if row["downstream_id"] in segments else None
        for _, row in zone_df.iterrows()
    }
    upstream_map = {
        row["segment_id"]: _parse_upstream(row.get("upstream_ids"), segments)
        for _, row in zone_df.iterrows()
    }

    outlet_candidates = [seg for seg, downstream in downstream_map.items() if downstream is None]
    if not outlet_candidates:
        outlet_candidates = [
            row["segment_id"]
            for _, row in zone_df.iterrows()
            if not row.get("downstream_id") or str(row.get("downstream_id")).startswith("P") is False
        ]
    if not outlet_candidates:
        outlet_candidates = [zone_df.iloc[0]["segment_id"]]
    outlet = outlet_candidates[0]

    memo: Dict[str, Tuple[float, List[str]]] = {}

    def longest_path(seg: str) -> Tuple[float, List[str]]:
        if seg in memo:
            return memo[seg]
        ups = upstream_map.get(seg, [])
        if not ups:
            result = (length_map.get(seg, 0.0), [seg])
        else:
            best_length = -1.0
            best_path: List[str] = []
            for upstream in ups:
                total, path = longest_path(upstream)
                if total > best_length:
                    best_length = total
                    best_path = path
            result = (best_length + length_map.get(seg, 0.0), best_path + [seg])
        memo[seg] = result
        return result

    _, path = longest_path(outlet)
    return path


def _summarise_main_channels(
    channel_csv: Path,
    zones: Sequence[str],
    cross_sections: "pd.DataFrame",
    output_dir: Path,
) -> "tuple[Path, Path, 'pd.DataFrame', 'pd.DataFrame', Dict[str, float]]":
    import pandas as pd

    channel_df = pd.read_csv(channel_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[Dict[str, object]] = []
    aggregated_frames: list[pd.DataFrame] = []
    zone_lengths: Dict[str, float] = {}

    for zone in zones:
        zone_df = channel_df[channel_df["zone_id"] == zone].copy()
        if zone_df.empty:
            continue
        path_segments = _resolve_main_path(zone_df)
        cumulative = 0.0
        for seg in path_segments:
            row = zone_df[zone_df["segment_id"] == seg].iloc[0]
            length = float(row["length_m"])
            start_offset = cumulative
            cumulative += length
            summary_rows.append(
                {
                    "zone_id": zone,
                    "segment_id": seg,
                    "subzone_id": row["subzone_id"],
                    "length_m": length,
                    "slope": float(row.get("slope", 0.0)),
                    "drop_m": float(row.get("drop_m", 0.0)),
                    "downstream_id": row.get("downstream_id"),
                    "upstream_ids": row.get("upstream_ids"),
                    "segment_start_m": start_offset,
                    "cumulative_length_m": cumulative,
                }
            )

            seg_sections = cross_sections[cross_sections["segment_id"] == seg].copy()
            if not seg_sections.empty:
                seg_sections["zone_id"] = zone
                seg_sections["segment_id"] = seg
                seg_sections["station_global_m"] = seg_sections["station_m"].astype(float) + start_offset
                aggregated_frames.append(seg_sections)
        zone_lengths[zone] = cumulative

    summary_df = pd.DataFrame(summary_rows)
    segments_path = output_dir / "main_channel_segments.csv"
    summary_df.to_csv(segments_path, index=False)

    if aggregated_frames:
        cross_df = pd.concat(aggregated_frames, ignore_index=True)
    else:
        cross_df = pd.DataFrame(
            columns=[
                "zone_id",
                "segment_id",
                "station_m",
                "station_global_m",
                "distance_from_center_m",
                "elevation_m",
                "x",
                "y",
            ]
        )
    cross_sections_path = output_dir / "main_channel_cross_sections.csv"
    cross_df.to_csv(cross_sections_path, index=False)

    return segments_path, cross_sections_path, summary_df, cross_df, zone_lengths


def run_step03_partitioning(config_path: Path | str) -> Dict[str, Path]:
    """Partition parameter zones and generate summary artefacts."""

    context = load_project_context(config_path)
    step_index = 3
    step_name = "partitioning"
    step_dir = _step_directory(context, step_index, step_name)
    intermediate_dir = step_dir / "intermediate"
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    log_path = context.logs_directory / "step03_partitioning.log"
    logger = _configure_logger("step03", log_path)
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
        resolved = _resolve_input_path(context, path)
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
        source_dir = _resolve_input_path(context, str(existing_source_entry))
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

            report_path = _build_report_path(context, step_index, step_name)
            builder = MarkdownReportBuilder("Step 03 – Parameter Partitioning")
            builder.add_paragraph(
                "This step reuses existing parameter zone results and validates key statistics and file structure."
            )
            builder.add_heading("Overview", level=2)
            highlight_items = [
                f"Parameter zones: {total_zones}",
                f"Parameter subzones: {total_subzones}",
            ]
            if largest_zone is not None:
                highlight_items.append(
                    f"Largest zone: {largest_zone['zone_id']} ({largest_zone['area_km2']:.2f} km², {largest_zone['subzone_count']} subzones)"
                )
            builder.add_list(highlight_items)

            builder.add_heading("Zone Preview", level=2)
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

            builder.add_heading("Output Files", level=2)
            builder.add_list(
                [
                    f"Parameter zones GeoJSON: `{context.to_relative(parameter_zones_geojson)}`",
                    f"Parameter subbasins GeoJSON: `{context.to_relative(parameter_subbasins_geojson)}`",
                    f"Parameter channels GeoJSON: `{context.to_relative(parameter_channels_geojson)}`",
                ]
            )
            builder.add_paragraph(f"Validation time: {timestamp.isoformat()}")
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

    _write_csv(
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

    report_path = _build_report_path(context, step_index, step_name)
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
    """Prepare channel profiles, cross-sections, and visualisations for key zones."""

    context = load_project_context(config_path)
    step_index = 4
    step_name = "channel_profile"
    step_dir = _step_directory(context, step_index, step_name)
    cross_section_dir = step_dir / "channel_cross_sections"
    cross_section_dir.mkdir(parents=True, exist_ok=True)
    log_path = context.logs_directory / "step04_channel_profile.log"
    logger = _configure_logger("step04", log_path)
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

    dem_path = _resolve_input_path(context, dem_entry)
    parameter_dir = _resolve_input_path(context, parameter_dir_entry)
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

    report_path = _build_report_path(context, step_index, step_name)
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
    """Generate spatial layout for rain gauges based on project settings."""

    context = load_project_context(config_path)
    step_index = 5
    step_name = "rain_gauge_layout"
    step_dir = _step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step05_rain_gauge_layout.log"
    logger = _configure_logger("step05", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 05 – Rain gauge layout started.")

    project_cfg = context.config.setdefault("project", {})
    gauge_cfg = project_cfg.setdefault("rain_gauge", {})
    rainfall_cfg = project_cfg.setdefault("rainfall", {})
    delineation_cfg = context.config.get("delineation") or {}
    parameter_dir_entry = delineation_cfg.get("parameter_directory")
    if not parameter_dir_entry:
        raise PipelineConfigurationError(
            "Parameter directory not configured. Run Step 03 before Step 05."
        )

    parameter_dir = _resolve_input_path(context, parameter_dir_entry)
    subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    subbasin_geometries = _load_subbasin_geometries(subbasin_geojson)

    default_base_series = rainfall_cfg.get(
        "base_series_path", "results/upper_truckee_channel_demo/storm_forcing.csv"
    )
    rainfall_cfg["base_series_path"] = default_base_series
    base_series_path = _resolve_input_path(context, default_base_series)
    base_column = rainfall_cfg.get("base_column")

    import json
    import numpy as np
    import pandas as pd
    from shapely.geometry import mapping
    from shapely.ops import unary_union

    base_series = _load_base_precipitation_series(base_series_path, column=base_column)
    if base_column is None:
        rainfall_cfg["base_column"] = base_series.name

    station_count = int(gauge_cfg.get("station_count", 10))
    heterogeneity = float(gauge_cfg.get("heterogeneity", 0.4))
    min_burst_events = int(gauge_cfg.get("min_burst_events", 1))
    max_burst_events = int(gauge_cfg.get("max_burst_events", 3))
    if max_burst_events < min_burst_events:
        max_burst_events = min_burst_events
    gauge_cfg["station_count"] = station_count
    gauge_cfg["heterogeneity"] = heterogeneity
    gauge_cfg["min_burst_events"] = min_burst_events
    gauge_cfg["max_burst_events"] = max_burst_events

    seed = gauge_cfg.get("seed")
    if seed is None:
        seed = 42
        gauge_cfg["seed"] = seed
    else:
        seed = int(seed)

    logger.info(
        "Generating rain gauge layout (stations=%d, seed=%d).", station_count, seed
    )
    rain_inputs = _generate_rain_gauge_inputs(
        base_series,
        subbasin_geometries,
        station_count=station_count,
        seed=seed,
        heterogeneity=heterogeneity,
        min_burst_events=min_burst_events,
        max_burst_events=max_burst_events,
    )

    locations_geojson = step_dir / "rain_gauge_locations.geojson"
    layout_csv = step_dir / "rain_gauge_layout.csv"
    coverage_csv = step_dir / "station_coverage.csv"
    polygons_geojson = step_dir / "rain_gauge_thiessen_polygons.geojson"
    layout_map_path = step_dir / "rain_gauge_layout_map.png"

    features = []
    for station_id, point in rain_inputs.station_positions.items():
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(point),
                "properties": {"station_id": station_id},
            }
        )
    locations_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )

    import math
    import numpy as np

    coverage_rows: list[tuple[str, float, float, float, float]] = []
    for station_id, polygon in rain_inputs.thiessen_polygons.items():
        area_m2 = float(polygon.area)
        area_km2 = area_m2 / 1_000_000.0
        radius_m = math.sqrt(area_m2 / math.pi) if area_m2 > 0 else 0.0
        coverage_rows.append(
            (
                station_id,
                rain_inputs.station_positions[station_id].x,
                rain_inputs.station_positions[station_id].y,
                area_km2,
                radius_m,
            )
        )

    layout_df = pd.DataFrame(
        coverage_rows,
        columns=["station_id", "x", "y", "coverage_area_km2", "coverage_radius_m"],
    )
    layout_df.to_csv(layout_csv, index=False)

    layout_df[["station_id", "coverage_radius_m"]].to_csv(coverage_csv, index=False)

    polygon_features = []
    for station_id, polygon in rain_inputs.thiessen_polygons.items():
        polygon_features.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {"station_id": station_id},
            }
        )
    polygons_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": polygon_features}, indent=2),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as MplPolygon

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    for geom in subbasin_geometries.values():
        if geom.geom_type == "Polygon":
            patch = MplPolygon(np.asarray(geom.exterior.coords), facecolor="#e0f3ff", edgecolor="#4a90e2", alpha=0.4, linewidth=0.6)
            ax.add_patch(patch)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                patch = MplPolygon(np.asarray(part.exterior.coords), facecolor="#e0f3ff", edgecolor="#4a90e2", alpha=0.4, linewidth=0.6)
                ax.add_patch(patch)

    for station_id, row in layout_df.iterrows():
        circle = Circle(
            (row["x"], row["y"]),
            radius=row["coverage_radius_m"],
            facecolor="none",
            edgecolor="orange",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.add_patch(circle)
        ax.scatter(row["x"], row["y"], c="crimson", s=40, edgecolors="white", linewidths=0.8)
        ax.text(row["x"], row["y"], row["station_id"], fontsize=8, color="black", ha="left", va="bottom")

    try:
        extent_geom = unary_union(list(subbasin_geometries.values()))
        minx, miny, maxx, maxy = extent_geom.bounds
    except Exception:  # pragma: no cover - fallback when geometry union fails
        all_points = np.array([[pt.x, pt.y] for pt in rain_inputs.station_positions.values()])
        if all_points.size:
            minx, miny = all_points[:, 0].min(), all_points[:, 1].min()
            maxx, maxy = all_points[:, 0].max(), all_points[:, 1].max()
        else:
            minx = miny = maxx = maxy = 0.0

    pad = max((maxx - minx), (maxy - miny)) * 0.05 + 1000.0
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Rain Gauge Layout")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(layout_map_path, dpi=220)
    plt.close()

    report_path = _build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 05 – Rain Gauge Layout")
    builder.add_paragraph(
        "Generate rain gauge layout based on reference precipitation sequence and subbasin geometry, "
        "outputting station coverage radius statistics and visualization."
    )
    builder.add_heading("Configuration Parameters", level=2)
    builder.add_list(
        [
            f"Station count: {station_count}",
            f"Random seed: {seed}",
            f"Spatial heterogeneity: {heterogeneity:.2f}",
            f"Burst event range: {min_burst_events}–{max_burst_events}",
        ]
    )
    builder.add_heading("Coverage Statistics", level=2)
    summary_table = TableData(
        headers=["Station", "Coverage Area (km²)", "Equivalent Radius (m)"],
        rows=[
            [
                row["station_id"],
                f"{row['coverage_area_km2']:.2f}",
                f"{row['coverage_radius_m']:.1f}",
            ]
            for _, row in layout_df.head(10).iterrows()
        ],
    )
    builder.add_table(summary_table)
    builder.add_paragraph(f"Layout completion time: {timestamp.isoformat()}")
    builder.write(report_path)

    gauge_cfg["layout_geojson"] = context.to_relative(locations_geojson)
    gauge_cfg["layout_csv"] = context.to_relative(layout_csv)
    gauge_cfg["thiessen_geojson"] = context.to_relative(polygons_geojson)
    project_cfg["last_step05_run"] = timestamp.isoformat()

    dump_project_config(context)
    logger.info("Step 05 – Rain gauge layout completed successfully.")

    return {
        "locations_geojson": locations_geojson,
        "layout_csv": layout_csv,
        "coverage_csv": coverage_csv,
        "thiessen_geojson": polygons_geojson,
        "layout_map": layout_map_path,
        "report": report_path,
        "log": log_path,
    }


def run_step06_rain_sequence(config_path: Path | str) -> Dict[str, Path]:
    """Generate synthetic rainfall time series for rain gauges and aggregate forcing."""

    context = load_project_context(config_path)
    step_index = 6
    step_name = "rain_sequence"
    step_dir = _step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step06_rain_sequence.log"
    logger = _configure_logger("step06", log_path)
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

    parameter_dir = _resolve_input_path(context, parameter_dir_entry)
    subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    subbasin_geometries = _load_subbasin_geometries(subbasin_geojson)

    base_series_path = rainfall_cfg.get(
        "base_series_path", "results/upper_truckee_channel_demo/storm_forcing.csv"
    )
    base_series = _load_base_precipitation_series(
        _resolve_input_path(context, base_series_path),
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

    report_path = _build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 06 – Rain Sequence Generation")
    builder.add_paragraph(
        "Generate time-series rainfall data based on gauge layout results, outputting station and "
        "aggregated intensity sequences, along with heatmaps, animations, and storm hyetographs."
    )
    builder.add_heading("Key Information", level=2)
    builder.add_list(
        [
            f"Time step: {time_step_hours:.2f} hours",
            f"Average total rainfall: {aggregated_total_depth:.2f} mm",
            f"Peak intensity: {aggregated_peak:.2f} mm/hr",
        ]
    )
    builder.add_heading("Representative Station Statistics", level=2)
    builder.add_table(
        TableData(
            headers=["Station", "Total Rainfall (mm)", "Peak Intensity (mm/hr)"],
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
    builder.add_paragraph(f"Sequence generation time: {timestamp.isoformat()}")
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
    """Compute Thiessen polygons and precipitation weights for parameter subbasins."""

    context = load_project_context(config_path)
    step_index = 7
    step_name = "thiessen_weights"
    step_dir = _step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step07_thiessen_weights.log"
    logger = _configure_logger("step07", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 07 – Thiessen weight computation started.")

    project_cfg = context.config.setdefault("project", {})
    gauge_cfg = project_cfg.setdefault("rain_gauge", {})
    rainfall_cfg = project_cfg.setdefault("rainfall", {})
    delineation_cfg = context.config.get("delineation") or {}
    parameter_dir_entry = delineation_cfg.get("parameter_directory")
    if not parameter_dir_entry:
        raise PipelineConfigurationError(
            "Parameter directory not configured. Run Step 03 before Step 07."
        )

    parameter_dir = _resolve_input_path(context, parameter_dir_entry)
    subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    subbasin_geometries = _load_subbasin_geometries(subbasin_geojson)

    import json
    import pandas as pd
    from shapely.geometry import shape, mapping

    thiessen_src = gauge_cfg.get("thiessen_geojson")
    layout_src = gauge_cfg.get("layout_geojson")
    if not thiessen_src or not layout_src:
        raise PipelineConfigurationError(
            "Rain gauge layout not available. Run Step 05 before Step 07."
        )
    thiessen_path = _resolve_input_path(context, thiessen_src)
    layout_path = _resolve_input_path(context, layout_src)
    locations_data = json.loads(layout_path.read_text(encoding="utf-8"))
    polygon_data = json.loads(thiessen_path.read_text(encoding="utf-8"))

    station_positions: Dict[str, "BaseGeometry"] = {}
    for feature in locations_data.get("features", []):
        props = feature.get("properties", {}) or {}
        station_id = str(props.get("station_id") or props.get("id") or "").strip()
        if not station_id:
            continue
        station_positions[station_id] = shape(feature.get("geometry"))

    thiessen_polygons: Dict[str, "BaseGeometry"] = {}
    for feature in polygon_data.get("features", []):
        props = feature.get("properties", {}) or {}
        station_id = str(props.get("station_id") or props.get("id") or "").strip()
        if not station_id:
            continue
        thiessen_polygons[station_id] = shape(feature.get("geometry"))

    if not thiessen_polygons:
        raise ValueError("No Thiessen polygons found in the provided layout.")

    from hydrosis.precipitation import compute_subbasin_station_weights

    weights = compute_subbasin_station_weights(subbasin_geometries, thiessen_polygons)

    weights_json_path = step_dir / "rain_gauge_weights.json"
    weights_json_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")

    weights_rows: list[tuple[str, str, float]] = []
    for sub_id, station_map in weights.items():
        for station_id, value in sorted(station_map.items()):
            weights_rows.append((sub_id, station_id, float(value)))
    weights_df = pd.DataFrame(
        weights_rows, columns=["subbasin_id", "station_id", "weight"]
    )
    weights_table_path = step_dir / "weights_table.csv"
    weights_df.to_csv(weights_table_path, index=False)

    polygons_out_path = step_dir / "rain_gauge_thiessen_polygons.geojson"
    polygons_out_path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": polygon_data.get("features", [])},
            indent=2,
        ),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    color_map = plt.get_cmap("tab20")
    station_ids = sorted(thiessen_polygons.keys())
    station_to_color = {sid: color_map(index % 20) for index, sid in enumerate(station_ids)}

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    patches = []
    colors = []
    xs_all: list[float] = []
    ys_all: list[float] = []

    for station_id, polygon in thiessen_polygons.items():
        if polygon.geom_type == "Polygon":
            coords = np.asarray(polygon.exterior.coords)
            patches.append(MplPolygon(coords, closed=True))
            colors.append(station_to_color[station_id])
            xs_all.extend(coords[:, 0])
            ys_all.extend(coords[:, 1])
        elif polygon.geom_type == "MultiPolygon":
            for part in polygon.geoms:
                coords = np.asarray(part.exterior.coords)
                patches.append(MplPolygon(coords, closed=True))
                colors.append(station_to_color[station_id])
                xs_all.extend(coords[:, 0])
                ys_all.extend(coords[:, 1])

    if patches:
        collection = PatchCollection(patches, facecolors=colors, edgecolor="black", linewidth=0.6, alpha=0.5)
        ax.add_collection(collection)

    for station_id, point in station_positions.items():
        ax.scatter(
            point.x,
            point.y,
            c=[station_to_color.get(station_id, "red")],
            s=40,
            edgecolors="white",
            linewidths=0.8,
            label=station_id,
        )
        ax.text(point.x, point.y, station_id, fontsize=8, ha="left", va="bottom")

    if xs_all and ys_all:
        ax.set_xlim(min(xs_all) - 1000, max(xs_all) + 1000)
        ax.set_ylim(min(ys_all) - 1000, max(ys_all) + 1000)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Thiessen Polygons by Station")
    ax.axis("off")
    plt.tight_layout()
    thiessen_map_path = step_dir / "thiessen_map.png"
    plt.savefig(thiessen_map_path, dpi=220)
    plt.close()

    report_path = _build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 07 – Thiessen Weights")
    builder.add_paragraph(
        "Compute Thiessen polygons and station weights based on gauge layout and subbasin geometry "
        "for areal precipitation interpolation."
    )
    builder.add_heading("Statistical Overview", level=2)
    top_rows = weights_df.sort_values("weight", ascending=False).head(10)
    builder.add_table(
        TableData(
            headers=["Subbasin", "Station", "Weight"],
            rows=[
                [row["subbasin_id"], row["station_id"], f"{row['weight']:.3f}"]
                for _, row in top_rows.iterrows()
            ],
        )
    )
    builder.add_paragraph(f"Weight computation time: {timestamp.isoformat()}")
    builder.write(report_path)

    rainfall_cfg["weights_json"] = context.to_relative(weights_json_path)
    rainfall_cfg["thiessen_polygons"] = context.to_relative(polygons_out_path)
    project_cfg["last_step07_run"] = timestamp.isoformat()
    dump_project_config(context)
    logger.info("Step 07 – Thiessen weights completed successfully.")

    return {
        "weights_json": weights_json_path,
        "weights_table": weights_table_path,
        "thiessen_polygons": polygons_out_path,
        "thiessen_map": thiessen_map_path,
        "report": report_path,
        "log": log_path,
    }


def run_step08_areal_precipitation(config_path: Path | str) -> Dict[str, Path]:
    """Interpolates station rainfall to parameter subbasins and generates diagnostics."""

    context = load_project_context(config_path)
    step_index = 8
    step_name = "areal_precipitation"
    step_dir = _step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step08_areal_precip.log"
    logger = _configure_logger("step08", log_path)
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

    weights_path = _resolve_input_path(context, weights_src)
    station_series_path = _resolve_input_path(context, station_series_src)
    weights = json.loads(weights_path.read_text(encoding="utf-8"))

    station_df = pd.read_csv(station_series_path, index_col=0)
    station_df.index = pd.to_datetime(station_df.index)

    parameter_dir = _resolve_input_path(context, parameter_dir_entry)
    subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    subbasin_geometries = _load_subbasin_geometries(subbasin_geojson)

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

    report_path = _build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 08 – Areal Precipitation")
    builder.add_paragraph(
        "Interpolate gauge time series to parameter subbasins using Thiessen weights, generating "
        "areal precipitation tables, heatmaps, cumulative plots, and animations."
    )
    builder.add_heading("Statistical Summary", level=2)
    summary_preview = areal_summary_df.sort_values("total_depth_mm", ascending=False).head(10)
    builder.add_table(
        TableData(
            headers=["Subbasin", "Total Rainfall (mm)", "Peak Intensity (mm/hr)"],
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
    builder.add_paragraph(f"Areal precipitation interpolation time: {timestamp.isoformat()}")
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
    """Execute the baseline hydrologic workflow using the prepared forcing data."""

    if yaml is None:
        raise ImportError("PyYAML is required to load the project configuration.")

    context = load_project_context(config_path)
    step_index = 9
    step_name = "hydrologic_run"
    step_dir = _step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step09_hydrologic_run.log"
    logger = _configure_logger("step09", log_path)
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
            parameter_dir = _resolve_input_path(context, parameter_dir_entry)
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
    _reset_runoff_initial_states(model_config)
    results_dir = step_dir / "hydro"
    figures_dir = step_dir / "figures"
    reports_dir = step_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_config.io.results_directory = results_dir
    model_config.io.figures_directory = figures_dir
    model_config.io.reports_directory = reports_dir

    precipitation_path = _resolve_input_path(
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

    report_path = _build_report_path(context, step_index, step_name)
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

    cross_sections_path = _resolve_input_path(
        context,
        cfg.get(
            "cross_sections_path",
            (default_channel_dir / "channel_cross_sections_corrected.csv").as_posix(),
        ),
    )
    centerline_path = _resolve_input_path(
        context,
        cfg.get(
            "centerline_path",
            (default_channel_dir / "channel_centerlines.csv").as_posix(),
        ),
    )
    flows_path = _resolve_input_path(
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
    """Execute hydrodynamic routing scenarios and compare against the baseline."""

    if yaml is None:
        raise ImportError("PyYAML is required to load the project configuration.")

    context = load_project_context(config_path)
    step_index = 10
    step_name = "hydrodynamic_run"
    step_dir = _step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step10_hydrodynamic_run.log"
    logger = _configure_logger("step10", log_path)
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
    _reset_runoff_initial_states(model_config)
    results_dir = step_dir / "hydro"
    figures_dir = step_dir / "figures"
    reports_dir = step_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_config.io.results_directory = results_dir
    model_config.io.figures_directory = figures_dir
    model_config.io.reports_directory = reports_dir

    precipitation_path = _resolve_input_path(
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

    report_path = _build_report_path(context, step_index, step_name)
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
    step_dir = _step_directory(context, step_index, step_name)
    step_dir.mkdir(parents=True, exist_ok=True)
    log_path = context.logs_directory / "step11_final_report.log"
    logger = _configure_logger("step11", log_path)
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
