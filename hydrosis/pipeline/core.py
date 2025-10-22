"""Core utilities and shared components for the ten-step pipeline.

This module provides common infrastructure used across all pipeline steps:
- Project configuration management
- Logger setup
- File path resolution
- Common data loading utilities
"""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore[assignment]


DEFAULT_RESULTS_ROOT = Path("results/upper_truckee_project")
LOGGER_NAMESPACE = "hydrosis.pipeline"


class PipelineConfigurationError(RuntimeError):
    """Raised when the project configuration is missing required fields."""


@dataclass(slots=True)
class ProjectContext:
    """Runtime view of the project configuration and directory layout.

    This class manages file paths and configuration for pipeline execution,
    automatically creating necessary directories and handling path resolution.
    """

    config_path: Path
    config: dict[str, Any]
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
        """Convert relative paths to absolute based on config directory."""
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
    """Read the YAML configuration and prepare the runtime context.

    Args:
        config_path: Path to the project YAML configuration file

    Returns:
        ProjectContext with loaded configuration and directory structure

    Raises:
        ImportError: If PyYAML is not installed
        FileNotFoundError: If config file doesn't exist
    """
    if yaml is None:
        raise ImportError("PyYAML must be installed to use the project pipeline.")

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Project configuration not found: {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    project_section = raw_config.get("project", {}) or {}
    results_root = Path(project_section.get("results_root") or DEFAULT_RESULTS_ROOT)

    # Ensure results_root is in config
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
    """Persist the (potentially modified) YAML configuration to disk.

    Args:
        context: Project context with config to save

    Raises:
        ImportError: If PyYAML is not installed
    """
    if yaml is None:
        raise ImportError("PyYAML must be installed to serialise the project config.")

    serialised = yaml.safe_dump(
        context.config,
        sort_keys=False,
        allow_unicode=False,
        indent=2,
    )
    context.config_path.write_text(serialised, encoding="utf-8")


def step_slug(index: int, name: str) -> str:
    """Generate a slug for a pipeline step (e.g., '01_dem_preprocessing')."""
    return f"{index:02d}_{name}"


def step_directory(context: ProjectContext, index: int, name: str) -> Path:
    """Create and return the directory for a specific pipeline step.

    Args:
        context: Project context
        index: Step number (1-10)
        name: Step name slug

    Returns:
        Path to step directory (created if doesn't exist)
    """
    directory = context.base_results / step_slug(index, name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logger(step_id: str, log_path: Path) -> logging.Logger:
    """Set up a namespaced logger that writes into the pipeline log directory.

    Args:
        step_id: Identifier for the step (e.g., 'step01')
        log_path: Path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"{LOGGER_NAMESPACE}.{step_id}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


def write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    """Write tabular data to CSV file.

    Args:
        path: Output CSV file path
        headers: Column headers
        rows: Data rows
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(row)


def build_report_path(context: ProjectContext, index: int, slug: str) -> Path:
    """Generate path for a step's markdown report.

    Args:
        context: Project context
        index: Step number
        slug: Report identifier

    Returns:
        Path to markdown report file
    """
    filename = f"step{index:02d}_{slug}.md"
    return context.reports_directory / filename


def resolve_input_path(context: ProjectContext, path_like: str | Path) -> Path:
    """Resolve a path relative to the project configuration directory.

    Args:
        context: Project context
        path_like: Path string or Path object (may be relative)

    Returns:
        Absolute resolved path
    """
    path = Path(path_like)
    if not path.is_absolute():
        path = context.config_directory / path
    return path.resolve()


def load_subbasin_geometries(path: Path) -> dict[str, Any]:
    """Load subbasin geometries from GeoJSON file.

    Args:
        path: Path to GeoJSON file containing subbasin polygons

    Returns:
        Dictionary mapping subbasin IDs to shapely geometry objects

    Raises:
        FileNotFoundError: If geometry file doesn't exist
        ValueError: If no valid geometries found
        ImportError: If shapely is not installed
    """
    try:
        from shapely.geometry import shape
        from shapely.geometry.base import BaseGeometry
    except ImportError:
        raise ImportError("Shapely must be installed for geometry operations")

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


def reset_runoff_initial_states(model_config: Any) -> None:
    """Ensure runoff models start from neutral storage states.

    This resets initial state parameters for models like HBV to ensure
    consistent starting conditions for pipeline runs.

    Args:
        model_config: ModelConfig instance with runoff_models attribute
    """
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


def load_base_precipitation_series(path: Path, column: Optional[str] = None) -> Any:
    """Load base precipitation time series from CSV file.

    Args:
        path: Path to precipitation CSV file
        column: Optional column name (auto-detected if None)

    Returns:
        pandas Series with datetime index

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or has no numeric columns
        ImportError: If pandas is not installed
    """
    if pd is None:
        raise ImportError("pandas must be installed for precipitation loading")

    if not path.exists():
        raise FileNotFoundError(f"Base precipitation series not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Precipitation series file is empty: {path}")

    # Parse timestamps
    timestamp_col = df.columns[0]
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    if df.empty:
        raise ValueError("Precipitation series contains no data rows.")

    # Select column
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


def compute_basic_stats(values: Any) -> dict[str, float]:
    """Compute basic statistics for an array of values.

    Args:
        values: Array-like numeric data

    Returns:
        Dictionary with min, max, mean, std, median, p90, p99
    """
    if np is None:
        raise ImportError("numpy must be installed for statistics")

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
