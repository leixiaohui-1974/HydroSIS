"""Pipeline orchestration utilities for the HydroSIS product.

This package provides both high-level orchestration (stages.py) and
step-by-step pipeline execution (step01-step10 modules).

The ten-step pipeline has been modularized for better maintainability:
- core.py: Shared utilities and configuration management
- step01_terrain.py: DEM preprocessing
- step02_pour_points.py: Pour point extraction
- step03_partitioning.py: Parameter partitioning
- step04_channel_profile.py: Channel profile analysis
- step05_rain_gauge_layout.py: Rain gauge layout
- step06_rain_sequence.py: Rain sequence generation
- step07_thiessen_weights.py: Thiessen polygon weights
- step08_areal_precipitation.py: Areal precipitation
- step09_hydrologic_run.py: Hydrologic simulation
- step10_hydrodynamic_run.py: Hydrodynamic routing
"""

# High-level pipeline orchestration
from .stages import (
    HydroProductPipeline,
    PipelineConfig,
    StageResult,
    StageTask,
    StageTaskResult,
)

# Core utilities
from .core import (
    PipelineConfigurationError,
    ProjectContext,
    load_project_context,
    dump_project_config,
)

# Individual pipeline steps
from .step01_terrain import run_step01_dem_preprocessing
from .step02_pour_points import run_step02_pour_points
from .step03_partitioning import run_step03_partitioning
from .step04_channel_profile import run_step04_channel_profile
from .step05_rain_gauge_layout import run_step05_rain_gauge_layout
from .step06_rain_sequence import run_step06_rain_sequence
from .step07_thiessen_weights import run_step07_thiessen_weights
from .step08_areal_precipitation import run_step08_areal_precipitation
from .step09_hydrologic_run import run_step09_hydrologic_run
from .step10_hydrodynamic_run import run_step10_hydrodynamic_run

# Backward compatibility: still support importing from ten_step_pipeline
# (deprecated, will be removed in future version)
try:
    from .ten_step_pipeline import run_final_pipeline_report
except (ImportError, AttributeError):
    # If ten_step_pipeline doesn't have this function, that's OK
    run_final_pipeline_report = None


__all__ = [
    # High-level orchestration
    "HydroProductPipeline",
    "PipelineConfig",
    "StageResult",
    "StageTask",
    "StageTaskResult",
    # Core utilities
    "PipelineConfigurationError",
    "ProjectContext",
    "load_project_context",
    "dump_project_config",
    # Pipeline steps
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
]

# Add optional function if it exists
if run_final_pipeline_report is not None:
    __all__.append("run_final_pipeline_report")
