"""Shared helpers for portal-related integration tests."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from hydrosis.config import (
    ComparisonPlanConfig,
    EvaluationConfig,
    IOConfig,
    ModelConfig,
    ScenarioConfig,
)
from hydrosis.delineation.dem_delineator import DelineationConfig
from hydrosis.parameters.zone import ParameterZoneConfig
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig


def build_demo_model_config(results_directory: Path) -> ModelConfig:
    """Construct a deterministic demo configuration used across portal tests."""

    delineation = DelineationConfig(
        dem_path=Path("dem.tif"),
        pour_points_path=Path("pour_points.geojson"),
        precomputed_subbasins=[
            {"id": "S1", "area_km2": 1.0, "downstream": "S3", "parameters": {}},
            {"id": "S2", "area_km2": 1.0, "downstream": "S3", "parameters": {}},
            {"id": "S3", "area_km2": 1.0, "downstream": None, "parameters": {}},
        ],
    )

    runoff_models = [
        RunoffModelConfig(
            id="curve",
            model_type="scs_curve_number",
            parameters={"curve_number": 75, "initial_abstraction_ratio": 0.2},
        ),
        RunoffModelConfig(
            id="reservoir",
            model_type="linear_reservoir",
            parameters={"recession": 0.85, "conversion": 1.0},
        ),
    ]

    routing_models = [
        RoutingModelConfig(
            id="lag_short",
            model_type="lag",
            parameters={"lag_steps": 1},
        ),
        RoutingModelConfig(
            id="lag_long",
            model_type="lag",
            parameters={"lag_steps": 2},
        ),
    ]

    parameter_zones = [
        ParameterZoneConfig(
            id="Z1",
            description="Headwater gauge",
            control_points=["S1"],
            parameters={"runoff_model": "curve", "routing_model": "lag_short"},
        ),
        ParameterZoneConfig(
            id="Z2",
            description="Outlet station",
            control_points=["S3"],
            parameters={"runoff_model": "reservoir", "routing_model": "lag_short"},
        ),
    ]

    io_config = IOConfig(
        precipitation=Path("data/forcing/precipitation.csv"),
        results_directory=results_directory,
        figures_directory=results_directory / "figures",
        reports_directory=results_directory / "reports",
    )

    scenarios = [
        ScenarioConfig(
            id="alternate_routing",
            description="Slow down routing through the middle catchment",
            modifications={"S2": {"routing_model": "lag_long"}},
        )
    ]

    evaluation = EvaluationConfig(
        metrics=["rmse", "nse"],
        comparisons=[
            ComparisonPlanConfig(
                id="baseline_vs_scenario",
                description="Compare baseline and scenario accuracy at the outlet",
                models=["baseline", "alternate_routing"],
                reference="observed",
                subbasins=["S3"],
                ranking_metric="rmse",
            )
        ],
    )

    return ModelConfig(
        delineation=delineation,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=io_config,
        scenarios=scenarios,
        evaluation=evaluation,
    )


def build_demo_forcing() -> Dict[str, List[float]]:
    """Return a compact forcing dataset shared by API tests."""

    return {
        "S1": [0.0, 10.0, 30.0, 0.0],
        "S2": [5.0, 5.0, 5.0, 5.0],
        "S3": [0.0, 0.0, 0.0, 0.0],
    }
