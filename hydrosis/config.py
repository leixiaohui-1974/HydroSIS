"""Configuration objects and helpers for HydroSIS."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

try:  # pragma: no cover - optional dependency
    import yaml
except ImportError:  # pragma: no cover - fallback for tests without PyYAML
    yaml = None

from .model import Subbasin
from .runoff.base import RunoffModelConfig
from .routing.base import RoutingModelConfig
from .delineation.dem_delineator import DelineationConfig
from .parameters.zone import ParameterZoneConfig


@dataclass
class IOConfig:
    """Input/output configuration for simulation data."""

    precipitation: Path
    evaporation: Optional[Path] = None
    discharge_observations: Optional[Path] = None
    results_directory: Path = Path("results")

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> "IOConfig":
        return cls(
            precipitation=Path(data["precipitation"]),
            evaporation=Path(data.get("evaporation")) if data.get("evaporation") else None,
            discharge_observations=Path(data["discharge_observations"]) if data.get("discharge_observations") else None,
            results_directory=Path(data.get("results_directory", "results")),
        )


@dataclass
class ScenarioConfig:
    """Hydrological scenario definition for what-if analyses."""

    id: str
    description: str
    modifications: Mapping[str, MutableMapping[str, float]] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Aggregate configuration for the HydroSIS model."""

    delineation: DelineationConfig
    runoff_models: List[RunoffModelConfig]
    routing_models: List[RoutingModelConfig]
    parameter_zones: List[ParameterZoneConfig]
    io: IOConfig
    scenarios: List[ScenarioConfig] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load configuration from YAML files."
            )

        data = yaml.safe_load(Path(path).read_text())

        delineation = DelineationConfig.from_dict(data["delineation"])
        runoff_models = [RunoffModelConfig.from_dict(cfg) for cfg in data.get("runoff_models", [])]
        routing_models = [RoutingModelConfig.from_dict(cfg) for cfg in data.get("routing_models", [])]
        parameter_zones = [ParameterZoneConfig.from_dict(cfg) for cfg in data.get("parameter_zones", [])]
        io_cfg = IOConfig.from_dict(data["io"])
        scenarios = [ScenarioConfig(**cfg) for cfg in data.get("scenarios", [])]

        return cls(
            delineation=delineation,
            runoff_models=runoff_models,
            routing_models=routing_models,
            parameter_zones=parameter_zones,
            io=io_cfg,
            scenarios=scenarios,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "delineation": self.delineation.to_dict(),
            "runoff_models": [cfg.to_dict() for cfg in self.runoff_models],
            "routing_models": [cfg.to_dict() for cfg in self.routing_models],
            "parameter_zones": [cfg.to_dict() for cfg in self.parameter_zones],
            "io": {
                "precipitation": str(self.io.precipitation),
                "evaporation": str(self.io.evaporation) if self.io.evaporation else None,
                "discharge_observations": str(self.io.discharge_observations)
                if self.io.discharge_observations
                else None,
                "results_directory": str(self.io.results_directory),
            },
            "scenarios": [
                {
                    "id": scenario.id,
                    "description": scenario.description,
                    "modifications": {k: dict(v) for k, v in scenario.modifications.items()},
                }
                for scenario in self.scenarios
            ],
        }

    def apply_scenario(self, scenario_id: str, subbasins: Iterable[Subbasin]) -> None:
        scenario = next((sc for sc in self.scenarios if sc.id == scenario_id), None)
        if scenario is None:
            raise KeyError(f"Scenario {scenario_id} not defined")

        for sub in subbasins:
            if sub.id in scenario.modifications:
                sub.update_parameters(scenario.modifications[sub.id])
