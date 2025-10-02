"""Configuration objects and helpers for HydroSIS."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

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
class ComparisonPlanConfig:
    """Configuration describing a model comparison experiment."""

    id: str
    description: str
    models: Sequence[str]
    reference: str
    subbasins: Optional[Sequence[str]] = None
    ranking_metric: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ComparisonPlanConfig":
        return cls(
            id=data["id"],
            description=data.get("description", ""),
            models=list(data.get("models", [])),
            reference=data.get("reference", "observed"),
            subbasins=list(data.get("subbasins", [])) or None,
            ranking_metric=data.get("ranking_metric"),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "models": list(self.models),
            "reference": self.reference,
            "subbasins": list(self.subbasins) if self.subbasins else None,
            "ranking_metric": self.ranking_metric,
        }


@dataclass
class EvaluationConfig:
    """Evaluation setup including metrics and comparison plans."""

    metrics: Sequence[str] = field(
        default_factory=lambda: ["rmse", "mae", "pbias", "nse"]
    )
    comparisons: List[ComparisonPlanConfig] = field(default_factory=list)
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EvaluationConfig":
        return cls(
            metrics=list(data.get("metrics", ["rmse", "mae", "pbias", "nse"])),
            comparisons=[
                ComparisonPlanConfig.from_dict(item)
                for item in data.get("comparisons", [])
            ],
            llm_provider=data.get("llm_provider"),
            llm_model=data.get("llm_model"),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": list(self.metrics),
            "comparisons": [cfg.to_dict() for cfg in self.comparisons],
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
        }


@dataclass
class IOConfig:
    """Input/output configuration for simulation data."""

    precipitation: Path
    evaporation: Optional[Path] = None
    discharge_observations: Optional[Path] = None
    results_directory: Path = Path("results")
    figures_directory: Optional[Path] = None
    reports_directory: Optional[Path] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, str]) -> "IOConfig":
        return cls(
            precipitation=Path(data["precipitation"]),
            evaporation=Path(data.get("evaporation")) if data.get("evaporation") else None,
            discharge_observations=Path(data["discharge_observations"]) if data.get("discharge_observations") else None,
            results_directory=Path(data.get("results_directory", "results")),
            figures_directory=Path(data["figures_directory"]) if data.get("figures_directory") else None,
            reports_directory=Path(data["reports_directory"]) if data.get("reports_directory") else None,
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
    evaluation: Optional[EvaluationConfig] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "ModelConfig":
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load configuration from YAML files."
            )

        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ModelConfig":
        delineation = DelineationConfig.from_dict(data["delineation"])
        runoff_models = [RunoffModelConfig.from_dict(cfg) for cfg in data.get("runoff_models", [])]
        routing_models = [RoutingModelConfig.from_dict(cfg) for cfg in data.get("routing_models", [])]
        parameter_zones = [ParameterZoneConfig.from_dict(cfg) for cfg in data.get("parameter_zones", [])]
        io_cfg = IOConfig.from_dict(data["io"])
        scenarios = [ScenarioConfig(**cfg) for cfg in data.get("scenarios", [])]
        evaluation = (
            EvaluationConfig.from_dict(data["evaluation"])
            if data.get("evaluation")
            else None
        )

        return cls(
            delineation=delineation,
            runoff_models=runoff_models,
            routing_models=routing_models,
            parameter_zones=parameter_zones,
            io=io_cfg,
            scenarios=scenarios,
            evaluation=evaluation,
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
                "figures_directory": str(self.io.figures_directory)
                if self.io.figures_directory
                else None,
                "reports_directory": str(self.io.reports_directory)
                if self.io.reports_directory
                else None,
            },
            "scenarios": [
                {
                    "id": scenario.id,
                    "description": scenario.description,
                    "modifications": {k: dict(v) for k, v in scenario.modifications.items()},
                }
                for scenario in self.scenarios
            ],
            "evaluation": self.evaluation.to_dict() if self.evaluation else None,
        }

    def apply_scenario(self, scenario_id: str, subbasins: Iterable[Subbasin]) -> None:
        scenario = next((sc for sc in self.scenarios if sc.id == scenario_id), None)
        if scenario is None:
            raise KeyError(f"Scenario {scenario_id} not defined")

        for sub in subbasins:
            if sub.id in scenario.modifications:
                sub.update_parameters(scenario.modifications[sub.id])
