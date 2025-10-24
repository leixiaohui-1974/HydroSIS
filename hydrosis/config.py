"""Configuration objects and helpers for HydroSIS."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field, replace
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

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "EvaluationConfig":
        return cls(
            metrics=list(data.get("metrics", ["rmse", "mae", "pbias", "nse"])),
            comparisons=[
                ComparisonPlanConfig.from_dict(item)
                for item in data.get("comparisons", [])
            ],
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "metrics": list(self.metrics),
            "comparisons": [cfg.to_dict() for cfg in self.comparisons],
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


@dataclass
class OutputArtifactsConfig:
    """Control which artefacts are generated during partitioning and reporting."""

    enable_figures: bool = True
    enable_tables: bool = True
    enable_reports: bool = True

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, object]]) -> "OutputArtifactsConfig":
        if not data:
            return cls()
        return cls(
            enable_figures=bool(data.get("enable_figures", True)),
            enable_tables=bool(data.get("enable_tables", True)),
            enable_reports=bool(data.get("enable_reports", True)),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "enable_figures": self.enable_figures,
            "enable_tables": self.enable_tables,
            "enable_reports": self.enable_reports,
        }


@dataclass
class SubbasinMethodAssignment:
    """Pattern-based override for runoff and routing methods."""

    targets: Sequence[str]
    runoff_model: Optional[str] = None
    routing_model: Optional[str] = None
    description: str = ""
    parameters: Dict[str, object] = field(default_factory=dict)

    def matches(self, subbasin_id: str) -> bool:
        return any(fnmatch.fnmatch(subbasin_id, pattern) for pattern in self.targets)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "SubbasinMethodAssignment":
        targets = list(data.get("targets", ["*"])) or ["*"]
        return cls(
            targets=targets,
            runoff_model=data.get("runoff_model"),
            routing_model=data.get("routing_model"),
            description=data.get("description", ""),
            parameters=dict(data.get("parameters", {})),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "targets": list(self.targets),
            "runoff_model": self.runoff_model,
            "routing_model": self.routing_model,
            "description": self.description,
            "parameters": dict(self.parameters),
        }


@dataclass
class ModelStructureConfig:
    """Library of runoff/routing models and assignment rules."""

    runoff_models: List[RunoffModelConfig] = field(default_factory=list)
    routing_models: List[RoutingModelConfig] = field(default_factory=list)
    default_runoff_model: Optional[str] = None
    default_routing_model: Optional[str] = None
    subbasin_assignments: List[SubbasinMethodAssignment] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ModelStructureConfig":
        return cls(
            runoff_models=[
                RunoffModelConfig.from_dict(item)
                for item in data.get("runoff_models", [])
            ],
            routing_models=[
                RoutingModelConfig.from_dict(item)
                for item in data.get("routing_models", [])
            ],
            default_runoff_model=data.get("default_runoff_model"),
            default_routing_model=data.get("default_routing_model"),
            subbasin_assignments=[
                SubbasinMethodAssignment.from_dict(item)
                for item in data.get("subbasin_assignments", [])
            ],
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "runoff_models": [cfg.to_dict() for cfg in self.runoff_models],
            "routing_models": [cfg.to_dict() for cfg in self.routing_models],
            "default_runoff_model": self.default_runoff_model,
            "default_routing_model": self.default_routing_model,
            "subbasin_assignments": [
                assignment.to_dict() for assignment in self.subbasin_assignments
            ],
        }


@dataclass
class ParameterPartitionConfig:
    """Controls how parameter zones and subzones are derived."""

    pour_points_path: Path
    target_subzone_area_km2: Optional[float] = None
    min_subzone_area_km2: Optional[float] = None
    max_subzones_per_zone: Optional[int] = None
    area_balance_tolerance: float = 0.25
    reuse_channel_network: bool = True
    subzone_accumulation_threshold: Optional[float] = None
    subzone_accumulation_thresholds: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ParameterPartitionConfig":
        threshold_value: Optional[float] = None
        threshold_map: Dict[str, float] = {}
        accum_value = data.get("subzone_accumulation_threshold")
        if isinstance(accum_value, Mapping):
            threshold_map.update(
                {
                    str(key): float(value)
                    for key, value in accum_value.items()
                    if value is not None
                }
            )
        elif accum_value is not None:
            threshold_value = float(accum_value)
        extra_map = data.get("subzone_accumulation_thresholds")
        if isinstance(extra_map, Mapping):
            threshold_map.update(
                {
                    str(key): float(value)
                    for key, value in extra_map.items()
                    if value is not None
                }
            )
        return cls(
            pour_points_path=Path(data["pour_points_path"]),
            target_subzone_area_km2=float(data["target_subzone_area_km2"])
            if data.get("target_subzone_area_km2") is not None
            else None,
            min_subzone_area_km2=float(data["min_subzone_area_km2"])
            if data.get("min_subzone_area_km2") is not None
            else None,
            max_subzones_per_zone=int(data["max_subzones_per_zone"])
            if data.get("max_subzones_per_zone") is not None
            else None,
            area_balance_tolerance=float(data.get("area_balance_tolerance", 0.25)),
            reuse_channel_network=bool(data.get("reuse_channel_network", True)),
            subzone_accumulation_threshold=threshold_value,
            subzone_accumulation_thresholds=threshold_map,
        )

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "pour_points_path": str(self.pour_points_path),
            "target_subzone_area_km2": self.target_subzone_area_km2,
            "min_subzone_area_km2": self.min_subzone_area_km2,
            "max_subzones_per_zone": self.max_subzones_per_zone,
            "area_balance_tolerance": self.area_balance_tolerance,
            "reuse_channel_network": self.reuse_channel_network,
            "subzone_accumulation_threshold": self.subzone_accumulation_threshold,
        }
        if self.subzone_accumulation_thresholds:
            payload["subzone_accumulation_thresholds"] = dict(
                self.subzone_accumulation_thresholds
            )
        return payload


@dataclass
class HydroProjectConfig:
    """Top-level configuration combining delineation, partition, and model inputs."""

    delineation: DelineationConfig
    partition: ParameterPartitionConfig
    model: ModelStructureConfig
    io: IOConfig
    outputs: OutputArtifactsConfig = field(default_factory=OutputArtifactsConfig)
    scenarios: List[ScenarioConfig] = field(default_factory=list)
    evaluation: Optional[EvaluationConfig] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "HydroProjectConfig":
        delineation = DelineationConfig.from_dict(data["delineation"])
        partition = ParameterPartitionConfig.from_dict(data["partition"])
        model = ModelStructureConfig.from_dict(data.get("model", {}))
        io_cfg = IOConfig.from_dict(data["io"])
        outputs_cfg = OutputArtifactsConfig.from_dict(data.get("outputs"))
        scenarios = [ScenarioConfig(**item) for item in data.get("scenarios", [])]
        evaluation = (
            EvaluationConfig.from_dict(data["evaluation"])
            if data.get("evaluation")
            else None
        )
        return cls(
            delineation=delineation,
            partition=partition,
            model=model,
            io=io_cfg,
            outputs=outputs_cfg,
            scenarios=scenarios,
            evaluation=evaluation,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "HydroProjectConfig":
        if yaml is None:
            raise ImportError("PyYAML is required to load project configuration files.")
        data = yaml.safe_load(Path(path).read_text())
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, object]:
        return {
            "delineation": self.delineation.to_dict(),
            "partition": self.partition.to_dict(),
            "model": self.model.to_dict(),
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
            "outputs": self.outputs.to_dict(),
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

    def build_model_config(
        self,
        parameter_zones: Optional[Sequence[ParameterZoneConfig]] = None,
    ) -> ModelConfig:
        zones = list(parameter_zones) if parameter_zones is not None else []
        return ModelConfig(
            delineation=self.delineation,
            runoff_models=list(self.model.runoff_models),
            routing_models=list(self.model.routing_models),
            parameter_zones=zones,
            io=self.io,
            scenarios=list(self.scenarios),
            evaluation=self.evaluation,
        )

# Validation configuration loading
# =================================

def load_validation_criteria(config_path: Path) -> Dict[str, object]:
    """Load validation criteria from YAML configuration file.

    This function loads validation criteria for all workflow steps,
    enabling configurable validation standards without hardcoding.

    Parameters
    ----------
    config_path : Path
        Path to validation_criteria.yaml file

    Returns
    -------
    dict
        Dictionary containing validation criteria for different
        validation categories (hydrologic, spatial, timeseries, etc.)

    Examples
    --------
    >>> from pathlib import Path
    >>> criteria = load_validation_criteria(Path("config/validation_criteria.yaml"))
    >>> hydrologic_criteria = criteria["hydrologic"]
    >>> print(hydrologic_criteria["runoff_coefficient_max"])
    1.0

    Notes
    -----
    The validation criteria file should define standards for:
    - hydrologic: Water balance, runoff coefficients, mass conservation
    - spatial: DEM quality, zone geometry, connectivity
    - timeseries: Precipitation, discharge data quality
    - rain_gauge: Station distribution and density
    - model_performance: NSE, PBIAS, R² thresholds
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load validation criteria from YAML files."
        )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Validation criteria file not found: {config_path}")

    data = yaml.safe_load(config_path.read_text())
    return data


def create_hydrologic_criteria(config_path: Optional[Path] = None) -> object:
    """Create HydrologicCriteria object from configuration file.

    Parameters
    ----------
    config_path : Path, optional
        Path to validation_criteria.yaml. If None, uses default criteria.

    Returns
    -------
    HydrologicCriteria
        Hydrologic validation criteria object

    Examples
    --------
    >>> criteria = create_hydrologic_criteria(Path("config/validation_criteria.yaml"))
    >>> from hydrosis.validation import validate_runoff_coefficient
    >>> result = validate_runoff_coefficient(coeffs, criteria=criteria)
    """
    from hydrosis.validation.hydrologic import HydrologicCriteria

    if config_path is None:
        return HydrologicCriteria()

    all_criteria = load_validation_criteria(config_path)
    hydrologic_data = all_criteria.get("hydrologic", {})

    return HydrologicCriteria.from_dict(hydrologic_data)


def load_workflow_config(config_path: Path) -> Dict:
    """Load workflow configuration from YAML file.

    This function loads the comprehensive workflow configuration that defines
    all paths, parameters, and settings for the complete HydroSIS workflow,
    eliminating the need for hardcoded values in scripts.

    Parameters
    ----------
    config_path : Path
        Path to workflow_config.yaml file

    Returns
    -------
    dict
        Workflow configuration with keys:
        - project: Project metadata
        - directories: Directory structure
        - steps: Step-specific configurations
        - hbv_model: HBV model parameters and calibration settings
        - rain_gauge_optimization: Rain gauge optimization settings
        - validation: Validation configuration
        - reporting: Report generation settings

    Raises
    ------
    ImportError
        If PyYAML is not installed
    FileNotFoundError
        If config_path does not exist

    Examples
    --------
    >>> from pathlib import Path
    >>> config = load_workflow_config(Path("config/workflow_config.yaml"))
    >>> base_dir = Path(config["directories"]["base_results"])
    >>> step9_config = config["steps"]["step_09_runoff"]
    >>> hbv_params = config["hbv_model"]["default_parameters"]
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load workflow configuration from YAML files."
        )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Workflow configuration file not found: {config_path}")

    data = yaml.safe_load(config_path.read_text())
    return data


def get_step_paths(workflow_config: Dict, step_name: str, base_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Get input and output paths for a workflow step.

    Parameters
    ----------
    workflow_config : dict
        Workflow configuration from load_workflow_config()
    step_name : str
        Step name (e.g., "step_09_runoff")
    base_dir : Path, optional
        Base results directory. If None, uses config["directories"]["base_results"]

    Returns
    -------
    dict
        Dictionary with keys:
        - output_dir: Output directory for this step
        - input: Dict of input file paths
        - output: Dict of output file paths

    Examples
    --------
    >>> config = load_workflow_config(Path("config/workflow_config.yaml"))
    >>> paths = get_step_paths(config, "step_09_runoff")
    >>> precip_path = paths["input"]["precipitation"]
    >>> output_dir = paths["output_dir"]
    """
    if base_dir is None:
        base_dir = Path(workflow_config["directories"]["base_results"])
    else:
        base_dir = Path(base_dir)

    step_config = workflow_config["steps"].get(step_name, {})

    # Build output directory
    output_dir = base_dir / step_config.get("output_dir", step_name)

    # Build input paths
    input_paths = {}
    for key, rel_path in step_config.get("input", {}).items():
        input_paths[key] = base_dir / rel_path

    # Build output paths
    output_paths = {}
    for key, filename in step_config.get("output", {}).items():
        output_paths[key] = output_dir / filename

    return {
        "output_dir": output_dir,
        "input": input_paths,
        "output": output_paths,
    }
