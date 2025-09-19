"""Run the bundled HydroSIS example configuration and persist outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis import (
    HydroSISModel,
    ModelConfig,
    default_evaluation_template,
    run_workflow,
)
from hydrosis.config import (
    EvaluationConfig,
    IOConfig,
    ScenarioConfig,
)
from hydrosis.delineation.dem_delineator import DelineationConfig
from hydrosis.io.inputs import load_forcing
from hydrosis.io.outputs import write_simulation_results
from hydrosis.parameters.zone import ParameterZoneConfig
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig


def _resolve_path(path: Path | None, repo_root: Path) -> Path | None:
    if path is None:
        return None
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_model_config(config_path: Path) -> ModelConfig:
    """Load the model configuration with a JSON fallback when PyYAML is absent."""

    try:
        return ModelConfig.from_yaml(config_path)
    except ImportError:
        json_path = config_path.with_suffix(".json")
        print(
            "PyYAML 未安装，改用 JSON 配置加载示例 (", json_path.as_posix(), ")",
            sep="",
        )
        data = json.loads(json_path.read_text(encoding="utf-8"))

    delineation = DelineationConfig.from_dict(data["delineation"])
    runoff_models = [
        RunoffModelConfig.from_dict(cfg) for cfg in data.get("runoff_models", [])
    ]
    routing_models = [
        RoutingModelConfig.from_dict(cfg) for cfg in data.get("routing_models", [])
    ]
    parameter_zones = [
        ParameterZoneConfig.from_dict(cfg)
        for cfg in data.get("parameter_zones", [])
    ]
    io_config = IOConfig.from_dict(data["io"])
    scenarios = [ScenarioConfig(**cfg) for cfg in data.get("scenarios", [])]
    evaluation = (
        EvaluationConfig.from_dict(data["evaluation"])
        if data.get("evaluation")
        else None
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


def _ensure_observations(
    config: ModelConfig,
    forcing: Mapping[str, Sequence[float]],
    repository_root: Path,
) -> Dict[str, Sequence[float]] | None:
    """Create synthetic observations when they are absent on disk."""

    observations_path = _resolve_path(config.io.discharge_observations, repository_root)
    if observations_path is None:
        return None

    if observations_path.is_file():
        # Single CSV file – treat it as a generic observation series.
        from hydrosis.io.inputs import load_time_series

        if not observations_path.exists():
            baseline = HydroSISModel.from_config(config)
            baseline_series = baseline.accumulate_discharge(baseline.run(forcing))
            first_series = next(iter(baseline_series.values()))
            observations_path.parent.mkdir(parents=True, exist_ok=True)
            write_simulation_results(observations_path.parent, {"observed": first_series})
        return {"observed": load_time_series(observations_path)}

    observations_path.mkdir(parents=True, exist_ok=True)
    existing = list(observations_path.glob("*.csv"))
    if existing:
        return load_forcing(observations_path)

    baseline_model = HydroSISModel.from_config(config)
    baseline_local = baseline_model.run(forcing)
    baseline_aggregated = baseline_model.accumulate_discharge(baseline_local)

    observed: Dict[str, Sequence[float]] = {}
    for sub_id, series in baseline_aggregated.items():
        adjusted: list[float] = []
        for idx, value in enumerate(series):
            if value == 0.0:
                adjusted.append(0.0)
                continue
            factor = 1.0 + (0.03 if idx % 4 == 0 else (-0.02 if idx % 3 == 0 else 0.0))
            adjusted.append(round(value * factor, 3))
        observed[sub_id] = adjusted

    write_simulation_results(observations_path, observed)
    return {key: list(values) for key, values in observed.items()}


def _load_observations(path: Path | None) -> Dict[str, Sequence[float]] | None:
    if path is None:
        return None
    if path.is_dir():
        return load_forcing(path)
    if path.is_file():
        from hydrosis.io.inputs import load_time_series

        return {"observed": load_time_series(path)}
    return None


def main() -> None:
    repo_root = REPO_ROOT
    config_path = repo_root / "config" / "example_model.yaml"
    config = _load_model_config(config_path)

    # Resolve IO paths relative to the repository root for consistent execution.
    config.io.precipitation = _resolve_path(config.io.precipitation, repo_root)
    config.io.discharge_observations = _resolve_path(
        config.io.discharge_observations, repo_root
    )
    config.io.results_directory = _resolve_path(config.io.results_directory, repo_root)
    config.io.figures_directory = _resolve_path(config.io.figures_directory, repo_root)
    config.io.reports_directory = _resolve_path(config.io.reports_directory, repo_root)

    forcing = load_forcing(config.io.precipitation)
    observations = _ensure_observations(config, forcing, repo_root)
    if observations is None:
        observations = _load_observations(config.io.discharge_observations)

    workflow_result = run_workflow(
        config,
        forcing,
        observations=observations,
        persist_outputs=True,
        generate_report=True,
        report_template=default_evaluation_template(),
        narrative_callback=lambda prompt: f"（示例 LLM 输出）{prompt}",
    )

    print("Baseline aggregated discharge (first 5 values):")
    for sub_id, series in workflow_result.baseline.aggregated.items():
        preview = ", ".join(f"{value:.2f}" for value in series[:5])
        print(f"  {sub_id}: {preview}")

    if workflow_result.overall_scores:
        print("\nAggregated evaluation metrics:")
        for score in workflow_result.overall_scores:
            metrics = ", ".join(
                f"{metric.upper()}={value:.4f}" for metric, value in score.aggregated.items()
            )
            print(f"  {score.model_id}: {metrics}")

    if workflow_result.scenarios:
        print("\nScenario summaries:")
        for scenario_id, scenario in workflow_result.scenarios.items():
            if scenario.aggregated:
                first_key = next(iter(scenario.aggregated))
                preview = ", ".join(f"{value:.2f}" for value in scenario.aggregated[first_key][:3])
            else:
                preview = "无数据"
            print(f"  {scenario_id}: first 3 aggregated flows -> {preview}")

    if workflow_result.report_path:
        print(f"\nMarkdown report saved to: {workflow_result.report_path}")
    print(f"Simulation outputs stored in: {config.io.results_directory}")


if __name__ == "__main__":
    main()
