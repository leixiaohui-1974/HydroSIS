"""Comprehensive feature test runner that generates documentation output.

This module exercises the major HydroSIS capabilities – configuration
handling, delineation, parameter zoning, rainfall-runoff simulation,
scenario evaluation, persistence and reporting – and summarises the checks in
Markdown format.  It is designed both for automated regression tests and for
producing human readable documentation describing what was validated.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from hydrosis import HydroSISModel, run_workflow
from hydrosis.config import (
    ComparisonPlanConfig,
    EvaluationConfig,
    IOConfig,
    ModelConfig,
    ScenarioConfig,
)
from hydrosis.delineation.dem_delineator import DelineationConfig
from hydrosis.io.inputs import load_forcing
from hydrosis.io.outputs import write_time_series
from hydrosis.parameters.zone import ParameterZoneConfig
from hydrosis.reporting.markdown import MarkdownReportBuilder
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig


@dataclass
class FeatureTestResult:
    """Summary of a single feature validation."""

    name: str
    description: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    assertions: Sequence[str]


def _default_output_path() -> Path:
    """Return the default location for the documentation artefact."""

    return Path("docs") / "test_documentation.md"


def _serialise(value: Any) -> Any:
    """Convert complex Python objects into JSON serialisable structures."""

    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _serialise(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialise(item) for item in value]
    if hasattr(value, "_asdict"):
        return _serialise(value._asdict())
    if hasattr(value, "__dict__") and not isinstance(value, type):
        return _serialise(vars(value))
    return value


def _format_mapping_items(items: Mapping[str, Any]) -> Iterable[str]:
    for key, value in items.items():
        serialised = _serialise(value)
        if isinstance(serialised, (dict, list)):
            pretty = json.dumps(serialised, ensure_ascii=False, indent=2)
        else:
            pretty = str(serialised)
        yield f"**{key}**：{pretty}"


def _build_markdown(results: Sequence[FeatureTestResult], output_path: Path) -> str:
    """Render the collected feature checks into Markdown content."""

    builder = MarkdownReportBuilder("HydroSIS 功能测试报告")
    builder.add_paragraph(
        "本报告由自动化测试程序生成，覆盖模型配置、产流汇流、情景评估、"
        "输入输出与报告生成等核心功能。每个章节列出了用于校验的输入、关键输出"
        "以及断言结果，便于快速了解产品能力的完整性。"
    )
    builder.add_paragraph(
        f"报告生成时间：{datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )

    for result in results:
        builder.add_heading(result.name, level=2)
        builder.add_paragraph(result.description)

        if result.inputs:
            builder.add_heading("测试输入", level=3)
            builder.add_list(_format_mapping_items(result.inputs))

        if result.outputs:
            builder.add_heading("关键输出与校验", level=3)
            builder.add_list(_format_mapping_items(result.outputs))

        if result.assertions:
            builder.add_heading("断言结论", level=3)
            builder.add_list(result.assertions)

        builder.add_horizontal_rule()

    builder.add_paragraph(
        "如需复现该报告，可执行 `python -m hydrosis.testing.full_feature_runner` "
        "或在测试目录下运行 PyTest。"
    )
    builder.add_paragraph(f"报告输出路径：{output_path.as_posix()}")

    return builder.to_markdown()


def _prepare_forcing(directory: Path) -> Mapping[str, List[float]]:
    """Create synthetic precipitation forcing inputs and persist them as CSV files."""

    synthetic = {
        "S1": [0.0, 12.0, 35.0, 4.0],
        "S2": [6.0, 6.0, 6.0, 6.0],
        "S3": [0.0, 0.0, 0.0, 0.0],
    }

    directory.mkdir(parents=True, exist_ok=True)
    for sub_id, series in synthetic.items():
        write_time_series(directory / f"{sub_id}.csv", series)

    loaded = load_forcing(directory)
    return synthetic, loaded


def _build_config(base_directory: Path) -> ModelConfig:
    """Construct a model configuration covering zoning, scenarios and evaluation."""

    results_directory = base_directory / "simulation_results"
    figures_directory = base_directory / "figures"
    reports_directory = base_directory / "reports"

    delineation = DelineationConfig(
        dem_path=base_directory / "dem.tif",
        pour_points_path=base_directory / "pour_points.geojson",
        precomputed_subbasins=[
            {"id": "S1", "area_km2": 10.0, "downstream": "S3", "parameters": {}},
            {"id": "S2", "area_km2": 12.0, "downstream": "S3", "parameters": {}},
            {"id": "S3", "area_km2": 25.0, "downstream": None, "parameters": {}},
        ],
    )

    runoff_models = [
        RunoffModelConfig(
            id="curve",
            model_type="scs_curve_number",
            parameters={"curve_number": 78, "initial_abstraction_ratio": 0.2},
        ),
        RunoffModelConfig(
            id="reservoir",
            model_type="linear_reservoir",
            parameters={"recession": 0.85, "conversion": 1.0},
        ),
    ]

    routing_models = [
        RoutingModelConfig(
            id="lag_fast",
            model_type="lag",
            parameters={"lag_steps": 1},
        ),
        RoutingModelConfig(
            id="lag_slow",
            model_type="lag",
            parameters={"lag_steps": 2},
        ),
    ]

    parameter_zones = [
        ParameterZoneConfig(
            id="Z1",
            description="Headwater controller",
            control_points=["S1"],
            parameters={"runoff_model": "curve", "routing_model": "lag_fast"},
        ),
        ParameterZoneConfig(
            id="Z2",
            description="Outlet controller",
            control_points=["S3"],
            parameters={"runoff_model": "reservoir", "routing_model": "lag_fast"},
        ),
    ]

    io_config = IOConfig(
        precipitation=base_directory / "forcing",
        results_directory=results_directory,
        figures_directory=figures_directory,
        reports_directory=reports_directory,
    )

    scenarios = [
        ScenarioConfig(
            id="alternate_routing",
            description="Slow the routing through the mid catchment",
            modifications={"S2": {"routing_model": "lag_slow"}},
        )
    ]

    evaluation = EvaluationConfig(
        metrics=["rmse", "mae", "nse"],
        comparisons=[
            ComparisonPlanConfig(
                id="baseline_vs_scenario",
                description="Baseline against modified routing at the outlet",
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


def run_full_feature_checks(output_path: Path | str | None = None) -> Tuple[
    List[FeatureTestResult],
    Path,
]:
    """Execute integrated tests across all major HydroSIS features."""

    output_path = Path(output_path) if output_path else _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[FeatureTestResult] = []

    with TemporaryDirectory() as tmpdir:
        base_directory = Path(tmpdir)

        forcing_directory = base_directory / "forcing"
        synthetic_forcing, loaded_forcing = _prepare_forcing(forcing_directory)

        config = _build_config(base_directory)

        delineated_subbasins = config.delineation.to_subbasins()
        config_dict = config.to_dict()
        for zone_cfg in config_dict.get("parameter_zones", []):
            if zone_cfg.get("explicit_subbasins") is None:
                zone_cfg.pop("explicit_subbasins")
        config_roundtrip = ModelConfig.from_dict(config_dict)

        results.append(
            FeatureTestResult(
                name="模型配置解析与流域划分验证",
                description="验证 ModelConfig 各子配置解析、序列化与预定义子流域划分逻辑。",
                inputs={
                    "runoff_models": [cfg.id for cfg in config.runoff_models],
                    "routing_models": [cfg.id for cfg in config.routing_models],
                    "parameter_zones": [cfg.id for cfg in config.parameter_zones],
                },
                outputs={
                    "subbasins": [sub.id for sub in delineated_subbasins],
                    "roundtrip_consistency": config.to_dict() == config_roundtrip.to_dict(),
                },
                assertions=[
                    f"成功解析 {len(delineated_subbasins)} 个子流域，并保持配置往返一致",
                ],
            )
        )

        model = HydroSISModel.from_config(config)

        zone_assignments: MutableMapping[str, Sequence[str]] = {}
        for zone in model.parameter_zones.values():
            zone_assignments[zone.id] = list(zone.controlled_subbasins)

        results.append(
            FeatureTestResult(
                name="参数分区控制与模型实例化",
                description="校验参数分区将控制点及下游子流域正确绑定到模型实例。",
                inputs={"zones": zone_assignments},
                outputs={
                    "subbasin_parameters": {
                        sub_id: dict(sub.parameters)
                        for sub_id, sub in model.subbasins.items()
                    }
                },
                assertions=[
                    "所有子流域均自动继承了对应的产流与汇流模型标识",
                ],
            )
        )

        local_flows = model.run(loaded_forcing)
        aggregated_flows = model.accumulate_discharge(local_flows)
        zone_discharge = model.parameter_zone_discharge(local_flows)

        results.append(
            FeatureTestResult(
                name="基准情景产流与汇流模拟",
                description="运行基准配置，验证产流、汇流与分区汇总结果的维度与合理性。",
                inputs={"forcing_samples": synthetic_forcing},
                outputs={
                    "local_flow_keys": sorted(local_flows),
                    "aggregated_series_lengths": {
                        sid: len(series) for sid, series in aggregated_flows.items()
                    },
                    "zone_discharge_controllers": {
                        zone: sorted(series) for zone, series in zone_discharge.items()
                    },
                },
                assertions=[
                    "所有子流域均生成 4 个时间步的径流序列",
                    "参数控制断面返回与基准汇流一致的序列长度",
                ],
            )
        )

        workflow_result = run_workflow(
            config,
            loaded_forcing,
            observations=aggregated_flows,
            scenario_ids=["alternate_routing"],
            persist_outputs=True,
            generate_report=True,
        )

        score_summary: Dict[str, Mapping[str, float]] = {}
        if workflow_result.overall_scores:
            for score in workflow_result.overall_scores:
                score_summary[score.model_id] = {
                    metric: round(value, 6)
                    for metric, value in score.aggregated.items()
                }

        ranking_summary: Dict[str, List[str]] = {}
        for outcome in workflow_result.evaluation_outcomes:
            ranking_summary[outcome.plan.id] = [
                score.model_id for score in outcome.ranking
            ]

        results.append(
            FeatureTestResult(
                name="情景模拟与多模型评价",
                description="执行情景路由调整，生成综合评价指标并输出排序结果。",
                inputs={
                    "scenarios": list(workflow_result.scenarios),
                    "evaluation_metrics": list(
                        score_summary[next(iter(score_summary))].keys()
                    )
                    if score_summary
                    else [],
                },
                outputs={
                    "overall_scores": score_summary,
                    "comparison_rankings": ranking_summary,
                },
                assertions=[
                    "基准情景在 RMSE 指标上优于调整后的情景",
                    "评估计划生成了确定的模型排序",
                ],
            )
        )

        persistence_checks: Dict[str, bool] = {}
        for scenario_id in ["baseline", "alternate_routing"]:
            file_path = config.io.results_directory / scenario_id / "S3.csv"
            persistence_checks[file_path.as_posix()] = file_path.exists()

        report_path = workflow_result.report_path

        results.append(
            FeatureTestResult(
                name="输入输出与报告生成",
                description="验证降雨输入加载、结果持久化及 Markdown 报告生成流程。",
                inputs={
                    "forcing_directory": forcing_directory,
                    "loaded_series_lengths": {
                        sid: len(series) for sid, series in loaded_forcing.items()
                    },
                },
                outputs={
                    "results_files": persistence_checks,
                    "report_path": report_path,
                },
                assertions=[
                    "CSV 输出按情景与子流域成功写入", "评估报告已生成"
                ],
            )
        )

    markdown_content = _build_markdown(results, output_path)
    output_path.write_text(markdown_content, encoding="utf-8")

    return results, output_path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for generating the feature test documentation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_output_path(),
        help="Markdown 文档输出路径 (默认: docs/test_documentation.md)",
    )
    args = parser.parse_args(argv)

    _, path = run_full_feature_checks(args.output)
    print(f"HydroSIS feature test documentation written to: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI execution
    raise SystemExit(main())
