"""Generate Markdown documentation for HydroSIS example validations."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from hydrosis import HydroSISModel, ModelComparator, SimulationEvaluator
from hydrosis.config import IOConfig, ModelConfig, ScenarioConfig
from hydrosis.delineation.dem_delineator import DelineationConfig
from hydrosis.delineation.simple_grid import delineate_from_json
from hydrosis.model import Subbasin
from hydrosis.parameters.zone import ParameterZoneBuilder, ParameterZoneConfig
from hydrosis.reporting.markdown import MarkdownReportBuilder
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig
from hydrosis.testing.synthetic_datasets import (
    SYNTHETIC_DEM_GRID,
    SYNTHETIC_POUR_POINTS,
    write_synthetic_delineation_inputs,
)



@dataclass
class ExampleDocumentation:
    """Structured details describing a single example validation."""

    slug: str
    title: str
    description: str
    inputs: Mapping[str, Any]
    outputs: Mapping[str, Any]
    assertions: Sequence[str]


def _serialise(value: Any) -> Any:
    """Convert values into JSON compatible structures for Markdown output."""

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
    if isinstance(value, float):
        return round(value, 6)
    return value


def _format_mapping_items(items: Mapping[str, Any]) -> Iterable[str]:
    for key, value in items.items():
        serialised = _serialise(value)
        if isinstance(serialised, (dict, list)):
            pretty = json.dumps(serialised, ensure_ascii=False, indent=2)
        else:
            pretty = str(serialised)
        yield f"**{key}**：{pretty}"


def _render_markdown(doc: ExampleDocumentation) -> str:
    builder = MarkdownReportBuilder(doc.title)
    builder.add_paragraph(doc.description)

    if doc.inputs:
        builder.add_heading("测试输入", level=2)
        builder.add_list(_format_mapping_items(doc.inputs))

    if doc.outputs:
        builder.add_heading("关键输出", level=2)
        builder.add_list(_format_mapping_items(doc.outputs))

    if doc.assertions:
        builder.add_heading("断言结论", level=2)
        builder.add_list(doc.assertions)

    builder.add_paragraph("该文档由自动化示例验证程序生成。")
    return builder.to_markdown()


def build_sample_config() -> ModelConfig:
    """Construct a minimal yet comprehensive configuration for examples."""

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
            description="Headwater zone controlled by gauge G1",
            control_points=["S1"],
            parameters={"runoff_model": "curve", "routing_model": "lag_short"},
        ),
        ParameterZoneConfig(
            id="Z2",
            description="Outlet control at station G2",
            control_points=["S3"],
            parameters={"runoff_model": "reservoir", "routing_model": "lag_short"},
        ),
    ]

    io_config = IOConfig(
        precipitation=Path("data/forcing/precipitation.csv"),
        results_directory=Path("results"),
    )

    scenarios = [
        ScenarioConfig(
            id="alternate_routing",
            description="Increase lag time for middle catchment",
            modifications={"S2": {"routing_model": "lag_long"}},
        )
    ]

    return ModelConfig(
        delineation=delineation,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=io_config,
        scenarios=scenarios,
    )


def build_comparison_config() -> ModelConfig:
    """Configuration with multiple parameter zones for comparison tests."""

    delineation = DelineationConfig(
        dem_path=Path("dem.tif"),
        pour_points_path=Path("pour_points.geojson"),
        precomputed_subbasins=[
            {"id": "S1", "area_km2": 1.5, "downstream": "S3", "parameters": {}},
            {"id": "S2", "area_km2": 2.0, "downstream": "S3", "parameters": {}},
            {"id": "S3", "area_km2": 3.0, "downstream": "S4", "parameters": {}},
            {"id": "S4", "area_km2": 4.0, "downstream": None, "parameters": {}},
        ],
    )

    runoff_models = [
        RunoffModelConfig(
            id="headwater",
            model_type="scs_curve_number",
            parameters={"curve_number": 70, "initial_abstraction_ratio": 0.1},
        ),
        RunoffModelConfig(
            id="headwater_wet",
            model_type="scs_curve_number",
            parameters={"curve_number": 82, "initial_abstraction_ratio": 0.05},
        ),
        RunoffModelConfig(
            id="mid_storage",
            model_type="linear_reservoir",
            parameters={"recession": 0.82, "conversion": 0.9},
        ),
        RunoffModelConfig(
            id="lowland_storage",
            model_type="linear_reservoir",
            parameters={"recession": 0.9, "conversion": 0.75},
        ),
    ]

    routing_models = [
        RoutingModelConfig(id="lag_fast", model_type="lag", parameters={"lag_steps": 1}),
        RoutingModelConfig(id="lag_medium", model_type="lag", parameters={"lag_steps": 2}),
        RoutingModelConfig(id="lag_slow", model_type="lag", parameters={"lag_steps": 3}),
    ]

    parameter_zones = [
        ParameterZoneConfig(
            id="ZU",
            description="Upstream gauge",
            control_points=["S1"],
            parameters={"runoff_model": "headwater", "routing_model": "lag_fast"},
        ),
        ParameterZoneConfig(
            id="ZM",
            description="Mid catchment reservoir",
            control_points=["S2"],
            parameters={"runoff_model": "mid_storage", "routing_model": "lag_medium"},
        ),
        ParameterZoneConfig(
            id="ZD",
            description="Downstream station",
            control_points=["S4"],
            parameters={"runoff_model": "lowland_storage", "routing_model": "lag_medium"},
        ),
    ]

    io_config = IOConfig(
        precipitation=Path("data/forcing/precipitation.csv"),
        results_directory=Path("results"),
    )

    return ModelConfig(
        delineation=delineation,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=io_config,
    )


def lag_route(values: Sequence[float], lag_steps: int) -> List[float]:
    """Shift a sequence forward by the requested lag steps, padding with zeros."""

    if lag_steps >= len(values):
        return [0.0] * lag_steps
    return [0.0] * lag_steps + list(values)[:-lag_steps]


def scs_runoff(precip: Sequence[float], curve_number: float, ia_ratio: float) -> List[float]:
    """SCS curve number runoff calculation used by the examples."""

    s = max(0.0, (1000.0 / curve_number - 10.0) * 25.4)
    ia = ia_ratio * s
    runoff: List[float] = []
    for p in precip:
        if p <= ia:
            runoff.append(0.0)
        else:
            q = (p - ia) ** 2 / (p - ia + s)
            runoff.append(q)
    return runoff


def linear_reservoir_runoff(
    precip: Sequence[float], recession: float, conversion: float, initial_storage: float
) -> List[float]:
    """Simple linear reservoir runoff implementation for the examples."""

    state = initial_storage
    flows: List[float] = []
    for p in precip:
        state = state * recession + p * conversion
        direct = (1.0 - recession) * state
        flows.append(direct)
    return flows


def _watershed_partition_example() -> ExampleDocumentation:
    """Validate watershed delineation areas and downstream relations."""

    with TemporaryDirectory() as tmpdir:
        base = Path(tmpdir)
        dem_path, pour_path = write_synthetic_delineation_inputs(base)
        delineation = DelineationConfig(
            dem_path=dem_path,
            pour_points_path=pour_path,
            accumulation_threshold=1.0,
        )
        subbasins = sorted(delineation.to_subbasins(), key=lambda sub: sub.id)
        dem_grid, _, downstream_map, _ = delineate_from_json(
            dem_path,
            pour_path,
            accumulation_threshold=1.0,
        )

        cell_area_km2 = abs(
            dem_grid.transform.pixel_width * dem_grid.transform.pixel_height
        ) / 1_000_000.0
        area_map = {sub.id: round(sub.area_km2, 3) for sub in subbasins}
        cell_counts = {
            sub.id: int(round(sub.area_km2 / cell_area_km2))
            for sub in subbasins
        }
        downstream_summary = {sub.id: downstream_map.get(sub.id) for sub in subbasins}

    expected_cells = {"S1": 3, "S2": 14, "S3": 16}
    if cell_counts != expected_cells:
        raise AssertionError(f"Unexpected watershed cell counts: {cell_counts}")

    expected_downstream = {"S1": "S2", "S2": "S3", "S3": None}
    if downstream_summary != expected_downstream:
        raise AssertionError(f"Unexpected downstream relations: {downstream_summary}")

    pour_ids = [
        feature["properties"]["id"] for feature in SYNTHETIC_POUR_POINTS["features"]
    ]
    grid_rows = len(SYNTHETIC_DEM_GRID["grid"])
    grid_cols = len(SYNTHETIC_DEM_GRID["grid"][0]) if grid_rows else 0
    cell_text = ", ".join(
        f"{basin}:{cell_counts[basin]}格" for basin in cell_counts
    )
    downstream_text = ", ".join(
        f"{basin}->{downstream_summary[basin] or '终点'}" for basin in downstream_summary
    )

    return ExampleDocumentation(
        slug="watershed_partition",
        title="流域划分示例：面积与下游关系验证",
        description="使用轻量 DEM 验证自动划分的子流域面积与下游拓扑。",
        inputs={
            "pour_points": pour_ids,
            "grid_shape": [grid_rows, grid_cols],
            "accumulation_threshold": 1.0,
            "cell_area_km2": round(cell_area_km2, 3),
        },
        outputs={
            "subbasin_areas_km2": area_map,
            "cell_counts": cell_counts,
            "downstream_relations": downstream_summary,
        },
        assertions=[
            f"面积单元格统计一致：{cell_text}",
            f"下游关系链：{downstream_text}",
        ],
    )


def _parameter_zone_assignment_example() -> ExampleDocumentation:
    subbasins = [
        Subbasin(id="S1", area_km2=10.0, downstream="S3"),
        Subbasin(id="S2", area_km2=12.0, downstream="S3"),
        Subbasin(id="S3", area_km2=20.0, downstream=None),
    ]

    zone_configs = [
        ParameterZoneConfig(
            id="Z1",
            description="Upstream gauge",
            control_points=["S1"],
            parameters={"runoff_model": "curve", "routing_model": "lag_short"},
        ),
        ParameterZoneConfig(
            id="Z2",
            description="Downstream control",
            control_points=["S3"],
            parameters={"runoff_model": "reservoir", "routing_model": "lag_short"},
        ),
    ]

    zones = ParameterZoneBuilder.from_config(zone_configs, subbasins)

    zone_map: Dict[str, List[str]] = {
        zone.id: list(zone.controlled_subbasins) for zone in zones
    }

    if zone_map.get("Z1") != ["S1"]:
        raise AssertionError("Zone Z1 should control only subbasin S1")
    if zone_map.get("Z2") != ["S2", "S3"]:
        raise AssertionError("Zone Z2 should automatically include downstream subbasins")

    return ExampleDocumentation(
        slug="parameter_zone_assignment",
        title="参数分区示例：控制点覆盖验证",
        description="验证参数分区在存在下游叠置时能保持控制范围的合理划分。",
        inputs={
            "subbasins": [
                {"id": sub.id, "area_km2": sub.area_km2, "downstream": sub.downstream}
                for sub in subbasins
            ],
            "zone_configs": [
                {
                    "id": cfg.id,
                    "control_points": cfg.control_points,
                    "parameters": cfg.parameters,
                }
                for cfg in zone_configs
            ],
        },
        outputs={"zone_map": zone_map},
        assertions=[
            "上游控制区 Z1 仅覆盖控制点 S1",
            "下游控制区 Z2 自动扩展包含中游与出口子流域",
        ],
    )


def _hand_calculated_run_example() -> ExampleDocumentation:
    config = build_sample_config()
    model = HydroSISModel.from_config(config)

    forcing: Dict[str, List[float]] = {
        "S1": [0.0, 20.0, 50.0],
        "S2": [5.0, 5.0, 5.0],
        "S3": [0.0, 0.0, 0.0],
    }

    routed = model.run(forcing)
    aggregated = model.accumulate_discharge(routed)

    expected_s1 = lag_route(scs_runoff(forcing["S1"], 75, 0.2), lag_steps=1)
    expected_s2 = lag_route(linear_reservoir_runoff(forcing["S2"], 0.85, 1.0, 0.0), 1)
    expected_s3 = lag_route(linear_reservoir_runoff(forcing["S3"], 0.85, 1.0, 0.0), 1)

    for actual, expected in zip(routed["S1"], expected_s1):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("S1 routing does not match analytical expectation")

    for actual, expected in zip(routed["S2"], expected_s2):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("S2 routing does not match analytical expectation")

    for actual, expected in zip(routed["S3"], expected_s3):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("S3 routing does not match analytical expectation")

    expected_total_s3 = [a + b + c for a, b, c in zip(expected_s1, expected_s2, expected_s3)]
    for actual, expected in zip(aggregated["S3"], expected_total_s3):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("Aggregated discharge does not match combined expectation")

    return ExampleDocumentation(
        slug="hand_calculated_run",
        title="模型运行示例：与手工计算结果一致",
        description="构建最小示例配置，验证模型模拟结果与手工计算完全一致。",
        inputs={"forcing": forcing},
        outputs={
            "routed": routed,
            "aggregated": {"S3": aggregated["S3"]},
            "expected": {
                "S1": expected_s1,
                "S2": expected_s2,
                "S3": expected_s3,
                "S3_total": expected_total_s3,
            },
        },
        assertions=[
            "每个子流域的产流与汇流序列与解析解逐项相符",
            "出口子流域的累积流量与分量叠加结果一致",
        ],
    )


def _scenario_modification_example() -> ExampleDocumentation:
    forcing = {
        "S1": [0.0, 20.0, 50.0],
        "S2": [5.0, 5.0, 5.0],
        "S3": [0.0, 0.0, 0.0],
    }

    baseline_config = build_sample_config()
    baseline_model = HydroSISModel.from_config(baseline_config)
    baseline_local = baseline_model.run(forcing)
    baseline_total = baseline_model.accumulate_discharge(baseline_local)

    scenario_config = build_sample_config()
    scenario_model = HydroSISModel.from_config(scenario_config)
    scenario_config.apply_scenario("alternate_routing", scenario_model.subbasins.values())
    scenario_local = scenario_model.run(forcing)
    scenario_total = scenario_model.accumulate_discharge(scenario_local)

    expected_baseline_s2 = lag_route(
        linear_reservoir_runoff(forcing["S2"], 0.85, 1.0, 0.0), 1
    )
    expected_scenario_s2 = lag_route(
        linear_reservoir_runoff(forcing["S2"], 0.85, 1.0, 0.0), 2
    )

    for actual, expected in zip(baseline_local["S2"], expected_baseline_s2):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("Baseline routing for S2 is incorrect")

    for actual, expected in zip(scenario_local["S2"], expected_scenario_s2):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("Scenario routing for S2 is incorrect")

    if baseline_local["S2"] == scenario_local["S2"]:
        raise AssertionError("Scenario should alter the routed discharge for S2")

    zone_baseline = baseline_model.parameter_zone_discharge(baseline_local)
    zone_scenario = scenario_model.parameter_zone_discharge(scenario_local)

    if zone_baseline["Z1"]["S1"] != baseline_total["S1"]:
        raise AssertionError("Baseline zone discharge mismatch for S1")
    if zone_scenario["Z1"]["S1"] != scenario_total["S1"]:
        raise AssertionError("Scenario zone discharge mismatch for S1")

    return ExampleDocumentation(
        slug="scenario_modification",
        title="情景修改示例：路由调整影响结果",
        description="应用情景修改后验证中游子流域路由延迟及参数区汇总的变化。",
        inputs={"forcing": forcing, "scenario_id": "alternate_routing"},
        outputs={
            "baseline_s2": baseline_local["S2"],
            "scenario_s2": scenario_local["S2"],
            "zone_baseline": zone_baseline,
            "zone_scenario": zone_scenario,
        },
        assertions=[
            "情景修改使得 S2 的路由序列出现明显滞后",
            "参数区 Z1 在基准与情景下均维持与 S1 累积流量一致",
        ],
    )


def _extended_runoff_models_example() -> ExampleDocumentation:
    precipitation = [10.0, 0.0, 5.0]
    subbasin = type("Subbasin", (), {"id": "TEST", "area_km2": 5.0, "downstream": None})

    configs = [
        (
            "xin_an_jiang",
            {
                "wm": 120.0,
                "b": 0.4,
                "imp": 0.02,
                "recession": 0.7,
            },
        ),
        (
            "wetspa",
            {
                "soil_storage_max": 250.0,
                "infiltration_coefficient": 0.5,
                "surface_runoff_coefficient": 0.35,
            },
        ),
        (
            "hymod",
            {
                "max_storage": 90.0,
                "beta": 1.2,
                "quickflow_ratio": 0.6,
                "num_quick_reservoirs": 3,
            },
        ),
    ]

    flow_preview: Dict[str, List[float]] = {}
    for idx, (model_type, parameters) in enumerate(configs):
        config = RunoffModelConfig(
            id=f"model_{idx}", model_type=model_type, parameters=parameters
        )
        model = config.build()
        flows = model.simulate(subbasin, precipitation)
        if len(flows) != len(precipitation):
            raise AssertionError(f"Runoff model {model_type} returned wrong length")
        if not all(math.isfinite(flow) for flow in flows):
            raise AssertionError(f"Runoff model {model_type} produced non-finite flow")
        flow_preview[model_type] = flows

    return ExampleDocumentation(
        slug="extended_runoff_models",
        title="扩展产流模型示例：统一接口验证",
        description="对多种新增产流模型进行统一调用，确认输出维度与数值稳定性。",
        inputs={"precipitation": precipitation},
        outputs={"flow_preview": flow_preview},
        assertions=[
            "所有扩展产流模型均输出与降雨序列等长的结果",
            "所有产流结果为有限数值，可用于后续汇流计算",
        ],
    )


def _multi_model_comparison_example() -> ExampleDocumentation:
    config = build_comparison_config()
    truth_model = HydroSISModel.from_config(config)

    forcing: Dict[str, List[float]] = {
        "S1": [5.0, 10.0, 20.0, 0.0],
        "S2": [0.0, 15.0, 10.0, 5.0],
        "S3": [0.0, 0.0, 5.0, 0.0],
        "S4": [1.0, 0.0, 2.0, 0.0],
    }

    observations = truth_model.accumulate_discharge(truth_model.run(forcing))

    calibrated_config = build_comparison_config()
    calibrated_model = HydroSISModel.from_config(calibrated_config)
    calibrated_results = calibrated_model.accumulate_discharge(
        calibrated_model.run(forcing)
    )

    biased_config = build_comparison_config()
    for runoff_cfg in biased_config.runoff_models:
        if runoff_cfg.id == "headwater":
            runoff_cfg.parameters["curve_number"] = 88
    biased_model = HydroSISModel.from_config(biased_config)
    biased_results = biased_model.accumulate_discharge(biased_model.run(forcing))

    sluggish_config = build_comparison_config()
    for routing_cfg in sluggish_config.routing_models:
        if routing_cfg.id == "lag_medium":
            routing_cfg.parameters["lag_steps"] = 4
    sluggish_model = HydroSISModel.from_config(sluggish_config)
    sluggish_results = sluggish_model.accumulate_discharge(
        sluggish_model.run(forcing)
    )

    simulations = {
        "calibrated": calibrated_results,
        "biased": biased_results,
        "sluggish": sluggish_results,
    }

    comparator = ModelComparator(SimulationEvaluator())
    scores = comparator.compare(simulations, observations)
    ranking = comparator.rank(scores, metric="rmse")

    if ranking[0].model_id != "calibrated":
        raise AssertionError("Calibrated model should rank first by RMSE")
    if ranking[-1].model_id != "biased":
        raise AssertionError("Biased model should be the least accurate")

    aggregated: Dict[str, Mapping[str, float]] = {
        score.model_id: score.aggregated for score in scores
    }

    if not (
        aggregated["calibrated"]["rmse"] < aggregated["sluggish"]["rmse"]
        and abs(aggregated["calibrated"]["pbias"]) < abs(aggregated["biased"]["pbias"])
    ):
        raise AssertionError("Calibrated model metrics should outperform alternatives")

    return ExampleDocumentation(
        slug="multi_model_comparison",
        title="多模型对比示例：指标排序验证",
        description="通过模型比较器验证不同模拟方案的精度排名与评价指标。",
        inputs={"forcing": forcing},
        outputs={
            "ranking": [score.model_id for score in ranking],
            "metrics": aggregated,
        },
        assertions=[
            "RMSE 排名将校准模型置于首位，偏差最大模型垫底",
            "综合误差指标显示校准模型在 RMSE 与 PBIAS 上均优于其他模型",
        ],
    )


def generate_example_documentation(
    output_directory: Optional[Path | str] = None,
) -> List[Path]:
    """Run all example validations and write Markdown documentation files."""

    output_dir = Path(output_directory) if output_directory else Path("docs") / "examples"
    output_dir.mkdir(parents=True, exist_ok=True)

    documents = [
        _watershed_partition_example(),
        _parameter_zone_assignment_example(),
        _hand_calculated_run_example(),
        _scenario_modification_example(),
        _extended_runoff_models_example(),
        _multi_model_comparison_example(),
    ]

    written: List[Path] = []
    for doc in documents:
        markdown = _render_markdown(doc)
        path = output_dir / f"{doc.slug}.md"
        path.write_text(markdown, encoding="utf-8")
        written.append(path)

    return written


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for generating example documentation."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "examples",
        help="示例 Markdown 文档输出目录 (默认: docs/examples)",
    )
    args = parser.parse_args(argv)

    paths = generate_example_documentation(args.output)
    for path in paths:
        print(f"Example documentation written to: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - allow CLI execution
    raise SystemExit(main())
