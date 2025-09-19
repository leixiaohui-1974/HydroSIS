"""Run the bundled HydroSIS example configuration and persist outputs."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List, Mapping, Sequence

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
from hydrosis.io.gis_report import (
    LeafletReportBuilder,
    accumulation_statistics,
    build_bullet_list,
    build_card,
    build_html_table,
    build_parameter_zone_geojson,
    embed_image,
    gather_gis_layers,
    summarise_paths,
    write_html,
)
from hydrosis.parameters.zone import ParameterZoneBuilder

import importlib.util

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
if HAS_MPL:
    from matplotlib import pyplot as plt
else:  # pragma: no cover - matplotlib is optional
    plt = None  # type: ignore[assignment]


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

    delineated = config.delineation.to_subbasins()
    zones = ParameterZoneBuilder.from_config(config.parameter_zones, delineated)

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

    _build_gis_summary(
        config,
        delineated,
        zones,
        workflow_result,
        repo_root,
    )

    print(f"Simulation outputs stored in: {config.io.results_directory}")


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _plot_subbasin_area(subbasins, output_path: Path) -> Path | None:
    if plt is None:
        return None
    labels = [sub.id for sub in subbasins]
    areas = [sub.area_km2 for sub in subbasins]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, areas, color="#2ca02c")
    ax.set_ylabel("面积 (km²)")
    ax.set_xlabel("子流域")
    ax.set_title("子流域面积分布")
    ax.grid(axis="y", alpha=0.3)
    _ensure_dir(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_scenario_hydrograph(
    baseline: Dict[str, List[float]],
    scenarios: Dict[str, Dict[str, List[float]]],
    subbasin_id: str,
    output_path: Path,
) -> Path | None:
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    time_steps = range(len(baseline.get(subbasin_id, [])))
    ax.plot(time_steps, baseline.get(subbasin_id, []), label="baseline", color="#1f77b4")
    colors = ["#d62728", "#9467bd", "#8c564b"]
    for idx, (scenario_id, data) in enumerate(scenarios.items()):
        ax.plot(time_steps, data.get(subbasin_id, []), label=scenario_id, color=colors[idx % len(colors)])
    ax.set_title(f"子流域 {subbasin_id} 情景累积流量对比")
    ax.set_xlabel("时间步")
    ax.set_ylabel("流量 (m³/s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")
    _ensure_dir(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _summarise_results(subbasins, zones, baseline: Dict[str, List[float]], accumulation_layer: dict) -> List[str]:
    total_area = sum(sub.area_km2 for sub in subbasins)
    largest = max(subbasins, key=lambda sub: sub.area_km2)
    stats = accumulation_statistics(accumulation_layer)
    peak_info = {
        sid: max(values) if values else 0.0 for sid, values in baseline.items()
    }
    peak_sub = max(peak_info.items(), key=lambda item: item[1]) if peak_info else ("—", 0.0)
    scenario_zones = {}
    for zone in zones:
        area = sum(
            next((sub.area_km2 for sub in subbasins if sub.id == sid), 0.0)
            for sid in zone.controlled_subbasins
        )
        scenario_zones[zone.id] = area
    return [
        f"总计 {len(subbasins)} 个子流域，覆盖面积 {total_area:.1f} km²，最大子流域 {largest.id}。",
        f"累积流量峰值出现在 {peak_sub[0]}，约 {peak_sub[1]:.2f} m³/s。",
        "参数区面积分布：" + ", ".join(f"{zone_id} {area:.1f} km²" for zone_id, area in scenario_zones.items()),
        f"栅格流量累积值范围 {stats['min']:.1f} – {stats['max']:.1f}，平均 {stats['mean']:.1f}。",
    ]


def _build_gis_summary(
    config: ModelConfig,
    subbasins,
    zones,
    workflow_result,
    repo_root: Path,
) -> None:
    layers = gather_gis_layers(config, repo_root)
    zone_geojson = build_parameter_zone_geojson(layers.get("subbasins", {}), zones)

    sub_rows = [
        (
            sub.id,
            f"{sub.area_km2:.2f}",
            sub.downstream or "—",
            ", ".join(zone.id for zone in zones if sub.id in zone.controlled_subbasins),
        )
        for sub in subbasins
    ]
    zone_rows = [
        (zone.id, zone.description, ", ".join(zone.controlled_subbasins))
        for zone in zones
    ]

    figures_dir = config.io.figures_directory
    area_fig = _plot_subbasin_area(subbasins, figures_dir / "subbasin_area.png")

    scenario_series = {
        scenario_id: result.aggregated
        for scenario_id, result in workflow_result.scenarios.items()
    }
    outlet = subbasins[-1].id if subbasins else ""
    scenario_fig = _plot_scenario_hydrograph(
        workflow_result.baseline.aggregated,
        scenario_series,
        outlet,
        figures_dir / "scenario_comparison.png",
    )

    report = LeafletReportBuilder(
        "HydroSIS 示例情景 GIS 报告",
        "展示示例情景的基础地理数据、流域划分、参数区及情景结果对比。",
    )
    report.add_section(
        build_card(
            "输入数据概览",
            build_html_table(
                ["路径", "类型", "大小", "备注"],
                summarise_paths(
                    [
                        config.delineation.dem_path,
                        config.delineation.pour_points_path,
                        config.delineation.burn_streams_path
                        if config.delineation.burn_streams_path
                        else repo_root / "data/sample/gis/streams.geojson",
                        config.io.precipitation,
                        config.io.discharge_observations,
                    ]
                ),
            ),
        )
    )
    report.add_section(
        build_card(
            "流域划分概览",
            build_html_table(["子流域", "面积 (km²)", "下游", "所属参数区"], sub_rows)
            + (embed_image(area_fig, "子流域面积分布") if area_fig else ""),
        )
    )
    report.add_section(
        build_card(
            "参数区与结果分析",
            build_html_table(["参数区", "描述", "覆盖子流域"], zone_rows)
            + build_bullet_list(
                _summarise_results(
                    subbasins,
                    zones,
                    workflow_result.baseline.aggregated,
                    layers.get("accumulation", {}),
                )
            )
            + (
                embed_image(scenario_fig, f"出口子流域 {outlet} 情景流量对比")
                if scenario_fig and outlet
                else ""
            ),
        )
    )

    report.add_geojson_layer(
        "DEM", layers.get("dem", {"type": "FeatureCollection", "features": []}), popup_fields=["elevation"]
    )
    report.add_geojson_layer(
        "土壤类型",
        layers.get("soil", {"type": "FeatureCollection", "features": []}),
        popup_fields=["soil", "hydrologic_group"],
    )
    report.add_geojson_layer(
        "土地利用",
        layers.get("landuse", {"type": "FeatureCollection", "features": []}),
        popup_fields=["landuse"],
    )
    report.add_geojson_layer(
        "水文站",
        layers.get("stations", {"type": "FeatureCollection", "features": []}),
        popup_fields=["name", "type"],
        point_radius=8,
    )
    report.add_geojson_layer(
        "水库",
        layers.get("reservoirs", {"type": "FeatureCollection", "features": []}),
        popup_fields=["name", "storage_mcm"],
        point_radius=8,
    )
    report.add_geojson_layer(
        "河网", layers.get("streams", {"type": "FeatureCollection", "features": []}), popup_fields=["name"], color="#0050b5"
    )
    report.add_geojson_layer(
        "流域划分",
        layers.get("subbasins", {"type": "FeatureCollection", "features": []}),
        popup_fields=["id"],
    )
    report.add_geojson_layer(
        "参数区划分",
        zone_geojson,
        popup_fields=["id", "description", "subbasins"],
    )

    output_html = config.io.reports_directory / "gis_overview.html"
    write_html(output_html, report.build())
    print(f"GIS report saved to: {output_html}")


if __name__ == "__main__":
    main()
