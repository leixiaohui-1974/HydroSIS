from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Dict, List, Optional
import sys

HAS_MPL = importlib.util.find_spec("matplotlib") is not None
if HAS_MPL:
    from matplotlib import pyplot as plt
else:  # pragma: no cover - matplotlib is optional
    plt = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydrosis import HydroSISModel
from hydrosis.config import ModelConfig
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
from hydrosis.io.inputs import load_forcing
from hydrosis.io.outputs import write_simulation_results
from hydrosis.parameters.zone import ParameterZoneBuilder

CONFIG_PATH = ROOT / "config" / "gis_demo.json"


def load_config() -> ModelConfig:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return ModelConfig.from_dict(data)


def _ensure_directory(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _plot_subbasin_area(subbasins, output_path: Path) -> Optional[Path]:
    if plt is None:
        return None
    labels = [sub.id for sub in subbasins]
    areas = [sub.area_km2 for sub in subbasins]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, areas, color="#1f77b4")
    ax.set_ylabel("面积 (km²)")
    ax.set_xlabel("子流域")
    ax.set_title("子流域面积分布")
    ax.grid(axis="y", alpha=0.3)
    _ensure_directory(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _plot_aggregated_flows(series: Dict[str, List[float]], output_path: Path) -> Optional[Path]:
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    time_steps = range(len(next(iter(series.values()), [])))
    for sub_id, values in series.items():
        ax.plot(time_steps, values, label=sub_id)
    ax.set_title("子流域累积流量过程")
    ax.set_xlabel("时间步")
    ax.set_ylabel("流量 (m³/s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize="small")
    _ensure_directory(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _analysis_bullets(subbasins, zones, flows: Dict[str, List[float]], accumulation_layer: dict) -> List[str]:
    total_area = sum(sub.area_km2 for sub in subbasins)
    largest = max(subbasins, key=lambda sub: sub.area_km2)
    flow_peaks = {sub_id: max(values) if values else 0.0 for sub_id, values in flows.items()}
    peak_sub = max(flow_peaks.items(), key=lambda item: item[1]) if flow_peaks else ("—", 0.0)
    stats = accumulation_statistics(accumulation_layer)

    zone_area = {}
    for zone in zones:
        area = sum(
            next((sub.area_km2 for sub in subbasins if sub.id == sid), 0.0)
            for sid in zone.controlled_subbasins
        )
        zone_area[zone.id] = area

    return [
        f"共划分 {len(subbasins)} 个子流域，总面积约 {total_area:.1f} km²，最大子流域 {largest.id} 面积 {largest.area_km2:.1f} km²。",
        f"子流域峰值流量由 {peak_sub[0]} 控制，峰值约 {peak_sub[1]:.2f} m³/s。",
        "参数区面积分布：" + ", ".join(f"{zone_id} {area:.1f} km²" for zone_id, area in zone_area.items()),
        f"流量累积栅格最小值 {stats['min']:.1f}、最大值 {stats['max']:.1f}，平均贡献 {stats['mean']:.1f}。",
    ]


def run_workflow() -> None:
    config = load_config()
    delineated = config.delineation.to_subbasins()
    zones = ParameterZoneBuilder.from_config(config.parameter_zones, delineated)
    model = HydroSISModel.from_config(config)
    forcing = load_forcing(ROOT / config.io.precipitation)
    routed = model.run(forcing)
    aggregated = model.accumulate_discharge(routed)

    write_simulation_results(ROOT / config.io.results_directory / "baseline", aggregated)

    layers = gather_gis_layers(config, ROOT)
    zone_geojson = build_parameter_zone_geojson(layers.get("subbasins", {}), zones)

    sub_rows = [
        (
            sub.id,
            f"{sub.area_km2:.2f}",
            sub.downstream or "—",
            ", ".join(zone.id for zone in zones if sub.id in zone.controlled_subbasins),
        )
        for sub in delineated
    ]
    zone_rows = [
        (zone.id, zone.description, ", ".join(zone.controlled_subbasins))
        for zone in zones
    ]

    figures_dir = ROOT / config.io.figures_directory
    area_fig = _plot_subbasin_area(delineated, figures_dir / "subbasin_area.png")
    flows_fig = _plot_aggregated_flows(aggregated, figures_dir / "aggregated_flows.png")

    report = LeafletReportBuilder(
        "HydroSIS GIS 演示报告",
        "示例数据集展示了 DEM 流域划分、参数分区、模拟结果与关键基础地理数据。",
    )
    report.add_section(
        build_card(
            "输入数据概览",
            build_html_table(
                ["路径", "类型", "大小", "备注"],
                summarise_paths(
                    [
                        ROOT / config.delineation.dem_path,
                        ROOT / config.delineation.pour_points_path,
                        ROOT / config.delineation.burn_streams_path
                        if config.delineation.burn_streams_path
                        else ROOT / "data/sample/gis/streams.geojson",
                        ROOT / config.io.precipitation,
                        ROOT / config.io.discharge_observations,
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
                _analysis_bullets(delineated, zones, aggregated, layers.get("accumulation", {}))
            )
            + (embed_image(flows_fig, "子流域累积流量过程") if flows_fig else ""),
        )
    )

    report.add_geojson_layer("DEM", layers.get("dem", {"type": "FeatureCollection", "features": []}), popup_fields=["elevation"])
    report.add_geojson_layer("土壤类型", layers.get("soil", {"type": "FeatureCollection", "features": []}), popup_fields=["soil", "hydrologic_group"])
    report.add_geojson_layer("土地利用", layers.get("landuse", {"type": "FeatureCollection", "features": []}), popup_fields=["landuse"])
    report.add_geojson_layer("水文站", layers.get("stations", {"type": "FeatureCollection", "features": []}), popup_fields=["name", "type"], point_radius=8)
    report.add_geojson_layer("水库", layers.get("reservoirs", {"type": "FeatureCollection", "features": []}), popup_fields=["name", "storage_mcm"], point_radius=8)
    report.add_geojson_layer("河网", layers.get("streams", {"type": "FeatureCollection", "features": []}), popup_fields=["name"], color="#0050b5")
    report.add_geojson_layer("流域划分", layers.get("subbasins", {"type": "FeatureCollection", "features": []}), popup_fields=["id"])
    report.add_geojson_layer("参数区划分", zone_geojson, popup_fields=["id", "description", "subbasins"])

    output_html = ROOT / config.io.reports_directory / "gis_report.html"
    write_html(output_html, report.build())
    print(f"GIS 报告已生成: {output_html}")


if __name__ == "__main__":
    run_workflow()

