"""Run the GIS enhanced HydroSIS demonstration workflow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydrosis import HydroSISModel
from hydrosis.config import ModelConfig
from hydrosis.delineation.simple_grid import grid_to_geojson_features, load_dem
from hydrosis.io.gis_report import LeafletReportBuilder, build_html_table, write_html
from hydrosis.io.inputs import load_forcing
from hydrosis.io.outputs import write_simulation_results
from hydrosis.parameters.zone import ParameterZoneBuilder
CONFIG_PATH = ROOT / "config" / "gis_demo.json"


def load_config() -> ModelConfig:
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return ModelConfig.from_dict(data)


def build_gis_layers(config: ModelConfig) -> Dict[str, dict]:
    """Create GeoJSON layers used in the final Leaflet report."""

    dem_path = ROOT / config.delineation.dem_path
    dem = load_dem(dem_path)

    cell_values = {(row, col): value for row, col, value in dem.iter_cells()}
    dem_features = grid_to_geojson_features(dem, cell_values, "elevation")

    derived_dir = dem_path.parent / "derived"
    subbasin_geojson = json.loads((derived_dir / "subbasins.geojson").read_text(encoding="utf-8"))
    accumulation_geojson = json.loads(
        (derived_dir / "flow_accumulation.geojson").read_text(encoding="utf-8")
    )

    base_dir = ROOT / config.delineation.dem_path.parent
    layers = {
        "dem": {"type": "FeatureCollection", "features": dem_features},
        "soil": json.loads((base_dir / "soil.geojson").read_text(encoding="utf-8")),
        "landuse": json.loads((base_dir / "landuse.geojson").read_text(encoding="utf-8")),
        "stations": json.loads((base_dir / "stations.geojson").read_text(encoding="utf-8")),
        "reservoirs": json.loads((base_dir / "reservoirs.geojson").read_text(encoding="utf-8")),
        "streams": json.loads((ROOT / config.delineation.burn_streams_path).read_text(encoding="utf-8"))
        if config.delineation.burn_streams_path
        else {"type": "FeatureCollection", "features": []},
        "subbasins": subbasin_geojson,
        "accumulation": accumulation_geojson,
    }

    return layers


def build_parameter_zone_geojson(subbasin_geojson: dict, zones) -> dict:
    """Aggregate subbasin polygons for each parameter zone."""

    polygon_lookup = {
        feature["properties"]["id"]: feature["geometry"]["coordinates"]
        for feature in subbasin_geojson.get("features", [])
    }
    features = []
    for zone in zones:
        multipolygon: List[List[List[float]]] = []
        for sub_id in zone.controlled_subbasins:
            coords = polygon_lookup.get(sub_id)
            if coords:
                multipolygon.extend(coords)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": multipolygon},
                "properties": {
                    "id": zone.id,
                    "description": zone.description,
                    "subbasins": ", ".join(zone.controlled_subbasins),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def run_workflow() -> None:
    config = load_config()
    # Trigger delineation and parameter zone construction
    delineated = config.delineation.to_subbasins()
    zones = ParameterZoneBuilder.from_config(config.parameter_zones, delineated)

    # Build the simulation model and execute baseline run
    model = HydroSISModel.from_config(config)
    forcing = load_forcing(ROOT / config.io.precipitation)
    routed = model.run(forcing)
    aggregated = model.accumulate_discharge(routed)

    write_simulation_results(ROOT / config.io.results_directory / "baseline", aggregated)

    # Prepare GIS layers
    layers = build_gis_layers(config)
    zone_geojson = build_parameter_zone_geojson(layers["subbasins"], zones)

    # Summary sections
    sub_rows = [
        (
            sub.id,
            f"{sub.area_km2:.2f}",
            sub.downstream or "—",
            ", ".join(
                zone.id for zone in zones if sub.id in zone.controlled_subbasins
            ),
        )
        for sub in delineated
    ]
    zone_rows = [
        (zone.id, zone.description, ", ".join(zone.controlled_subbasins))
        for zone in zones
    ]

    report = LeafletReportBuilder(
        "HydroSIS GIS 演示报告",
        "示例数据集展示了 DEM 流域划分、参数分区及关键基础地理数据。",
    )
    report.add_section(
        "<div class=\"section\">"
        + "<h2>流域划分概览</h2>"
        + build_html_table(["子流域", "面积 (km²)", "下游", "所属参数区"], sub_rows)
        + "</div>"
    )
    report.add_section(
        "<div class=\"section\">"
        + "<h2>参数分区</h2>"
        + build_html_table(["参数区", "描述", "覆盖子流域"], zone_rows)
        + "</div>"
    )

    report.add_geojson_layer("DEM", layers["dem"], popup_fields=["elevation"])
    report.add_geojson_layer("土壤类型", layers["soil"], popup_fields=["soil", "hydrologic_group"])
    report.add_geojson_layer("土地利用", layers["landuse"], popup_fields=["landuse"])
    report.add_geojson_layer("水文站", layers["stations"], popup_fields=["name", "type"], point_radius=8)
    report.add_geojson_layer("水库", layers["reservoirs"], popup_fields=["name", "storage_mcm"], point_radius=8)
    report.add_geojson_layer("河网", layers["streams"], popup_fields=["name"], color="#0050b5")
    report.add_geojson_layer("流域划分", layers["subbasins"], popup_fields=["id"])
    report.add_geojson_layer("参数区划分", zone_geojson, popup_fields=["id", "description", "subbasins"])

    output_html = ROOT / config.io.reports_directory / "gis_report.html"
    write_html(output_html, report.build())
    print(f"GIS 报告已生成: {output_html}")


if __name__ == "__main__":
    run_workflow()

