from __future__ import annotations

import base64
import json
from pathlib import Path

from hydrosis.config import ModelConfig
from hydrosis.io.gis_report import (
    accumulation_statistics,
    build_parameter_zone_geojson,
    embed_image,
    gather_gis_layers,
    summarise_paths,
)
from hydrosis.parameters.zone import ParameterZoneBuilder


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_demo_config() -> ModelConfig:
    config_path = REPO_ROOT / "config" / "gis_demo.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    return ModelConfig.from_dict(data)


def test_summarise_paths(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("demo", encoding="utf-8")
    dir_path = tmp_path / "folder"
    dir_path.mkdir()
    (dir_path / "a.txt").write_text("x", encoding="utf-8")

    rows = summarise_paths([file_path, dir_path, tmp_path / "missing.txt"])
    assert rows[0][1] == "文件"
    assert "KB" in rows[0][2] or rows[0][2].endswith("B")
    assert rows[1][1] == "目录"
    assert "包含" in rows[1][3]
    assert rows[2][1] == "不存在"


def test_embed_image_creates_data_url(tmp_path: Path) -> None:
    # 1x1 transparent PNG
    png_bytes = base64.b64decode(
        b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    img_path = tmp_path / "pixel.png"
    img_path.write_bytes(png_bytes)

    html = embed_image(img_path)
    assert "data:image/png;base64," in html


def test_gather_layers_and_zones() -> None:
    config = _load_demo_config()
    subbasins = config.delineation.to_subbasins()
    zones = ParameterZoneBuilder.from_config(config.parameter_zones, subbasins)

    layers = gather_gis_layers(config, REPO_ROOT)
    assert "dem" in layers and layers["dem"]["type"] == "FeatureCollection"
    assert layers["subbasins"]["features"]

    zone_geojson = build_parameter_zone_geojson(layers["subbasins"], zones)
    assert zone_geojson["features"], "zone features should be created"


def test_accumulation_statistics() -> None:
    config = _load_demo_config()
    layers = gather_gis_layers(config, REPO_ROOT)
    stats = accumulation_statistics(layers["accumulation"])
    assert stats["max"] >= stats["min"]
    assert stats["mean"] >= 0
