"""Helpers to produce lightweight Leaflet based GIS reports."""
from __future__ import annotations

from dataclasses import dataclass, field
import base64
import json
import mimetypes
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

try:  # pragma: no cover - optional dependency used only when available
    import numpy as _np
except Exception:  # pragma: no cover - keep working without numpy
    _np = None  # type: ignore

from ..delineation.simple_grid import grid_to_geojson_features, load_dem

if _np is None:  # pragma: no cover - fallback sum implementation
    def _np_sum(values: Iterable[float]) -> float:
        total = 0.0
        for value in values:
            total += float(value)
        return total
else:  # pragma: no cover - delegate to numpy when available
    def _np_sum(values: Iterable[float]) -> float:
        return float(_np.sum(list(values)))


@dataclass
class LayerDefinition:
    label: str
    data: Mapping[str, object]
    color: str
    popup_fields: Sequence[str] = field(default_factory=tuple)
    point_radius: int = 6


class LeafletReportBuilder:
    """Build a single page HTML report embedding GIS layers."""

    def __init__(self, title: str, description: str = "") -> None:
        self.title = title
        self.description = description
        self.layers: List[LayerDefinition] = []
        self.sections: List[str] = []
        self._palette = iter(
            [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
            ]
        )

    def _next_color(self) -> str:
        try:
            return next(self._palette)
        except StopIteration:
            self._palette = iter(
                [
                    "#1f77b4",
                    "#ff7f0e",
                    "#2ca02c",
                    "#d62728",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                ]
            )
            return next(self._palette)

    def add_geojson_layer(
        self,
        label: str,
        data: Mapping[str, object],
        popup_fields: Optional[Sequence[str]] = None,
        color: Optional[str] = None,
        point_radius: int = 6,
    ) -> None:
        self.layers.append(
            LayerDefinition(
                label=label,
                data=data,
                color=color or self._next_color(),
                popup_fields=tuple(popup_fields or ()),
                point_radius=point_radius,
            )
        )

    def add_section(self, html_content: str) -> None:
        self.sections.append(html_content)

    def build(self) -> str:
        layer_js_snippets: List[str] = []

        for idx, layer in enumerate(self.layers):
            var_name = f"layer{idx}"
            geojson_str = json.dumps(layer.data)
            popup_fields = json.dumps(list(layer.popup_fields))
            snippet = f"""
const {var_name} = L.geoJSON({geojson_str}, {{
    style: function(feature) {{
        return {{ color: '{layer.color}', weight: 1, fillOpacity: 0.35 }};
    }},
    pointToLayer: function(feature, latlng) {{
        return L.circleMarker(latlng, {{
            radius: {layer.point_radius},
            color: '{layer.color}',
            weight: 1,
            fillOpacity: 0.85
        }});
    }},
    onEachFeature: function(feature, layer) {{
        const fields = {popup_fields};
        if (!fields.length) {{
            return;
        }}
        const rows = fields
            .map(name => `<tr><th>${{name}}</th><td>${{feature.properties?.[name] ?? ''}}</td></tr>`)
            .join('');
        if (rows) {{
            layer.bindPopup(`<table class=\"popup-table\">${{rows}}</table>`);
        }}
    }}
}}).addTo(map);
if ({var_name}.getBounds && {var_name}.getBounds().isValid()) {{
    bounds = bounds ? bounds.extend({var_name}.getBounds()) : {var_name}.getBounds();
}}
layers['{layer.label}'] = {var_name};
"""
            layer_js_snippets.append(snippet)

        sections_html = "\n".join(self.sections)

        return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>{self.title}</title>
  <link
    rel=\"stylesheet\"
    href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\"
    integrity=\"sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=\"
    crossorigin=\"\"
  />
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0 1rem; background: #f9fafc; }}
    h1 {{ margin-top: 1rem; }}
    #map {{ height: 640px; margin: 1rem 0; border: 1px solid #ccd; }}
    .section {{ margin-bottom: 1.5rem; background: #fff; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
    .section h2 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #dde; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #eef2f7; }}
    .popup-table {{ border-collapse: collapse; }}
    .popup-table th, .popup-table td {{ border: 1px solid #ccc; padding: 0.25rem 0.35rem; }}
    ul {{ padding-left: 1.25rem; }}
    figure {{ margin: 1rem 0; text-align: center; }}
    figure img {{ max-width: 100%; border: 1px solid #ccd; border-radius: 4px; }}
    figure figcaption {{ font-size: 0.9rem; color: #555; margin-top: 0.4rem; }}
    .file-meta {{ font-size: 0.9rem; color: #555; margin-top: 0.3rem; }}
  </style>
</head>
<body>
  <h1>{self.title}</h1>
  <p>{self.description}</p>
  {sections_html}
  <div id=\"map\"></div>
  <script
    src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"
    integrity=\"sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=\"
    crossorigin=\"\"
  ></script>
  <script>
    const map = L.map('map');
    let bounds = null;
    const base = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
        maxZoom: 19,
        attribution: '© OpenStreetMap contributors'
    }}).addTo(map);
    const layers = {{}};
    {''.join(layer_js_snippets)}
    L.control.layers(null, layers, {{ collapsed: false }}).addTo(map);
    if (bounds && bounds.isValid()) {{
        map.fitBounds(bounds.pad(0.15));
    }} else {{
        map.setView([0, 0], 3);
    }}
  </script>
</body>
</html>
"""


def build_html_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> str:
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    row_html = "".join(
        "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{row_html}</tbody></table>"


def write_html(path: Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_card(title: str, body: str) -> str:
    """Wrap ``body`` HTML inside a styled card with a heading."""

    return f"<div class=\"section\"><h2>{title}</h2>{body}</div>"


def build_bullet_list(items: Sequence[str]) -> str:
    """Create a HTML unordered list from the provided strings."""

    return "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"


def _format_size(num_bytes: float) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def summarise_paths(paths: Sequence[Path]) -> List[Sequence[object]]:
    """Return table rows describing the provided files or directories."""

    rows: List[Sequence[object]] = []
    for raw in paths:
        path = Path(raw)
        kind = "不存在"
        size_text = "—"
        details = ""
        if path.is_file():
            kind = "文件"
            size_text = _format_size(path.stat().st_size)
        elif path.is_dir():
            kind = "目录"
            total_size = 0.0
            file_count = 0
            try:
                for entry in path.rglob("*"):
                    if entry.is_file():
                        file_count += 1
                        total_size += entry.stat().st_size
            except FileNotFoundError:
                file_count = 0
            size_text = _format_size(total_size)
            details = f"包含 {file_count} 个文件"
        rows.append((path.as_posix(), kind, size_text, details or ""))
    return rows


def encode_image(path: Path) -> str:
    """Return a ``data:`` URL for embedding the specified image."""

    path = Path(path)
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def embed_image(path: Path, caption: Optional[str] = None, embed_data: bool = True) -> str:
    """Create a HTML ``<figure>`` element referencing the image."""

    img_src = encode_image(path) if embed_data else Path(path).as_posix()
    caption_html = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f"<figure><img src=\"{img_src}\" alt=\"{Path(path).stem}\" />{caption_html}</figure>"


def load_geojson(path: Path) -> Optional[Mapping[str, object]]:
    """Safely read a GeoJSON file returning ``None`` when missing."""

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


def gather_gis_layers(config, repository_root: Path) -> Mapping[str, Mapping[str, object]]:
    """Collect DEM, thematic layers and derived products for reporting."""

    repo_root = Path(repository_root)
    layers: dict[str, Mapping[str, object]] = {}

    dem_path = Path(getattr(config.delineation, "dem_path", ""))
    if not dem_path.is_absolute():
        dem_path = repo_root / dem_path

    if dem_path.suffix.lower() == ".json" and dem_path.exists():
        dem = load_dem(dem_path)
        values = {(row, col): value for row, col, value in dem.iter_cells()}
        layers["dem"] = {
            "type": "FeatureCollection",
            "features": grid_to_geojson_features(dem, values, "elevation"),
        }
        derived_dir = dem_path.parent / "derived"
    else:
        # Attempt to fall back to the bundled synthetic grid for richer visuals.
        fallback_dem = repo_root / "data/sample/gis/dem_grid.json"
        derived_dir = fallback_dem.parent / "derived"
        if fallback_dem.exists():
            dem = load_dem(fallback_dem)
            values = {(row, col): value for row, col, value in dem.iter_cells()}
            layers["dem"] = {
                "type": "FeatureCollection",
                "features": grid_to_geojson_features(dem, values, "elevation"),
            }

    base_dirs = [
        dem_path.parent,
        Path(getattr(config.delineation, "pour_points_path", "")).resolve().parent
        if getattr(config.delineation, "pour_points_path", None)
        else dem_path.parent,
        repo_root / "data/sample/gis",
    ]

    def _load_from_candidates(filename: str) -> Optional[Mapping[str, object]]:
        for directory in base_dirs:
            candidate = directory / filename
            geojson = load_geojson(candidate)
            if geojson:
                return geojson
        return None

    layers["soil"] = _load_from_candidates("soil.geojson") or {"type": "FeatureCollection", "features": []}
    layers["landuse"] = _load_from_candidates("landuse.geojson") or {"type": "FeatureCollection", "features": []}
    layers["stations"] = _load_from_candidates("stations.geojson") or {"type": "FeatureCollection", "features": []}
    layers["reservoirs"] = _load_from_candidates("reservoirs.geojson") or {"type": "FeatureCollection", "features": []}
    layers["streams"] = _load_from_candidates("streams.geojson") or {"type": "FeatureCollection", "features": []}

    subbasins_geojson = None
    accumulation_geojson = None
    derived_candidates = [
        derived_dir,
        dem_path.parent / "derived",
        repo_root / "data/sample/gis/derived",
    ]
    for directory in derived_candidates:
        sub_candidate = directory / "subbasins.geojson"
        acc_candidate = directory / "flow_accumulation.geojson"
        if sub_candidate.exists() and subbasins_geojson is None:
            subbasins_geojson = load_geojson(sub_candidate)
        if acc_candidate.exists() and accumulation_geojson is None:
            accumulation_geojson = load_geojson(acc_candidate)

    layers["subbasins"] = subbasins_geojson or {"type": "FeatureCollection", "features": []}
    layers["accumulation"] = accumulation_geojson or {"type": "FeatureCollection", "features": []}

    return layers


def build_parameter_zone_geojson(subbasin_geojson: Mapping[str, object], zones: Sequence[object]) -> Mapping[str, object]:
    """Aggregate subbasin polygons for each parameter zone."""

    features = subbasin_geojson.get("features", []) if isinstance(subbasin_geojson, Mapping) else []
    polygon_lookup = {}
    for feature in features:
        props = feature.get("properties", {}) if isinstance(feature, Mapping) else {}
        polygon_lookup[props.get("id")] = feature.get("geometry", {}).get("coordinates", [])

    zone_features: List[Mapping[str, object]] = []
    for zone in zones:
        zone_id = getattr(zone, "id", None) or zone.get("id")  # type: ignore[attr-defined]
        description = getattr(zone, "description", None) or zone.get("description", "")  # type: ignore[attr-defined]
        controlled = getattr(zone, "controlled_subbasins", None) or zone.get("controlled_subbasins", [])  # type: ignore[attr-defined]
        coords: List[List] = []
        for sub_id in controlled:
            coordinates = polygon_lookup.get(sub_id)
            if coordinates:
                coords.extend(coordinates)
        zone_features.append(
            {
                "type": "Feature",
                "geometry": {"type": "MultiPolygon", "coordinates": coords},
                "properties": {
                    "id": zone_id,
                    "description": description,
                    "subbasins": ", ".join(controlled),
                },
            }
        )
    return {"type": "FeatureCollection", "features": zone_features}


def accumulation_statistics(accumulation_geojson: Mapping[str, object]) -> Mapping[str, float]:
    """Compute simple statistics of the flow accumulation layer."""

    features = accumulation_geojson.get("features", []) if isinstance(accumulation_geojson, Mapping) else []
    values = [feature.get("properties", {}).get("accumulation", 0.0) for feature in features]
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": _np_sum(values) / max(len(values), 1),
    }


__all__ = [
    "LeafletReportBuilder",
    "build_html_table",
    "write_html",
    "build_card",
    "build_bullet_list",
    "summarise_paths",
    "embed_image",
    "gather_gis_layers",
    "build_parameter_zone_geojson",
    "accumulation_statistics",
]
