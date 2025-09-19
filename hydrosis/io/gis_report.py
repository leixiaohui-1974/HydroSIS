"""Helpers to produce lightweight Leaflet based GIS reports."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence


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
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #dde; padding: 0.35rem 0.5rem; text-align: left; }}
    th {{ background: #eef2f7; }}
    .popup-table {{ border-collapse: collapse; }}
    .popup-table th, .popup-table td {{ border: 1px solid #ccc; padding: 0.25rem 0.35rem; }}
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


__all__ = ["LeafletReportBuilder", "build_html_table", "write_html"]
