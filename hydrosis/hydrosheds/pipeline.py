from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Mapping, Optional

from .client import HydroSHEDSClient
from .processor import HydroSHEDSProcessor
from .dem import run_dem_flow_pipeline
from ..io.spatial_db import init_db, write_layers


class HydroSHEDSPipeline:
    """Batch pipeline to download/select a basin and produce standardized outputs."""

    def __init__(self, cache_root: Path, results_root: Path) -> None:
        self.cache_root = Path(cache_root)
        self.results_root = Path(results_root)
        self.client = HydroSHEDSClient(self.cache_root)
        self.processor = HydroSHEDSProcessor()

    def run(self, input_params: Mapping[str, object]) -> Dict[str, Path]:
        """Execute the pipeline.

        input_params supports keys:
        - bounding_box: ignored for WBD query in this minimal implementation
        - basin_code: optional HUC12 code to prioritize (not used here)
        - output_format: e.g. "geojson"
        """
        self.results_root.mkdir(parents=True, exist_ok=True)
        process_log: Dict[str, object] = {
            "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": [],
        }

        # Step 1: Download candidates with adjustable area range
        candidates_path = self.cache_root / "wbd_candidates.geojson"
        bbox = input_params.get("bounding_box")
        area_range = input_params.get("area_range") or (480, 550)
        force_bbox = bool(input_params.get("force_bbox") or False)
        min_a, max_a = (float(area_range[0]), float(area_range[1])) if isinstance(area_range, (list, tuple)) else (480.0, 550.0)
        where = f"AreaSqKm between {min_a} and {max_a}"
        self.client.note(f"Downloading WBD candidates for {where}" + (" with bbox" if bbox else ""))
        fc = {"type": "FeatureCollection", "features": []}
        try:
            self.client.download_wbd_subset(where, candidates_path, bounding_box=bbox if isinstance(bbox, (list, tuple)) else None)
            process_log["steps"].append({"download_candidates": {"path": str(candidates_path), "where": where, "bbox": bbox if isinstance(bbox, (list, tuple)) else None}})
            fc = json.loads(candidates_path.read_text(encoding="utf-8"))
        except Exception as e:
            process_log["steps"].append({"download_candidates_error": str(e)})

        # Step 2: Pick near 500 km²
        # ensure fc is a FeatureCollection, even when upstream download failed
        if not isinstance(fc, dict) or fc.get("type") != "FeatureCollection":
            fc = {"type": "FeatureCollection", "features": []}
        # Optional: filter by inland states to avoid coastal units
        states_whitelist = input_params.get("states_whitelist")
        if isinstance(states_whitelist, (list, tuple)):
            # normalize to both abbreviations and full names
            us_states = {
                "AL":"AL","ALABAMA":"AL","AK":"AK","ALASKA":"AK","AZ":"AZ","ARIZONA":"AZ","AR":"AR","ARKANSAS":"AR",
                "CA":"CA","CALIFORNIA":"CA","CO":"CO","COLORADO":"CO","CT":"CT","CONNECTICUT":"CT","DE":"DE","DELAWARE":"DE",
                "FL":"FL","FLORIDA":"FL","GA":"GA","GEORGIA":"GA","HI":"HI","HAWAII":"HI","ID":"ID","IDAHO":"ID",
                "IL":"IL","ILLINOIS":"IL","IN":"IN","INDIANA":"IN","IA":"IA","IOWA":"IA","KS":"KS","KANSAS":"KS",
                "KY":"KY","KENTUCKY":"KY","LA":"LA","LOUISIANA":"LA","ME":"ME","MAINE":"ME","MD":"MD","MARYLAND":"MD",
                "MA":"MA","MASSACHUSETTS":"MA","MI":"MI","MICHIGAN":"MI","MN":"MN","MINNESOTA":"MN","MS":"MS","MISSISSIPPI":"MS",
                "MO":"MO","MISSOURI":"MO","MT":"MT","MONTANA":"MT","NE":"NE","NEBRASKA":"NE","NV":"NV","NEVADA":"NV",
                "NH":"NH","NEW HAMPSHIRE":"NH","NJ":"NJ","NEW JERSEY":"NJ","NM":"NM","NEW MEXICO":"NM","NY":"NY","NEW YORK":"NY",
                "NC":"NC","NORTH CAROLINA":"NC","ND":"ND","NORTH DAKOTA":"ND","OH":"OH","OHIO":"OH","OK":"OK","OKLAHOMA":"OK",
                "OR":"OR","OREGON":"OR","PA":"PA","PENNSYLVANIA":"PA","RI":"RI","RHODE ISLAND":"RI","SC":"SC","SOUTH CAROLINA":"SC",
                "SD":"SD","SOUTH DAKOTA":"SD","TN":"TN","TENNESSEE":"TN","TX":"TX","TEXAS":"TX","UT":"UT","UTAH":"UT",
                "VT":"VT","VERMONT":"VT","VA":"VA","VIRGINIA":"VA","WA":"WA","WASHINGTON":"WA","WV":"WV","WEST VIRGINIA":"WV",
                "WI":"WI","WISCONSIN":"WI","WY":"WY","WYOMING":"WY"
            }
            allowed = {us_states.get(str(s).upper(), str(s).upper()) for s in states_whitelist}
            def _in_allowed(feature: Mapping[str, object]) -> bool:
                props = feature.get("properties", {}) if isinstance(feature, Mapping) else {}
                states = (props.get("States") or props.get("STATE") or "")
                parts = [us_states.get(p.strip().upper(), p.strip().upper()) for p in str(states).split(",") if p.strip()]
                return bool(parts) and all(p in allowed for p in parts)
            filtered = [f for f in fc.get("features", []) if _in_allowed(f)]
            fc = {"type": "FeatureCollection", "features": filtered}
            process_log["steps"].append({"filter_states": list(allowed)})
        if not fc.get("features"):
            # if forcing bbox, try expanding area range within the bbox first
            if force_bbox and isinstance(bbox, (list, tuple)):
                for expand in [(300.0, 800.0), (200.0, 1200.0)]:
                    where2 = f"AreaSqKm between {expand[0]} and {expand[1]}"
                    self.client.note(f"No candidates; retry with expanded area {where2} in bbox")
                    self.client.download_wbd_subset(where2, candidates_path, bounding_box=bbox)
                    fc = json.loads(candidates_path.read_text(encoding="utf-8"))
                    process_log["steps"].append({"download_candidates_retry": {"path": str(candidates_path), "where": where2, "bbox": bbox}})
                    if fc.get("features"):
                        break
            # if still none, fallback: broaden query without bbox (unless force_bbox)
        if not fc.get("features") and not force_bbox:
            self.client.note("No candidates within bbox; retrying without spatial filter")
            self.client.download_wbd_subset(f"AreaSqKm between {min_a} and {max_a}", candidates_path, bounding_box=None)
            fc = json.loads(candidates_path.read_text(encoding="utf-8"))
            if isinstance(states_whitelist, (list, tuple)):
                allowed = {str(s).upper() for s in states_whitelist}
                def _in_allowed2(feature: Mapping[str, object]) -> bool:
                    props = feature.get("properties", {}) if isinstance(feature, Mapping) else {}
                    states = (props.get("States") or props.get("STATE") or "")
                    parts = [p.strip().upper() for p in str(states).split(",") if p.strip()]
                    return bool(parts) and all(p in allowed for p in parts)
                filtered2 = [f for f in fc.get("features", []) if _in_allowed2(f)]
                fc = {"type": "FeatureCollection", "features": filtered2}
                process_log["steps"].append({"filter_states": list(allowed)})
        # 将候选子流域写入结果目录，便于前端图层加载
        try:
            (self.results_root / "wbd_candidates.geojson").write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")
            process_log["steps"].append({"write_wbd_candidates": str(self.results_root / "wbd_candidates.geojson")})
        except Exception as e:
            process_log["steps"].append({"write_wbd_candidates_error": str(e)})
        selected = self.processor.select_near_area(fc, 500.0)
        basin_geojson_path = self.results_root / "case_basin.geojson"
        self.processor.save_geojson(selected, basin_geojson_path)
        process_log["steps"].append({"select_basin": str(basin_geojson_path)})

        # Step 3: Build metadata
        metadata = self.processor.build_metadata("USGS WBD HUC12 REST", selected)
        metadata_path = self.results_root / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        process_log["steps"].append({"write_metadata": str(metadata_path)})

        # Step 4: Persist to SQLite
        db_path = self.results_root / "hydrosheds_pipeline.sqlite"
        init_db(db_path)
        layers = {"case_basin": selected}
        write_layers(db_path, layers)
        process_log["steps"].append({"write_sqlite": str(db_path)})

        # Step 5: Simple HTML report
        report_path = self.results_root / "report.html"
        name = metadata.get("name") or metadata.get("basin_code") or "Case Basin"
        html = self._build_leaflet_report(name, selected, metadata)
        report_path.write_text(html, encoding="utf-8")
        process_log["steps"].append({"write_report": str(report_path)})

        # Step 6 (optional): DEM flow analysis if dem_path provided
        dem_path = input_params.get("dem_path")
        if dem_path:
            dem_out_dir = self.results_root / "dem"
            try:
                # 提高自动降采样阈值，尽量避免 DEM 过度降采样导致“直边”
                max_pixels_no_decimate = int(input_params.get("max_pixels_no_decimate", 256_000_000))
                max_cells_d8 = int(input_params.get("max_cells_d8", 16_000_000))
                dem_result = run_dem_flow_pipeline(
                    Path(dem_path),
                    bbox if isinstance(bbox, (list, tuple)) else None,
                    dem_out_dir,
                    max_pixels_no_decimate=max_pixels_no_decimate,
                    max_cells_d8=max_cells_d8,
                )
                process_log["steps"].append({
                    "dem_flow": {
                        "flow_direction_tif": str(dem_result.flow_direction_tif),
                        "flow_accumulation_tif": str(dem_result.flow_accumulation_tif),
                        "flow_accumulation_geojson": str(dem_result.flow_accumulation_geojson),
                        "stream_network_geojson": str(getattr(dem_result, "stream_network_geojson", "")),
                        "stats": dem_result.stats,
                    }
                })
            except Exception as e:  # pragma: no cover
                process_log["steps"].append({"dem_flow_error": str(e)})

        # Finalize log
        process_log["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        log_path = self.results_root / "process_log.json"
        log_path.write_text(json.dumps(process_log, ensure_ascii=False), encoding="utf-8")

        outputs = {
            "basin_geojson": basin_geojson_path,
            "metadata": metadata_path,
            "sqlite": db_path,
            "report": report_path,
            "process_log": log_path,
        }
        if dem_path:
            outputs.update({
                "flow_direction_tif": dem_out_dir / "flow_direction.tif",
                "flow_accumulation_tif": dem_out_dir / "flow_accumulation.tif",
                "flow_accumulation_geojson": dem_out_dir / "flow_accumulation.geojson",
                "stream_network_geojson": dem_out_dir / "stream_network.geojson",
                "parameter_zones_tif": dem_out_dir / "parameter_zones.tif",
                "parameter_zones_geojson": dem_out_dir / "parameter_zones.geojson",
            })
        return outputs

    @staticmethod
    def _build_leaflet_report(title: str, fc: Mapping[str, object], metadata: Mapping[str, object]) -> str:
        # 使用纯模板字符串和占位符避免 Python f-string 与 HTML/JS 花括号冲突
        name = metadata.get('name') or '—'
        code = metadata.get('basin_code') or '—'
        area = metadata.get('area_km2') or '—'
        source = metadata.get('source')
        fc_json = json.dumps(fc)
        html = """
<!DOCTYPE html>
<html lang=\"zh\">
<head>
  <meta charset=\"utf-8\" />
  <title>HydroSHEDS 管线报告 - __TITLE__</title>
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <link rel=\"stylesheet\" href=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.css\" />
  <style> html, body, #map {{ height: 85%; margin: 0; }} body {{ font-family: Arial, sans-serif; }} .card {{ padding: 12px; }} .coord {{ position: fixed; right: 12px; bottom: 12px; background: #fff; padding: 6px 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 12px; }} </style>
</head>
<body>
  <div class=\"card\">
    <h2>HydroSHEDS 处理报告</h2>
    <p>名称: __NAME__ | 代码: __CODE__ | 面积(km²): __AREA__</p>
    <p>来源: __SOURCE__</p>
    <p id="dem-stats">DEM统计: —</p>
    <div class="toggle-group">
      <label><input type="checkbox" id="toggle-dem" checked> 显示DEM栅格</label>
      <label><input type="checkbox" id="toggle-fdir" checked> 显示流向栅格</label>
      <label><input type="checkbox" id="toggle-facc-raster" checked> 显示累积栅格</label>
      <label><input type="checkbox" id="toggle-acc"> 显示累积点</label>
      <label><input type="checkbox" id="toggle-streams" checked> 显示水系矢量</label>
      <label><input type="checkbox" id="toggle-wbd" checked> 显示子流域(WBD)</label>
      <label><input type="checkbox" id="toggle-zones-raster"> 显示参数分区栅格</label>
      <label><input type="checkbox" id="toggle-zones"> 显示参数分区矢量</label>
    </div>
    <p style="margin-top:8px;color:#555;">图例：河网(蓝线)；高累积点(紫点)；参数分区(灰/绿/橙网格)</p>
  </div>
  <div id=\"map\"></div>
  <div class=\"coord\" id=\"coord\">经纬度: —</div>
  <script src=\"https://unpkg.com/leaflet@1.9.4/dist/leaflet.js\"></script>
  <!-- GeoTIFF 渲染依赖 -->
  <script src=\"https://cdn.jsdelivr.net/npm/geotiff@2.1.3/dist/geotiff.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/georaster@1.7.1/dist/georaster.min.js\"></script>
  <script src=\"https://cdn.jsdelivr.net/npm/georaster-layer-for-leaflet@1.7.0/dist/georaster-layer-for-leaflet.min.js\"></script>
  <script>
    const map = L.map('map');
    const osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{ maxZoom: 19 }}).addTo(map);
    // 图层面板与叠放顺序
    map.createPane('zones'); map.getPane('zones').style.zIndex = 400;
    map.createPane('acc');   map.getPane('acc').style.zIndex = 600;
    map.createPane('streams'); map.getPane('streams').style.zIndex = 650;
    const fc = __FC_JSON__;
    const layer = L.geoJSON(fc, {{ style: {{ color: '#d62728', weight: 2, fillOpacity: 0.35 }} }}).addTo(map);
    try {{ map.fitBounds(layer.getBounds().pad(0.2)); }} catch(e) {{ map.setView([0,0], 2); }}
    L.control.scale().addTo(map);
    map.on('mousemove', (e) => {{ const s = `经度: ${{e.latlng.lng.toFixed(5)}} | 纬度: ${{e.latlng.lat.toFixed(5)}}`; document.getElementById('coord').textContent = s; }});

    // -------- 栅格图层加载助手 --------
    async function addGeoTiff(url, options) {{
      const resp = await fetch(url);
      const arrayBuffer = await resp.arrayBuffer();
      const georasterObj = await parseGeoraster(arrayBuffer);
      const layer = new GeoRasterLayer(Object.assign({{
        georaster: georasterObj,
        opacity: options.opacity ?? 0.6,
        pane: options.pane ?? undefined,
        pixelValuesToColor: options.pixelValuesToColor ?? undefined,
      }}, options));
      return layer;
    }}

    // 加载 DEM 派生图层（相对 report.html 所在目录）
    fetch('dem/flow_accumulation.geojson').then(r => r.json()).then(acc => {{
      const accLayer = L.geoJSON(acc, {{ pane: 'acc', pointToLayer: (f, latlng) => L.circleMarker(latlng, {{ radius: 3, color: '#9467bd' }}),
                                          onEachFeature: (f,l) => l.bindPopup('累积: ' + (f.properties?.acc ?? f.properties?.accumulation ?? 'N/A')) }});
      document.getElementById('toggle-acc').onchange = (ev) => {{ if (ev.target.checked) accLayer.addTo(map); else map.removeLayer(accLayer); }};
    }}).catch(() => {{}});

    // DEM统计信息（来自 process_log.json）
    fetch('process_log.json').then(r => r.json()).then(log => {{
      const steps = (log && log.steps) ? log.steps : [];
      let stats = null;
      for (const s of steps) {{ if (s.dem_flow && s.dem_flow.stats) {{ stats = s.dem_flow.stats; break; }} }}
      if (stats) {{
        document.getElementById('dem-stats').textContent = `DEM统计: 高程[min/max]=${{stats.dem_min}}/${{stats.dem_max}} | 累积最大=${{stats.acc_max}} | 累积点数=${{stats.points_exported}}`;
      }}
    }}).catch(() => {{}});
    fetch('dem/stream_network.geojson').then(r => r.json()).then(streams => {{
      const streamsLayer = L.geoJSON(streams, {{ pane: 'streams', style: {{ color: '#0050b5', weight: 3 }}, onEachFeature: (f,l) => l.bindPopup(`河网线 | 阈值: ${{f.properties?.threshold ?? 'N/A'}} | 点数: ${{f.properties?.length_vertices ?? 'N/A'}}`) }});
      // 默认显示水系矢量；通过复选框控制
      streamsLayer.addTo(map);
      const $streams = document.getElementById('toggle-streams');
      $streams.onchange = (ev) => {{ if (ev.target.checked) streamsLayer.addTo(map); else map.removeLayer(streamsLayer); }};
      try {{ map.fitBounds(streamsLayer.getBounds().pad(0.2)); }} catch(e) {{}}
    }}).catch(() => {{}});

    fetch('dem/parameter_zones.geojson').then(r => r.json()).then(zones => {{
      const style = (f) => {{
        const z = f.properties?.zone;
        if (z === 'high') return {{ color: '#e6550d', weight: 1, fillOpacity: 0.05 }};
        if (z === 'mid') return {{ color: '#31a354', weight: 1, fillOpacity: 0.04 }};
        return {{ color: '#636363', weight: 1, fillOpacity: 0.03 }};
      }};
      const zonesLayer = L.geoJSON(zones, {{ pane: 'zones', style }});
      document.getElementById('toggle-zones').onchange = (ev) => {{ if (ev.target.checked) zonesLayer.addTo(map); else map.removeLayer(zonesLayer); }};
    }}).catch(() => {{}});

    // --- WBD 子流域（候选 HUC12） ---
    fetch('wbd_candidates.geojson').then(r => r.json()).then(wbd => {{
      let data = wbd;
      if (!data || !Array.isArray(data.features) || data.features.length === 0) {{
        // 候选为空时，用已选 case_basin 作为回退，避免“看不到”的情况
        return fetch('case_basin.geojson').then(rr => rr.json()).then(basin => {{ return basin; }});
      }}
      return data;
    }}).then(data => {{
      const wbdLayer = L.geoJSON(data, {{ style: {{ color: '#444', weight: 1, fillOpacity: 0 }} }});
      const $wbd = document.getElementById('toggle-wbd');
      if ($wbd.checked) wbdLayer.addTo(map);
      $wbd.onchange = (ev) => {{ if (ev.target.checked) wbdLayer.addTo(map); else map.removeLayer(wbdLayer); }};
    }}).catch(() => {{}});

    // --- 栅格：DEM、流向、累积、参数分区 ---
    (async () => {{
      try {{
        const demLayer = await addGeoTiff('dem/dem_cropped.tif', {{ opacity: 0.6, pixelValuesToColor: (vals) => {{
          const v = vals[0]; if (v == null || isNaN(v)) return null; const t = Math.max(0, Math.min(1, (v - 500) / 2000));
          const r = Math.round(255 * t); const g = Math.round(255 * (1 - Math.abs(t - 0.5) * 2)); const b = Math.round(255 * (1 - t));
          return `rgba(${{r}},${{g}},${{b}},0.6)`; }} }});
        const fdLayer = await addGeoTiff('dem/flow_direction.tif', {{ opacity: 0.5, pixelValuesToColor: (vals) => {{ const d = (vals[0]||0)|0; const palette = {{1:'#2166ac',2:'#67a9cf',4:'#d1e5f0',8:'#fddbc7',16:'#ef8a62',32:'#b2182b',64:'#762a83',128:'#1b7837'}}; return palette[d] || '#999'; }} }});
        const faLayer = await addGeoTiff('dem/flow_accumulation.tif', {{ opacity: 0.5, pixelValuesToColor: (vals) => {{ const v = vals[0]; if (!isFinite(v)) return null; const t = Math.min(1, Math.log10(1+v)/5); const c = Math.round(255*(1-t)); return `rgba(${{c}},${{c}},255,0.5)`; }} }});
        const zonesRasterLayer = await addGeoTiff('dem/parameter_zones.tif', {{ opacity: 0.35, pixelValuesToColor: (vals) => {{ const z = (vals[0]||0)|0; if (z===3) return 'rgba(230,85,13,0.35)'; if (z===2) return 'rgba(49,163,84,0.35)'; if (z===1) return 'rgba(99,99,99,0.35)'; return null; }} }});

        const $dem = document.getElementById('toggle-dem');
        const $fd  = document.getElementById('toggle-fdir');
        const $faR = document.getElementById('toggle-facc-raster');
        const $zonesR = document.getElementById('toggle-zones-raster');
        // 默认显示这些栅格图层，提升“能看到”的体验
        if ($dem.checked) demLayer.addTo(map);
        if ($fd.checked)  fdLayer.addTo(map);
        if ($faR.checked) faLayer.addTo(map);
        if ($zonesR.checked) zonesRasterLayer.addTo(map);
        $dem.onchange = (ev) => {{ if (ev.target.checked) demLayer.addTo(map); else map.removeLayer(demLayer); }};
        $fd.onchange  = (ev) => {{ if (ev.target.checked) fdLayer.addTo(map); else map.removeLayer(fdLayer); }};
        $faR.onchange = (ev) => {{ if (ev.target.checked) faLayer.addTo(map); else map.removeLayer(faLayer); }};
        $zonesR.onchange = (ev) => {{ if (ev.target.checked) zonesRasterLayer.addTo(map); else map.removeLayer(zonesRasterLayer); }};
      }} catch (e) {{ console.warn('GeoTIFF 图层加载失败:', e); }}
    })();
  </script>
</body>
</html>
"""
        return (
            html
            .replace("__TITLE__", str(title))
            .replace("__NAME__", str(name))
            .replace("__CODE__", str(code))
            .replace("__AREA__", str(area))
            .replace("__SOURCE__", str(source))
            .replace("__FC_JSON__", fc_json)
            # 统一将模板遗留的双花括号转换为单花括号，避免浏览器语法错误
            .replace("{{", "{")
            .replace("}}", "}")
        )