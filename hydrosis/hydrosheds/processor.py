from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence


class HydroSHEDSProcessor:
    """Processing helpers for standardizing basin boundaries and metadata."""

    @staticmethod
    def select_near_area(fc: Mapping[str, object], target_km2: float) -> Mapping[str, object]:
        features = fc.get("features", []) if isinstance(fc, Mapping) else []
        if not features:
            return {"type": "FeatureCollection", "features": []}
        def area(feat: Mapping[str, object]) -> float:
            props = feat.get("properties", {}) if isinstance(feat, Mapping) else {}
            try:
                return float(props.get("AreaSqKm", 0.0))
            except Exception:
                return 0.0
        chosen = sorted(features, key=lambda f: abs(area(f) - target_km2))[0]
        return {"type": "FeatureCollection", "features": [chosen]}

    @staticmethod
    def build_metadata(source: str, fc: Mapping[str, object]) -> Dict[str, object]:
        features = fc.get("features", []) if isinstance(fc, Mapping) else []
        props = (features[0].get("properties", {}) if features else {})
        return {
            "source": source,
            "basin_code": props.get("HUC12"),
            "name": props.get("Name"),
            "area_km2": props.get("AreaSqKm"),
            "states": props.get("States"),
        }

    @staticmethod
    def save_geojson(fc: Mapping[str, object], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fc, ensure_ascii=False), encoding="utf-8")