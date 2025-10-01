from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    candidates_path = ROOT / "data/sample/hydrosheds/case_basin_candidates.geojson"
    output_path = ROOT / "data/sample/hydrosheds/case_basin.geojson"
    data = json.loads(candidates_path.read_text(encoding="utf-8"))
    features = data.get("features", [])
    if not features:
        raise SystemExit("No candidates found in case_basin_candidates.geojson")
    # Choose feature whose AreaSqKm is closest to 500
    def area_km2(feat: dict) -> float:
        try:
            return float(feat.get("properties", {}).get("AreaSqKm", 0.0))
        except Exception:
            return 0.0

    chosen = sorted(features, key=lambda f: abs(area_km2(f) - 500.0))[0]
    out = {"type": "FeatureCollection", "features": [chosen]}
    output_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    props = chosen.get("properties", {})
    print(f"Selected basin: {props.get('HUC12')} | AreaSqKm={props.get('AreaSqKm')}")


if __name__ == "__main__":
    main()