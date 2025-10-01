from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hydrosis.hydrosheds import HydroSHEDSPipeline


def main() -> None:
    cache_root = ROOT / "data/sample/hydrosheds/cache"
    results_root = ROOT / "results/hydrosheds_demo"
    pipeline = HydroSHEDSPipeline(cache_root, results_root)

    # 可通过文件或环境传入，这里使用默认参数进行演示
    dem_env = os.environ.get("HYDROSHEDS_DEM_PATH") or str((ROOT / "data/sample/dem/n00e070_con.tif").resolve())
    # 扩大到黄石-博兹曼-杰克逊一带的山区范围
    input_params = {
        # 为你提供的 DEM (EPSG:4326, 70E~80E, 0~10N) 设置对应范围
        "bounding_box": [70.0, 0.0, 80.0, 10.0],
        "states_whitelist": ["WY", "MT", "ID"],
        "area_range": [0, 100000],
        "force_bbox": True,
        "output_format": "geojson",
        **({"dem_path": dem_env} if dem_env else {}),
    }
    outputs = pipeline.run(input_params)
    print("HydroSHEDS 管线完成，输出：")
    for k, v in outputs.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()