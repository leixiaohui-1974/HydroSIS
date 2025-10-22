# 子流域划分诊断输出说明

新版 DEM 划分流程会在 `derived/` 目录下生成 `delineation_diagnostics.json`，用于快速核查各流域面积、重叠情况以及下游拓扑是否闭合。

## 诊断文件结构

```json
{
  "dem_path": "data/sample/dem/typical_watershed_dem.json",
  "pour_points_path": "data/sample/gis/typical_watershed_pour_points.geojson",
  "accumulation_threshold": 1000.0,
  "cell_area_km2": 0.0001,
  "total_area_cells": 48621,
  "total_area_km2": 4.8621,
  "total_overlap_cells": 0,
  "pour_points": [
    {
      "id": "P1",
      "row": 412,
      "col": 275,
      "area_cells": 15834,
      "area_km2": 1.5834,
      "forced_outlet": false,
      "overlap_cells_removed": 0,
      "downstream_id": "P3"
    }
  ]
}
```

- `total_overlap_cells`：若大于 0，表示存在多个子流域共享像元，系统已按先后顺序裁剪；建议调整入流点避免过度重叠。
- `forced_outlet`：若为 `true`，说明原始流向结果未覆盖出流点，系统已强制包含；通常发生在平坦或空值像元。
- `downstream_id`：自动沿 D8 流向追踪到的下游子流域 ID，若为 `null` 则代表该流域直接出湖或汇入域外。

## 使用建议

1. 在生成 `Subbasin` 后，首先查看 `delineation_diagnostics.json`，确认面积汇总与预期一致。
2. 若存在 `forced_outlet` 或 `total_overlap_cells > 0`，考虑调整阈值（`accumulation_threshold`）或重新布设入流点，以免出现“空流域”或拓扑断裂。
3. `Subbasins.geojson` 与诊断 JSON 可结合加载到 GIS 中，校核边界与面积统计；必要时导入外部分区覆盖进行复核。

## 案例脚本

运行 `python examples/delineation_quality_case.py` 将基于示例 DEM 执行划分，并在 `results/delineation_quality_case/` 中生成栅格、矢量图以及 `delineation_summary.md`，直观展示本诊断流程的全部中间结果。
