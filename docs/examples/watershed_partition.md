# 流域划分示例：面积与下游关系验证

使用轻量 DEM 验证自动划分的子流域面积与下游拓扑。

## 测试输入

- **pour_points**：[
  "S1",
  "S2",
  "S3"
]
- **grid_shape**：[
  6,
  6
]
- **accumulation_threshold**：1.0
- **cell_area_km2**：1.0

## 关键输出

- **subbasin_areas_km2**：{
  "S1": 3.0,
  "S2": 14.0,
  "S3": 16.0
}
- **cell_counts**：{
  "S1": 3,
  "S2": 14,
  "S3": 16
}
- **downstream_relations**：{
  "S1": "S2",
  "S2": "S3",
  "S3": null
}

## 断言结论

- 面积单元格统计一致：S1:3格, S2:14格, S3:16格
- 下游关系链：S1->S2, S2->S3, S3->终点

该文档由自动化示例验证程序生成。
