# 参数分区示例：控制点覆盖验证

验证参数分区在存在下游叠置时能保持控制范围的合理划分。

## 测试输入

- **subbasins**：[
  {
    "id": "S1",
    "area_km2": 10.0,
    "downstream": "S3"
  },
  {
    "id": "S2",
    "area_km2": 12.0,
    "downstream": "S3"
  },
  {
    "id": "S3",
    "area_km2": 20.0,
    "downstream": null
  }
]
- **zone_configs**：[
  {
    "id": "Z1",
    "control_points": [
      "S1"
    ],
    "parameters": {
      "runoff_model": "curve",
      "routing_model": "lag_short"
    }
  },
  {
    "id": "Z2",
    "control_points": [
      "S3"
    ],
    "parameters": {
      "runoff_model": "reservoir",
      "routing_model": "lag_short"
    }
  }
]

## 关键输出

- **zone_map**：{
  "Z1": [
    "S1"
  ],
  "Z2": [
    "S2",
    "S3"
  ]
}

## 断言结论

- 上游控制区 Z1 仅覆盖控制点 S1
- 下游控制区 Z2 自动扩展包含中游与出口子流域

该文档由自动化示例验证程序生成。
