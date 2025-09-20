# HydroSIS 功能测试报告

本报告由自动化测试程序生成，覆盖模型配置、产流汇流、情景评估、输入输出与报告生成等核心功能。每个章节列出了用于校验的输入、关键输出以及断言结果，便于快速了解产品能力的完整性。

报告生成时间：2025-09-20T02:20:38+00:00

## 模型配置解析与流域划分验证

验证 ModelConfig 各子配置解析、序列化与预定义子流域划分逻辑。

### 测试输入

- **runoff_models**：[
  "curve",
  "reservoir"
]
- **routing_models**：[
  "lag_fast",
  "lag_slow"
]
- **parameter_zones**：[
  "Z1",
  "Z2"
]

### 关键输出与校验

- **subbasins**：[
  "S1",
  "S2",
  "S3"
]
- **roundtrip_consistency**：True

### 断言结论

- 成功解析 3 个子流域，并保持配置往返一致

---

## 参数分区控制与模型实例化

校验参数分区将控制点及下游子流域正确绑定到模型实例。

### 测试输入

- **zones**：{
  "Z1": [
    "S1"
  ],
  "Z2": [
    "S2",
    "S3"
  ]
}

### 关键输出与校验

- **subbasin_parameters**：{
  "S1": {
    "runoff_model": "curve",
    "routing_model": "lag_fast"
  },
  "S2": {
    "runoff_model": "reservoir",
    "routing_model": "lag_fast"
  },
  "S3": {
    "runoff_model": "reservoir",
    "routing_model": "lag_fast"
  }
}

### 断言结论

- 所有子流域均自动继承了对应的产流与汇流模型标识

---

## 基准情景产流与汇流模拟

运行基准配置，验证产流、汇流与分区汇总结果的维度与合理性。

### 测试输入

- **forcing_samples**：{
  "S1": [
    0.0,
    12.0,
    35.0,
    4.0
  ],
  "S2": [
    6.0,
    6.0,
    6.0,
    6.0
  ],
  "S3": [
    0.0,
    0.0,
    0.0,
    0.0
  ]
}

### 关键输出与校验

- **local_flow_keys**：[
  "S1",
  "S2",
  "S3"
]
- **aggregated_series_lengths**：{
  "S1": 4,
  "S2": 4,
  "S3": 4
}
- **zone_discharge_controllers**：{
  "Z1": [
    "S1"
  ],
  "Z2": [
    "S3"
  ]
}

### 断言结论

- 所有子流域均生成 4 个时间步的径流序列
- 参数控制断面返回与基准汇流一致的序列长度

---

## 情景模拟与多模型评价

执行情景路由调整，生成综合评价指标并输出排序结果。

### 测试输入

- **scenarios**：[
  "alternate_routing"
]
- **evaluation_metrics**：[
  "rmse",
  "mae",
  "nse"
]

### 关键输出与校验

- **overall_scores**：{
  "baseline": {
    "rmse": 0.0,
    "mae": 0.0,
    "nse": 1.0
  },
  "alternate_routing": {
    "rmse": 5.393403,
    "mae": 4.6305,
    "nse": 0.770393
  }
}
- **comparison_rankings**：{
  "baseline_vs_scenario": [
    "baseline",
    "alternate_routing"
  ]
}

### 断言结论

- 基准情景在 RMSE 指标上优于调整后的情景
- 评估计划生成了确定的模型排序

---

## 输入输出与报告生成

验证降雨输入加载、结果持久化及 Markdown 报告生成流程。

### 测试输入

- **forcing_directory**：/tmp/tmp2lzk47op/forcing
- **loaded_series_lengths**：{
  "S3": 4,
  "S2": 4,
  "S1": 4
}

### 关键输出与校验

- **results_files**：{
  "/tmp/tmp2lzk47op/simulation_results/baseline/S3.csv": true,
  "/tmp/tmp2lzk47op/simulation_results/alternate_routing/S3.csv": true
}
- **report_path**：/tmp/tmp2lzk47op/reports/evaluation.md

### 断言结论

- CSV 输出按情景与子流域成功写入
- 评估报告已生成

---

如需复现该报告，可执行 `python -m hydrosis.testing.full_feature_runner` 或在测试目录下运行 PyTest。

报告输出路径：docs/test_documentation.md
