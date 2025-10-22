# HydroSIS 功能测试报告

本报告由自动化测试程序生成，覆盖模型配置、产流汇流、情景评估、输入输出与报告生成等核心功能。每个章节列出了用于校验的输入、关键输出以及断言结果，便于快速了解产品能力的完整性。

报告生成时间：2025-10-10 12:13 UTC

## 情景模拟与多模型评价

执行情景路由调整，生成综合评价指标并输出排序结果。

### 测试输入

- **scenarios**：[
  "alternate_routing"
]
- **evaluation_metrics**：[]

### 关键输出与校验

- **overall_scores**：{
  "baseline": "{}",
  "alternate_routing": "{}"
}
- **comparison_rankings**：{
  "baseline_vs_scenario": []
}

### 断言结论

- 基准情景在 RMSE 指标上优于调整后的情景
- 评估计划生成了确定的模型排序

---

## 输入输出与报告生成

验证降雨输入加载、结果持久化及 Markdown 报告生成流程。

### 测试输入

- **forcing_directory**：forcing
- **loaded_series_lengths**：{
  "S1": 4,
  "S2": 4,
  "S3": 4
}

### 关键输出与校验

- **results_files**：{
  "baseline": false,
  "alternate_routing": false
}
- **report_path**：evaluation.md

### 断言结论

- CSV 输出按情景与子流域成功写入
- 评估报告已生成

---

如需复现该报告，可执行 `python -m hydrosis.testing.full_feature_runner` 或在测试目录下运行 PyTest。

报告输出路径：FEATURE_CHECKS.md
