# 洪水过程验证示例：产流与汇流耦合对比

基于合成暴雨事件生成完整洪水过程，将多种产流-汇流组合与参考结果对比验证模型正确性。

## 测试输入

- **降雨序列(mm/step)**：[
  0.0,
  0.0,
  0.0,
  5.0,
  12.0,
  25.0,
  40.0,
  65.0,
  90.0,
  110.0,
  95.0,
  80.0,
  60.0,
  40.0,
  25.0,
  15.0,
  8.0,
  5.0,
  3.0,
  1.0,
  0.0,
  0.0
]
- **集水面积(km^2)**：42.0
- **总降雨量(mm)**：679.0
- **降雨体积(面积加权)**：28518.0

## 关键输出

- **参考洪水统计**：{
  "peak": 1403.888,
  "time_to_peak": 17,
  "volume": 14788.556
}
- **模型洪水峰值对比**：{
  "reference_hymod_dynamic": {
    "peak": 1403.888,
    "time_to_peak": 17,
    "volume": 14788.556
  },
  "hymod_muskingum": {
    "peak": 1637.982,
    "time_to_peak": 18,
    "volume": 13261.847
  },
  "scs_dynamic": {
    "peak": 860.841,
    "time_to_peak": 11,
    "volume": 8021.635
  },
  "xinan_dynamic": {
    "peak": 2367.961,
    "time_to_peak": 12,
    "volume": 32778.903
  },
  "scs_lag": {
    "peak": 2346.91,
    "time_to_peak": 11,
    "volume": 9710.938
  }
}
- **误差指标**：{
  "reference_hymod_dynamic": {
    "rmse": 0.0,
    "mae": 0.0,
    "pbias": 0.0,
    "nse": 1.0
  },
  "hymod_muskingum": {
    "rmse": 266.919994,
    "mae": 202.541835,
    "pbias": 10.32358,
    "nse": 0.785825
  },
  "scs_dynamic": {
    "rmse": 547.887934,
    "mae": 375.744677,
    "pbias": 45.757817,
    "nse": 0.097619
  },
  "xinan_dynamic": {
    "rmse": 1020.537608,
    "mae": 892.492084,
    "pbias": 121.650468,
    "nse": -2.130867
  },
  "scs_lag": {
    "rmse": 869.54961,
    "mae": 637.151878,
    "pbias": 34.334777,
    "nse": -1.272979
  }
}
- **NSE 排名(由好到差)**：[
  "reference_hymod_dynamic",
  "hymod_muskingum",
  "scs_dynamic",
  "scs_lag",
  "xinan_dynamic"
]

## 断言结论

- HYMOD + 动态波组合与观测一致 (NSE = 1.0)
- 仅调整汇流 (Muskingum) 导致峰值推迟且 NSE 降至约 0.79
- 纯延时 (lag) 方案峰现时间比参考更早
- XinAnJiang 组合产生显著峰值高估与正偏差

该文档由自动化示例验证程序生成。
