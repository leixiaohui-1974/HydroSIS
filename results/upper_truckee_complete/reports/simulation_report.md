# Upper Truckee River 水文模拟报告

**生成时间**: 2025-10-23 00:15:03

## 1. 流域概况

- **流域名称**: Upper Truckee River
- **流域面积**: 371.5 km²
- **子流域数量**: 4
- **参数分区**: 3 个

### 子流域信息

| 子流域 | 面积 (km²) | 下游 | 产流模型 | 汇流模型 |
|--------|-----------|------|----------|----------|
| W170 | 45.2 | W180 | mountain_zone | muskingum_upper |
| W160 | 35.8 | W180 | mountain_zone | muskingum_upper |
| W180 | 125.5 | W190 | forest_zone | muskingum_main |
| W190 | 165.0 | 出口 | valley_zone | muskingum_outlet |

## 2. 降雨设计

- **模拟时长**: 120 小时
- **暴雨开始**: 第 48 小时
- **暴雨持续**: 24 小时
- **总降雨量**: 75.00 mm
- **峰值强度**: 7.50 mm/h

## 3. 模型配置

### 产流模型

**mountain_zone** (hbv)

- degree_day_factor: 3.5
- snow_threshold: 2.0
- field_capacity: 180
- beta: 1.8
- k0: 0.5
- k1: 0.15
- k2: 0.03
- percolation: 3.0
- initial_snow: 0.0
- initial_soil: 0.0
- initial_upper: 0.0
- initial_lower: 0.0

**forest_zone** (scs_curve_number)

- curve_number: 65
- initial_abstraction_ratio: 0.05

**valley_zone** (scs_curve_number)

- curve_number: 72
- initial_abstraction_ratio: 0.05

### 汇流模型

**muskingum_upper** (muskingum)

- travel_time: 8
- weighting_factor: 0.25
- time_step: 1

**muskingum_main** (muskingum)

- travel_time: 12
- weighting_factor: 0.2
- time_step: 1

**muskingum_outlet** (muskingum)

- travel_time: 15
- weighting_factor: 0.15
- time_step: 1

## 4. 模拟结果

### 本地产流统计

| 子流域 | 总产流 (m³/s·h) | 峰值流量 (m³/s) | 峰现时间 (h) |
|--------|----------------|----------------|-------------|
| W170 | 156.10 | 4.17 | 76 |
| W160 | 108.08 | 2.89 | 76 |
| W180 | 0.83 | 0.08 | 62 |
| W190 | 28.06 | 1.92 | 63 |

### 累积流量统计

| 子流域 | 总流量 (m³/s·h) | 峰值流量 (m³/s) | 峰现时间 (h) |
|--------|----------------|----------------|-------------|
| W170 | 156.10 | 4.17 | 76 |
| W160 | 108.08 | 2.89 | 76 |
| W180 | 265.01 | 7.07 | 76 |
| W190 | 293.07 | 7.83 | 75 |

## 5. 模型评估

评估站点: **W190**

| 指标 | 数值 | 说明 |
|------|------|------|
| RMSE | 3.6331 | - |
| MAE | 2.4374 | - |
| NSE | -894.6818 | 不满意 |
| PBIAS | 1455.6634 | 不满意 |

## 6. 文件清单

### 数据文件

- 降雨数据: `forcing/`
- 观测数据: `observed_flow.csv`
- 结果数据: `results/`

### 图表文件

- `rainfall_distribution.png` - 降雨分布图
- `subbasin_hydrographs.png` - 子流域流量过程线
- `outlet_comparison.png` - 出口流量对比图
- `zone_discharge.png` - 参数区流量汇总图
- `scatter_plot.png` - 散点对比图

## 7. 结论

本次模拟成功完成了Upper Truckee River流域的完整水文过程模拟，包括：

- ✓ 流域划分与参数分区
- ✓ 降雨数据生成与空间分布
- ✓ 产流模拟（HBV雪融模型 + SCS-CN模型）
- ✓ 河道汇流演算（Muskingum模型）
- ✓ 模型评估与验证
- ✓ 结果可视化与报告生成

---

*本报告由 HydroSIS 自动生成*
