# Upper Truckee River 详细水文工作流结果

**生成时间**: 2025-10-23
**流域名称**: Upper Truckee River
**总面积**: 371.5 km²

---

## 📋 工作流概览

本工作流为 Upper Truckee River 流域的每个关键步骤生成了详细的可视化图表和数据表。

### 工作流步骤

| 步骤 | 内容 | 图表数 | 数据表数 |
|------|------|--------|----------|
| 步骤1 | DEM和地形分析 | 0 | 0 |
| 步骤2 | 子流域划分 | 2 | 1 |
| 步骤3 | 降雨数据生成 | 2 | 1 |
| 步骤4 | 产流模型配置 | 1 | 3 |
| 步骤5 | 汇流模型配置 | 1 | 1 |
| 步骤6 | 水文模拟运行 | 0 | 0 |
| 步骤7 | 产流结果分析 | 1 | 1 |
| 步骤8 | 汇流结果分析 | 1 | 1 |
| 步骤9 | 模型评估验证 | 2 | 1 |
| 步骤10 | 综合结果报告 | 1 | 0 |
| **总计** | **10个步骤** | **11张图表** | **9个数据表** |

---

## 📊 详细内容

### 步骤1: DEM和地形分析
**目录**: `step_01/`
- *注: 需要安装 rasterio 库才能生成DEM可视化*

### 步骤2: 子流域划分
**目录**: `step_02/`

**图表**:
- `2.1_subbasin_area_pie.png` - 子流域面积饼图
- `2.2_subbasin_topology.png` - 子流域拓扑关系图

**数据表**:
- `2.3_subbasin_info.csv` - 子流域信息表（面积、下游、参数区、高程范围）

**关键信息**:
- W170: 45.2 km² (上游高山区)
- W160: 35.8 km² (上游高山区)
- W180: 125.5 km² (中游森林区)
- W190: 165.0 km² (下游河谷区)

### 步骤3: 降雨数据生成和空间分布
**目录**: `step_03/` 和 `forcing/`

**图表**:
- `3.1_rainfall_timeseries.png` - 各子流域降雨时序图
- `3.2_rainfall_spatial_distribution.png` - 面雨量空间分布图

**数据表**:
- `3.3_rainfall_statistics.csv` - 降雨统计表
- `forcing/W170.csv` - W170子流域降雨数据
- `forcing/W160.csv` - W160子流域降雨数据
- `forcing/W180.csv` - W180子流域降雨数据
- `forcing/W190.csv` - W190子流域降雨数据

**降雨设计**:
- 总时长: 120小时
- 暴雨开始: 第48小时
- 暴雨持续: 24小时
- 设计总降雨量: 75mm

### 步骤4: 产流模型配置
**目录**: `step_04/`

**图表**:
- `4.4_runoff_model_distribution.png` - 产流模型分布图

**数据表**:
- `4.1_mountain_zone_parameters.csv` - 高山区HBV模型参数
- `4.2_forest_zone_parameters.csv` - 森林区SCS-CN模型参数
- `4.3_valley_zone_parameters.csv` - 河谷区SCS-CN模型参数

**模型配置**:
- mountain_zone: HBV雪融模型 (W170, W160)
- forest_zone: SCS-CN模型, CN=65 (W180)
- valley_zone: SCS-CN模型, CN=72 (W190)

### 步骤5: 汇流模型配置
**目录**: `step_05/`

**图表**:
- `5.2_routing_parameters_comparison.png` - 汇流参数对比图（演进时间K、权重因子x）

**数据表**:
- `5.1_routing_parameters.csv` - 汇流模型参数表

**模型配置**:
- muskingum_upper: K=8h, x=0.25 (W170, W160)
- muskingum_main: K=12h, x=0.2 (W180)
- muskingum_outlet: K=15h, x=0.15 (W190)

### 步骤6: 水文模拟运行
**目录**: `step_06/`
- 模拟计算步骤，无可视化输出

### 步骤7: 产流结果分析
**目录**: `step_07/`

**图表**:
- `7.1_runoff_generation.png` - 各子流域产流过程线（4个子图）

**数据表**:
- `7.2_runoff_statistics.csv` - 产流统计表（总产流、峰值流量、峰现时间）

**主要结果**:
- W170: 峰值4.17 m³/s (第76小时)
- W160: 峰值2.89 m³/s (第76小时)
- W180: 峰值0.08 m³/s (第62小时)
- W190: 峰值1.92 m³/s (第63小时)

### 步骤8: 汇流结果分析
**目录**: `step_08/`

**图表**:
- `8.1_routing_results.png` - 各子流域累积流量过程线（4个子图）

**数据表**:
- `8.2_routing_statistics.csv` - 汇流统计表（总流量、峰值流量、峰现时间）

**主要结果**:
- W170: 峰值4.17 m³/s (第76小时)
- W160: 峰值2.89 m³/s (第76小时)
- W180: 峰值7.07 m³/s (第76小时) - 汇集上游
- W190: 峰值7.83 m³/s (第75小时) - 出口

### 步骤9: 模型评估和验证
**目录**: `step_09/`

**图表**:
- `9.1_model_validation.png` - 模拟vs观测流量对比图
- `9.2_scatter_plot.png` - 模拟vs观测散点图

**数据表**:
- `9.3_evaluation_metrics.csv` - 评估指标表

**评估指标**:
- RMSE: 均方根误差
- MAE: 平均绝对误差
- NSE: Nash-Sutcliffe效率系数
- PBIAS: 百分比偏差

### 步骤10: 综合结果报告
**目录**: `step_10/`

**图表**:
- `10.1_water_balance.png` - 水量平衡图

**文档**:
- `10.2_summary_report.md` - 工作流总结报告

---

## 📁 文件结构

```
upper_truckee_detailed/
├── README.md                    # 本文件
├── pour_points.geojson          # 倾泻点数据
├── observed_flow.csv            # 观测流量数据
├── forcing/                     # 降雨数据
│   ├── W170.csv
│   ├── W160.csv
│   ├── W180.csv
│   └── W190.csv
├── step_01/                     # DEM和地形分析
├── step_02/                     # 子流域划分
│   ├── 2.1_subbasin_area_pie.png
│   ├── 2.2_subbasin_topology.png
│   └── 2.3_subbasin_info.csv
├── step_03/                     # 降雨数据
│   ├── 3.1_rainfall_timeseries.png
│   ├── 3.2_rainfall_spatial_distribution.png
│   └── 3.3_rainfall_statistics.csv
├── step_04/                     # 产流模型
│   ├── 4.1_mountain_zone_parameters.csv
│   ├── 4.2_forest_zone_parameters.csv
│   ├── 4.3_valley_zone_parameters.csv
│   └── 4.4_runoff_model_distribution.png
├── step_05/                     # 汇流模型
│   ├── 5.1_routing_parameters.csv
│   └── 5.2_routing_parameters_comparison.png
├── step_06/                     # 模拟运行
├── step_07/                     # 产流结果
│   ├── 7.1_runoff_generation.png
│   └── 7.2_runoff_statistics.csv
├── step_08/                     # 汇流结果
│   ├── 8.1_routing_results.png
│   └── 8.2_routing_statistics.csv
├── step_09/                     # 模型评估
│   ├── 9.1_model_validation.png
│   ├── 9.2_scatter_plot.png
│   └── 9.3_evaluation_metrics.csv
└── step_10/                     # 综合报告
    ├── 10.1_water_balance.png
    └── 10.2_summary_report.md
```

---

## 🔧 运行说明

本工作流使用脚本 `run_upper_truckee_detailed_viz.py` 生成。

**依赖包**:
- numpy, pandas, matplotlib (必需)
- rasterio (可选，用于DEM可视化)

**运行命令**:
```bash
python run_upper_truckee_detailed_viz.py
```

---

## 📖 使用建议

1. **浏览图表**: 按步骤顺序查看各个PNG图表，了解完整的水文建模流程
2. **查看数据**: CSV文件可用Excel或Python pandas读取，进行进一步分析
3. **理解流程**: 结合图表和数据表，深入理解流域水文过程
4. **定制分析**: 基于提供的数据，可进行自定义分析和可视化

---

## 📝 结论

本次详细可视化工作流成功完成了Upper Truckee River流域的完整水文分析，为每个关键步骤生成了清晰的图表和完整的数据表。这些成果可用于：

- ✅ 教学演示
- ✅ 科研分析
- ✅ 项目报告
- ✅ 决策支持

---

*本报告由 HydroSIS 自动生成*
*生成时间: 2025-10-23*
