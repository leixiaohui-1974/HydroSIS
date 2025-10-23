# Upper Truckee River 10步水文建模工作流

**生成时间**: 2025-10-23
**流域**: Upper Truckee River, California
**总面积**: 371.5 km²

---

## 📋 工作流概览

本工作流基于HydroSIS的`run_workflow`函数，实现了完整的10步水文建模流程，每一步都生成详细的图表和数据表。

### 10个工作流步骤

| 步骤 | 名称 | 输出图表 | 输出数据 | 主要内容 |
|------|------|----------|----------|----------|
| 1 | DEM数据读取与地形分析 | 1 | 1 | DEM高程图、地形统计 |
| 2 | 流域划分与子流域定义 | 1 | 1 | 子流域面积分布、拓扑关系 |
| 3 | 河网提取与流向分析 | 1 | 1 | 河网提取、流量累积 |
| 4 | 参数分区定义 | 1 | 1 | 参数区划分、特征描述 |
| 5 | 降雨输入数据生成 | 1 | 5 | 降雨时序、空间分布 |
| 6 | 产流模型配置 | 0 | 1 | HBV、SCS-CN模型参数 |
| 7 | 汇流模型配置 | 1 | 1 | Muskingum参数配置 |
| 8 | 模型构建与完整配置 | 0 | 1 | 模型组装、配置摘要 |
| 9 | 模型执行与模拟 | 1 | 1 | 运行run_workflow、生成流量过程 |
| 10 | 结果分析与报告生成 | 0 | 2 | 评估指标、总结报告 |
| **总计** | **10个步骤** | **7张图表** | **15个文件** | **完整工作流** |

---

## 📊 详细步骤说明

### 步骤1: DEM数据读取与地形分析
**目录**: `step_01/`

**输出**:
- `dem_elevation.png` - DEM高程图（terrain colormap）
- `dem_statistics.csv` - 地形统计（高程范围、分辨率等）

**关键信息**:
- 高程范围: 6040.8 - 10034.0 m
- 分辨率: 112.63 m
- 数据规模: 884 × 590 像元

---

### 步骤2: 流域划分与子流域定义
**目录**: `step_02/`

**输出**:
- `subbasin_areas.png` - 子流域面积饼图
- `subbasins.csv` - 子流域信息表（ID、面积、下游、高程）

**子流域列表**:
- W170: 45.2 km² (上游高山区)
- W160: 35.8 km² (上游高山区)
- W180: 125.5 km² (中游森林区)
- W190: 165.0 km² (下游河谷区，出口)

---

### 步骤3: 河网提取与流向分析
**目录**: `step_03/`

**输出**:
- `stream_network.png` - 河网提取图（DEM叠加河网）
- `stream_statistics.csv` - 河网统计（河网像元数、密度、阈值）

**河网特征**:
- 总河网像元: 17,386
- 提取方法: 95百分位流量累积阈值

---

### 步骤4: 参数分区定义
**目录**: `step_04/`

**输出**:
- `parameter_zones.png` - 参数区分布柱状图
- `parameter_zones.csv` - 参数区信息表

**参数区**:
1. **Upper_Mountain** (上游高山区 2000-3000m)
   - 子流域: W170, W160
   - 特征: 高海拔、陡坡、积雪

2. **Middle_Forest** (中游森林区 1500-2000m)
   - 子流域: W180
   - 特征: 森林覆盖、中等坡度

3. **Lower_Valley** (下游河谷区 1200-1500m)
   - 子流域: W190
   - 特征: 河谷、缓坡、草地

---

### 步骤5: 降雨输入数据生成
**目录**: `step_05/` 和 `forcing/`

**输出**:
- `rainfall_timeseries.png` - 4个子流域的降雨时序图
- `rainfall_statistics.csv` - 降雨统计表
- `forcing/W170.csv` - W170降雨数据
- `forcing/W160.csv` - W160降雨数据
- `forcing/W180.csv` - W180降雨数据
- `forcing/W190.csv` - W190降雨数据

**降雨设计**:
- 总时长: 120小时
- 暴雨开始: 第48小时
- 暴雨持续: 24小时
- 设计总降雨量: 75 mm
- 海拔梯度: 高山区 +10%, 下游 -5%

---

### 步骤6: 产流模型配置
**目录**: `step_06/`

**输出**:
- `runoff_models.csv` - 产流模型配置表

**模型配置**:
1. **mountain_zone**: HBV雪融模型
   - 应用于: W170, W160
   - 参数: degree_day_factor=3.5, snow_threshold=2.0°C等

2. **forest_zone**: SCS-CN模型
   - 应用于: W180
   - CN=65 (森林，良好条件)

3. **valley_zone**: SCS-CN模型
   - 应用于: W190
   - CN=72 (草地，一般条件)

---

### 步骤7: 汇流模型配置
**目录**: `step_07/`

**输出**:
- `routing_parameters.png` - Muskingum参数对比图（K和x）
- `routing_models.csv` - 汇流模型参数表

**模型配置**:
1. **muskingum_upper**: K=8h, x=0.25 (W170, W160)
2. **muskingum_main**: K=12h, x=0.2 (W180)
3. **muskingum_outlet**: K=15h, x=0.15 (W190)

---

### 步骤8: 模型构建与完整配置
**目录**: `step_08/`

**输出**:
- `model_configuration.csv` - 完整模型配置摘要

**配置内容**:
- 子流域: 4个
- 参数区: 3个
- 产流模型: 3个
- 汇流模型: 3个
- 模拟时长: 120小时

---

### 步骤9: 模型执行与模拟
**目录**: `step_09/`

**输出**:
- `discharge_hydrographs.png` - 所有子流域流量过程线
- `simulation_results.csv` - 模拟结果统计（峰值、总量、峰现时间）

**核心功能**: 调用HydroSIS的`run_workflow`函数执行模拟

**模拟结果**:
- W170峰值: ~4.17 m³/s (第76小时)
- W160峰值: ~2.89 m³/s (第76小时)
- W180峰值: ~7.07 m³/s (第76小时)
- W190峰值: 7.83 m³/s (第75小时) - **出口**

---

### 步骤10: 结果分析与报告生成
**目录**: `step_10/`

**输出**:
- `evaluation_metrics.csv` - 模型评估指标（RMSE, MAE, NSE, PBIAS）
- `workflow_summary.md` - 工作流总结报告

**评估指标**:
- RMSE: 均方根误差
- MAE: 平均绝对误差
- NSE: Nash-Sutcliffe效率系数
- PBIAS: 百分比偏差

---

## 📁 文件结构

```
upper_truckee_10steps/
├── README.md                    # 本文件
├── pour_points.geojson          # 倾泻点数据
├── forcing/                     # 降雨输入数据
│   ├── W170.csv
│   ├── W160.csv
│   ├── W180.csv
│   └── W190.csv
├── step_01/                     # DEM分析
│   ├── dem_elevation.png
│   └── dem_statistics.csv
├── step_02/                     # 子流域划分
│   ├── subbasin_areas.png
│   └── subbasins.csv
├── step_03/                     # 河网提取
│   ├── stream_network.png
│   └── stream_statistics.csv
├── step_04/                     # 参数分区
│   ├── parameter_zones.png
│   └── parameter_zones.csv
├── step_05/                     # 降雨数据
│   ├── rainfall_timeseries.png
│   └── rainfall_statistics.csv
├── step_06/                     # 产流模型
│   └── runoff_models.csv
├── step_07/                     # 汇流模型
│   ├── routing_parameters.png
│   └── routing_models.csv
├── step_08/                     # 模型配置
│   └── model_configuration.csv
├── step_09/                     # 模拟执行
│   ├── discharge_hydrographs.png
│   └── simulation_results.csv
└── step_10/                     # 结果分析
    ├── evaluation_metrics.csv
    └── workflow_summary.md
```

---

## 🔧 运行说明

**脚本**: `run_upper_truckee_10step.py`

**依赖**:
- numpy, pandas, matplotlib (必需)
- rasterio (必需，用于DEM读取)
- hydrosis (本项目核心库)

**运行命令**:
```bash
python run_upper_truckee_10step.py
```

**预计运行时间**: 约30-60秒

---

## 📖 工作流特点

### 1. 基于HydroSIS标准工作流
- 使用`run_workflow`函数执行核心模拟
- 遵循HydroSIS的配置规范
- 完整的产流-汇流耦合

### 2. 完整的10步流程
- 从DEM分析到结果报告的完整链条
- 每步都有明确的输入输出
- 步骤之间逻辑清晰、可追溯

### 3. 详细的可视化
- 7张高质量PNG图表
- 清晰的数据表格
- 完整的Markdown报告

### 4. 真实案例
- 使用Upper Truckee River真实DEM
- 实际地形数据（已预处理）
- 符合实际的水文参数

---

## 📝 使用建议

1. **学习水文建模**: 按步骤1-10顺序理解完整建模流程
2. **数据分析**: 使用CSV文件进行进一步分析
3. **模型调参**: 修改步骤6-7的模型参数，重新运行
4. **结果对比**: 对比不同参数配置的模拟结果
5. **扩展应用**: 基于此框架应用到其他流域

---

## ✨ 与其他工作流的区别

### vs `run_upper_truckee_complete.py`
- 本工作流: 10个明确步骤，强调流程清晰度
- complete版: 11个步骤，更多细节配置

### vs `run_upper_truckee_detailed_viz.py`
- 本工作流: 基于run_workflow的标准流程
- detailed_viz版: 独立实现，更多可视化细节

---

## 🎯 核心成果

- ✅ **完整性**: 覆盖水文建模全流程
- ✅ **规范性**: 基于HydroSIS标准工作流
- ✅ **清晰性**: 10步逻辑清楚，易于理解
- ✅ **实用性**: 真实案例，可直接应用
- ✅ **可追溯**: 每步都有详细记录

---

*本工作流由HydroSIS自动生成*
*生成时间: 2025-10-23*
