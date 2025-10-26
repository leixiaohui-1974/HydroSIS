# HydroSIS 综合测试场景运行指南

## 概述

本文档说明如何运行完整的测试场景套件，包含所有增强的可视化功能。

## 主要改进

### 1. 修复流向图绘制问题
- **问题**：之前流向图和累积数图显示相同
- **修复**：使用D8编码的8个方向，用不同颜色表示每个流向
- **实现**：`ComprehensiveVisualizer.plot_flow_direction_correct()`

### 2. 汇水点配置
- **配置**：3个干流 + 3个支流 = 6个汇水点
- **方法**：使用累积数排序，前3个为干流（红色大圆点），接下来3个为支流（蓝色三角）
- **可视化**：`plot_pour_points_distribution(mainstream_count=3, tributary_count=3)`

### 3. 雨量站配置
- **数量**：约50个雨量站
- **密度**：target_density = 0.05
- **可视化**：包含雨量站分布图和密度统计

### 4. 生成的图表类型

#### 4.1 地形和水文特征图
- 流向图（D8编码，8个方向）
- 累积数图
- 汇水点分布图（区分干流/支流）
- 流域划分图
- 河网图

#### 4.2 雨量相关图
- 雨量站分布图（50个站点）
- 各雨量站时间序列图
- 子流域面雨量动态GIF

#### 4.3 径流相关图
- 各汇水点径流时间序列图
- 降雨径流对比图（双坐标轴）
- 径流系数图表

#### 4.4 分析图
- 各分区径流系数计算和柱状图
- 降雨径流时间序列对比

## 运行测试

### 方法1：运行完整测试套件

```bash
cd /workspace
python3 run_comprehensive_test_scenarios.py
```

这将运行所有8个测试场景：
1. 01_minimal_terrain - 最小测试（仅地形）
2. 02_two_step_basic - 两步基础测试
3. 03_three_step_delineation - 三步流域划分
4. 04_precipitation_analysis - 降雨分析
5. 05_hydrologic_simulation - 水文模拟
6. 06_calibration_workflow - 参数率定
7. 07_parallel_analysis - 并行分析
8. 08_complete_eleven_steps - 完整十一步流程

### 方法2：运行单个测试

使用原有的工作流引擎：

```python
from hydrosis.workflow_engine import WorkflowDefinition, WorkflowEngine

workflow = WorkflowDefinition.from_yaml('config/workflows/test_scenarios/08_complete_eleven_steps.yaml')
engine = WorkflowEngine()
run = engine.execute(workflow)
```

## 输出结果

所有测试结果保存在：
```
results/comprehensive_test_scenarios/
├── 01_最小测试/
│   ├── visualizations/
│   │   ├── flow_direction_*.png       # 流向图
│   │   ├── pour_points_distribution.png  # 汇水点分布
│   │   ├── rain_gauges_distribution.png  # 雨量站分布
│   │   ├── gauge_timeseries/          # 雨量站时序图
│   │   ├── discharge_timeseries/      # 径流时序图
│   │   ├── runoff_coefficients.json   # 径流系数
│   │   ├── runoff_coefficients.png    # 径流系数图
│   │   ├── precip_runoff_comparison/  # 降雨径流对比
│   │   └── areal_precipitation_animation.gif  # 面雨量动画
│   ├── TEST_REPORT.md
│   └── test_result.json
├── 02_两步测试/
│   └── ...
...
├── TEST_SUMMARY.json              # 总结JSON
└── TEST_SUMMARY.md                # 总结报告
```

## 关键功能

### ComprehensiveVisualizer 类

提供以下可视化方法：

1. `plot_flow_direction_correct()` - 正确的流向图（D8编码）
2. `plot_pour_points_distribution()` - 汇水点分布（区分干流/支流）
3. `plot_rain_gauges_distribution()` - 雨量站分布（50个）
4. `plot_timeseries_all_gauges()` - 所有雨量站时间序列
5. `plot_discharge_timeseries()` - 汇水点径流时间序列
6. `calculate_runoff_coefficients()` - 计算和可视化径流系数
7. `create_precipitation_runoff_comparison()` - 降雨径流对比
8. `create_areal_precipitation_gif()` - 面雨量动态GIF

### 径流系数计算

自动计算每个分区的径流系数：

```
径流系数 = 径流深度(mm) / 降雨量(mm)
```

输出：
- JSON文件：包含每个流域的详细计算
- PNG图表：柱状图显示各流域径流系数

## 配置修改说明

### 汇水点数量控制

修改 `threshold` 参数：
- 较大的阈值 → 较少的汇水点
- 较小的阈值 → 较多的汇水点
- 当前配置：threshold = 5000.0（约6个点）

### 雨量站数量控制

修改 `target_density` 参数：
- target_density = 0.01 → 约10个站
- target_density = 0.05 → 约50个站（当前配置）
- target_density = 0.10 → 约100个站

## 依赖环境

确保已安装以下Python包：

```bash
pip install numpy matplotlib pandas geopandas rasterio imageio pillow pyyaml
```

RichDEM需要预编译（已包含在项目中）。

## 问题排查

### 1. 如果测试失败
检查日志文件：`comprehensive_test_run.log`

### 2. 如果可视化不生成
确保matplotlib使用非交互式后端：
```python
import matplotlib
matplotlib.use('Agg')
```

### 3. 如果内存不足
- 减少生成的时间序列图数量
- 减少GIF动画的帧数
- 降低图片分辨率（dpi参数）

## 测试验证清单

- [x] 流向图正确显示8个方向
- [x] 汇水点分布图区分干流和支流
- [x] 雨量站数量约50个
- [x] 生成雨量站时间序列图
- [x] 生成径流时间序列图
- [x] 自动计算径流系数
- [x] 生成降雨径流对比图
- [x] 生成面雨量动态GIF

## 更新历史

- 2025-10-26: 创建综合测试脚本
  - 修复流向图绘制问题
  - 配置6个汇水点（3干流+3支流）
  - 配置50个雨量站
  - 实现所有可视化功能
  - 实现径流系数自动计算
