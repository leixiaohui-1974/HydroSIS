# HydroSIS 工作流运行结果

本目录包含HydroSIS完整10步工作流的运行结果。

## 快速查看

### 📊 主要报告
- **[WORKFLOW_RESULTS_SUMMARY.md](WORKFLOW_RESULTS_SUMMARY.md)** - 完整结果总结
- **[reports/evaluation.md](reports/evaluation.md)** - 模型评估报告
- **[reports/gis_overview.html](reports/gis_overview.html)** - 交互式GIS地图（在浏览器中打开）

### 🎨 可视化结果

所有图表位于 `figures/` 目录：

#### 水文过程线
查看各子流域的流量随时间变化过程：
- `hydrograph_SB1.png` - 子流域SB1（120.5 km²）
- `hydrograph_SB2.png` - 子流域SB2（80.2 km²）
- `hydrograph_SB3.png` - 子流域SB3（210.7 km²）
- `hydrograph_SB4.png` - 子流域SB4（540.0 km², 出口）

#### 模型评估指标
对比不同情景的性能：
- `metric_rmse.png` - 均方根误差
- `metric_mae.png` - 平均绝对误差
- `metric_nse.png` - 纳什效率系数（0.9903，优秀！）
- `metric_pbias.png` - 百分比偏差

#### 空间分布
- `subbasin_area.png` - 子流域面积分布柱状图
- `subbasin_map.png` - 流域分区地图
- `parameter_zone_map.png` - 参数区分布图
- `scenario_comparison.png` - 情景流量对比

### 📈 数据文件

所有数值结果以CSV格式保存，可用Excel、Python pandas等打开：

```
baseline/              # 基准情景
├── SB1.csv           # 子流域流量
├── SB2.csv
├── SB3.csv
├── SB4.csv
├── Z1.csv            # 参数区流量
├── Z2.csv
└── Z3.csv

baseline_local/       # 本地产流
baseline_subbasin/    # 子流域汇流

reservoir_reoperation/     # 水库调度情景
vic_hbv_tuning/           # VIC/HBV调优情景
```

## 工作流步骤

本次运行完成的10个步骤：

1. ✅ **DEM预处理** - 地形分析、流向和汇流累积计算
2. ✅ **汇水点提取** - 识别流域出口点
3. ✅ **参数分区（流域划分）** - 子流域和参数区划分
4. ✅ **河道断面提取** - 河道几何数据提取
5. ✅ **雨量站点生成** - 雨量站网络设计
6. ✅ **降雨序列生成** - 降雨时间序列
7. ✅ **泰森权重计算** - 雨量站点空间权重
8. ✅ **面雨量计算** - 子流域平均降雨
9. ✅ **产汇流计算** - 水文模拟
10. ✅ **水动力路由** - 河道演算

## 流域概况

### 子流域结构
```
        SB1 (120.5 km²) ──┐
                          ├──> SB3 (210.7 km²) ──> SB4 (540.0 km², 出口)
        SB2 (80.2 km²) ───┘
```

**总流域面积**: 951.4 km²

### 参数区
- **Z1**: 上游山区集水区（控制SB1）
- **Z2**: 中游水库控制区（控制SB3）
- **Z3**: 下游水文站控制区（控制SB4）

## 模型性能

| 指标 | 数值 | 评级 |
|------|------|------|
| **NSE** (纳什效率系数) | **0.9903** | ⭐ 优秀 (>0.9) |
| RMSE (均方根误差) | 66.32 m³/s | |
| MAE (平均绝对误差) | 45.61 m³/s | |
| PBIAS (百分比偏差) | 0.21% | ⭐ 非常好 (<10%) |

## 情景分析

完成了3个情景的对比分析：

1. **Baseline** - 基准情景
2. **Reservoir Reoperation** - 水库优化调度情景
3. **VIC/HBV Tuning** - 模型参数调优情景

每个情景包含完整的流量过程、产流和汇流数据。

## 使用Python查看数据

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取出口流量数据
baseline = pd.read_csv('baseline/SB4.csv', index_col=0)

# 绘制流量过程线
plt.figure(figsize=(12, 6))
plt.plot(baseline.index, baseline.values)
plt.xlabel('Time Step')
plt.ylabel('Discharge (m³/s)')
plt.title('SB4 (Outlet) Hydrograph - Baseline Scenario')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 计算统计量
print(f"Peak discharge: {baseline.max().values[0]:.2f} m³/s")
print(f"Mean discharge: {baseline.mean().values[0]:.2f} m³/s")
print(f"Total volume: {baseline.sum().values[0]:.2f} m³")
```

## 技术细节

### 产流模型
- SCS Curve Number
- Xin'anjiang (新安江)
- HBV
- VIC
- WETSPA
- HYMOD

### 汇流模型
- Muskingum
- Dynamic Wave
- Lag

### 评估指标
- RMSE - Root Mean Square Error
- MAE - Mean Absolute Error
- NSE - Nash-Sutcliffe Efficiency
- PBIAS - Percent Bias

## 相关文档

- **[../../WORKFLOW_EXECUTION_LOG.md](../../WORKFLOW_EXECUTION_LOG.md)** - 完整执行日志
- **[../../README.md](../../README.md)** - HydroSIS项目说明
- **[../../docs/product_pipeline.md](../../docs/product_pipeline.md)** - 三阶段产品体系说明

## 如何重现

要重新运行工作流：

```bash
cd /home/user/HydroSIS
python run_complete_workflow.py
```

或使用原始示例脚本：

```bash
python examples/run_sample_workflow.py
```

## 引用

如果使用本系统的结果，请引用：

```
HydroSIS - Hydrological and Hydrodynamic Integrated Simulation System
GitHub: https://github.com/leixiaohui-1974/HydroSIS
```

## 许可证

本项目遵循项目根目录的LICENSE文件。

---

**生成日期**: 2025-10-22
**HydroSIS版本**: 0.1.0
**分支**: claude/watershed-workflow-execution-011CUP5q458VvcZqQJq5FDF8
