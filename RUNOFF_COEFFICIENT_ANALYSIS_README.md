# 径流系数对比分析与自动率定系统

## 概述

这是一个全面的径流系数分析和模型率定系统，用于对比观测径流系数与多种水文模型的模拟结果，并通过自动优化使模拟值接近观测值。

## 主要功能

### 1. 观测径流系数计算
从生成的降雨径流数据自动计算各分区的观测径流系数：
```
径流系数 = 径流深度(mm) / 降雨量(mm)
```

### 2. 多模型模拟对比

**产流模型**：
- **HBV** - 概念性土壤水分模型（5参数）
- **SCS-CN** - Curve Number方法（1参数）
- **Green-Ampt** - 物理入渗模型（4参数）

**汇流模型**：
- **Muskingum** - 河道汇流
- **Kinematic Wave** - 运动波汇流

**自动组合测试**：产流模型 × 汇流模型 = 多种组合

### 3. 按分区详细对比
- 每个分区独立分析
- 所有模型组合vs观测值
- 误差定量评估
- 最佳模型推荐

### 4. 参数敏感性分析
- 系统测试各参数影响
- 生成参数-RC关系曲线
- 识别敏感参数
- 指导率定范围

### 5. 自动率定优化
- 使用差分进化全局优化算法
- 目标：最小化 |模拟RC - 观测RC|
- 自动搜索最优参数
- 验证率定效果

### 6. 完整可视化

生成的图表：
```
results/runoff_coefficient_analysis/
├── rc_comparison_all_zones.png          # 总体RC对比柱状图
├── model_comparison_Zone*.png           # 各分区详细对比
├── error_analysis.png                   # 误差热力图和箱线图
├── best_model_selection.png             # 最佳模型选择
├── sensitivity_*.csv                    # 敏感性分析结果
├── runoff_coefficient_analysis_report.md # 详细Markdown报告
└── analysis_results.json                # 完整JSON结果
```

## 快速开始

### 1. 运行综合测试生成数据
```bash
# 首先运行综合测试场景生成降雨径流数据
python3 run_comprehensive_test_scenarios.py
```

### 2. 运行径流系数分析
```bash
# 自动分析所有分区
python3 runoff_coefficient_analysis.py
```

### 3. 查看结果
```bash
# 查看Markdown报告
cat results/runoff_coefficient_analysis/runoff_coefficient_analysis_report.md

# 查看图表
ls results/runoff_coefficient_analysis/*.png

# 查看JSON结果
cat results/runoff_coefficient_analysis/analysis_results.json
```

## 作为模块使用

```python
from runoff_coefficient_analysis import RunoffCoefficientAnalyzer

# 创建分析器
analyzer = RunoffCoefficientAnalyzer(Path("results/my_analysis"))

# 加载数据
analyzer.load_data(
    precipitation_file=Path("path/to/precipitation.csv"),
    discharge_file=Path("path/to/discharge.csv"),
    watershed_file=Path("path/to/watersheds.geojson")
)

# 1. 分析所有分区
results = analyzer.analyze_all_zones()

# 2. 敏感性分析
sensitivity_df = analyzer.sensitivity_analysis(
    zone_id='Zone1',
    model_name='HBV',
    param_ranges={
        'field_capacity': (50, 500),
        'beta': (1, 5)
    }
)

# 3. 自动率定
optimal_params = analyzer.calibrate_model(
    zone_id='Zone1',
    model_name='HBV',
    target_rc=0.45  # 可选，默认使用观测RC
)

# 4. 生成图表
analyzer.generate_comparison_plots()

# 5. 生成报告
analyzer.generate_report()
```

## 输出解读

### 1. RC对比图 (rc_comparison_all_zones.png)
- **深蓝色柱**: 观测径流系数（真值）
- **彩色柱**: 各模型模拟的径流系数
- **越接近深蓝色越好**

### 2. 分区模型对比图 (model_comparison_*.png)
- **左图**: 所有模型的RC值，红虚线为观测值
- **右图**: 各模型的绝对误差
- **颜色编码**:
  - 绿色: 优秀 (误差 < 0.05)
  - 橙色: 良好 (误差 < 0.1)
  - 红色: 较差 (误差 > 0.1)

### 3. 误差分析图 (error_analysis.png)
- **上图**: 热力图 - 所有分区×所有模型的误差矩阵
  - 绿色区域: 误差小，性能好
  - 红色区域: 误差大，性能差
- **下图**: 箱线图 - 各模型误差分布
  - 箱体越低越好
  - 离群点表示个别分区表现异常

### 4. 最佳模型选择图 (best_model_selection.png)
- 每个分区的最佳模型及其误差
- 柱高 = 最小误差
- 柱顶文字 = 最佳模型名称
- 颜色 = 性能等级

### 5. 分析报告 (*.md)
- 观测RC汇总表
- 各分区详细模型性能表
- 最佳模型推荐
- 分区特定建议

## HBV模型参数说明

| 参数 | 名称 | 典型范围 | 单位 | 说明 |
|------|------|----------|------|------|
| FC | 田间持水量 | 50-500 | mm | 土壤最大含水能力 |
| Beta | 形状系数 | 1-5 | - | 控制产流非线性 |
| K0 | 快速出流系数 | 0.01-0.2 | 1/h | 地表快速流 |
| K1 | 中速出流系数 | 0.001-0.05 | 1/h | 壤中流 |
| K2 | 慢速出流系数 | 0.0001-0.01 | 1/h | 基流 |

**调整建议**：
- **RC偏低**: 增大K0, K1或减小FC
- **RC偏高**: 减小K0, K1或增大FC
- **响应太快**: 减小K0, 增大K1
- **响应太慢**: 增大K0, 减小K1

## SCS-CN模型参数说明

| 参数 | 名称 | 典型范围 | 说明 |
|------|------|----------|------|
| CN | Curve Number | 40-98 | 土地利用和土壤综合特征 |

**CN值参考**：
- **林地，良好状况**: CN = 55-70
- **草地，良好状况**: CN = 61-74
- **农田，中等状况**: CN = 78-85
- **城市，不透水**: CN = 90-98

**调整建议**：
- **RC偏低**: 增大CN值
- **RC偏高**: 减小CN值

## 敏感性分析结果解读

敏感性分析CSV文件格式：
```csv
parameter,value,runoff_coefficient,total_runoff_mm
field_capacity,50.0,0.5234,78.51
field_capacity,100.0,0.4891,73.37
...
```

**如何使用**：
1. 绘制 parameter value vs runoff_coefficient 曲线
2. 曲线斜率大 = 参数敏感，需要精确率定
3. 曲线平坦 = 参数不敏感，可以使用默认值

## 自动率定使用技巧

### 1. 基本率定
```python
# 自动率定匹配观测RC
optimal_params = analyzer.calibrate_model('Zone1', 'HBV')
```

### 2. 指定目标RC
```python
# 率定到特定目标RC
optimal_params = analyzer.calibrate_model('Zone1', 'HBV', target_rc=0.4)
```

### 3. 率定所有分区
```python
for result in analyzer.results.values():
    optimal_params = analyzer.calibrate_model(result.zone_id, 'HBV')
    # 保存或使用 optimal_params
```

### 4. 评估率定效果
率定后会自动输出：
```
Calibrating HBV for Zone1...
  Target RC: 0.4523
  Running optimization...
  Optimization complete!
  Final error: 0.0001
  Calibrated RC: 0.4524 (Target: 0.4523)
  Calibrated parameters:
    field_capacity = 198.45
    beta = 2.34
    k0 = 0.048
    k1 = 0.012
    k2 = 0.0015
```

## 常见问题

### Q1: 某个模型误差特别大怎么办？
**A**: 
1. 检查该模型是否适合该流域
2. 尝试自动率定优化参数
3. 进行敏感性分析找出关键参数
4. 如果仍然不好，选择其他模型

### Q2: 所有模型都偏高或偏低？
**A**:
1. 检查观测数据质量（面积、单位转换）
2. 检查降雨数据合理性
3. 可能需要调整时间步长
4. 考虑是否遗漏了某些损失项（蒸发等）

### Q3: 不同分区最佳模型不一样？
**A**: 
这是正常的！因为：
- 不同分区土地利用不同
- 不同分区土壤类型不同
- 不同分区地形坡度不同
建议：为每个分区使用其最佳模型

### Q4: 如何选择产流和汇流模型组合？
**A**:
1. 先看单独产流模型性能
2. 再看加汇流后的改善
3. 选择改善最明显的组合
4. 优先选择物理机制合理的组合

### Q5: 率定后参数是否合理？
**A**: 检查：
1. 参数是否在合理范围内
2. 参数之间是否有矛盾（如K0<K1<K2）
3. 是否符合流域物理特征
4. 在其他时段是否仍然有效

## 依赖安装

```bash
# 基础依赖
pip install numpy pandas matplotlib

# GIS支持
pip install geopandas

# 优化算法
pip install scipy

# 全部安装
pip install numpy pandas matplotlib geopandas scipy
```

## 扩展开发

### 添加新的产流模型

```python
class MyNewModel(HydrologicModel):
    def __init__(self):
        super().__init__("MyNewModel")
        self.default_params = {
            'param1': 1.0,
            'param2': 2.0
        }
    
    def run(self, precipitation: np.ndarray, **params) -> np.ndarray:
        p = {**self.default_params, **params}
        # 实现您的模型
        runoff = precipitation * p['param1']  # 示例
        return runoff

# 注册模型
analyzer.runoff_models['MyNewModel'] = MyNewModel()
```

### 添加新的汇流模型

```python
class MyRoutingModel:
    def __init__(self):
        self.name = "MyRouting"
        self.default_params = {'lag': 1.0}
    
    def route(self, inflow: np.ndarray, dt: float = 1.0, **params) -> np.ndarray:
        # 实现汇流算法
        outflow = np.roll(inflow, int(params.get('lag', 1)))
        return outflow

# 注册汇流模型
analyzer.routing_models['MyRouting'] = MyRoutingModel()
```

## 性能优化建议

1. **减少测试的模型组合数量**
   ```python
   # 只测试关键模型
   analyzer.runoff_models = {'HBV': HBVModel()}
   analyzer.routing_models = {'Muskingum': MuskingumRouting()}
   ```

2. **并行率定多个分区**
   ```python
   from multiprocessing import Pool
   
   with Pool(4) as pool:
       results = pool.map(calibrate_zone, zone_ids)
   ```

3. **减少优化迭代次数**（快速测试时）
   ```python
   # 修改 differential_evolution 参数
   result = differential_evolution(
       objective, bounds, 
       maxiter=50,  # 减少迭代（默认100）
       popsize=10   # 减少种群（默认15）
   )
   ```

## 参考文献

1. **HBV模型**: Bergström, S. (1992). The HBV model - its structure and applications.
2. **SCS-CN方法**: USDA (1986). Urban Hydrology for Small Watersheds.
3. **Green-Ampt**: Green, W. H., & Ampt, G. A. (1911). Studies on Soil Physics.
4. **Muskingum汇流**: McCarthy, G. T. (1938). The unit hydrograph and flood routing.
5. **差分进化**: Storn, R., & Price, K. (1997). Differential evolution.

## 许可和贡献

本模块是HydroSIS项目的一部分。欢迎贡献新的模型和改进！

## 联系方式

如有问题或建议，请通过GitHub Issues反馈。
