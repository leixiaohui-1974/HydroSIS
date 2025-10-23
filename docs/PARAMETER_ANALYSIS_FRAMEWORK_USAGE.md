# HydroSIS 参数分析框架使用指南

本文档介绍如何使用HydroSIS参数分析框架进行水文模型参数率定、敏感性分析和不确定性分析。

## 框架概述

HydroSIS参数分析框架提供了完整的工具链用于水文模型参数分析：

```
hydrosis/analysis/
├── metrics.py                    # 性能评估指标
├── sensitivity.py                # 参数敏感性分析
├── uncertainty.py                # 不确定性分析
├── calibration_optimization.py  # 参数率定优化
└── visualization.py              # 结果可视化
```

---

## 快速开始

### 1. 基础导入

```python
from hydrosis.analysis import (
    # 性能评估
    calculate_metrics, nse, rmse,
    # 敏感性分析
    one_at_a_time_sensitivity,
    # 参数率定
    calibrate_model,
    # 不确定性分析
    monte_carlo_analysis, glue_analysis,
    # 可视化
    plot_hydrograph_comparison, plot_convergence_history
)
```

---

## 模块1: 性能评估指标

### 支持的指标

| 指标 | 说明 | 最优值 | 范围 |
|------|------|--------|------|
| NSE | Nash-Sutcliffe效率系数 | 1.0 | (-∞, 1] |
| RMSE | 均方根误差 | 0 | [0, ∞) |
| MAE | 平均绝对误差 | 0 | [0, ∞) |
| PBIAS | 百分比偏差 | 0 | (-∞, ∞) |
| KGE | Kling-Gupta效率 | 1.0 | (-∞, 1] |
| log-NSE | 对数NSE（强调低流量） | 1.0 | (-∞, 1] |
| VE | 体积误差 | 0 | (-100, ∞) |

### 使用示例

```python
import numpy as np
from hydrosis.analysis import calculate_metrics

# 观测和模拟数据
observed = np.array([10.5, 12.3, 15.8, 20.2, 14.2])
simulated = np.array([10.0, 12.0, 16.5, 19.5, 13.5])

# 计算所有指标
metrics = calculate_metrics(observed, simulated)

print(f"NSE:   {metrics['nse']:.4f}")
print(f"RMSE:  {metrics['rmse']:.4f} m³/s")
print(f"PBIAS: {metrics['pbias']:.2f}%")
```

---

## 模块2: 参数敏感性分析

### OAT敏感性分析

一次变动一个参数，分析对模型输出的影响。

```python
from hydrosis.analysis import one_at_a_time_sensitivity

def model_function(FC, K0, BETA):
    """你的模型函数"""
    simulated = run_model(FC, K0, BETA)
    return nash_sutcliffe_efficiency(observed, simulated)

# 运行敏感性分析
results = one_at_a_time_sensitivity(
    model_function=model_function,
    parameters={'FC': 150, 'K0': 0.3, 'BETA': 1.0},
    param_ranges={'FC': [100, 200], 'K0': [0.1, 0.5], 'BETA': [0.5, 2.0]},
    variations=[0.8, 0.9, 1.0, 1.1, 1.2]
)

# 查看参数重要性排序
for i, (param, si) in enumerate(results['ranked_parameters'], 1):
    print(f"{i}. {param}: SI = {si:.4f}")
```

### Sobol全局敏感性（可选，需要SALib）

```python
from hydrosis.analysis import sobol_sensitivity

results = sobol_sensitivity(
    model_function=model_function,
    param_ranges={'FC': [100, 200], 'K0': [0.1, 0.5]},
    n_samples=1000,
    calc_second_order=True
)

print("一阶指数 (S1):", results['S1'])
print("总阶指数 (ST):", results['ST'])
```

---

## 模块3: 参数率定

### 支持的优化算法

| 算法 | 特点 | 适用场景 |
|------|------|----------|
| **SCE-UA** | 水文学标准算法，全局搜索能力强 | 中低维问题（<10参数），需要高质量解 |
| **PSO** | 粒子群优化，计算效率高 | 中等维度，快速率定 |
| **DE** | 差分进化（scipy），适合高维 | 高维问题（>10参数） |

### 基本用法

```python
from hydrosis.analysis import calibrate_model

# 定义目标函数（最大化NSE）
def objective(FC, K0, BETA):
    simulated = run_hbv_model(FC, K0, BETA)
    return nash_sutcliffe_efficiency(observed, simulated)

# 参数搜索范围
param_bounds = {
    'FC': [100, 200],
    'K0': [0.1, 0.5],
    'BETA': [0.5, 2.0]
}

# 率定（SCE-UA）
result = calibrate_model(
    objective_function=objective,
    param_bounds=param_bounds,
    method='sce_ua',
    maximize=True,
    max_iterations=100
)

print(result.summary())
print(f"最优参数: {result.best_params}")
print(f"最优NSE: {result.best_score:.4f}")
```

### 算法对比

```python
# 测试不同算法
algorithms = ['sce_ua', 'pso', 'differential_evolution']
results = {}

for method in algorithms:
    result = calibrate_model(
        objective_function=objective,
        param_bounds=param_bounds,
        method=method,
        maximize=True,
        max_iterations=50
    )
    results[method] = result
    print(f"{method}: NSE={result.best_score:.4f}, "
          f"Time={result.computation_time:.2f}s")
```

### PSO参数调优

```python
# PSO算法特定参数
result = calibrate_model(
    objective_function=objective,
    param_bounds=param_bounds,
    method='pso',
    maximize=True,
    n_particles=30,      # 粒子数量
    w=0.7,               # 惯性权重
    c1=1.5,              # 认知系数
    c2=1.5,              # 社会系数
    max_iterations=100
)
```

---

## 模块4: 不确定性分析

### Monte Carlo分析

```python
from hydrosis.analysis import monte_carlo_analysis

# 定义参数分布
param_distributions = {
    'FC': {'dist': 'uniform', 'min': 100, 'max': 200},
    'K0': {'dist': 'normal', 'mean': 0.3, 'std': 0.05, 'min': 0.1, 'max': 0.5}
}

# 运行Monte Carlo
mc_results = monte_carlo_analysis(
    model_function=run_model,
    param_distributions=param_distributions,
    n_samples=1000,
    sampling_method='lhs',  # 或 'monte_carlo'
    observed_data=observed,
    metric_function=nash_sutcliffe_efficiency
)

print(f"NSE均值: {mc_results['metric_mean']:.4f}")
print(f"NSE标准差: {mc_results['metric_std']:.4f}")
print(f"90%置信区间: [{mc_results['metric_percentiles'][0]:.4f}, "
      f"{mc_results['metric_percentiles'][4]:.4f}]")
```

### GLUE不确定性估计

```python
from hydrosis.analysis import glue_analysis

glue_results = glue_analysis(
    model_function=run_model,
    param_distributions=param_distributions,
    observed_data=observed,
    likelihood_function=nash_sutcliffe_efficiency,
    likelihood_threshold=0.7,  # NSE > 0.7为"行为参数"
    n_samples=5000,
    confidence_levels=[0.05, 0.95]
)

print(f"行为参数集: {glue_results['n_behavioral']}/{glue_results['n_total']}")
print(f"接受率: {glue_results['acceptance_rate']*100:.1f}%")

# 参数后验分布
for param, stats in glue_results['posterior_param_stats'].items():
    print(f"{param}: {stats['mean']:.2f} ± {stats['std']:.2f}")
```

---

## 模块5: 可视化

### 水文过程对比图

```python
from hydrosis.analysis import plot_hydrograph_comparison

plot_hydrograph_comparison(
    observed=observed,
    simulated=simulated,
    time_index=times,
    metrics={'NSE': 0.85, 'PBIAS': 3.2},
    title="模型率定结果",
    save_path="hydrograph.png"
)
```

### 散点图

```python
from hydrosis.analysis import plot_scatter

plot_scatter(
    observed=observed,
    simulated=simulated,
    metrics={'NSE': 0.85, 'R²': 0.88},
    save_path="scatter.png"
)
```

### 收敛历史

```python
from hydrosis.analysis import plot_convergence_history

plot_convergence_history(
    convergence_history=result.convergence_history,
    title="SCE-UA收敛历史",
    ylabel="NSE",
    save_path="convergence.png"
)
```

### 不确定性包络

```python
from hydrosis.analysis import plot_uncertainty_envelope

plot_uncertainty_envelope(
    time_index=times,
    observed=observed,
    mean_simulation=mean_sim,
    percentiles={'p5': lower, 'p25': q1, 'p75': q3, 'p95': upper},
    title="预测不确定性包络",
    save_path="uncertainty.png"
)
```

### 参数后验分布

```python
from hydrosis.analysis import plot_parameter_distributions

plot_parameter_distributions(
    param_samples=glue_results['behavioral_params'],
    true_values={'FC': 150, 'K0': 0.3},  # 可选
    save_path="param_distributions.png"
)
```

---

## 完整工作流示例

```python
from hydrosis.analysis import *

# 1. 定义模型和目标函数
def run_hbv(FC, K0, BETA):
    # 运行HBV模型
    return model_output

def objective(FC, K0, BETA):
    sim = run_hbv(FC, K0, BETA)
    return nash_sutcliffe_efficiency(observed, sim)

# 2. 参数敏感性分析
sensitivity = one_at_a_time_sensitivity(
    model_function=objective,
    parameters={'FC': 150, 'K0': 0.3, 'BETA': 1.0},
    param_ranges={'FC': [100, 200], 'K0': [0.1, 0.5], 'BETA': [0.5, 2.0]}
)
print("敏感性排序:", sensitivity['ranked_parameters'])

# 3. 参数率定
calibration = calibrate_model(
    objective_function=objective,
    param_bounds={'FC': [100, 200], 'K0': [0.1, 0.5], 'BETA': [0.5, 2.0]},
    method='sce_ua',
    maximize=True
)
print(f"最优NSE: {calibration.best_score:.4f}")
print(f"最优参数: {calibration.best_params}")

# 4. 不确定性分析
uncertainty = glue_analysis(
    model_function=run_hbv,
    param_distributions={...},
    observed_data=observed,
    likelihood_function=nash_sutcliffe_efficiency,
    likelihood_threshold=0.7
)

# 5. 可视化
calibrated_sim = run_hbv(**calibration.best_params)
plot_hydrograph_comparison(observed, calibrated_sim,
                          metrics={'NSE': calibration.best_score})
plot_convergence_history(calibration.convergence_history)
plot_parameter_distributions(uncertainty['behavioral_params'])
```

---

## 实际案例

### 案例1: 简单演示（5分钟）

```bash
python example_complete_analysis_workflow.py
```

这个示例使用简化的水文模型，演示完整的5步工作流：
- 性能评估
- 敏感性分析
- 参数率定
- 不确定性分析
- 可视化

输出：11个分析图表 + 完整报告

### 案例2: HBV模型率定（需要先生成数据）

```bash
# 第1步：生成估计径流数据
python generate_estimated_runoff.py

# 第2步：运行HBV率定案例
python example_hbv_calibration_case.py
```

这个示例使用真实的HBV模型，展示实际的参数率定流程。

---

## 性能优化建议

### 1. 减少函数评估次数

```python
# SCE-UA: 减少复杂度和迭代次数
result = calibrate_model(
    objective_function=objective,
    param_bounds=param_bounds,
    method='sce_ua',
    n_complexes=3,        # 默认5，减少到3
    max_iterations=30,    # 默认1000，快速测试用30
    patience=10           # 早停
)
```

### 2. 使用LHS代替Monte Carlo

```python
# LHS提供更好的参数空间覆盖
mc_results = monte_carlo_analysis(
    model_function=model,
    param_distributions=params,
    n_samples=100,
    sampling_method='lhs'  # 而不是 'monte_carlo'
)
```

### 3. 缓存模型结果

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_model(FC, K0, BETA):
    # 如果参数相同，直接返回缓存的结果
    return run_expensive_model(FC, K0, BETA)
```

---

## 依赖安装

### 必需依赖

```bash
pip install numpy scipy
```

### 可选依赖

```bash
# Sobol和Morris敏感性分析
pip install SALib

# 更优的Latin Hypercube采样
pip install pyDOE

# 可视化
pip install matplotlib
```

---

## 常见问题

### Q1: 率定结果不收敛怎么办？

**A:**
1. 增加迭代次数：`max_iterations=200`
2. 增加粒子数/复杂度：`n_particles=50`或`n_complexes=10`
3. 扩大参数搜索范围
4. 检查目标函数是否返回合理值
5. 尝试不同的优化算法

### Q2: 哪个优化算法最好？

**A:** 没有绝对最好的算法，建议：
- **SCE-UA**: 首选，水文学标准
- **PSO**: 快速测试，计算资源有限时
- **DE**: 参数维度高（>10个参数）时

### Q3: 如何选择参数范围？

**A:**
1. 查阅文献获取经验范围
2. 使用敏感性分析确定重要参数
3. 先用宽范围初步率定，再用窄范围精细率定
4. 确保范围包含物理合理值

### Q4: 率定时间太长怎么办？

**A:**
1. 减少数据长度（使用代表性时段）
2. 减少率定参数数量（固定不敏感参数）
3. 使用PSO代替SCE-UA
4. 减少迭代次数和种群大小

### Q5: 如何评估率定结果的可靠性？

**A:**
1. 查看多个性能指标（NSE, PBIAS, KGE）
2. 绘制水文过程对比图（目视检查）
3. 运行不确定性分析（参数置信区间）
4. 分割数据集（率定期 vs 验证期）
5. 多次率定检查稳定性

---

## 引用

如果在论文中使用本框架，请引用：

```bibtex
@software{hydrosis_analysis_framework,
  title={HydroSIS Parameter Analysis Framework},
  author={HydroSIS Development Team},
  year={2024},
  url={https://github.com/your-repo/HydroSIS}
}
```

---

## 联系与支持

- **文档**: [docs/](../docs/)
- **示例**: [example_*.py](../)
- **问题**: GitHub Issues

---

**最后更新**: 2024-10-23
**版本**: v1.0.0
