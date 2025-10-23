# HydroSIS参数分析框架设计

## 总体架构

```
hydrosis/analysis/
├── __init__.py
├── metrics.py              # 性能评估指标（NSE, RMSE等）
├── sensitivity.py          # 参数敏感性分析
├── uncertainty.py          # 不确定性分析（蒙特卡洛）
├── calibration.py          # 自动率定算法（SCE-UA, DREAM）
└── visualization.py        # 可视化工具
```

## 模块1: 性能评估指标 (metrics.py)

### 功能
计算模拟与观测之间的统计指标。

### 指标清单

#### 1. Nash-Sutcliffe效率系数 (NSE)
```
NSE = 1 - Σ(Qobs - Qsim)² / Σ(Qobs - Q̄obs)²
```
- 范围：(-∞, 1]
- 1 = 完美拟合
- 0 = 模型等同于平均值
- < 0 = 模型比平均值更差

#### 2. 均方根误差 (RMSE)
```
RMSE = √[Σ(Qobs - Qsim)² / n]
```
- 单位：与流量相同 (m³/s)
- 越小越好

#### 3. 平均绝对误差 (MAE)
```
MAE = Σ|Qobs - Qsim| / n
```

#### 4. 百分比偏差 (PBIAS)
```
PBIAS = 100 × Σ(Qobs - Qsim) / Σ(Qobs)
```
- < ±10%: 非常好
- ±10-15%: 好
- ±15-25%: 满意
- > ±25%: 不满意

#### 5. Kling-Gupta效率 (KGE)
```
KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]
```
其中：
- r = 相关系数
- α = 标准差比
- β = 均值比

#### 6. 对数NSE (log-NSE)
强调低流量拟合

#### 7. 体积误差 (VE)
```
VE = (Vsim - Vobs) / Vobs × 100%
```

### 使用示例
```python
from hydrosis.analysis.metrics import calculate_metrics

metrics = calculate_metrics(
    observed=estimated_runoff,
    simulated=model_output,
    metrics=['nse', 'rmse', 'pbias', 'kge']
)
```

## 模块2: 参数敏感性分析 (sensitivity.py)

### 2.1 单参数敏感性分析 (OAT - One-at-a-Time)

逐个改变参数，观察对输出的影响。

**步骤**：
1. 设定基准参数集
2. 逐个参数在其范围内变化（±10%, ±20%, ±50%）
3. 运行模型，计算性能指标变化
4. 计算敏感性指数

**敏感性指数**：
```
SI = (ΔOutput / Output) / (ΔParam / Param)
```

**优点**：简单直观
**缺点**：忽略参数交互

### 2.2 全局敏感性分析 (Sobol指数)

评估每个参数对输出方差的贡献。

**Sobol指数**：
- First-order (Si): 单个参数的直接影响
- Total-order (STi): 包括参数交互的总影响

**使用SALib库**：
```python
from SALib.sample import saltelli
from SALib.analyze import sobol

# 定义问题
problem = {
    'num_vars': 3,
    'names': ['FC', 'K0', 'BETA'],
    'bounds': [[100, 200], [0.1, 0.5], [0.5, 2.0]]
}

# 采样
param_values = saltelli.sample(problem, N=1000)

# 运行模型
Y = np.array([run_model(params) for params in param_values])

# 分析
Si = sobol.analyze(problem, Y)
```

### 2.3 Morris筛选法

快速筛选重要参数（不需要大量样本）。

**步骤**：
1. 在参数空间中随机采样轨迹
2. 沿每个轨迹逐步改变参数
3. 计算基本效应（EE）
4. 分析μ*（平均绝对效应）和σ（标准差）

**解释**：
- 高μ*: 重要参数
- 高σ: 参数交互或非线性

### 使用示例
```python
from hydrosis.analysis.sensitivity import (
    one_at_a_time_sensitivity,
    sobol_sensitivity,
    morris_screening
)

# OAT
oat_results = one_at_a_time_sensitivity(
    parameters=['FC', 'K0', 'K1'],
    base_values={'FC': 150, 'K0': 0.3, 'K1': 0.1},
    variations=[0.5, 0.8, 1.0, 1.2, 1.5],
    target_metric='nse'
)

# Sobol
sobol_results = sobol_sensitivity(
    parameters={'FC': [100, 200], 'K0': [0.1, 0.5]},
    n_samples=1000,
    target_metric='nse'
)

# Morris
morris_results = morris_screening(
    parameters={'FC': [100, 200], 'K0': [0.1, 0.5]},
    n_trajectories=10,
    n_levels=4
)
```

## 模块3: 不确定性分析 (uncertainty.py)

### 3.1 蒙特卡洛采样 (Monte Carlo)

随机采样参数空间，评估输出不确定性。

**步骤**：
1. 为每个参数定义概率分布（均匀/正态）
2. 随机采样N组参数（N=1000-10000）
3. 运行模型得到N个输出
4. 统计分析：均值、标准差、置信区间

```python
from hydrosis.analysis.uncertainty import monte_carlo_analysis

mc_results = monte_carlo_analysis(
    parameters={
        'FC': {'dist': 'uniform', 'min': 100, 'max': 200},
        'K0': {'dist': 'normal', 'mean': 0.3, 'std': 0.05},
        'BETA': {'dist': 'uniform', 'min': 0.5, 'max': 2.0}
    },
    n_samples=5000,
    output_metrics=['nse', 'peak_flow', 'total_volume']
)
```

### 3.2 拉丁超立方采样 (LHS)

更高效的采样方法（覆盖参数空间更均匀）。

**优点**：
- 比蒙特卡洛需要更少样本
- 参数空间覆盖更全面

```python
from hydrosis.analysis.uncertainty import latin_hypercube_sampling

lhs_results = latin_hypercube_sampling(
    parameters={'FC': [100, 200], 'K0': [0.1, 0.5]},
    n_samples=500
)
```

### 3.3 GLUE (Generalized Likelihood Uncertainty Estimation)

基于可能性的不确定性估计。

**步骤**：
1. 蒙特卡洛采样
2. 设定似然阈值（如NSE > 0.5为"行为参数集"）
3. 用行为参数集估计输出的不确定性区间

```python
from hydrosis.analysis.uncertainty import glue_analysis

glue_results = glue_analysis(
    parameters={'FC': [100, 200], 'K0': [0.1, 0.5]},
    n_samples=10000,
    likelihood_threshold={'nse': 0.5},
    confidence_levels=[0.05, 0.95]  # 90%置信区间
)
```

### 输出结果
- 参数后验分布
- 输出不确定性包络线
- 置信区间（5%-95%）
- 参数相关性矩阵

## 模块4: 自动参数率定 (calibration.py)

### 4.1 SCE-UA算法 (Shuffled Complex Evolution)

全局优化算法，适合多峰、非凸问题。

**算法特点**：
- 结合随机采样和进化算法
- 复形shuffling提高全局搜索能力
- 不需要梯度信息

**参数**：
- ngs: 复形数量（建议 2×参数数）
- npg: 每个复形的点数（建议 2×参数数+1）
- nps: 每个子复形的点数（建议 参数数+1）
- max_iter: 最大迭代次数

```python
from hydrosis.analysis.calibration import sce_ua_calibration

best_params = sce_ua_calibration(
    parameters={
        'FC': {'min': 100, 'max': 200, 'initial': 150},
        'K0': {'min': 0.1, 'max': 0.5, 'initial': 0.3},
        'BETA': {'min': 0.5, 'max': 2.0, 'initial': 1.0}
    },
    objective='nse',  # 最大化NSE
    max_iter=1000,
    ngs=4,  # 4个复形
    npg=10,
    nps=5
)
```

### 4.2 DREAM算法 (DiffeRential Evolution Adaptive Metropolis)

贝叶斯MCMC方法，给出参数后验分布。

**算法特点**：
- 同时给出最优参数和不确定性
- 基于差分进化的自适应MCMC
- 适合高维参数空间

**输出**：
- 参数后验分布
- 参数相关性
- 收敛诊断（Gelman-Rubin统计量）

```python
from hydrosis.analysis.calibration import dream_calibration

dream_results = dream_calibration(
    parameters={
        'FC': {'min': 100, 'max': 200, 'prior': 'uniform'},
        'K0': {'min': 0.1, 'max': 0.5, 'prior': 'uniform'}
    },
    n_chains=3,  # MCMC链数
    n_iter=10000,
    burn_in=2000,  # 预烧期
    thin=10  # 稀释因子
)
```

### 4.3 多目标优化 (NSGA-II)

同时优化多个目标（如峰值误差+体积误差）。

**输出**：
- Pareto前沿（非支配解集）
- 折中解选择

```python
from hydrosis.analysis.calibration import multi_objective_calibration

pareto_front = multi_objective_calibration(
    parameters={'FC': [100, 200], 'K0': [0.1, 0.5]},
    objectives=['nse', 'pbias', 'peak_error'],
    algorithm='nsga2',
    population_size=100,
    n_generations=200
)
```

## 可视化工具 (visualization.py)

### 1. 时间序列对比图
```python
plot_hydrograph_comparison(observed, simulated, metrics)
```

### 2. 散点图 (Observed vs Simulated)
```python
plot_scatter_comparison(observed, simulated)
```

### 3. 参数敏感性图
- 龙卷风图（Tornado plot）
- 雷达图
- 热力图

### 4. 不确定性包络线
```python
plot_uncertainty_envelope(observed, simulated_ensemble, percentiles=[5, 25, 75, 95])
```

### 5. 参数后验分布
```python
plot_posterior_distributions(dream_results)
```

### 6. Pareto前沿
```python
plot_pareto_front(pareto_results, objectives=['NSE', 'PBIAS'])
```

## 工作流示例

### 完整分析流程

```python
# Step 1: 加载数据
from hydrosis.analysis import load_observations, run_model

observed = load_observations('zone_1_estimated_runoff.csv')

# Step 2: 基准运行
baseline_params = {'FC': 150, 'K0': 0.3, 'BETA': 1.0}
baseline_output = run_model(baseline_params)
baseline_metrics = calculate_metrics(observed, baseline_output)

# Step 3: 参数敏感性分析
sensitivity = one_at_a_time_sensitivity(
    parameters=['FC', 'K0', 'K1', 'K2', 'BETA'],
    base_values=baseline_params,
    variations=[0.5, 0.8, 1.0, 1.2, 1.5],
    target_metric='nse'
)

# Step 4: 不确定性分析
uncertainty = monte_carlo_analysis(
    parameters={
        'FC': {'dist': 'uniform', 'min': 100, 'max': 200},
        'K0': {'dist': 'uniform', 'min': 0.1, 0.5}
    },
    n_samples=1000
)

# Step 5: 自动率定
calibrated = sce_ua_calibration(
    parameters={'FC': [100, 200], 'K0': [0.1, 0.5]},
    objective='nse',
    max_iter=500
)

# Step 6: 生成报告
generate_analysis_report(
    baseline=baseline_metrics,
    sensitivity=sensitivity,
    uncertainty=uncertainty,
    calibration=calibrated,
    output_dir='results/analysis_report'
)
```

## 性能优化

### 并行化
```python
from multiprocessing import Pool

def run_model_parallel(param_sets):
    with Pool(processes=8) as pool:
        results = pool.map(run_model, param_sets)
    return results
```

### 缓存
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def run_model_cached(params_tuple):
    params = dict(params_tuple)
    return run_model(params)
```

## 依赖库

- numpy, pandas: 数据处理
- scipy: 优化算法
- SALib: Sobol敏感性分析
- spotpy: SCE-UA, DREAM等率定算法
- matplotlib, seaborn: 可视化
- tqdm: 进度条
