# HydroSIS AI Development Guide

**重要：任何AI辅助开发必须优先查阅并使用本指南！**

## 核心原则

### 1. **Always Use Base Library First (优先使用基础库)**

在开发任何功能之前，必须按以下顺序检查：

1. **检查基础库是否已有类似功能**
   - 不要重复造轮子
   - 不要在脚本中内联实现已有的功能
   - 不要创建临时的"工具函数"

2. **如果基础库没有，考虑扩展基础库**
   - 将新功能添加到适当的基础库模块
   - 确保新功能是可重用的、通用的

3. **只有在特定情况下才在脚本中实现**
   - 功能非常特定于当前任务
   - 不具有通用性和可重用性

## HydroSIS基础库结构

### 性能指标 (Performance Metrics)
**位置**: `hydrosis/evaluation/metrics.py`

**可用函数**:
```python
from hydrosis.evaluation.metrics import (
    nash_sutcliffe_efficiency,      # NSE
    log_nash_sutcliffe_efficiency,  # Log NSE (强调低流量)
    kling_gupta_efficiency,          # KGE
    rmse,                            # 均方根误差
    mae,                             # 平均绝对误差
    percent_bias,                    # 百分比偏差
    pearson_correlation,             # Pearson相关系数
)

# 使用示例
nse = nash_sutcliffe_efficiency(simulated, observed)
kge = kling_gupta_efficiency(simulated, observed)
log_nse = log_nash_sutcliffe_efficiency(simulated, observed, epsilon=1e-6)
```

**何时使用**:
- ✅ 任何需要评估模拟vs观测精度的地方
- ✅ 参数率定的目标函数
- ✅ 模型验证和比较
- ❌ **不要**在脚本中重新实现这些函数

### 参数率定 (Parameter Calibration)
**位置**: `hydrosis/calibration/`

**可用函数**:
```python
from hydrosis.calibration import (
    calibrate_parameters,              # 通用率定接口
    differential_evolution_calibrate,  # DE算法
    CalibrationResult,                 # 结果对象
)

# 使用示例 - 简单方式
result = calibrate_parameters(
    objective_function=my_objective,
    param_bounds=[(100, 500), (1.0, 3.0)],
    algorithm="differential_evolution",
    maximize=True,
    maxiter=100,
    popsize=15,
    seed=42
)

# 使用示例 - 直接调用特定算法
result = differential_evolution_calibrate(
    objective_function=my_objective,
    param_bounds=[(100, 500), (1.0, 3.0)],
    maximize=True,
    maxiter=100
)

# 访问结果
print(f"Best parameters: {result.best_params}")
print(f"Best score: {result.best_score}")
print(f"Convergence: {result.convergence_history}")
```

**CalibrationResult 属性**:
- `success`: 是否成功
- `best_params`: 最优参数列表
- `best_score`: 最优目标函数值
- `n_iterations`: 迭代次数
- `n_evaluations`: 函数评估次数
- `computation_time`: 计算时间(秒)
- `convergence_history`: 收敛历史
- `message`: 状态消息
- `algorithm`: 算法名称

**何时使用**:
- ✅ HBV、HYMOD等模型的参数率定
- ✅ 任何需要优化参数的场景
- ❌ **不要**直接调用scipy的differential_evolution
- ❌ **不要**实现自己的优化算法包装器

### 可视化 (Visualization)
**位置**: `hydrosis/reporting/charts.py`

**可用函数**:
```python
from hydrosis.reporting.charts import (
    plot_hydrograph,     # 水文过程线
    plot_scatter,        # 散点图
    plot_convergence,    # 收敛历史
    plot_metric_bars,    # 指标柱状图
)
from pathlib import Path

# 水文过程对比图
plot_hydrograph(
    output_path=Path("output/hydrograph.png"),
    simulations={"HBV": simulated_series},
    observed=observed_series,
    title="Model Calibration",
    xlabel="Time Step (hours)",
    ylabel="Discharge (m³/s)"
)

# 散点图
plot_scatter(
    output_path=Path("output/scatter.png"),
    observed=observed_series,
    simulated=simulated_series,
    title="Observed vs Simulated",
    equal_axis=True
)

# 收敛历史
plot_convergence(
    output_path=Path("output/convergence.png"),
    convergence_history=result.convergence_history,
    title="Calibration Convergence",
    ylabel="NSE",
    maximize=True
)
```

**何时使用**:
- ✅ 展示率定结果
- ✅ 模型对比
- ✅ 诊断分析
- ❌ **不要**使用matplotlib直接绘图（除非基础库功能不满足）

### 径流生成模型 (Runoff Generation)
**位置**: `hydrosis/runoff/`

**可用模型**:
```python
from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator

# HBV模型
hbv = HBVRunoff(parameters_dict)
runoff_m3s = hbv.simulate(subbasin, precipitation_series)

# 增强型径流生成器（用于生成观测数据）
generator = EnhancedRunoffGenerator(
    soil_capacity=420.0,
    soil_beta=2.0,
    fast_threshold=0.70,
    fast_ratio=0.40,
    inter_ratio=0.35,
    base_ratio=0.25,
    k_fast=0.30,
    k_inter=0.09,
    k_base=0.022,
    initial_soil=170.0,
    et_rate=0.9
)
runoff_m3s, stats = generator.generate(
    precipitation_series=precip,
    area_km2=140.0,
    return_components=True
)
```

**何时使用**:
- ✅ 产流模拟
- ✅ 生成合成观测数据（用于率定验证）
- ❌ **不要**实现简单的Q=Rc×P线性模型（使用EnhancedRunoffGenerator）

## 开发工作流程

### 步骤1: 需求分析
在开始编码前，问自己：
1. 这个功能是否已经在基础库中？
2. 基础库中是否有类似的功能可以扩展？
3. 这个功能是否具有通用性？

### 步骤2: 搜索基础库
```bash
# 搜索相关功能
grep -r "function_name" hydrosis/
grep -r "class.*Name" hydrosis/

# 查看模块结构
ls hydrosis/evaluation/
ls hydrosis/calibration/
ls hydrosis/reporting/
ls hydrosis/runoff/
```

### 步骤3: 使用或扩展基础库
**优先级**:
1. ✅ **最优**: 直接使用现有功能
2. ✅ **次优**: 扩展现有模块
3. ⚠️ **可接受**: 在脚本中实现（仅当非通用功能时）

### 步骤4: 如果基础库不够用
1. **上网搜索最佳实践**
   - 搜索学术论文和行业标准
   - 查看成熟的水文建模库（如hydroeval, spotpy等）
   - 参考scipy、numpy的实现

2. **升级基础库**
   - 将新功能添加到合适的模块
   - 编写清晰的文档字符串
   - 确保函数签名一致

3. **测试和验证**
   - 确保新功能正确工作
   - 与已知结果对比验证

## 代码规范

### 避免硬编码
❌ **错误示例**:
```python
# 硬编码路径
data = pd.read_csv("results/upper_truckee/step_08/data.csv")

# 硬编码参数
zone1_area = 139.995
zone1_subbasins = [101, 102, 103, ...]
```

✅ **正确示例**:
```python
# 从配置读取
from hydrosis.config import load_config
config = load_config(config_path)
data = pd.read_csv(config["data_path"])

# 从数据文件读取
zones_df = pd.read_csv(zones_stats_file)
zone1_data = zones_df[zones_df['Zone_ID'] == 1]
zone1_area = zone1_data['Area_km2'].sum()
```

### 函数设计原则
1. **单一职责**: 一个函数只做一件事
2. **参数化**: 不要硬编码值，使用参数
3. **可重用**: 设计时考虑其他场景
4. **文档齐全**: 使用docstring说明参数和返回值

✅ **好的函数设计**:
```python
def calibrate_zone_parameters(
    zone_id: int,
    observed_runoff: np.ndarray,
    precipitation: np.ndarray,
    area_km2: float,
    param_bounds: dict,
    output_dir: Path,
    algorithm: str = "differential_evolution",
    **kwargs
) -> CalibrationResult:
    """
    Calibrate hydrological model parameters for a specific zone.

    Parameters
    ----------
    zone_id : int
        Zone identifier
    observed_runoff : np.ndarray
        Observed runoff series (m³/s)
    precipitation : np.ndarray
        Precipitation series (mm/h)
    area_km2 : float
        Zone area in km²
    param_bounds : dict
        Parameter bounds {param_name: (min, max)}
    output_dir : Path
        Directory to save results
    algorithm : str, optional
        Calibration algorithm, default "differential_evolution"
    **kwargs
        Additional arguments passed to calibration algorithm

    Returns
    -------
    CalibrationResult
        Calibration results including best parameters and metrics
    """
    # 实现...
```

## 常见场景指南

### 场景1: 需要计算性能指标
```python
# ❌ 不要这样做
def my_nse(obs, sim):
    mean_obs = np.mean(obs)
    return 1 - np.sum((obs-sim)**2) / np.sum((obs-mean_obs)**2)

# ✅ 应该这样做
from hydrosis.evaluation.metrics import nash_sutcliffe_efficiency
nse = nash_sutcliffe_efficiency(simulated, observed)
```

### 场景2: 需要参数率定
```python
# ❌ 不要这样做
from scipy.optimize import differential_evolution
result = differential_evolution(obj_func, bounds, maxiter=100, ...)

# ✅ 应该这样做
from hydrosis.calibration import calibrate_parameters
result = calibrate_parameters(
    objective_function=obj_func,
    param_bounds=bounds,
    maximize=True,
    maxiter=100
)
# 或者
from hydrosis.calibration import differential_evolution_calibrate
result = differential_evolution_calibrate(
    objective_function=obj_func,
    param_bounds=bounds,
    maximize=True
)
```

### 场景3: 需要绘制对比图
```python
# ❌ 不要这样做
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(observed, label='Observed')
ax.plot(simulated, label='Simulated')
plt.savefig('output.png')

# ✅ 应该这样做
from hydrosis.reporting.charts import plot_hydrograph
plot_hydrograph(
    output_path=Path('output.png'),
    simulations={'Model': simulated},
    observed=observed,
    title='Calibration Result'
)
```

### 场景4: 需要生成观测数据
```python
# ❌ 不要这样做（简单线性模型）
runoff = precipitation * runoff_coefficient * area / 3.6

# ✅ 应该这样做（物理基础更强）
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator
generator = EnhancedRunoffGenerator(
    soil_capacity=420.0,
    soil_beta=2.0,
    # ... 其他参数
)
runoff, stats = generator.generate(precipitation, area_km2)
```

## 扩展基础库时的注意事项

### 1. 选择正确的模块
- **性能指标** → `hydrosis/evaluation/metrics.py`
- **率定算法** → `hydrosis/calibration/optimizers.py`
- **可视化** → `hydrosis/reporting/charts.py`
- **径流模型** → `hydrosis/runoff/`
- **数据处理** → `hydrosis/utils/`

### 2. 保持一致的API风格
```python
# 遵循现有函数的签名模式
def new_metric(
    simulated: Sequence[float],
    observed: Sequence[float],
    **kwargs
) -> float:
    """
    Clear description.

    Parameters
    ----------
    simulated : Sequence[float]
        Simulated values
    observed : Sequence[float]
        Observed values
    **kwargs
        Additional parameters

    Returns
    -------
    float
        Metric value
    """
    pass
```

### 3. 更新__all__和模块导入
```python
# 在模块文件末尾
__all__ = [
    "existing_function",
    "new_function",  # 添加新函数
]

# 在__init__.py中
from .module import (
    existing_function,
    new_function,  # 添加导入
)
```

## 记住：优先级顺序

1. ✅ **使用现有基础库功能** (最优)
2. ✅ **扩展基础库** (次优)
3. ⚠️ **搜索最佳实践，升级基础库** (如果基础库不够)
4. ❌ **在脚本中重复实现** (最差，应避免)

## 检查清单

在提交代码前，确认：
- [ ] 已检查基础库是否有类似功能
- [ ] 如果使用基础库，导入路径正确
- [ ] 如果扩展基础库，添加了文档和导出
- [ ] 没有硬编码路径、参数或配置
- [ ] 函数设计遵循单一职责原则
- [ ] 代码具有良好的可重用性

---

**记住：好的代码是可重用的代码。优先使用基础库！**
