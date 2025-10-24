# HydroSIS 框架指南

本文档介绍 HydroSIS 项目中新实现的统一框架，帮助您快速上手。

---

## 📚 目录

1. [诊断框架](#1-诊断框架)
2. [并行框架](#2-并行框架)
3. [校准框架](#3-校准框架)
4. [配置管理](#4-配置管理)
5. [示例代码](#5-示例代码)

---

## 1. 诊断框架

统一的模型诊断接口，自动识别和报告模型问题。

### 核心概念

- **BaseDiagnostic**: 诊断器抽象基类
- **DiagnosticResult**: 标准化诊断结果
- **DiagnosticIssue**: 问题描述（包含严重程度、详情和建议）
- **IssueSeverity**: 4级严重程度（INFO/WARNING/ERROR/CRITICAL）

### 可用诊断器

#### 1.1 水量平衡诊断 (WaterBalanceDiagnostic)

检测HBV模型的水量平衡问题。

```python
from hydrosis.diagnostics import WaterBalanceDiagnostic

diagnostic = WaterBalanceDiagnostic(
    output_dir="results/diagnostics",
    target_runoff_coefficient=0.5
)

result = diagnostic.run(
    precipitation=precipitation_array,  # mm/h
    runoff=runoff_array,               # mm/h
    initial_lower=1000.0,               # mm
    k2=0.02                            # 1/h
)

# 查看问题
for issue in result.issues:
    print(f"{issue.severity}: {issue.message}")
    if issue.suggestion:
        print(f"  建议: {issue.suggestion}")

# 查看指标
print(f"径流系数: {result.metrics['runoff_coefficient']:.4f}")

# 查看修正建议
for rec in result.recommendations:
    print(f"• {rec}")
```

**检测问题**:
- ✅ 径流系数异常 (RC > 1.0)
- ✅ 初始储量配置不合理
- ✅ 基流参数验证
- ✅ 储量释放分析

**输出文件**:
- `水量平衡诊断_report.txt` - 文本报告
- `水量平衡诊断_report.json` - JSON报告
- `水量平衡诊断_visualization.png` - 4图可视化

#### 1.2 降雨空间分布诊断 (PrecipitationDiagnostic)

检测降雨数据的空间分布问题。

```python
from hydrosis.diagnostics import PrecipitationDiagnostic

diagnostic = PrecipitationDiagnostic(
    output_dir="results/diagnostics",
    anomaly_threshold=0.3  # 30%差异视为异常
)

result = diagnostic.run(
    precipitation_df=precip_df,      # DataFrame, 列为子流域ID
    subbasins_df=subbasins_df        # 必须包含: subzone_id, zone_id, area_km2
)

# 获取异常分区
anomalies = [
    issue for issue in result.issues
    if issue.category == "zone_precipitation_anomaly"
]

for issue in anomalies:
    zone_id = issue.details['zone_id']
    diff = issue.details['relative_difference']
    print(f"Zone {zone_id} 降雨异常，差异 {diff*100:.1f}%")
```

**检测问题**:
- ✅ 分区间降雨差异过大
- ✅ 子流域降雨变异系数异常
- ✅ 数据质量问题（缺失值、负值、异常值）
- ✅ 雨量站覆盖不足

**输出文件**:
- 分区对比可视化
- 各分区详细诊断图
- 数据质量报告

---

## 2. 并行框架

通用并行任务执行框架，支持多种执行模式。

### 核心概念

- **ParallelExecutor**: 并行执行器抽象基类
- **ExecutionMode**: 执行模式（串行/多进程/多线程）
- **ExecutionConfig**: 执行配置
- **TaskResult**: 任务执行结果

### 2.1 简单并行映射

使用 `parallel_map` 快速并行化函数：

```python
from hydrosis.parallel import parallel_map, ExecutionConfig, ExecutionMode

def process_data(data):
    # 耗时操作
    return result

# 并行执行
config = ExecutionConfig(
    mode=ExecutionMode.MULTIPROCESS,
    max_workers=4
)

results = parallel_map(
    process_data,
    data_list,
    config=config
)
```

### 2.2 自定义并行执行器

继承 `ParallelExecutor` 创建复杂任务执行器：

```python
from hydrosis.parallel import ParallelExecutor, ExecutionConfig

class MyExecutor(ParallelExecutor[InputType, OutputType]):
    """自定义执行器"""

    def execute_task(self, task: InputType) -> OutputType:
        """执行单个任务"""
        # 实现具体逻辑
        return process(task)

    def get_task_id(self, task: InputType) -> str:
        """可选: 自定义任务ID"""
        return task.id

# 使用
executor = MyExecutor(
    config=ExecutionConfig(
        mode=ExecutionMode.MULTIPROCESS,
        max_workers=8,
        retry_on_failure=True,
        max_retries=3
    )
)

results = executor.run(tasks)

# 获取成功结果
successful = executor.get_successful_results(results)

# 获取失败任务
failed = executor.get_failed_tasks(results)
```

### 2.3 并行数据验证

使用 `ParallelValidator` 加速数据验证：

```python
from hydrosis.parallel import validate_datasets_parallel
from hydrosis.validation import BaseValidator

class MyValidator(BaseValidator):
    def validate(self, data):
        # 验证逻辑
        return ValidationResult(...)

# 并行验证多个数据集
datasets = {
    '2020': df_2020,
    '2021': df_2021,
    '2022': df_2022,
}

validator = MyValidator()
results = validate_datasets_parallel(validator, datasets)

for name, result in results.items():
    print(f"{name}: {'✓' if result.is_valid else '✗'}")
```

### 执行配置选项

```python
ExecutionConfig(
    mode=ExecutionMode.MULTIPROCESS,  # 执行模式
    max_workers=4,                    # worker数量
    chunk_size=1,                     # 分块大小
    timeout=None,                     # 超时时间（秒）
    retry_on_failure=False,           # 是否重试
    max_retries=3,                    # 最大重试次数
    show_progress=True,               # 显示进度
    raise_on_error=False             # 遇错是否抛出异常
)
```

### 最佳实践

| 任务类型 | 推荐模式 | 说明 |
|---------|---------|------|
| CPU密集型 | MULTIPROCESS | 绕过GIL，充分利用多核 |
| I/O密集型 | MULTITHREADED | 减少进程开销 |
| 调试/简单 | SEQUENTIAL | 便于调试 |

---

## 3. 校准框架

统一的模型参数校准接口。

### 核心概念

- **BaseCalibrator**: 校准器抽象基类
- **CalibrationData**: 校准数据容器
- **CalibrationConfig**: 校准配置
- **CalibrationResult**: 校准结果

### 3.1 HBV模型校准

```python
from hydrosis.calibration import HBVCalibrator, CalibrationConfig, CalibrationData

# 准备数据
calib_data = CalibrationData(
    precipitation=precip_array,      # mm/h
    observed_runoff=runoff_array,   # mm/h
    area_km2=100.0,
    temperature=temp_array,          # 可选
    metadata={'station': 'XXX'}      # 可选
)

# 配置校准
config = CalibrationConfig(
    param_bounds={
        'FC': [100, 500],      # 校准参数及边界
        'BETA': [1.0, 4.0],
    },
    fixed_params={             # 固定参数
        'LP': 0.7,
        'K0': 0.1,
        # ... 其他固定参数
    },
    algorithm='differential_evolution',  # 优化算法
    algorithm_params={
        'maxiter': 100,
        'popsize': 15
    },
    objective_metric='nse',    # 目标指标
    maximize=True,             # 是否最大化
    warmup_steps=24,          # 预热步数
    seed=42                   # 随机种子
)

# 运行校准
calibrator = HBVCalibrator(
    data=calib_data,
    config=config,
    output_dir="results/calibration"
)

result = calibrator.run_calibration()

# 查看结果
print(f"最优NSE: {result.metrics['nse']:.4f}")
print(f"最优参数: {result.best_params}")
print(f"计算时间: {result.computation_time:.2f}秒")
print(f"评估次数: {result.n_evaluations}")

# 保存结果
calibrator.save_results(result, prefix="basin_01")
```

### 3.2 可用优化算法

| 算法 | 类型 | 特点 | 适用场景 |
|-----|------|------|---------|
| `differential_evolution` | 全局优化 | 不需要初始值，鲁棒性强 | **推荐**，适合大多数情况 |
| `nelder_mead` | 局部优化 | 需要好的初始值 | 精细调优 |
| `powell` | 局部优化 | 不需要梯度 | 特定问题 |

### 3.3 性能指标

支持的目标指标：
- `nse`: Nash-Sutcliffe Efficiency (推荐)
- `kge`: Kling-Gupta Efficiency
- `rmse`: Root Mean Square Error

---

## 4. 配置管理

统一的配置加载和管理。

### 4.1 ConfigManager

```python
from hydrosis.config import ConfigManager

# 获取单例实例
config_manager = ConfigManager()

# 加载配置
config = config_manager.load_config(
    "config/workflow_config.yaml",
    use_cache=True,
    resolve_paths=True,
    resolve_env_vars=True
)

# 访问配置
value = config_manager.get("paths.dem", default="/default/path")

# 嵌套访问
nested = config_manager.get("model.parameters.hbv.FC")

# 清除缓存
config_manager.clear_cache()
```

### 4.2 环境变量支持

配置文件中使用环境变量：

```yaml
# workflow_config.yaml
paths:
  dem: ${DEM_PATH}/elevation.tif
  output: ${OUTPUT_DIR}/results

model:
  threads: ${NUM_THREADS:4}  # 带默认值
```

使用：

```python
import os
os.environ['DEM_PATH'] = '/data/dems'
os.environ['OUTPUT_DIR'] = '/results'

config = config_manager.load_config("config/workflow_config.yaml")
# paths.dem 自动解析为 '/data/dems/elevation.tif'
```

---

## 5. 示例代码

### 5.1 运行示例

```bash
# 诊断框架示例
python examples/diagnostics_example.py

# 并行框架示例
python examples/parallel_example.py

# 综合工作流示例
python examples/integrated_workflow_example.py
```

### 5.2 完整工作流

```python
from pathlib import Path
from hydrosis.parallel import ExecutionConfig, ExecutionMode
from hydrosis.diagnostics import WaterBalanceDiagnostic
from hydrosis.calibration import HBVCalibrator, CalibrationConfig, CalibrationData

# 1. 诊断模型问题
diagnostic = WaterBalanceDiagnostic(output_dir="results/diagnostics")
diag_result = diagnostic.run(
    precipitation=precip,
    runoff=runoff,
    initial_lower=2000.0,
    k2=0.02
)

# 2. 根据诊断结果调整配置
if diag_result.recommendations:
    print("修正建议:")
    for rec in diag_result.recommendations:
        print(f"  • {rec}")

# 3. 校准模型参数
calib_data = CalibrationData(
    precipitation=precip,
    observed_runoff=observed,
    area_km2=100.0
)

config = CalibrationConfig(
    param_bounds={'FC': [100, 500], 'BETA': [1.0, 4.0]},
    fixed_params={...},
    algorithm='differential_evolution'
)

calibrator = HBVCalibrator(calib_data, config)
calib_result = calibrator.run_calibration()

# 4. 使用最优参数运行模型
best_params = calib_result.best_params
# ... 继续模拟
```

---

## 📊 性能提示

### 并行化建议

1. **CPU密集型任务**: 使用 `MULTIPROCESS`
   ```python
   config = ExecutionConfig(
       mode=ExecutionMode.MULTIPROCESS,
       max_workers=cpu_count()
   )
   ```

2. **I/O密集型任务**: 使用 `MULTITHREADED`
   ```python
   config = ExecutionConfig(
       mode=ExecutionMode.MULTITHREADED,
       max_workers=cpu_count() * 2
   )
   ```

3. **混合任务**: 根据瓶颈选择

### 优化算法选择

- **首次校准**: `differential_evolution` (全局搜索)
- **精细调优**: `nelder_mead` (从好的初始值开始)
- **大参数空间**: 增加 `popsize` 和 `maxiter`

---

## 🔧 故障排除

### 常见问题

**Q: 并行执行失败 "Can't pickle..."**

A: 确保任务类定义在模块级别，不要在函数内定义：

```python
# ✓ 正确
class MyExecutor(ParallelExecutor):
    ...

# ✗ 错误
def main():
    class MyExecutor(ParallelExecutor):  # 无法pickle
        ...
```

**Q: 校准结果不理想**

A: 尝试：
1. 增加迭代次数 `maxiter`
2. 增加种群大小 `popsize`
3. 调整参数边界
4. 检查数据质量
5. 增加预热步数 `warmup_steps`

**Q: 诊断没有发现问题**

A: 可能：
1. 阈值设置过宽
2. 数据本身质量很好
3. 需要使用其他诊断器

---

## 📚 参考

- [API文档](API.md)
- [开发指南](CONTRIBUTING.md)
- [测试指南](TESTING.md)
- [任务清单](../TASKS.md)

---

**版本**: 1.0.0
**更新日期**: 2025-01-24
**维护者**: HydroSIS Team
