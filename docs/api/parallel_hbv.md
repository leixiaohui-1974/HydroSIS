# HBV模型并行化 API Reference

## 概述

HydroSIS提供了HBV模型的并行化执行框架，通过Python的multiprocessing实现多核并行计算，显著提升大规模流域模拟的效率。

## 核心功能

- ✅ **多核并行**: 基于ProcessPoolExecutor的多进程并行
- ✅ **自动负载均衡**: 动态任务分配，最大化CPU利用率
- ✅ **结果一致性**: 并行和串行执行结果完全一致
- ✅ **灵活配置**: 支持worker数量、进度显示等配置
- ✅ **性能基准**: 内置性能测试工具

---

## API Reference

### ParallelHBVConfig

并行配置数据类。

```python
from hydrosis.runoff.parallel_hbv import ParallelHBVConfig

config = ParallelHBVConfig(
    max_workers=4,              # 最大worker数（建议=CPU核心数）
    chunk_size=100,             # 子流域分组大小（暂未使用）
    use_multiprocessing=True,   # 使用多进程（vs多线程）
    show_progress=True          # 显示进度信息
)
```

**参数说明:**
- `max_workers` (int): 最大并行worker数量
  - 1: 串行执行
  - 2-8: 典型并行度
  - 建议设置为CPU核心数

- `chunk_size` (int): 子流域分组大小（保留参数，暂未实现分组优化）

- `use_multiprocessing` (bool): 是否使用多进程
  - True: 多进程（推荐，适用于CPU密集型任务）
  - False: 多线程（暂未实现）

- `show_progress` (bool): 是否显示进度信息

---

### run_hbv_parallel

并行运行多个分区的HBV模拟。

```python
from hydrosis.runoff.parallel_hbv import run_hbv_parallel, ParallelHBVConfig
import numpy as np

# 准备分区数据
zones = [
    {'zone_id': 1, 'area_km2': 100.5},
    {'zone_id': 2, 'area_km2': 150.2},
    {'zone_id': 3, 'area_km2': 200.8},
]

# 准备降雨数据 {zone_id: precipitation_array}
precipitation_data = {
    1: np.array([0, 5, 10, 8, 3, 0]),  # 6个时间步
    2: np.array([0, 6, 12, 9, 4, 0]),
    3: np.array([0, 4, 8, 7, 2, 0]),
}

# HBV参数
hbv_params = {
    'FC': 300,      # Field capacity (mm)
    'BETA': 2.0,    # Shape coefficient
    'LP': 0.7,      # Evapotranspiration threshold
    'K0': 0.1,      # Recession coefficient 0 (1/h)
    'K1': 0.05,     # Recession coefficient 1 (1/h)
    'K2': 0.01,     # Recession coefficient 2 (1/h)
    'PERC': 2.0,    # Percolation rate (mm/h)
    'UZL': 50.0,    # Upper zone threshold (mm)
    'TT': 0.0,      # Temperature threshold (°C)
    'CFMAX': 3.0,   # Degree-day factor (mm/°C/day)
    'CFR': 0.05,    # Refreezing coefficient
    'CWH': 0.1      # Water holding capacity
}

# 并行执行
config = ParallelHBVConfig(max_workers=4)
results = run_hbv_parallel(
    zones,
    precipitation_data,
    hbv_params,
    config
)

# 结果格式: {zone_id: result_dict}
for zone_id, result in results.items():
    print(f"分区 {zone_id}:")
    print(f"  径流系数: {result['runoff_coefficient']:.4f}")
    print(f"  总降雨: {result['total_precip_mm']:.2f} mm")
    print(f"  总径流: {result['total_runoff_mm']:.2f} mm")
    print(f"  峰值径流: {result['peak_runoff_m3s']:.2f} m³/s")
```

**参数:**
- `zones` (List[Dict]): 分区列表
  - 每个分区字典必须包含: `zone_id`, `area_km2`
  - 示例: `[{'zone_id': 1, 'area_km2': 100.5}, ...]`

- `precipitation_data` (Dict[int, np.ndarray]): 分区降雨字典
  - 键: zone_id
  - 值: 降雨序列 (numpy array, 单位: mm/h)

- `hbv_params` (Dict): HBV参数字典
  - 必须包含所有HBV模型所需参数

- `config` (ParallelHBVConfig, optional): 并行配置
  - 默认: `ParallelHBVConfig()`

**返回:** Dict[int, Dict]

结果字典格式:
```python
{
    zone_id: {
        'zone_id': int,                      # 分区ID
        'runoff_coefficient': float,         # 径流系数
        'total_precip_mm': float,            # 总降雨 (mm)
        'total_runoff_mm': float,            # 总径流 (mm)
        'peak_runoff_m3s': float,            # 峰值径流 (m³/s)
        'mean_runoff_m3s': float,            # 平均径流 (m³/s)
        'runoff_series': np.ndarray          # 径流时间序列 (m³/s)
    }
}
```

**执行逻辑:**
1. `max_workers=1`: 串行执行
2. `max_workers>1`: 并行执行
   - 使用ProcessPoolExecutor创建进程池
   - 动态任务提交和结果收集
   - 实时进度显示（如果show_progress=True）

---

### benchmark_parallel_performance

性能基准测试工具。

```python
from hydrosis.runoff.parallel_hbv import benchmark_parallel_performance

# 测试不同worker数量的性能
performance = benchmark_parallel_performance(
    zones,
    precipitation_data,
    hbv_params,
    worker_counts=[1, 2, 4, 8]
)

# 输出结果
for num_workers, elapsed_time in performance.items():
    print(f"{num_workers} workers: {elapsed_time:.2f} 秒")
```

**参数:**
- `zones` (List[Dict]): 分区列表
- `precipitation_data` (Dict[int, np.ndarray]): 降雨数据
- `hbv_params` (Dict): HBV参数
- `worker_counts` (List[int]): 要测试的worker数量列表

**返回:** Dict[int, float]
- 键: worker数量
- 值: 执行时间（秒）

**输出示例:**
```
测试 1 workers...
  完成时间: 45.23 秒

测试 2 workers...
  完成时间: 24.15 秒
  加速比: 1.87x
  并行效率: 93.68%

测试 4 workers...
  完成时间: 13.42 秒
  加速比: 3.37x
  并行效率: 84.25%

测试 8 workers...
  完成时间: 8.91 秒
  加速比: 5.08x
  并行效率: 63.45%
```

---

## 使用场景

### 1. 大规模流域模拟

```python
from hydrosis.runoff.parallel_hbv import run_hbv_parallel, ParallelHBVConfig
import json

# 加载参数分区（例如100个分区）
with open('parameter_zones.geojson') as f:
    data = json.load(f)
    zones = [
        {
            'zone_id': f['properties']['zone_id'],
            'area_km2': f['properties']['area_km2']
        }
        for f in data['features']
    ]

# 加载降雨数据
precipitation_data = load_precipitation_for_zones(zones)

# 使用8核并行
config = ParallelHBVConfig(max_workers=8, show_progress=True)
results = run_hbv_parallel(zones, precipitation_data, hbv_params, config)

print(f"✓ 完成 {len(results)} 个分区的模拟")
```

### 2. 参数敏感性分析

```python
import numpy as np
from itertools import product

# 生成参数组合
fc_values = [250, 300, 350]
beta_values = [1.5, 2.0, 2.5]

param_combinations = []
for fc, beta in product(fc_values, beta_values):
    params = hbv_params.copy()
    params['FC'] = fc
    params['BETA'] = beta
    param_combinations.append(params)

# 并行测试每组参数
all_results = []
for params in param_combinations:
    results = run_hbv_parallel(
        zones,
        precipitation_data,
        params,
        ParallelHBVConfig(max_workers=4, show_progress=False)
    )
    all_results.append(results)

# 分析参数敏感性...
```

### 3. 集成到工作流

```python
def run_step_09_runoff_simulation(config, hbv_params):
    """Step 9: 径流模拟（使用并行HBV）"""

    # 加载分区
    zones = load_zones(config['zones_path'])

    # 加载降雨
    precip_data = load_precipitation(config['precip_path'])

    # 并行HBV模拟
    parallel_config = ParallelHBVConfig(
        max_workers=config.get('max_workers', 4),
        show_progress=True
    )

    results = run_hbv_parallel(
        zones,
        precip_data,
        hbv_params,
        parallel_config
    )

    # 保存结果
    save_runoff_results(results, config['output_dir'])

    return results
```

---

## 性能优化建议

### 1. Worker数量选择

- **小规模（<20个分区）**: max_workers=1-2
- **中等规模（20-100个分区）**: max_workers=4-8
- **大规模（>100个分区）**: max_workers=8-16

**最佳实践:**
```python
import multiprocessing

# 使用CPU核心数-1（留一个核心给系统）
optimal_workers = max(1, multiprocessing.cpu_count() - 1)
config = ParallelHBVConfig(max_workers=optimal_workers)
```

### 2. 内存管理

每个worker会复制一份数据，注意内存消耗:
```python
# 估算内存需求
single_zone_memory_mb = 10  # 单个分区约10MB
total_memory_mb = single_zone_memory_mb * max_workers

# 如果内存不足，减少worker数量
if total_memory_mb > available_memory_mb:
    max_workers = available_memory_mb // single_zone_memory_mb
```

### 3. 任务粒度

- HBV模拟属于CPU密集型任务，适合多进程并行
- 单个分区模拟时间 > 1秒时，并行效果最佳
- 如果单个分区太快（<0.1秒），并行开销可能抵消收益

---

## 测试验证

运行并行模块测试:

```bash
python test_parallel_hbv.py
```

**测试内容:**
1. 串行执行功能
2. 并行执行功能
3. 串行/并行结果一致性
4. 结果格式验证

**预期输出:**
```
================================================================================
HBV并行化模块测试
================================================================================

⚙ 准备测试数据...
  ✓ 创建 4 个分区
  ✓ 每个分区 120 个时间步

⚙ 测试串行执行...
  ✓ 串行执行完成: 4 个分区

⚙ 测试并行执行 (2 workers)...
  ✓ 并行执行完成: 4 个分区

⚙ 验证结果一致性...
  ✓ 串行和并行结果完全一致

✅ HBV并行化模块测试通过！
```

---

## 注意事项

1. **进程开销**: 并行执行有进程创建和通信开销，分区数量太少时可能不如串行快

2. **结果顺序**: 并行执行时结果返回顺序不确定，但这不影响正确性

3. **异常处理**: 单个分区失败不会中断整个流程，会记录错误并继续

4. **随机数种子**: 如果HBV模型内部使用随机数，需要在worker函数中设置独立的随机种子

5. **I/O操作**: 并行模块适用于计算密集型任务，不适合I/O密集型任务

---

## 故障排查

### 问题1: 并行比串行慢

**可能原因:**
- 分区数量太少（<4个）
- 单个分区模拟时间太短（<0.1秒）
- CPU核心数不足

**解决方案:**
```python
# 测试性能
performance = benchmark_parallel_performance(
    zones, precipitation_data, hbv_params,
    worker_counts=[1, 2, 4, 8]
)

# 选择最快的worker数量
best_workers = min(performance, key=performance.get)
```

### 问题2: 内存溢出

**解决方案:**
```python
# 减少worker数量
config = ParallelHBVConfig(max_workers=2)

# 或分批处理
batch_size = 50
for i in range(0, len(zones), batch_size):
    batch_zones = zones[i:i+batch_size]
    batch_results = run_hbv_parallel(batch_zones, ...)
```

### 问题3: 结果不一致

**检查清单:**
- [ ] HBV参数是否完全相同
- [ ] 降雨数据是否一致
- [ ] 是否有随机过程（需要固定种子）
- [ ] 浮点精度设置是否一致

---

## 参考

- [HBV模型原理](../theory/hbv_model.md)
- [性能优化指南](../performance_optimization.md)
- [Python multiprocessing文档](https://docs.python.org/3/library/concurrent.futures.html)
