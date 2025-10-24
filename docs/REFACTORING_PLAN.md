# HydroSIS 代码重构计划

## 当前问题分析

### 1. 违反开发指南的问题

根据AI开发指南审查，发现以下严重问题：

#### ❌ 硬编码泛滥
- **enhance_step_05_rain_gauges.py**: 所有路径、密度标准硬编码
- **enhance_step_09_runoff.py**: 所有路径、验证标准硬编码
- **optimize_rain_gauges.py**: 优化参数、路径全部硬编码
- **run_upper_truckee_complete_11steps.py**: 大量硬编码参数和路径

#### ❌ 未使用基础库功能
- 自己实现了降雨聚合逻辑（应该检查hydrosis是否已有）
- 自己实现了验证逻辑（应该在基础库中）
- 未使用hydrosis.reporting.charts进行可视化

#### ❌ 外部脚本分离
- 增强功能在外部脚本中，未集成到主工作流
- 验证逻辑分散，难以维护

### 2. Step 9 数据问题根源

**问题**: 径流系数>1（2.12），违反物理规律

**可能原因**:
1. HBV模型初始状态设置不当（initial_soil=0可能导致前期产流过多）
2. 模型参数未率定（使用默认值）
3. Muskingum路由可能放大了流量
4. 时间步长或单位转换问题

**正确的解决方法**（遵循开发指南）:
```python
# 应该使用基础库的rate定功能
from hydrosis.calibration import calibrate_parameters
from hydrosis.evaluation.metrics import nash_sutcliffe_efficiency

# 使用EnhancedRunoffGenerator生成观测数据
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator

# 使用可视化工具
from hydrosis.reporting.charts import plot_hydrograph, plot_scatter
```

## 重构计划

### 阶段1: 创建通用验证框架（基础库）

**位置**: `hydrosis/validation/`

```
hydrosis/validation/
  ├── __init__.py
  ├── base.py              # 基础验证类
  │   ├── ValidationCriteria (dataclass)
  │   ├── ValidationResult (dataclass)
  │   └── BaseValidator (abstract class)
  ├── spatial.py           # 空间数据验证
  │   ├── validate_dem_quality
  │   ├── validate_zone_geometry
  │   └── validate_connectivity
  ├── timeseries.py        # 时间序列验证
  │   ├── validate_continuity
  │   ├── validate_value_range
  │   └── validate_temporal_consistency
  ├── hydrologic.py        # 水文验证
  │   ├── validate_water_balance
  │   ├── validate_runoff_coefficient
  │   └── validate_mass_conservation
  └── reporting.py         # 验证报告生成
      ├── generate_validation_report
      └── ValidationReporter
```

**设计原则**:
- ✅ 所有验证标准通过配置文件设置
- ✅ 通用的ValidationResult数据结构
- ✅ 一致的API接口
- ✅ 支持自定义验证规则

### 阶段2: 创建验证配置文件

**文件**: `config/validation_criteria.yaml`

```yaml
# 空间验证标准
spatial:
  dem:
    min_valid_pixels_ratio: 0.95
    max_nodata_ratio: 0.05
    elevation_range_check: true
  zones:
    expected_count: 6
    count_tolerance: 0
    min_area_km2: 10.0
    max_area_km2: 1000.0
  connectivity:
    require_connected_network: true
    allow_disconnected_zones: false

# 时间序列验证标准
timeseries:
  precipitation:
    min_value: 0.0
    max_value: 100.0  # mm/hr
    require_continuous: true
  discharge:
    min_value: 0.0
    max_value: 10000.0  # m³/s

# 水文验证标准
hydrologic:
  runoff_coefficient:
    min_value: 0.0
    max_value: 1.0
    warning_low: 0.05
    warning_high: 0.9
  water_balance:
    max_error_ratio: 0.01  # 1%
  mass_conservation:
    tolerance: 0.001  # 0.1%

# 雨量站密度标准
rain_gauge:
  density_grades:
    excellent: 0.02  # stations/km²
    good: 0.01
    fair: 0.005
    poor: 0.001
  min_stations_per_zone: 1
  target_density: 0.01

# 优化参数
optimization:
  rain_gauge:
    max_iterations: 10
    min_spacing_m: 2000
    target_density: 0.01
```

### 阶段3: 重构主工作流

将验证集成到每个步骤：

```python
# run_upper_truckee_complete_11steps.py

from hydrosis.validation import (
    validate_dem_quality,
    validate_zone_geometry,
    validate_water_balance,
    validate_runoff_coefficient,
    ValidationCriteria,
)
from hydrosis.config import load_validation_criteria

def step01_dem_processing(...):
    """Step 1: DEM处理"""
    # ... 现有代码 ...

    # 添加验证
    criteria = load_validation_criteria(config_path)
    validation_result = validate_dem_quality(
        dem_path,
        flow_dir_path,
        criteria.spatial.dem
    )

    if not validation_result.is_valid:
        print(validation_result.summary())
        # 保存报告
        report_path = output_dir / "1.4_validation_report.txt"
        report_path.write_text(validation_result.summary())

    return {"validation": validation_result, ...}

def step09_hydrologic_simulation(...):
    """Step 9: 水文模拟"""
    # ... 模拟代码 ...

    # 添加水文验证
    validation_result = validate_water_balance(
        precipitation=zone_precip,
        runoff=zone_runoff,
        area_km2=zone_area,
        criteria=criteria.hydrologic
    )

    runoff_coeff_validation = validate_runoff_coefficient(
        runoff_coefficients=coefficients,
        criteria=criteria.hydrologic.runoff_coefficient
    )

    # 保存验证报告
    ...
```

### 阶段4: 消除硬编码

#### 4.1 工作流配置文件
**文件**: `examples/upper_truckee_complete_workflow.yaml`

```yaml
workflow:
  name: "Upper Truckee River Complete 11-Step Analysis"
  description: "Complete hydrologic modeling workflow"

  paths:
    data_dir: "examples/data/upper_truckee"
    results_dir: "results/upper_truckee_complete_11steps"
    dem: "examples/data/upper_truckee/dem.tif"
    flow_dir: null  # 自动生成
    flow_acc: null  # 自动生成

  parameters:
    step02_pour_points:
      num_main_stream_zones: 3
      num_tributary_zones: 3
      threshold_ratio: 0.05

    step03_partition:
      subbasins_per_zone: 16
      target_subzone_area_km2: 12.0

    step05_rain_gauges:
      num_stations: 10
      heterogeneity: 0.4
      seed: 42

    step06_precipitation:
      simulation_hours: 120
      base_intensity: 5.0

  validation_criteria_file: "config/validation_criteria.yaml"
```

#### 4.2 重构示例脚本

```python
# examples/run_upper_truckee_workflow.py

from pathlib import Path
from hydrosis.config import load_workflow_config
from hydrosis.workflow import run_complete_workflow
from hydrosis.validation import load_validation_criteria

def main():
    # 从配置加载
    config_path = Path("examples/upper_truckee_complete_workflow.yaml")
    config = load_workflow_config(config_path)

    # 加载验证标准
    criteria = load_validation_criteria(
        config.validation_criteria_file
    )

    # 运行工作流
    results = run_complete_workflow(
        config=config,
        validation_criteria=criteria,
        enable_validation=True,
        stop_on_validation_error=False
    )

    # 生成综合报告
    generate_comprehensive_report(
        results=results,
        output_path=config.paths.results_dir / "comprehensive_report.md"
    )

if __name__ == "__main__":
    main()
```

### 阶段5: 修复Step 9数据问题

**正确方法**（遵循开发指南）:

```python
# 在基础库中添加validation模块
# hydrosis/validation/hydrologic.py

from hydrosis.evaluation.metrics import nash_sutcliffe_efficiency
from hydrosis.calibration import calibrate_parameters
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator

def diagnose_runoff_issue(
    precipitation: np.ndarray,
    discharge: np.ndarray,
    area_km2: float,
    config: dict
) -> dict:
    """诊断径流系数异常问题"""

    # 1. 计算径流系数
    runoff_coeff = calculate_runoff_coefficient(...)

    # 2. 如果异常，生成观测数据用于率定
    if runoff_coeff > 1.0:
        # 使用EnhancedRunoffGenerator生成合理的观测数据
        generator = EnhancedRunoffGenerator(...)
        synthetic_runoff, stats = generator.generate(
            precipitation, area_km2
        )

        # 3. 率定HBV参数
        from hydrosis.calibration import calibrate_parameters

        result = calibrate_parameters(
            objective_function=lambda p: nse_objective(p, precip, synthetic_runoff),
            param_bounds=param_bounds,
            algorithm="differential_evolution",
            maximize=True
        )

        return {
            "issue": "runoff_coefficient_too_high",
            "original_coeff": runoff_coeff,
            "calibrated_params": result.best_params,
            "calibrated_nse": result.best_score
        }
```

## 实施优先级

### 高优先级（必须立即完成）
1. ✅ 创建`hydrosis/validation/`模块
2. ✅ 创建`config/validation_criteria.yaml`
3. ✅ 在`hydrosis/config.py`中添加加载验证配置的函数
4. ✅ 修复Step 9径流系数问题（使用率定）

### 中优先级（应该尽快完成）
1. ✅ 重构所有外部脚本，移除硬编码
2. ✅ 创建`examples/upper_truckee_complete_workflow.yaml`
3. ✅ 为每个步骤添加验证逻辑
4. ✅ 生成综合验证报告

### 低优先级（可以逐步完成）
1. ✅ 改进可视化（使用hydrosis.reporting.charts）
2. ✅ 添加更多验证规则
3. ✅ 性能优化

## 预期效果

### 代码质量提升
- ✅ 无硬编码
- ✅ 高度可配置
- ✅ 可重用的验证框架
- ✅ 符合开发指南

### 工作流改进
- ✅ 每步都有闭环验证
- ✅ 自动发现和报告问题
- ✅ 生成详细的验证报告
- ✅ 支持自定义验证标准

### 用户体验
- ✅ 配置文件驱动，易于定制
- ✅ 清晰的错误和警告信息
- ✅ 自动化诊断和修复建议
- ✅ 完整的文档和示例

## 下一步行动

1. **立即**: 创建基础验证框架
2. **今日**: 创建验证配置文件
3. **本周**: 重构主工作流集成验证
4. **下周**: 修复Step 9问题并完成文档

---

**注意**: 所有改进必须严格遵循AI开发指南，优先使用基础库功能，避免硬编码。
