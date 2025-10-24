# Validation Framework API Reference

## 概述

HydroSIS验证框架提供了全面的数据质量检查和验证功能，适用于水文模型的各个环节。所有验证标准均通过配置文件定义，遵循零硬编码原则。

## 核心类

### ValidationResult

验证结果容器类，统一管理验证过程中的错误、警告和指标。

```python
from hydrosis.validation import ValidationResult

result = ValidationResult(step_name="降雨数据验证")

# 添加错误（严重问题，会导致is_valid=False）
result.add_error("存在负降雨值: -5.2 mm/h")

# 添加警告（轻微问题，不影响is_valid）
result.add_warning("部分站点缺失率偏高: 15%")

# 记录指标
result.metrics['spatial_cv'] = 0.35
result.metrics['mean_correlation'] = 0.62

# 检查验证状态
if result.is_valid:
    print("验证通过")
else:
    print(f"验证失败: {result.errors}")

# 打印完整报告
print(result)
```

**属性:**
- `step_name` (str): 验证步骤名称
- `is_valid` (bool): 是否通过验证（无错误时为True）
- `errors` (List[str]): 错误列表
- `warnings` (List[str]): 警告列表
- `metrics` (Dict[str, Any]): 指标字典

**方法:**
- `add_error(message: str)`: 添加错误
- `add_warning(message: str)`: 添加警告
- `__str__()`: 生成格式化报告

---

### ValidationCriteria

验证标准基类，所有具体验证标准类都继承自此。

```python
from dataclasses import dataclass
from hydrosis.validation import ValidationCriteria

@dataclass
class CustomCriteria(ValidationCriteria):
    max_value: float = 100.0
    min_count: int = 10

    @classmethod
    def from_dict(cls, data: Dict) -> "CustomCriteria":
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})
```

---

## 水文过程验证

### validate_water_balance

验证水量平衡是否守恒。

```python
from hydrosis.validation import validate_water_balance

result = validate_water_balance(
    precipitation_mm=500.0,
    runoff_mm=350.0,
    evapotranspiration_mm=120.0,
    storage_change_mm=25.0,
    tolerance=0.05  # 5% tolerance
)

print(result)
```

**参数:**
- `precipitation_mm` (float): 降水量 (mm)
- `runoff_mm` (float): 径流量 (mm)
- `evapotranspiration_mm` (float): 蒸散发量 (mm)
- `storage_change_mm` (float): 蓄水变化量 (mm)
- `tolerance` (float): 容许误差比例
- `step_name` (str): 步骤名称

**返回:** ValidationResult

---

### validate_runoff_coefficient

验证径流系数是否在物理合理范围内。

```python
from hydrosis.validation import validate_runoff_coefficient

result = validate_runoff_coefficient(
    runoff_coefficient=0.65,
    min_rc=0.0,
    max_rc=1.0,
    step_name="径流系数检查"
)
```

**参数:**
- `runoff_coefficient` (float): 径流系数
- `min_rc` (float): 最小合理值（默认0.0）
- `max_rc` (float): 最大合理值（默认1.0）
- `step_name` (str): 步骤名称

**返回:** ValidationResult

---

## 降雨数据验证

### PrecipitationCriteria

降雨验证标准类。

```python
from hydrosis.validation import PrecipitationCriteria

criteria = PrecipitationCriteria(
    min_value=0.0,              # 最小值 (mm/h)
    max_value=100.0,            # 最大值 (mm/h)
    max_daily_value=500.0,      # 最大日降雨 (mm)
    max_spatial_cv=0.5,         # 最大空间变异系数
    min_spatial_correlation=0.3,# 最小站点间相关系数
    max_missing_ratio=0.1,      # 最大缺测率
    max_consecutive_zeros=24    # 最大连续零值小时数
)
```

---

### validate_precipitation_data

验证降雨数据质量。

```python
import pandas as pd
from hydrosis.validation import validate_precipitation_data, PrecipitationCriteria

# 准备降雨数据 (时间×站点/子流域)
precip_df = pd.DataFrame({
    'station_1': [0, 2.5, 5.0, 3.2, 0],
    'station_2': [0, 3.1, 4.8, 2.9, 0],
    'station_3': [0, 2.8, 5.2, 3.5, 0]
}, index=pd.date_range('2024-01-01', periods=5, freq='h'))

# 使用默认标准验证
result = validate_precipitation_data(precip_df)

# 使用自定义标准验证
custom_criteria = PrecipitationCriteria(max_spatial_cv=0.3)
result = validate_precipitation_data(
    precip_df,
    criteria=custom_criteria,
    step_name="降雨质量检查"
)

print(result)
```

**参数:**
- `precipitation_df` (pd.DataFrame): 降雨数据（时间×站点）
- `criteria` (PrecipitationCriteria, optional): 验证标准
- `step_name` (str): 步骤名称

**返回:** ValidationResult

**检查项目:**
1. 数值范围（负值、异常高值）
2. 空间一致性（变异系数、站点间相关性）
3. 时间连续性（缺测率、连续零值）
4. 累积降雨合理性

---

### identify_precipitation_outliers

识别降雨异常值。

```python
from hydrosis.validation import identify_precipitation_outliers

# 使用Z-score方法
outliers = identify_precipitation_outliers(
    precip_df,
    method="zscore",
    threshold=3.0
)

# 使用IQR方法
outliers = identify_precipitation_outliers(
    precip_df,
    method="iqr",
    threshold=1.5
)

# 输出异常值
for station, anomalies in outliers.items():
    print(f"站点 {station}:")
    for timestamp, value in anomalies:
        print(f"  {timestamp}: {value} mm/h")
```

**参数:**
- `precipitation_df` (pd.DataFrame): 降雨数据
- `method` (str): 检测方法 ("zscore" 或 "iqr")
- `threshold` (float): 阈值（zscore用3.0，iqr用1.5）

**返回:** Dict[str, List[Tuple[timestamp, value]]]

---

### suggest_precipitation_fixes

基于验证结果建议修复措施。

```python
from hydrosis.validation import (
    validate_precipitation_data,
    suggest_precipitation_fixes
)

result = validate_precipitation_data(precip_df)
suggestions = suggest_precipitation_fixes(result, precip_df)

for issue, suggestion in suggestions.items():
    print(f"\n{issue}:")
    print(suggestion)
```

**参数:**
- `validation_result` (ValidationResult): 验证结果
- `precipitation_df` (pd.DataFrame): 降雨数据

**返回:** Dict[str, str] - 问题类型到修复建议的映射

---

## 空间数据验证

### SpatialCriteria

空间验证标准类。

```python
from hydrosis.validation import SpatialCriteria

criteria = SpatialCriteria(
    min_area_km2=1.0,               # 最小面积
    max_area_km2=10000.0,           # 最大面积
    allow_invalid_geometry=False,   # 是否允许无效几何
    min_valid_geometry_ratio=0.95,  # 最小有效几何比例
    allow_gaps=False,               # 是否允许空隙
    allow_overlaps=False,           # 是否允许重叠
    overlap_tolerance_m=1.0         # 重叠容差(米)
)
```

---

### validate_basin_geometry

验证流域几何数据。

```python
import json
from hydrosis.validation import validate_basin_geometry, SpatialCriteria

# 加载GeoJSON
with open('basin.geojson') as f:
    data = json.load(f)
    features = data['features']

# 验证
result = validate_basin_geometry(
    features,
    criteria=SpatialCriteria(min_area_km2=10.0),
    step_name="流域边界验证"
)

print(result)
```

**参数:**
- `basin_features` (List[Dict]): GeoJSON features列表
- `criteria` (SpatialCriteria, optional): 验证标准
- `step_name` (str): 步骤名称

**返回:** ValidationResult

**检查项目:**
1. 几何有效性
2. 面积范围
3. 有效几何比例

---

### validate_network_topology

验证河网拓扑一致性。

```python
from hydrosis.validation import validate_network_topology

result = validate_network_topology(
    network_features,
    step_name="河网拓扑检查"
)

print(result)
print(f"出口点数量: {result.metrics['num_outlets']}")
```

**参数:**
- `network_features` (List[Dict]): 河网要素列表
- `criteria` (SpatialCriteria, optional): 验证标准
- `step_name` (str): 步骤名称

**返回:** ValidationResult

**检查项目:**
1. 下游引用有效性
2. 环路检测
3. 出口点统计

---

## 时间序列验证

### TimeSeriesCriteria

时间序列验证标准类。

```python
from hydrosis.validation import TimeSeriesCriteria

criteria = TimeSeriesCriteria(
    max_missing_ratio=0.1,          # 最大缺测率
    max_consecutive_missing=24,     # 最大连续缺失
    min_value=0.0,                  # 最小值
    max_value=1e6,                  # 最大值
    max_hourly_change_ratio=0.5,    # 最大小时变化率
    max_daily_change_ratio=2.0,     # 最大日变化率
    allow_negative_trend=True,      # 是否允许负趋势
    max_trend_slope=1e6             # 最大趋势斜率
)
```

---

### validate_time_series

验证单个时间序列。

```python
import pandas as pd
from hydrosis.validation import validate_time_series, TimeSeriesCriteria

# 准备时间序列
series = pd.Series(
    [10, 12, 15, 14, 13],
    index=pd.date_range('2024-01-01', periods=5, freq='h')
)

# 验证
result = validate_time_series(
    series,
    criteria=TimeSeriesCriteria(max_missing_ratio=0.05),
    series_name="流量站点A",
    step_name="流量数据验证"
)

print(result)
```

**参数:**
- `series` (pd.Series): 时间序列
- `criteria` (TimeSeriesCriteria, optional): 验证标准
- `series_name` (str): 序列名称
- `step_name` (str): 步骤名称

**返回:** ValidationResult

**检查项目:**
1. 缺失值统计和连续缺失
2. 数值范围
3. 变化率
4. 趋势检测

---

### validate_multiple_series

批量验证多个时间序列。

```python
from hydrosis.validation import validate_multiple_series

# DataFrame (时间×变量)
df = pd.DataFrame({
    'flow_A': [10, 12, 15, 14, 13],
    'flow_B': [8, 9, 11, 10, 9],
    'flow_C': [12, 14, 17, 16, 15]
}, index=pd.date_range('2024-01-01', periods=5, freq='h'))

# 批量验证
results = validate_multiple_series(
    df,
    step_name="多站点流量验证"
)

# 检查每个序列的验证结果
for series_name, result in results.items():
    print(f"\n{series_name}:")
    print(result)
```

**参数:**
- `df` (pd.DataFrame): 时间序列数据框
- `criteria` (TimeSeriesCriteria, optional): 验证标准
- `step_name` (str): 步骤名称

**返回:** Dict[str, ValidationResult]

---

## 使用示例

### 完整工作流验证

```python
import pandas as pd
from hydrosis.validation import (
    ValidationResult,
    validate_precipitation_data,
    validate_water_balance,
    validate_runoff_coefficient,
    PrecipitationCriteria
)

# 1. 降雨数据验证
precip_result = validate_precipitation_data(
    precip_df,
    criteria=PrecipitationCriteria(max_spatial_cv=0.4)
)

if not precip_result.is_valid:
    print(f"降雨数据验证失败: {precip_result.errors}")
    exit(1)

# 2. 径流系数验证
rc_result = validate_runoff_coefficient(
    runoff_coefficient=0.65,
    min_rc=0.3,
    max_rc=0.9
)

if not rc_result.is_valid:
    print(f"径流系数异常: {rc_result.errors}")

# 3. 水量平衡验证
wb_result = validate_water_balance(
    precipitation_mm=500,
    runoff_mm=325,
    evapotranspiration_mm=150,
    storage_change_mm=20,
    tolerance=0.05
)

# 4. 汇总所有验证结果
all_results = {
    'precipitation': precip_result,
    'runoff_coefficient': rc_result,
    'water_balance': wb_result
}

for name, result in all_results.items():
    print(f"\n{'='*60}")
    print(f"{name.upper()} 验证结果:")
    print(result)
```

---

## 配置文件集成

验证标准可以从YAML配置文件加载:

```yaml
# config/validation_criteria.yaml
precipitation:
  min_value: 0.0
  max_value: 100.0
  max_spatial_cv: 0.5
  min_spatial_correlation: 0.3

spatial:
  min_area_km2: 1.0
  max_area_km2: 10000.0
  allow_invalid_geometry: false

timeseries:
  max_missing_ratio: 0.1
  max_consecutive_missing: 24
  max_hourly_change_ratio: 0.5
```

```python
import yaml
from hydrosis.validation import (
    PrecipitationCriteria,
    SpatialCriteria,
    TimeSeriesCriteria
)

# 加载配置
with open('config/validation_criteria.yaml') as f:
    config = yaml.safe_load(f)

# 创建验证标准
precip_criteria = PrecipitationCriteria.from_dict(config['precipitation'])
spatial_criteria = SpatialCriteria.from_dict(config['spatial'])
timeseries_criteria = TimeSeriesCriteria.from_dict(config['timeseries'])
```

---

## 最佳实践

1. **始终记录验证结果**: 使用ValidationResult.metrics记录关键指标，便于后续分析
2. **自定义验证标准**: 根据具体流域特征调整验证阈值
3. **分级验证**: 区分error（必须修复）和warning（可选修复）
4. **批量验证**: 对多个站点/子流域使用批量验证函数提高效率
5. **配置文件管理**: 将验证标准写入YAML配置文件，避免硬编码

---

## 参考

- [HydroSIS开发指南](../development_guide.md)
- [配置文件说明](../configuration.md)
- [水文过程验证理论](../theory/hydrologic_validation.md)
