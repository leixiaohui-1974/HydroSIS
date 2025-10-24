# 雨量站分布优化 API Reference

## 概述

HydroSIS提供了配置驱动的雨量站分布优化工具，基于参数分区和目标密度自动生成优化的雨量站网络。完全消除硬编码，所有参数从配置文件加载。

## 核心功能

- ✅ **配置驱动**: 所有参数从workflow_config.yaml加载
- ✅ **自动分区**: 基于参数分区自动分配站点
- ✅ **空间约束**: 确保站点间最小距离
- ✅ **质量验证**: 集成validation framework验证结果
- ✅ **GeoJSON输出**: 标准地理数据格式

---

## 配置文件

### workflow_config.yaml

```yaml
# Rain Gauge Configuration
rain_gauge:
  # 目标密度 (站点数 per 100 km²)
  target_density: 0.01  # 1站点/100km² (一般质量)

  # 质量等级参考
  excellent_density: 0.02  # 优秀: 2站点/100km²
  good_density: 0.01       # 良好: 1站点/100km²
  fair_density: 0.005      # 一般: 0.5站点/100km²
  poor_density: 0.002      # 较差: 0.2站点/100km²

  # 空间约束
  min_distance_m: 1000     # 站点间最小距离 (米)
  buffer_distance_m: 500   # 缓冲区距离

  # 优化参数
  random_seed: 42          # 随机种子（保证可复现）
  max_iterations: 1000     # 最大迭代次数
  optimization_method: "random_uniform"  # 优化方法
```

**参数说明:**

| 参数 | 类型 | 说明 |
|-----|------|------|
| `target_density` | float | 目标站点密度 (站点数/100km²) |
| `excellent_density` | float | 优秀密度参考值 |
| `good_density` | float | 良好密度参考值 |
| `fair_density` | float | 一般密度参考值 |
| `poor_density` | float | 较差密度参考值 |
| `min_distance_m` | float | 站点间最小距离 (米) |
| `buffer_distance_m` | float | 边界缓冲区 (米) |
| `random_seed` | int | 随机种子 |
| `max_iterations` | int | 最大尝试次数 |
| `optimization_method` | str | 优化算法 |

---

## API Reference

### calculate_target_gauge_count

计算目标雨量站数量。

```python
from optimize_rain_gauges_refactored import calculate_target_gauge_count

# 计算100 km²区域在0.01密度下需要的站点数
count = calculate_target_gauge_count(
    area_km2=100.0,
    target_density=0.01
)
print(count)  # 1

# 计算500 km²区域在0.02密度下需要的站点数
count = calculate_target_gauge_count(
    area_km2=500.0,
    target_density=0.02
)
print(count)  # 10
```

**参数:**
- `area_km2` (float): 区域面积 (平方千米)
- `target_density` (float): 目标密度 (站点数/100km²)

**返回:** int - 目标站点数（最少为1）

**公式:**
```
count = max(1, int(area_km2 * target_density / 100))
```

---

### generate_optimized_gauges

生成优化的雨量站分布。

```python
from optimize_rain_gauges_refactored import (
    generate_optimized_gauges,
    load_parameter_zones
)

# 加载参数分区
zones = load_parameter_zones('parameter_zones.geojson')

# 生成优化的雨量站
gauges = generate_optimized_gauges(
    zones=zones,
    target_density=0.01,      # 1站点/100km²
    min_distance_m=1000,      # 站点间最小距离1km
    max_iterations=1000,      # 最大尝试次数
    random_seed=42            # 随机种子
)

print(f"生成 {len(gauges)} 个雨量站")
```

**参数:**
- `zones` (List[Dict]): 参数分区列表
  - 每个分区包含: `zone_id`, `area_km2`, `geometry`
- `target_density` (float): 目标密度
- `min_distance_m` (float): 最小站间距 (米)
- `max_iterations` (int): 最大迭代次数
- `random_seed` (int): 随机种子

**返回:** List[Dict] - 雨量站列表

雨量站格式:
```python
{
    'id': '1_1',         # 站点ID (zone_id_序号)
    'zone_id': 1,        # 所属分区ID
    'lon': -120.123,     # 经度
    'lat': 38.456        # 纬度
}
```

**算法流程:**
1. 遍历每个参数分区
2. 根据面积和目标密度计算该分区需要的站点数
3. 在分区内随机生成候选点
4. 检查候选点是否在分区多边形内
5. 检查与已有站点的距离是否满足最小距离要求
6. 达到目标数量或超过最大迭代次数后停止

---

### validate_gauge_distribution

验证雨量站分布质量。

```python
from optimize_rain_gauges_refactored import validate_gauge_distribution
from hydrosis.config import load_workflow_config

# 加载配置
config = load_workflow_config('config/workflow_config.yaml')

# 验证雨量站分布
result = validate_gauge_distribution(
    gauges=gauges,
    zones=zones,
    config=config
)

print(result)
print(f"总站点数: {result.metrics['total_gauges']}")
print(f"总体密度: {result.metrics['overall_density']:.4f}")
```

**参数:**
- `gauges` (List[Dict]): 雨量站列表
- `zones` (List[Dict]): 参数分区列表
- `config` (Dict): 配置字典

**返回:** ValidationResult

**验证指标:**
- 各分区实际站点数 vs 目标数
- 各分区实际密度 vs 目标密度
- 总体站点密度
- 分区覆盖率

**示例输出:**
```
================================================================================
验证结果: 雨量站分布验证
================================================================================

状态: ⚠️ 警告

警告:
  - 分区1: 站点数不足 (0/1), 密度0.0000 < 目标0.01
  - 分区3: 站点数不足 (1/2), 密度0.0067 < 目标0.01
  - 总体密度偏低: 0.0075 < 目标0.01

指标:
  zone_1_count: 0
  zone_1_density: 0.0
  zone_2_count: 2
  zone_2_density: 0.0133
  zone_3_count: 1
  zone_3_density: 0.0067
  total_gauges: 3
  overall_density: 0.0075
  target_density: 0.01
```

---

### save_gauges_geojson

保存雨量站为GeoJSON格式。

```python
from optimize_rain_gauges_refactored import save_gauges_geojson
from pathlib import Path

output_path = Path('results/rain_gauges/optimized_gauges.geojson')
save_gauges_geojson(gauges, output_path)
print(f"已保存: {output_path}")
```

**参数:**
- `gauges` (List[Dict]): 雨量站列表
- `output_path` (Path): 输出文件路径

**GeoJSON格式:**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-120.123, 38.456]
      },
      "properties": {
        "id": "1_1",
        "zone_id": 1
      }
    }
  ]
}
```

---

### load_parameter_zones

加载参数分区GeoJSON。

```python
from optimize_rain_gauges_refactored import load_parameter_zones
from pathlib import Path

zones_path = Path('results/parameters/parameter_zones.geojson')
zones = load_parameter_zones(zones_path)

print(f"加载 {len(zones)} 个分区")
for zone in zones:
    print(f"  分区 {zone['zone_id']}: {zone['area_km2']:.2f} km²")
```

**参数:**
- `geojson_path` (Path): GeoJSON文件路径

**返回:** List[Dict]

分区格式:
```python
{
    'zone_id': 1,              # 分区ID
    'area_km2': 150.5,         # 面积
    'geometry': Polygon(...)   # Shapely几何对象
}
```

---

## 命令行工具

### 基本用法

```bash
python optimize_rain_gauges_refactored.py \
    --config config/workflow_config.yaml \
    --validation-config config/validation_criteria.yaml
```

**参数:**
- `--config`: 工作流配置文件路径（默认: config/workflow_config.yaml）
- `--validation-config`: 验证标准配置（默认: config/validation_criteria.yaml）

### 执行流程

1. **加载配置**
   - 读取workflow_config.yaml
   - 提取rain_gauge配置参数

2. **加载分区数据**
   - 从results/{project}/parameters/parameter_zones.geojson加载

3. **生成优化分布**
   - 基于目标密度和空间约束
   - 为每个分区生成站点

4. **验证质量**
   - 检查各分区密度
   - 验证总体覆盖率

5. **保存结果**
   - 雨量站GeoJSON: optimized_gauges.geojson
   - 验证结果JSON: validation_result.json

---

## 使用示例

### 示例1: 基本优化

```python
from pathlib import Path
from optimize_rain_gauges_refactored import (
    load_parameter_zones,
    generate_optimized_gauges,
    validate_gauge_distribution,
    save_gauges_geojson
)
from hydrosis.config import load_workflow_config

# 1. 加载配置
config = load_workflow_config('config/workflow_config.yaml')
rain_gauge_config = config['rain_gauge']

# 2. 加载分区
base_dir = Path(config['directories']['base_results'])
zones_path = base_dir / 'parameters' / 'parameter_zones.geojson'
zones = load_parameter_zones(zones_path)

# 3. 生成优化分布
gauges = generate_optimized_gauges(
    zones,
    target_density=rain_gauge_config['target_density'],
    min_distance_m=rain_gauge_config['min_distance_m'],
    random_seed=rain_gauge_config['random_seed']
)

# 4. 验证
result = validate_gauge_distribution(gauges, zones, config)
print(result)

# 5. 保存
output_dir = base_dir / 'rain_gauge_optimization'
output_dir.mkdir(exist_ok=True)
save_gauges_geojson(gauges, output_dir / 'optimized_gauges.geojson')
```

### 示例2: 多密度对比

```python
densities = [0.005, 0.01, 0.015, 0.02]  # 不同密度等级

results = {}
for density in densities:
    gauges = generate_optimized_gauges(
        zones,
        target_density=density,
        min_distance_m=1000,
        random_seed=42
    )

    validation = validate_gauge_distribution(gauges, zones, config)

    results[density] = {
        'gauge_count': len(gauges),
        'overall_density': validation.metrics['overall_density'],
        'is_valid': validation.is_valid
    }

# 选择最优密度
for density, result in results.items():
    print(f"密度 {density:.3f}: {result['gauge_count']} 个站点, "
          f"实际密度 {result['overall_density']:.4f}")
```

### 示例3: 自定义空间约束

```python
# 山区流域: 站点间距需更大
mountain_gauges = generate_optimized_gauges(
    zones,
    target_density=0.01,
    min_distance_m=2000,  # 2km最小距离
    random_seed=42
)

# 平原流域: 可以更密集
plain_gauges = generate_optimized_gauges(
    zones,
    target_density=0.02,
    min_distance_m=500,   # 500m最小距离
    random_seed=42
)
```

---

## 质量等级建议

根据WMO（世界气象组织）和中国气象局标准:

| 等级 | 密度 (站点/100km²) | 适用场景 |
|-----|-------------------|---------|
| **优秀** | ≥0.02 | 研究级流域、高精度模拟 |
| **良好** | 0.01-0.02 | 业务化模拟、一般研究 |
| **一般** | 0.005-0.01 | 粗略模拟、数据稀缺地区 |
| **较差** | <0.005 | 仅用于初步评估 |

**地形修正因子:**
- 平原: 密度 × 0.8
- 丘陵: 密度 × 1.0
- 山地: 密度 × 1.5
- 高山: 密度 × 2.0

---

## 输出文件

### 1. optimized_gauges.geojson

标准GeoJSON格式的雨量站分布文件。

**用途:**
- GIS软件可视化（QGIS, ArcGIS）
- Web地图展示（Leaflet, Mapbox）
- 空间分析

### 2. validation_result.json

验证结果JSON文件。

```json
{
  "is_valid": false,
  "errors": [],
  "warnings": [
    "分区1: 站点数不足 (0/1), 密度0.0000 < 目标0.01"
  ],
  "metrics": {
    "zone_1_count": 0,
    "zone_1_density": 0.0,
    "total_gauges": 3,
    "overall_density": 0.0075,
    "target_density": 0.01
  }
}
```

---

## 故障排查

### 问题1: 部分分区站点数不足

**症状:** 警告"站点数不足"

**原因:**
1. 分区面积过小
2. 分区形状不规则（狭长型）
3. min_distance_m设置过大

**解决方案:**
```python
# 方案1: 降低最小距离
gauges = generate_optimized_gauges(
    zones,
    target_density=0.01,
    min_distance_m=500,  # 从1000降低到500
    max_iterations=2000   # 增加尝试次数
)

# 方案2: 针对小分区调整密度
for zone in zones:
    if zone['area_km2'] < 50:
        # 小分区使用更高密度
        density = 0.02
    else:
        density = 0.01
```

### 问题2: 站点过于集中

**解决方案:**
```python
# 增加最小距离
gauges = generate_optimized_gauges(
    zones,
    target_density=0.01,
    min_distance_m=2000,  # 增加到2km
    random_seed=42
)
```

### 问题3: 结果不可复现

**解决方案:**
```python
# 确保使用固定随机种子
gauges = generate_optimized_gauges(
    zones,
    target_density=0.01,
    min_distance_m=1000,
    random_seed=42  # 固定种子
)
```

---

## 高级用法

### 集成到工作流

```python
from hydrosis.workflow import WorkflowStep

class RainGaugeOptimizationStep(WorkflowStep):
    """雨量站优化步骤"""

    def execute(self):
        # 加载配置
        config = self.load_config()

        # 生成优化分布
        gauges = generate_optimized_gauges(
            self.zones,
            **config['rain_gauge']
        )

        # 验证
        result = validate_gauge_distribution(
            gauges, self.zones, config
        )

        if not result.is_valid:
            self.logger.warning(f"验证未通过: {result.errors}")

        # 保存
        self.save_results(gauges, result)

        return gauges
```

---

## 参考

- [WMO雨量站网设计指南](https://library.wmo.int/)
- [空间优化算法](../theory/spatial_optimization.md)
- [配置文件说明](../configuration.md)
