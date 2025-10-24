# HydroSIS 文档中心

欢迎使用HydroSIS (Hydrological Simulation and Information System) 文档中心！

## 📚 文档导航

### 🚀 快速入门

| 文档 | 说明 | 适合人群 |
|-----|------|---------|
| [用户手册](用户手册.md) | 快速开始、工作流步骤、常见问题 | 所有用户 |
| [开发指南](开发指南.md) | 核心设计原则、开发规范、最佳实践 | 开发者 |

### 📖 API参考

| 模块 | 文档 | 主要功能 |
|-----|------|---------|
| **验证框架** | [validation_framework.md](api/validation_framework.md) | 数据质量检查、水文过程验证 |
| **HBV并行化** | [parallel_hbv.md](api/parallel_hbv.md) | 多核并行径流模拟 |
| **雨量站优化** | [rain_gauge_optimization.md](api/rain_gauge_optimization.md) | 配置驱动的站点分布优化 |

---

## 🎯 按任务查找文档

### 我想运行HydroSIS工作流
→ 阅读 [用户手册 - 快速开始](用户手册.md#快速开始)

### 我遇到了径流系数>1的问题
→ 阅读 [用户手册 - Q1: 径流系数大于1怎么办](用户手册.md#q1-径流系数大于1怎么办)

### 我想优化雨量站分布
→ 阅读 [雨量站优化API](api/rain_gauge_optimization.md)

### 我想验证数据质量
→ 阅读 [验证框架API](api/validation_framework.md)

### 我想加速HBV模拟
→ 阅读 [HBV并行化API](api/parallel_hbv.md)

### 我想开发新功能
→ 阅读 [开发指南](开发指南.md)

---

## 📋 文档概览

### [用户手册](用户手册.md)

**内容包括**:
- ✅ 快速开始指南
- ✅ 11步工作流详解
- ✅ 配置文件说明
- ✅ 数据质量检查
- ✅ 结果分析方法
- ✅ 常见问题解答
- ✅ 完整命令参考

**适用场景**: 第一次使用HydroSIS、运行工作流、解决常见问题

---

### [开发指南](开发指南.md)

**内容包括**:
- ✅ 核心设计原则（零硬编码、配置驱动、验证为先）
- ✅ 配置驱动架构
- ✅ 验证框架使用
- ✅ 模块开发规范
- ✅ 性能优化建议
- ✅ 测试规范
- ✅ 代码审查清单

**适用场景**: 开发新功能、修改现有代码、代码审查

---

### [验证框架 API](api/validation_framework.md)

**主要类和函数**:
- `ValidationResult` - 验证结果容器
- `ValidationCriteria` - 验证标准基类
- `validate_water_balance()` - 水量平衡验证
- `validate_runoff_coefficient()` - 径流系数验证
- `validate_precipitation_data()` - 降雨数据验证
- `validate_basin_geometry()` - 流域几何验证
- `validate_network_topology()` - 河网拓扑验证
- `validate_time_series()` - 时间序列验证

**适用场景**:
- 验证数据质量
- 检查水文合理性
- 集成验证到工作流

**代码示例**:
```python
from hydrosis.validation import validate_precipitation_data

result = validate_precipitation_data(precip_df)
if result.is_valid:
    print("✓ 降雨数据质量合格")
else:
    print(f"✗ 发现问题: {result.errors}")
```

---

### [HBV并行化 API](api/parallel_hbv.md)

**主要类和函数**:
- `ParallelHBVConfig` - 并行配置
- `run_hbv_parallel()` - 并行HBV模拟
- `benchmark_parallel_performance()` - 性能基准测试

**适用场景**:
- 大规模流域模拟（>20个分区）
- 需要加速计算
- 参数敏感性分析

**性能提升**:
- 4核并行: 3-4倍加速
- 8核并行: 5-6倍加速

**代码示例**:
```python
from hydrosis.runoff.parallel_hbv import run_hbv_parallel, ParallelHBVConfig

config = ParallelHBVConfig(max_workers=4)
results = run_hbv_parallel(zones, precip_data, hbv_params, config)
```

---

### [雨量站优化 API](api/rain_gauge_optimization.md)

**主要函数**:
- `generate_optimized_gauges()` - 生成优化分布
- `validate_gauge_distribution()` - 验证站点密度
- `calculate_target_gauge_count()` - 计算目标站点数
- `save_gauges_geojson()` - 保存GeoJSON

**适用场景**:
- 优化雨量站网络设计
- 评估站点密度
- 生成站点分布方案

**质量等级**:
- 优秀: ≥0.02 站点/100km²
- 良好: 0.01-0.02
- 一般: 0.005-0.01
- 较差: <0.005

**代码示例**:
```python
from optimize_rain_gauges_refactored import generate_optimized_gauges

gauges = generate_optimized_gauges(
    zones,
    target_density=0.01,
    min_distance_m=1000,
    random_seed=42
)
```

---

## 🔧 工具脚本

| 脚本 | 功能 | 使用场景 |
|-----|------|---------|
| `run_upper_truckee_complete_11steps.py` | 运行完整11步工作流 | 主工作流执行 |
| `optimize_rain_gauges_refactored.py` | 雨量站分布优化 | Step 5优化 |
| `calibrate_hbv_all_zones.py` | HBV参数校准 | 参数率定 |
| `diagnose_zone2_precipitation.py` | 降雨数据诊断 | 问题排查 |
| `test_parallel_hbv.py` | HBV并行化测试 | 性能验证 |

---

## 📊 数据格式

### 输入数据

| 数据类型 | 格式 | 示例文件 |
|---------|------|---------|
| 流域边界 | GeoJSON | `subbasins.geojson` |
| 河网 | GeoJSON | `river_network.geojson` |
| 参数分区 | GeoJSON | `parameter_zones.geojson` |
| 降雨数据 | CSV | `precipitation.csv` |
| 配置文件 | YAML | `workflow_config.yaml` |

### 输出数据

| 数据类型 | 格式 | 示例文件 |
|---------|------|---------|
| 雨量站 | GeoJSON | `optimized_gauges.geojson` |
| 径流结果 | CSV | `runoff_results/*.csv` |
| 统计汇总 | CSV | `runoff_summary.csv` |
| 验证报告 | JSON | `validation_result.json` |

---

## 🎓 学习路径

### 初学者路径

1. **第1天**: 阅读[用户手册 - 快速开始](用户手册.md#快速开始)
2. **第2天**: 运行示例工作流 `python run_upper_truckee_complete_11steps.py`
3. **第3天**: 理解[配置文件](用户手册.md#配置文件详解)
4. **第4天**: 学习[数据质量检查](用户手册.md#数据质量检查)
5. **第5天**: 实践[结果分析](用户手册.md#结果分析)

### 进阶用户路径

1. 阅读[开发指南 - 核心设计原则](开发指南.md#核心设计原则)
2. 学习[验证框架API](api/validation_framework.md)
3. 实践[HBV并行化](api/parallel_hbv.md)
4. 掌握[雨量站优化](api/rain_gauge_optimization.md)
5. 开发自定义功能模块

### 开发者路径

1. 阅读完整[开发指南](开发指南.md)
2. 研究模块源代码
3. 编写单元测试
4. 提交代码改进
5. 参与文档维护

---

## 🔍 关键概念索引

### 水文概念

- **径流系数 (RC)**: 径流量与降雨量的比值，必须在0-1之间
- **HBV模型**: 概念性降雨-径流模型
- **参数分区**: 基于流域特征的空间离散化
- **水量平衡**: P = R + ET + ΔS

### 系统概念

- **配置驱动**: 所有参数从YAML配置文件加载，零硬编码
- **验证框架**: 统一的数据质量检查API
- **并行化**: 使用多核加速大规模计算
- **GeoJSON**: 标准地理数据格式

### 验证指标

- **空间变异系数 (CV)**: 评估降雨空间分布均匀性
- **站点密度**: 雨量站数量/流域面积，单位: 站点/100km²
- **缺测率**: 缺失数据占比
- **相关系数**: 站点间数据相关性

---

## ⚠️ 常见警告解读

| 警告信息 | 含义 | 解决方法 |
|---------|------|---------|
| `径流系数 > 1` | 径流超过降雨，物理不合理 | 检查降雨数据质量，运行诊断工具 |
| `空间变异系数过大` | 降雨分布不均匀 | 增加雨量站，改进插值方法 |
| `站点密度不足` | 雨量站过少 | 增加目标密度配置 |
| `缺测率偏高` | 数据缺失过多 | 数据补全或降低阈值 |

---

## 📈 性能优化

| 场景 | 方法 | 文档链接 |
|-----|------|---------|
| 大规模流域 | 使用并行HBV | [HBV并行化](api/parallel_hbv.md) |
| 参数敏感性分析 | 并行执行多组参数 | [开发指南 - 性能优化](开发指南.md#性能优化建议) |
| 减少内存占用 | 批量处理分区 | [HBV并行化 - 故障排查](api/parallel_hbv.md#故障排查) |

---

## 🆕 最新更新

### 版本 1.0.0 (2025-01-24)

**新增功能**:
- ✅ 完整的验证框架（降雨、空间、时序验证）
- ✅ HBV模型并行化支持
- ✅ 配置驱动的雨量站优化
- ✅ 降雨数据诊断工具
- ✅ 完整的API文档和用户手册

**改进**:
- ♻️ 重构雨量站优化脚本，消除硬编码
- ♻️ 扩展验证框架，支持更多验证类型
- 🚀 HBV并行化，提升3-4倍性能

**修复**:
- 🐛 识别Zone 2径流系数异常的根本原因（降雨CV过大）

---

## 📞 获取帮助

### 优先级顺序

1. **查阅文档** - 本文档中心包含95%常见问题的答案
2. **运行诊断工具** - 使用诊断脚本定位问题
3. **查看验证结果** - ValidationResult包含详细错误信息
4. **查看示例代码** - API文档中的代码示例

### 文档反馈

欢迎对文档提出改进建议！

---

## 📄 许可证

本项目采用 MIT 许可证。

---

**文档版本**: v1.0.0
**最后更新**: 2025-01-24
**维护者**: HydroSIS开发团队
