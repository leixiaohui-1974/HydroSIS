# 高级校准脚本说明

本目录包含几个高级的流域级参数校准脚本。这些脚本演示了复杂的多分区流域校准策略。

## 已重构的脚本（推荐使用）

这些脚本已经重构为使用统一的校准框架：

### 1. calibrate_with_realistic_observations.py
**功能**: 使用增强模型生成观测数据，用HBV模型率定
**特点**: 测试模型结构不匹配的情况
**代码行数**: 298行（原507行，减少41%）
**使用场景**: 评估HBV在结构误差存在时的性能

### 2. calibrate_upstream_zone_60day.py
**功能**: 基于60天数据的高精度参数率定
**特点**: 参数恢复验证，评估率定精度
**代码行数**: 319行（原525行，减少39%）
**使用场景**: 验证校准算法的参数恢复能力

### 3. calibrate_zone1_with_enhanced_obs.py
**功能**: 使用增强观测数据的基础校准
**代码行数**: 约300行
**使用场景**: 标准的单分区校准

### 4. calibrate_zone1_with_sensitivity.py
**功能**: 带敏感性分析的参数校准
**代码行数**: 约400行
**使用场景**: 了解参数敏感性

### 5. calibrate_zone1_with_warmup.py
**功能**: 带预热期的HBV校准
**代码行数**: 约400行
**使用场景**: 需要模型状态初始化的情况

### 6. calibrate_zone1_hbv_enhanced.py
**功能**: 增强型HBV校准（多参数）
**代码行数**: 约450行
**使用场景**: 全参数集校准

## 高级流域级脚本（专业用途）

以下脚本实现了复杂的多分区流域校准策略。它们是研究和高级应用的专业工具。

### 7. calibrate_watershed_cascading.py
**功能**: 流域分区逐级参数率定
**方法**: 从上游到下游逐个率定，上游结果作为下游输入
**代码行数**: 547行
**优点**:
- 简单直观，易于理解
- 计算效率高
**缺点**:
- 误差可能累积
- 下游分区性能可能受上游影响

**流域结构**: 6个分区 (P3→P6→P2→P4→P1, P5→P1)

### 8. calibrate_watershed_joint.py
**功能**: 多目标联合参数率定
**方法**: 同时优化所有分区参数（24维优化问题）
**代码行数**: 604行
**优点**:
- 避免误差累积
- 利用多个观测点信息
- 参数空间连续性
**缺点**:
- 计算成本高
- 可能收敛慢

**优化维度**: 24个参数（6分区 × 4参数）

### 9. calibrate_watershed_hybrid.py
**功能**: 混合策略参数率定
**方法**: 两阶段优化
  - 阶段1: 快速逐级率定获取初始参数
  - 阶段2: 用初始参数进行联合精细优化
**代码行数**: 658行
**优点**:
- 结合两种方法优点
- 更快收敛
- 更高质量解
**适用场景**: 需要高质量结果且可以接受较长计算时间

### 10. calibrate_hbv_all_zones.py
**功能**: 配置驱动的多分区HBV参数率定
**特点**:
- 完全配置驱动，无硬编码
- 使用EnhancedRunoffGenerator生成观测数据
- 集成验证框架
**代码行数**: 566行
**使用场景**: 生产环境中的标准化校准流程

## 使用建议

1. **快速入门**: 使用 `calibrate_with_realistic_observations.py` 或 `calibrate_upstream_zone_60day.py`
2. **单分区校准**: 使用 `calibrate_zone1_*` 系列脚本
3. **多分区流域（简单）**: 使用 `calibrate_watershed_cascading.py`
4. **多分区流域（高质量）**: 使用 `calibrate_watershed_hybrid.py`
5. **生产环境**: 使用 `calibrate_hbv_all_zones.py`

## 重构说明

这些高级流域脚本由于其复杂性和专业性，暂时保留原有实现：

- 它们实现了研究级别的复杂校准策略
- 涉及多分区协调和复杂的优化算法
- 需要extensive domain knowledge to use effectively
- 完全重构可能改变算法行为

如需使用这些高级脚本，建议：
1. 先理解其算法原理
2. 从简单的单分区校准开始
3. 逐步过渡到多分区场景
4. 根据具体需求选择合适的策略

## 依赖要求

部分脚本依赖自定义模块：
- `simple_runoff_generator.py`: 简化的径流生成器
- 这些模块可能需要单独安装或实现

如果缺少依赖，脚本会自动降级到简化实现。

## 更多信息

参考统一校准框架文档：
- `hydrosis/calibration/README.md`
- `hydrosis/calibration/base_calibrator.py`
- `hydrosis/calibration/hbv_calibrator.py`
