# HydroSIS 工作流测试场景设计

## 概述

本文档描述了HydroSIS模块化API系统的8种工作流测试场景，这些场景充分利用了模块化架构的灵活性，从最简单的单模块测试到完整的十一步工作流，全面验证系统的各项功能。

## 测试场景列表

### 场景1: 最小测试 - 仅地形处理

**工作流ID**: `test_minimal_terrain`  
**配置文件**: `config/workflows/test_scenarios/01_minimal_terrain.yaml`

**目的**: 
- 测试单个模块的独立运行能力
- 验证最基础的地形处理功能
- 确认模块可以完全独立使用

**步骤**:
1. 地形处理（DEM→流向+流量累积+坡度）

**测试重点**:
- ✅ 模块独立运行
- ✅ DEM数据读取
- ✅ 流向计算正确性
- ✅ 流量累积计算
- ✅ 坡度计算
- ✅ 输出文件生成

**预期输出**:
- `flow_direction.tif` - 流向栅格
- `flow_accumulation.tif` - 流量累积栅格
- `slope.tif` - 坡度栅格

---

### 场景2: 两步基础测试 - 地形+汇水点

**工作流ID**: `test_two_step_basic`  
**配置文件**: `config/workflows/test_scenarios/02_two_step_basic.yaml`

**目的**:
- 测试两个模块的串联
- 验证模块间数据传递
- 测试变量引用机制

**步骤**:
1. 地形处理
2. 汇水点生成（基于步骤1的流量累积）

**测试重点**:
- ✅ 模块串联执行
- ✅ 数据传递（`${steps.terrain.outputs.flow_accumulation}`）
- ✅ 依赖关系管理
- ✅ 汇水点自动识别
- ✅ GeoJSON输出

**预期输出**:
- `pour_points.geojson` - 汇水点空间数据
- 汇水点数量统计

---

### 场景3: 三步流域划分测试

**工作流ID**: `test_three_step_delineation`  
**配置文件**: `config/workflows/test_scenarios/03_three_step_delineation.yaml`

**目的**:
- 测试完整的流域划分链路
- 验证多步骤依赖关系
- 测试拓扑关系计算

**步骤**:
1. 地形处理
2. 汇水点生成
3. 流域划分（需要步骤1和2的输出）

**测试重点**:
- ✅ 多步骤依赖（步骤3依赖步骤1和2）
- ✅ 流域边界提取
- ✅ 流域拓扑关系
- ✅ 面积计算
- ✅ 流域-汇水点对应关系

**预期输出**:
- `watersheds.geojson` - 流域多边形
- `topology` - 流域拓扑关系（上下游）

---

### 场景4: 降雨分析工作流

**工作流ID**: `test_precipitation_analysis`  
**配置文件**: `config/workflows/test_scenarios/04_precipitation_analysis.yaml`

**目的**:
- 测试降雨相关模块的组合
- 验证空间降雨分析能力
- 测试泰森多边形方法

**步骤**:
1. 雨量站布局优化
2. 降雨序列生成（并行于步骤1）
3. 面雨量计算（基于步骤1和2）

**测试重点**:
- ✅ 并行步骤执行（步骤1和2无依赖关系）
- ✅ 雨量站空间优化
- ✅ 泰森多边形生成
- ✅ 面雨量权重计算
- ✅ 时间序列处理

**预期输出**:
- `rain_gauge_locations.geojson` - 雨量站位置
- `thiessen_polygons.geojson` - 泰森多边形
- `areal_precipitation.csv` - 流域面雨量时间序列

---

### 场景5: 水文模拟工作流

**工作流ID**: `test_hydrologic_simulation`  
**配置文件**: `config/workflows/test_scenarios/05_hydrologic_simulation.yaml`

**目的**:
- 测试产流和汇流的完整链路
- 验证HBV模型计算
- 测试Muskingum汇流

**步骤**:
1. 产流模拟（HBV模型）
2. 汇流演算（Muskingum方法）

**测试重点**:
- ✅ HBV模型执行
- ✅ 产流参数设置
- ✅ 径流系数计算
- ✅ 汇流过程模拟
- ✅ 洪峰演进
- ✅ 水量平衡验证

**预期输出**:
- `runoff_timeseries.csv` - 产流时间序列
- `discharge_timeseries.csv` - 流量过程

---

### 场景6: 参数率定工作流

**工作流ID**: `test_calibration_workflow`  
**配置文件**: `config/workflows/test_scenarios/06_calibration_workflow.yaml`

**目的**:
- 测试参数率定功能
- 验证多目标评估
- 测试优化算法

**步骤**:
1. 模型评估（计算初始指标）
2. 参数率定（基于评估结果优化）

**测试重点**:
- ✅ NSE、KGE等指标计算
- ✅ 参数优化算法
- ✅ 参数约束处理
- ✅ 目标函数优化
- ✅ 率定结果验证

**预期输出**:
- 初始性能指标（NSE、KGE、RMSE等）
- 率定后的最优参数
- 率定后的性能提升

---

### 场景7: 并行分析工作流

**工作流ID**: `test_parallel_analysis`  
**配置文件**: `config/workflows/test_scenarios/07_parallel_analysis.yaml`

**目的**:
- 测试并行执行能力
- 验证分支工作流
- 测试多阈值对比分析

**步骤**:
1. 地形处理（基础步骤）
2-4. 不同阈值的汇水点生成（并行执行）
   - 阈值500（精细）
   - 阈值1000（中等）
   - 阈值2000（粗略）
5. 河网提取（并行于步骤2-4）

**测试重点**:
- ✅ 并行步骤执行
- ✅ 共享基础数据
- ✅ 不同参数对比
- ✅ 多分支独立运行
- ✅ 结果对比分析

**预期输出**:
- 3个不同粒度的汇水点集
- 河网数据
- 可用于敏感性分析

---

### 场景8: 完整十一步工作流

**工作流ID**: `test_complete_eleven_steps`  
**配置文件**: `config/workflows/test_scenarios/08_complete_eleven_steps.yaml`

**目的**:
- 测试完整的端到端水文建模流程
- 验证所有模块的协同工作
- 模拟真实的生产环境使用

**步骤**:
1. DEM地形处理
2. 汇水点生成
3. 流域划分
4. 河网提取（并行于步骤2-3）
5. 雨量站布局
6. 降雨序列生成（并行于步骤5）
7. （泰森多边形-包含在步骤5中）
8. 面雨量计算
9. 产流模拟
10. 汇流演算
11. 率定与验证

**测试重点**:
- ✅ 完整工作流执行
- ✅ 复杂依赖关系
- ✅ 长时间运行稳定性
- ✅ 内存管理
- ✅ 中间结果传递
- ✅ 最终结果验证

**预期输出**:
- 所有中间结果
- 最终流量过程
- 完整的评估报告

---

## 测试矩阵

| 场景 | 模块数 | 并行步骤 | 复杂度 | 预期耗时 | 主要验证点 |
|------|--------|---------|--------|---------|-----------|
| 1. 最小测试 | 1 | 0 | ⭐ | <10s | 单模块独立性 |
| 2. 两步基础 | 2 | 0 | ⭐⭐ | <20s | 模块串联 |
| 3. 三步划分 | 3 | 0 | ⭐⭐ | <30s | 多步依赖 |
| 4. 降雨分析 | 3 | 2 | ⭐⭐⭐ | <30s | 并行执行 |
| 5. 水文模拟 | 2 | 0 | ⭐⭐⭐ | <40s | 模型计算 |
| 6. 参数率定 | 2 | 0 | ⭐⭐⭐⭐ | <60s | 优化算法 |
| 7. 并行分析 | 5 | 4 | ⭐⭐⭐⭐ | <40s | 多分支并行 |
| 8. 完整十一步 | 11 | 2 | ⭐⭐⭐⭐⭐ | <180s | 端到端 |

## 运行测试

### 方式1: 运行所有测试

```bash
python tests/test_multiple_workflows.py
```

### 方式2: 运行单个测试

```bash
# 运行最小测试
python tests/test_multiple_workflows.py \
  --test config/workflows/test_scenarios/01_minimal_terrain.yaml

# 运行完整十一步
python tests/test_multiple_workflows.py \
  --test config/workflows/test_scenarios/08_complete_eleven_steps.yaml
```

### 方式3: 使用CLI

```bash
# 列出所有工作流
hydrosis workflow list

# 运行指定工作流
hydrosis workflow run test_minimal_terrain \
  --config config/workflows/test_scenarios/01_minimal_terrain.yaml
```

## 测试报告

测试完成后会生成两种报告：

1. **JSON报告**: `results/workflow_tests/workflow_test_report.json`
   - 机器可读
   - 包含详细的执行数据
   - 可用于自动化分析

2. **Markdown报告**: `results/workflow_tests/workflow_test_report.md`
   - 人类可读
   - 包含测试总结
   - 包含每个场景的详细结果

## 验证标准

每个测试场景都会进行以下验证：

### 基础验证
- ✅ 工作流成功加载
- ✅ 所有步骤执行完成
- ✅ 无异常和错误
- ✅ 输出文件生成

### 数据验证
- ✅ 输出文件存在
- ✅ 文件格式正确
- ✅ 数据值合理
- ✅ 数据完整性

### 性能验证
- ✅ 执行时间在预期范围内
- ✅ 内存使用合理
- ✅ 无内存泄漏

### 逻辑验证
- ✅ 步骤依赖关系正确
- ✅ 数据传递无误
- ✅ 变量引用解析正确
- ✅ 物理约束满足

## 扩展测试场景

可以轻松创建新的测试场景：

1. 复制现有的工作流配置文件
2. 修改步骤和参数
3. 添加到测试套件中

示例：创建敏感性分析工作流

```yaml
workflow:
  id: "sensitivity_analysis"
  name: "参数敏感性分析"
  
  steps:
    # 运行多组不同参数的模拟
    - id: "run_param_set_1"
      module: "runoff_generation"
      inputs:
        parameters:
          field_capacity: 100.0
    
    - id: "run_param_set_2"
      module: "runoff_generation"
      inputs:
        parameters:
          field_capacity: 200.0
    
    - id: "run_param_set_3"
      module: "runoff_generation"
      inputs:
        parameters:
          field_capacity: 300.0
```

## 持续集成

这些测试可以集成到CI/CD流程中：

```yaml
# .github/workflows/test-workflows.yml
name: Workflow Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run workflow tests
        run: python tests/test_multiple_workflows.py
      - name: Upload test reports
        uses: actions/upload-artifact@v2
        with:
          name: test-reports
          path: results/workflow_tests/
```

## 总结

这8个测试场景涵盖了：

1. **简单性测试** - 单模块、双模块
2. **复杂性测试** - 多步骤、长链路
3. **并行性测试** - 多分支、独立执行
4. **完整性测试** - 端到端流程
5. **功能性测试** - 各类专业功能
6. **性能测试** - 执行效率
7. **灵活性测试** - 不同组合方式

通过这套测试，可以：
- ✅ 全面验证系统功能
- ✅ 发现潜在问题
- ✅ 确保系统可靠性
- ✅ 验证架构设计合理性
- ✅ 为用户提供使用示例

---

**版本**: 1.0.0  
**更新日期**: 2025-10-26  
**维护者**: HydroSIS Team
