# HydroSIS 模块化API测试策略

## 测试概述

本文档描述了HydroSIS模块化API系统的完整测试策略，包括单元测试、集成测试和闭环验证。

## 测试层次

```
测试金字塔
┌─────────────────┐
│   E2E测试        │  完整的十一步工作流
├─────────────────┤
│  集成测试         │  工作流引擎、多模块组合
├─────────────────┤
│  单元测试         │  单个模块功能
└─────────────────┘
```

## 1. 单元测试

### 1.1 模块测试

每个模块都需要独立测试：

```python
def test_terrain_module():
    """测试地形处理模块"""
    terrain = TerrainModule()
    
    # 测试正常输入
    output = terrain.run({
        "dem_path": "test_data/dem.tif",
        "method": "d8"
    })
    
    assert output.flow_direction is not None
    assert output.flow_accumulation is not None
    
    # 测试错误输入
    with pytest.raises(ValueError):
        terrain.run({"dem_path": "nonexistent.tif"})
```

### 1.2 工作流引擎测试

```python
def test_workflow_engine():
    """测试工作流引擎"""
    engine = WorkflowEngine()
    
    # 测试简单工作流
    workflow = WorkflowDefinition(...)
    run = engine.execute(workflow)
    
    assert run.status == "completed"
    assert len(run.step_results) == len(workflow.steps)
```

## 2. 集成测试

### 2.1 模块链测试

测试多个模块的组合：

```python
def test_terrain_to_pour_points():
    """测试地形处理->汇水点生成链"""
    # 步骤1
    terrain = TerrainModule()
    terrain_output = terrain.run({...})
    
    # 步骤2
    pour_points = PourPointsModule()
    pp_output = pour_points.run({
        "flow_accumulation": terrain_output.flow_accumulation
    })
    
    assert len(pp_output.points) > 0
```

### 2.2 工作流测试

测试预定义工作流：

```python
def test_pour_points_workflow():
    """测试汇水点生成工作流"""
    engine = WorkflowEngine()
    template = WorkflowTemplates.get_template("pour_points_only")
    workflow = WorkflowDefinition.from_dict(template)
    
    run = engine.execute(workflow, parameters={
        "dem_path": "test_data/dem.tif"
    })
    
    assert run.status == "completed"
    assert "pour_points" in run.outputs
```

## 3. 闭环验证

### 3.1 验证原则

每一步都需要验证：

1. **输出文件存在性** - 所有声明的输出文件都应该存在
2. **格式正确性** - 文件格式符合规范（GeoJSON、GeoTIFF等）
3. **数据合理性** - 数值在合理范围内
4. **物理约束** - 满足物理规律（如水量平衡）
5. **上下游一致性** - 与上游步骤的输出一致

### 3.2 验证框架

```python
class ClosedLoopValidator:
    """闭环验证器"""
    
    def validate_step(self, step_id: str, output_dir: Path) -> ValidationResult:
        """验证单个步骤"""
        result = ValidationResult(step_id)
        
        # 1. 文件存在性检查
        self._check_files_exist(output_dir, result)
        
        # 2. 格式检查
        self._check_format(output_dir, result)
        
        # 3. 数据范围检查
        self._check_data_range(output_dir, result)
        
        # 4. 物理约束检查
        self._check_physical_constraints(output_dir, result)
        
        # 5. 一致性检查
        self._check_consistency(output_dir, result)
        
        return result
```

### 3.3 具体验证项

#### 步骤1: 地形处理

- ✅ 流向文件存在且格式正确
- ✅ 流量累积值全部≥0
- ✅ 坡度值在合理范围(0-1)
- ✅ 栅格尺寸与原始DEM一致
- ✅ 坐标系统正确

#### 步骤2: 汇水点生成

- ✅ GeoJSON格式正确
- ✅ 汇水点数量>0
- ✅ 每个点都有流量累积值
- ✅ 流量累积值大于阈值
- ✅ 坐标在DEM范围内

#### 步骤3: 流域划分

- ✅ 流域数量与汇水点数量一致
- ✅ 流域面积>0
- ✅ 流域不重叠
- ✅ 流域覆盖整个研究区

#### 步骤4: 河网提取

- ✅ 河网连通性
- ✅ 河道长度>0
- ✅ 河道坡度>0
- ✅ 河网拓扑正确

#### 步骤5: 雨量站布局

- ✅ 站点密度满足要求
- ✅ 站点在流域内
- ✅ 泰森多边形覆盖完整
- ✅ 站点间距合理

#### 步骤6: 降雨生成

- ✅ 时间序列长度正确
- ✅ 降雨值≥0
- ✅ 时间步长一致
- ✅ 总降雨量合理

#### 步骤7: 泰森多边形

- ✅ 多边形数量与站点数一致
- ✅ 多边形覆盖完整无重叠
- ✅ 每个流域都被覆盖

#### 步骤8: 面雨量计算

- ✅ 时间序列长度与降雨一致
- ✅ 面雨量≥0
- ✅ 面雨量≤最大点雨量
- ✅ 每个流域都有数据

#### 步骤9: 产流模拟

- ✅ 径流序列长度与降雨一致
- ✅ 径流值≥0
- ✅ **水量平衡**: 0 < 径流系数 < 1
- ✅ **物理约束**: 总径流 ≤ 总降雨
- ✅ 峰值时间合理

#### 步骤10: 汇流演算

- ✅ 流量序列连续
- ✅ 流量≥0
- ✅ 下游流量≥上游流量
- ✅ 峰值延迟合理
- ✅ 流量守恒

#### 步骤11: 率定验证

- ✅ NSE在合理范围(-∞, 1]
- ✅ RMSE>0
- ✅ 参数在约束范围内
- ✅ 率定结果优于初始参数

## 4. 测试数据

### 4.1 测试数据集

使用Upper Truckee River真实数据：

```
data/Upper_Truckee_River/
├── terrain/
│   └── elevation.tif          # DEM数据
├── observed/
│   └── discharge.csv          # 实测流量
└── ...
```

### 4.2 配置文件

```yaml
# config/test_config.yaml
test:
  data_dir: "data/Upper_Truckee_River"
  output_dir: "results/test_output"
  validation_criteria:
    runoff_coefficient:
      min: 0.1
      max: 0.9
    nse:
      min: 0.5
```

## 5. 测试执行

### 5.1 运行所有测试

```bash
# 单元测试
pytest tests/test_modules/ -v

# 集成测试
pytest tests/test_integration/ -v

# 完整工作流测试
python tests/test_complete_eleven_steps.py

# 带闭环验证的综合测试
pytest tests/test_modular_api_comprehensive.py -v -s
```

### 5.2 生成测试报告

```bash
# 生成HTML报告
pytest --html=report.html --self-contained-html

# 生成覆盖率报告
pytest --cov=hydrosis --cov-report=html
```

## 6. 持续集成

### 6.1 CI配置

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest -v
```

### 6.2 测试覆盖率目标

- 模块代码覆盖率: ≥80%
- 工作流引擎覆盖率: ≥90%
- 接口层覆盖率: ≥70%

## 7. 性能测试

### 7.1 性能基准

记录每个模块的性能指标：

| 模块 | 输入规模 | 期望时间 | 内存使用 |
|------|---------|---------|---------|
| terrain | 1000x1000 | <10s | <1GB |
| pour_points | 1000个候选点 | <5s | <500MB |
| watershed | 100个流域 | <30s | <2GB |

### 7.2 性能测试脚本

```python
@pytest.mark.benchmark
def test_terrain_performance(benchmark):
    """测试地形处理性能"""
    terrain = TerrainModule()
    
    result = benchmark(
        terrain.run,
        {"dem_path": "large_dem.tif"}
    )
    
    assert result is not None
```

## 8. 回归测试

### 8.1 回归测试套件

保存已知正确的输出作为基准：

```
tests/fixtures/
├── terrain_output/
│   ├── flow_direction.tif
│   └── flow_accumulation.tif
└── expected_results.json
```

### 8.2 回归检测

```python
def test_regression_terrain():
    """回归测试：确保输出与基准一致"""
    terrain = TerrainModule()
    output = terrain.run({...})
    
    # 比较与基准输出
    expected = load_baseline("terrain_output")
    assert compare_rasters(output.flow_direction, expected.flow_direction)
```

## 9. 测试文档

### 9.1 测试报告模板

```markdown
# 测试报告

## 测试概览
- 测试日期: YYYY-MM-DD
- 测试者: XXX
- 版本: v1.0.0

## 测试结果
- 总测试数: XX
- 通过: XX
- 失败: XX
- 跳过: XX

## 失败测试详情
...

## 性能指标
...
```

### 9.2 问题追踪

使用GitHub Issues追踪测试发现的问题。

## 10. 最佳实践

### 10.1 测试编写规范

1. **命名规范**: `test_<module>_<功能>.py`
2. **文档字符串**: 每个测试都要有清晰的说明
3. **断言信息**: 提供有意义的断言消息
4. **测试隔离**: 测试之间互不影响
5. **清理资源**: 测试后清理临时文件

### 10.2 测试数据管理

1. 使用小型测试数据集
2. 数据版本控制
3. 敏感数据脱敏
4. 提供数据生成脚本

### 10.3 测试维护

1. 定期运行测试
2. 及时修复失败的测试
3. 更新测试以匹配新功能
4. 删除过时的测试

## 11. 测试检查清单

- [ ] 所有模块都有单元测试
- [ ] 所有工作流都有集成测试
- [ ] 实现了闭环验证
- [ ] 测试覆盖率达标
- [ ] 性能测试通过
- [ ] 回归测试通过
- [ ] 测试文档完整
- [ ] CI/CD配置正确

## 总结

完善的测试体系确保了HydroSIS模块化API系统的质量和可靠性。通过单元测试、集成测试和闭环验证的三层测试策略，我们能够：

1. ✅ 确保每个模块功能正确
2. ✅ 验证模块间协作正常
3. ✅ 保证完整工作流可靠运行
4. ✅ 满足物理约束和水文规律
5. ✅ 及早发现和修复问题

---

**版本**: 1.0.0  
**更新日期**: 2025-10-26
