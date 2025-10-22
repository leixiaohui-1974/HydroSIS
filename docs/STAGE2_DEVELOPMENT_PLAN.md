# Stage 2 开发计划

**创建日期**: 2025-10-22
**项目状态**: Stage 1 已完成 (100%)
**当前代码质量**: 7.0/10 → **目标**: 8.5/10

---

## Stage 1 完成情况回顾

### ✅ 已完成任务

1. **Task 1.1: 分解超大文件** - ✅ 部分完成
   - Pipeline 已拆分为模块化步骤 (14个文件)
   - **遗留**: `ten_step_pipeline.py` (4,582行) 仍需重构或废弃

2. **Task 1.2: 统一语言规范** - ✅ 完成 (hydrodynamics模块)
   - 所有 hydrodynamics 模块已转为英文
   - 创建了双语术语表
   - **遗留**: 其他模块仍有中文注释

3. **Task 1.3: 添加参数验证** - ✅ 完全完成
   - 11个模型全部添加验证
   - 28个测试用例 (100%通过)
   - 完整文档

### 📊 代码质量提升

```
Stage 0 (初始):    6.3/10
Stage 1 (当前):    7.0/10  (+11%)
Stage 2 (目标):    8.5/10  (+35%)
```

---

## Stage 2 任务清单

### 优先级 1: 核心质量提升 🔴

#### Task 2.1: 完成语言国际化 (剩余模块)

**优先级**: 🔴 CRITICAL
**预计工时**: 12-16小时
**影响范围**: 国际化协作 +100%

**发现**: 以下模块仍包含大量中文注释
```bash
# 检测到中文的文件:
hydrosis/analysis/runoff_coefficients.py
hydrosis/analysis/channel_profile.py
hydrosis/analysis/channel_flow.py
hydrosis/pipeline/step07_thiessen_weights.py
hydrosis/pipeline/step03_partitioning.py
hydrosis/pipeline/step10_hydrodynamic_run.py
hydrosis/pipeline/ten_step_pipeline.py
hydrosis/parameters/partition.py
hydrosis/delineation/utils.py
... 以及更多文件
```

**行动计划**:

**Phase 1: Analysis 模块 (4-5小时)**
```python
# 文件清单:
- analysis/runoff_coefficients.py
- analysis/channel_profile.py
- analysis/channel_flow.py
- analysis/__init__.py

# 标准化内容:
- 所有中文注释 → 英文
- 函数/类文档字符串 → Google Style
- 变量名保持英文 (已是英文)
```

**Phase 2: Pipeline 模块 (5-6小时)**
```python
# 文件清单:
- pipeline/step01_terrain.py
- pipeline/step02_pour_points.py
- pipeline/step03_partitioning.py
- pipeline/step04_channel_profile.py
- pipeline/step05_rain_gauge_layout.py
- pipeline/step06_rain_sequence.py
- pipeline/step07_thiessen_weights.py
- pipeline/step08_areal_precipitation.py
- pipeline/step09_hydrologic_run.py
- pipeline/step10_hydrodynamic_run.py
- pipeline/stages.py
```

**Phase 3: Parameters & Delineation 模块 (3-4小时)**
```python
# 文件清单:
- parameters/partition.py (1,395行)
- delineation/utils.py (1,263行)
```

**验收标准**:
- ✅ 无中文字符: `grep -r "[\u4e00-\u9fff]" hydrosis/ --include="*.py"` 返回空
- ✅ 所有公共API有英文文档
- ✅ 创建完整的双语术语对照表 (扩展版)

**可交付成果**:
- `docs/LANGUAGE_STANDARDIZATION_STAGE2.md` - 第二阶段语言标准化文档
- `docs/BILINGUAL_GLOSSARY_COMPLETE.md` - 完整双语术语表

---

#### Task 2.2: 重构大型文件

**优先级**: 🔴 CRITICAL
**预计工时**: 10-14小时
**影响范围**: 可维护性 +50%

**目标文件**:

**1. `pipeline/ten_step_pipeline.py` (4,582行)**

**问题分析**:
```bash
# 文件包含10个步骤的完整实现
# 每个步骤 300-600 行
# 导致单一文件过于庞大
```

**重构方案 A: 废弃该文件 (推荐)**
```python
# 现状: pipeline/ 目录已经有独立的步骤文件
# step01_terrain.py, step02_pour_points.py, ... step10_hydrodynamic_run.py

# 行动:
# 1. 验证所有功能已迁移到独立文件
# 2. 添加 @deprecated 装饰器到 ten_step_pipeline.py
# 3. 创建迁移指南: docs/MIGRATION_GUIDE.md
# 4. 1个版本后删除该文件
```

**重构方案 B: 保留为高级接口**
```python
# 如果需要向后兼容，保留为包装器:

# pipeline/ten_step_pipeline.py (重构后 ~200行)
from .step01_terrain import run_step01
from .step02_pour_points import run_step02
# ... 导入所有步骤

def run_ten_step_pipeline(config: PipelineConfig) -> PipelineResult:
    """High-level wrapper for complete pipeline execution.

    DEPRECATED: Use individual step functions for better control.
    This function will be removed in v1.0.0.

    See: docs/MIGRATION_GUIDE.md for migration instructions.
    """
    warnings.warn(
        "ten_step_pipeline is deprecated. Use individual steps.",
        DeprecationWarning,
        stacklevel=2
    )

    # 委托给各个步骤
    result01 = run_step01(config)
    result02 = run_step02(config, result01)
    # ... 依次执行

    return aggregate_results(...)
```

**2. `parameters/partition.py` (1,395行)**

**重构方案**:
```python
# 拆分为:
parameters/
├── partition/
│   ├── __init__.py (导出公共API)
│   ├── builder.py (分区构建逻辑, ~400行)
│   ├── optimizer.py (分区优化算法, ~400行)
│   ├── validator.py (分区验证, ~300行)
│   └── utils.py (工具函数, ~200行)
└── ...
```

**3. `delineation/utils.py` (1,263行)**

**重构方案**:
```python
# 拆分为:
delineation/
├── utils/
│   ├── __init__.py (导出公共API)
│   ├── pour_point_generation.py (~400行)
│   ├── watershed_analysis.py (~400行)
│   ├── network_topology.py (~300行)
│   └── geojson_export.py (~150行)
└── ...
```

**验收标准**:
- ✅ 无文件超过 800 行
- ✅ 所有模块职责单一、清晰
- ✅ 保持100%向后兼容 (通过 `__init__.py` 导出)
- ✅ 所有现有测试继续通过

**可交付成果**:
- 重构后的模块化代码
- `docs/MIGRATION_GUIDE.md` - 迁移指南 (如果有破坏性更改)
- 更新的测试

---

#### Task 2.3: 消除代码重复

**优先级**: 🟠 HIGH
**预计工时**: 6-8小时
**影响范围**: 可维护性 +25%

**识别的重复代码**:

**重复 1: 配置序列化逻辑**

**位置**:
- `config.py:ModelConfig.to_dict()`
- `config.py:HydroProjectConfig.to_dict()`

**修复方案**:
```python
# 创建 config/serialization.py

from typing import Any, Dict
from pathlib import Path
from dataclasses import fields

def serialize_config(config_obj: Any) -> Dict[str, Any]:
    """Generic configuration serialization.

    Converts dataclass instances to dictionaries, handling:
    - Path objects → strings
    - Nested dataclasses → recursive serialization
    - None values → excluded from output
    """
    result = {}
    for field in fields(config_obj):
        value = getattr(config_obj, field.name)
        if value is None:
            continue

        if isinstance(value, Path):
            result[field.name] = str(value)
        elif hasattr(value, '__dataclass_fields__'):
            result[field.name] = serialize_config(value)
        else:
            result[field.name] = value

    return result

# 在 ModelConfig 中使用:
def to_dict(self) -> Dict[str, Any]:
    return serialize_config(self)
```

**重复 2: 时间序列加载逻辑**

**位置**:
- `io/inputs.py`
- `analysis/runoff_coefficients.py`
- 可能还有其他地方

**修复方案**:
```python
# 统一到 io/time_series.py

from typing import Tuple, Optional
import numpy as np
from pathlib import Path

def load_time_series(
    path: Path,
    time_column: int = 0,
    value_column: int = -1,
    skip_header: int = 1,
    delimiter: str = ",",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load time series data from CSV file.

    Args:
        path: Path to CSV file
        time_column: Column index for timestamps
        value_column: Column index for values (-1 for last column)
        skip_header: Number of header lines to skip
        delimiter: Column delimiter

    Returns:
        Tuple of (timestamps, values) as numpy arrays

    Example:
        >>> times, values = load_time_series(Path("precip.csv"))
        >>> assert len(times) == len(values)
    """
    data = np.loadtxt(
        path,
        delimiter=delimiter,
        skiprows=skip_header
    )

    times = data[:, time_column]
    values = data[:, value_column]

    return times, values
```

**重复 3: GeoJSON 导出逻辑**

**位置**: 多个模块重复实现 GeoJSON 导出

**修复方案**:
```python
# 创建 io/geojson.py

from typing import List, Dict, Any
import json
from pathlib import Path

def export_features_geojson(
    features: List[Dict[str, Any]],
    output_path: Path,
    crs: str = "EPSG:4326"
) -> None:
    """Export features to GeoJSON file.

    Args:
        features: List of GeoJSON features
        output_path: Output file path
        crs: Coordinate reference system
    """
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": crs}
        },
        "features": features
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(geojson, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
```

**验收标准**:
- ✅ 所有重复代码已提取到共享模块
- ✅ 原有调用处已更新为使用共享函数
- ✅ 添加单元测试覆盖新的工具函数
- ✅ 文档更新

---

### 优先级 2: 测试与文档 🟡

#### Task 2.4: 扩展测试覆盖

**优先级**: 🟡 MEDIUM
**预计工时**: 12-16小时
**影响范围**: 可靠性 +40%

**当前测试状态**:
```bash
# 现有测试:
tests/
├── test_portal_*.py (Portal API 测试)
├── test_workflow*.py (工作流测试)
├── test_delineation_*.py (流域划分测试)
├── test_distributed_green_ampt.py (单一模型测试)
└── test_examples.py (示例测试)

# 缺失:
- 集成测试 (端到端)
- 性能测试
- 边界条件测试
- 参数验证测试 (除了我们刚添加的)
```

**新增测试套件**:

**1. 集成测试 (tests/integration/)**
```python
# test_full_pipeline.py
def test_complete_dem_to_report_pipeline():
    """Test complete workflow from DEM to final report."""

def test_multi_scenario_comparison():
    """Test scenario comparison functionality."""

def test_parameter_calibration_workflow():
    """Test parameter optimization workflow."""

# test_model_chains.py
def test_runoff_routing_integration():
    """Test integration between runoff and routing models."""

def test_hydrologic_hydrodynamic_coupling():
    """Test coupling between hydrologic and hydrodynamic models."""
```

**2. 性能测试 (tests/performance/)**
```python
# test_benchmarks.py
@pytest.mark.slow
def test_large_basin_performance():
    """Benchmark performance on large basin (1000+ subbasins)."""

@pytest.mark.slow
def test_parallel_scaling():
    """Test parallel computation speedup."""

def test_memory_usage():
    """Monitor memory usage during simulation."""
```

**3. 边界条件测试 (tests/test_edge_cases.py)**
```python
def test_zero_precipitation():
    """Test model behavior with zero rainfall."""

def test_extreme_rainfall():
    """Test model stability with extreme precipitation."""

def test_missing_data_handling():
    """Test handling of missing/invalid data."""

def test_empty_basin():
    """Test behavior with minimal basin configuration."""

def test_single_timestep():
    """Test single timestep simulation."""
```

**4. 参数验证测试扩展 (tests/test_validation_extended.py)**
```python
def test_cross_model_parameter_consistency():
    """Verify parameter consistency across model types."""

def test_physical_constraint_violations():
    """Test all physical constraint validations."""

def test_validation_error_messages():
    """Verify error messages are clear and actionable."""
```

**目标覆盖率**:
```
当前覆盖率: ~60% (估计)
目标覆盖率: 85%+

核心模块要求:
- hydrosis/runoff/: 90%+
- hydrosis/routing/: 90%+
- hydrosis/validation.py: 100%
- hydrosis/config.py: 85%+
```

**验收标准**:
- ✅ 总体测试覆盖率 ≥ 85%
- ✅ 所有核心模块覆盖率达标
- ✅ CI/CD 集成测试通过
- ✅ 性能基准建立并记录

---

#### Task 2.5: 完善项目文档

**优先级**: 🟡 MEDIUM
**预计工时**: 10-14小时
**影响范围**: 用户体验 +60%

**文档结构规划**:

```
docs/
├── README.md (主文档索引)
│
├── user_guide/
│   ├── 01_installation.md
│   ├── 02_quick_start.md
│   ├── 03_configuration.md
│   ├── 04_running_simulations.md
│   ├── 05_analyzing_results.md
│   └── tutorials/
│       ├── tutorial_01_simple_basin.md
│       ├── tutorial_02_parameter_calibration.md
│       └── tutorial_03_scenario_comparison.md
│
├── api_reference/
│   ├── runoff_models.md
│   ├── routing_models.md
│   ├── parameter_zones.md
│   ├── workflow_api.md
│   └── portal_api.md
│
├── developer_guide/
│   ├── architecture.md
│   ├── adding_new_models.md
│   ├── testing_guidelines.md
│   ├── contributing.md
│   └── code_style.md
│
├── technical/
│   ├── LANGUAGE_STANDARDIZATION.md (已存在)
│   ├── LANGUAGE_STANDARDIZATION_STAGE2.md (待创建)
│   ├── PARAMETER_VALIDATION.md (已存在)
│   ├── STAGE1_COMPLETION_SUMMARY.md (已存在)
│   ├── STAGE2_DEVELOPMENT_PLAN.md (本文档)
│   └── BILINGUAL_GLOSSARY_COMPLETE.md (待创建)
│
└── examples/
    ├── example_01_basic_simulation.md
    ├── example_02_multi_scenario.md
    ├── example_03_hydrodynamics.md
    └── example_04_api_usage.md
```

**关键文档内容**:

**1. User Guide (用户指南)**

**`01_installation.md`**:
```markdown
# Installation Guide

## System Requirements
- Python 3.9+
- GDAL 3.0+
- ...

## Installation Methods

### Method 1: pip install (recommended)
\`\`\`bash
pip install hydrosis
\`\`\`

### Method 2: From source
\`\`\`bash
git clone https://github.com/user/HydroSIS
cd HydroSIS
pip install -e .
\`\`\`

### Method 3: Docker
\`\`\`bash
docker pull hydrosis/hydrosis:latest
\`\`\`

## Verification
...
```

**`02_quick_start.md`**:
```markdown
# Quick Start Guide

## Your First Simulation in 5 Minutes

### Step 1: Prepare Data
...

### Step 2: Create Configuration
...

### Step 3: Run Simulation
\`\`\`python
from hydrosis import run_workflow
from hydrosis.config import ModelConfig

config = ModelConfig.from_yaml("config.yaml")
result = run_workflow(config, forcing_data)
\`\`\`

### Step 4: Analyze Results
...
```

**2. API Reference (API文档)**

使用自动文档生成工具 (Sphinx + autodoc):
```bash
# 配置 Sphinx
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints

# 生成配置
sphinx-quickstart docs/

# 配置 conf.py
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # 支持 Google style docstrings
    'sphinx_autodoc_typehints',
]

# 生成文档
sphinx-apidoc -o docs/api_reference hydrosis/
make html
```

**3. Developer Guide (开发者指南)**

**`adding_new_models.md`**:
```markdown
# Adding New Hydrological Models

## Step-by-Step Guide

### 1. Create Model Class
\`\`\`python
from hydrosis.runoff.base import RunoffModel

class MyNewModel(RunoffModel):
    \"\"\"Your model description.\"\"\"

    def validate_parameters(self) -> None:
        # Add parameter validation
        pass

    def simulate(self, subbasin, precipitation):
        # Implement simulation logic
        pass
\`\`\`

### 2. Add Parameter Validation
...

### 3. Register Model
...

### 4. Add Tests
...

### 5. Update Documentation
...
```

**验收标准**:
- ✅ 所有文档使用一致的格式 (Markdown)
- ✅ API文档自动生成并保持最新
- ✅ 至少3个完整的教程示例
- ✅ 开发者指南涵盖常见任务
- ✅ 文档可通过 ReadTheDocs 或 GitHub Pages 托管

---

### 优先级 3: 高级优化 🟢

#### Task 2.6: 类型注解完善

**优先级**: 🟢 LOW
**预计工时**: 4-6小时

**目标**: 达到 mypy strict 模式兼容

**行动**:
```python
# 1. 添加类型别名 (io/types.py)
from typing import TypeAlias, Callable, Sequence, Mapping

TimeSeriesData: TypeAlias = Sequence[float]
ForcingData: TypeAlias = Mapping[str, TimeSeriesData]
ProgressCallback: TypeAlias = Callable[[str, str, Optional[Dict]], None]

# 2. 启用 mypy strict 检查
# mypy.ini
[mypy]
python_version = 3.9
strict = True
warn_return_any = True
warn_unused_configs = True

# 3. 逐模块修复类型问题
mypy hydrosis/runoff/ --strict
mypy hydrosis/routing/ --strict
# ... 依次处理
```

---

#### Task 2.7: 性能优化

**优先级**: 🟢 LOW
**预计工时**: 8-12小时

**优化方向**:

**1. 向量化计算**
```python
# Before: Loop over subbasins
for sub_id, sub in self.subbasins.items():
    runoff[sub_id] = model.simulate(sub, forcing[sub_id])

# After: Vectorized batch processing
runoff = model.simulate_batch(
    subbasins=list(self.subbasins.values()),
    forcing=np.array([forcing[sid] for sid in self.subbasins])
)
```

**2. 缓存机制**
```python
from functools import lru_cache

@lru_cache(maxsize=128)
def compute_flow_accumulation(dem_path: str):
    """Cache expensive DEM computations."""
    ...
```

**3. 并行处理优化**
```python
# 使用 multiprocessing 或 dask
from concurrent.futures import ProcessPoolExecutor

def simulate_subbasins_parallel(subbasins, forcing, n_workers=4):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(simulate_one, subbasins, forcing)
    return list(results)
```

---

## 实施时间表

### Week 1-2: 核心质量提升 (Task 2.1-2.3)
```
Week 1:
  Mon-Tue: Task 2.1 Phase 1 (Analysis 模块国际化)
  Wed-Thu: Task 2.1 Phase 2 (Pipeline 模块国际化)
  Fri:     Task 2.1 Phase 3 (其余模块国际化)

Week 2:
  Mon-Tue: Task 2.2 (重构大型文件)
  Wed-Thu: Task 2.3 (消除代码重复)
  Fri:     代码审查 & 测试验证
```

### Week 3-4: 测试与文档 (Task 2.4-2.5)
```
Week 3:
  Mon-Wed: Task 2.4 (扩展测试覆盖)
  Thu-Fri: Task 2.5 Phase 1 (用户指南)

Week 4:
  Mon-Tue: Task 2.5 Phase 2 (API文档)
  Wed-Thu: Task 2.5 Phase 3 (开发者指南)
  Fri:     文档审查 & 发布
```

### Week 5: 高级优化 & 收尾 (Optional)
```
  Mon-Tue: Task 2.6 (类型注解)
  Wed-Thu: Task 2.7 (性能优化)
  Fri:     Stage 2 总结 & 发布
```

**总预计工时**: 54-76 小时 (约 1.5-2 个全职工作周)

---

## 质量提升预期

### Stage 1 (当前)
```
代码质量评分: 7.0/10

├── 功能完整性: 8/10 ✓
├── 模块设计:   7/10 ✓
├── 类型安全:   7/10 ✓
├── 代码质量:   7/10 ✓
├── 测试覆盖:   6/10 ✗
├── 文档完整:   5/10 ✗
└── 国际化:     7/10 ✓ (部分)
```

### Stage 2 目标
```
代码质量评分: 8.5/10 (+21%)

├── 功能完整性: 9/10 ✓✓
├── 模块设计:   9/10 ✓✓✓
├── 类型安全:   8/10 ✓✓
├── 代码质量:   9/10 ✓✓✓
├── 测试覆盖:   9/10 ✓✓✓
├── 文档完整:   9/10 ✓✓✓
└── 国际化:     10/10 ✓✓✓
```

---

## 验收标准

### Stage 2 完成标准

#### 必须满足 (MUST)
- ✅ 所有 Python 文件无中文字符 (仅英文)
- ✅ 无文件超过 800 行
- ✅ 测试覆盖率 ≥ 85%
- ✅ 所有公共 API 有完整文档
- ✅ 所有测试通过 (包括新增测试)

#### 应该满足 (SHOULD)
- ✅ 用户指南至少 5 篇文档
- ✅ 至少 3 个完整教程
- ✅ API 文档自动生成
- ✅ 开发者指南完整

#### 可以满足 (COULD)
- ✅ mypy strict 模式兼容
- ✅ 性能提升 20%+
- ✅ ReadTheDocs 或 GitHub Pages 部署

---

## 风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 重构破坏现有功能 | 中 | 高 | 保持100%测试覆盖，严格代码审查 |
| 时间估算不准 | 中 | 中 | 采用敏捷方法，分批交付 |
| 文档维护困难 | 低 | 中 | 使用自动生成工具，建立维护流程 |
| 性能优化收益低 | 低 | 低 | 先建立性能基准，再决定是否优化 |

---

## 下一步行动

### 立即开始 (本周)
1. ✅ 创建 Stage 2 开发分支: `git checkout -b stage2/language-internationalization`
2. ✅ 运行中文字符检测: `grep -r "[\u4e00-\u9fff]" hydrosis/ --include="*.py" > chinese_files.txt`
3. ✅ 开始 Task 2.1 Phase 1: Analysis 模块国际化

### 本月目标
- 完成 Task 2.1-2.3 (核心质量提升)
- 测试覆盖率提升至 75%+
- 初步文档框架建立

### 三个月目标
- Stage 2 全部完成
- 代码质量达到 8.5/10
- 准备 Stage 3 (生产化部署)

---

**负责人**: Claude Code
**审核**: 项目负责人
**下次审查**: 完成 Task 2.1 后

**让我们开始 Stage 2 的旅程！** 🚀
