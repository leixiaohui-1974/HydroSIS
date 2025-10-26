# HydroSIS 模块化API系统

## 🎉 新特性

HydroSIS已经完成模块化重构！现在您可以：

- ✅ **独立使用任何功能模块** - 只需要哪个功能就用哪个
- ✅ **灵活组合工作流** - 通过YAML配置自由编排业务流程
- ✅ **多种接口方式** - REST API、CLI、Python SDK，随您选择
- ✅ **完全配置驱动** - 无需修改代码，只需调整配置

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone <repository-url>
cd HydroSIS

# 安装依赖
pip install -r requirements.txt

# 安装HydroSIS
pip install -e .
```

### 5分钟入门

#### 方式1: Python SDK

```python
from hydrosis.modules import TerrainModule

# 创建模块
terrain = TerrainModule()

# 执行
output = terrain.run({
    "dem_path": "data/dem.tif",
    "method": "d8",
    "output_dir": "results/"
})

print(f"流向文件: {output.flow_direction}")
```

#### 方式2: CLI命令行

```bash
# 列出所有模块
hydrosis module list

# 执行模块
hydrosis module run terrain \
  --param dem_path=data/dem.tif \
  --output-dir results/

# 执行工作流
hydrosis workflow run pour_points_only \
  --param dem_path=data/dem.tif
```

#### 方式3: REST API

```bash
# 启动API服务
python -m hydrosis.api.rest

# 执行模块
curl -X POST http://localhost:8000/api/v1/modules/terrain/execute \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"dem_path": "data/dem.tif"}}'
```

## 📦 可用模块

| 模块 | 功能 | 示例用途 |
|------|------|---------|
| `terrain` | 地形处理 | DEM处理、流向计算 |
| `pour_points` | 汇水点生成 | 识别流域出口 |
| `watershed_delineation` | 流域划分 | 划分子流域 |
| `channel_network` | 河网提取 | 提取河道网络 |
| `rain_gauge_layout` | 雨量站布局 | 优化站点分布 |
| `precipitation_generation` | 降雨生成 | 生成降雨序列 |
| `areal_precipitation` | 面雨量计算 | 计算流域降雨 |
| `runoff_generation` | 产流模拟 | HBV、SCS等模型 |
| `routing` | 汇流演算 | 河道汇流计算 |
| `calibration` | 参数率定 | 自动参数优化 |
| `evaluation` | 结果评估 | 性能评估 |

## 🔧 工作流示例

### 预定义工作流

1. **`pour_points_only`** - 仅生成汇水点
2. **`complete_simulation`** - 完整水文模拟流程

### 自定义工作流

创建 `my_workflow.yaml`:

```yaml
workflow:
  id: "my_workflow"
  name: "我的自定义工作流"
  
  parameters:
    dem_path: "data/dem.tif"
    output_dir: "results/"
  
  steps:
    - id: "terrain"
      module: "terrain"
      inputs:
        dem_path: "${parameters.dem_path}"
    
    - id: "pour_points"
      module: "pour_points"
      depends_on: ["terrain"]
      inputs:
        flow_accumulation: "${steps.terrain.outputs.flow_accumulation}"
  
  outputs:
    result: "${steps.pour_points.outputs.pour_points_geojson}"
```

执行：

```bash
hydrosis workflow run my_workflow --config my_workflow.yaml
```

## 📚 文档

- **[架构设计](docs/MODULAR_API_ARCHITECTURE.md)** - 详细的架构说明
- **[API使用指南](docs/API_USAGE_GUIDE.md)** - 完整的使用文档
- **[重构总结](docs/REFACTORING_SUMMARY_ZH.md)** - 重构说明
- **[示例代码](examples/modular_api_example.py)** - 代码示例

## 🎯 使用场景

### 场景1: 单纯生成汇水点

```python
from hydrosis.workflow_engine import WorkflowEngine, WorkflowTemplates, WorkflowDefinition

engine = WorkflowEngine()
template = WorkflowTemplates.get_template("pour_points_only")
workflow = WorkflowDefinition.from_dict(template)

run = engine.execute(workflow, parameters={
    "dem_path": "data/dem.tif"
})
```

### 场景2: 面雨量计算

组合模块：`rain_gauge_layout` → `precipitation_generation` → `areal_precipitation`

### 场景3: 完整水文模拟

使用预定义的 `complete_simulation` 工作流，或自定义组合。

### 场景4: 参数率定

```python
from hydrosis.modules import CalibrationModule

calibration = CalibrationModule()
output = calibration.run({
    "model_config": "config.json",
    "observed_data": "observed.csv",
    "parameters_to_calibrate": ["field_capacity", "beta"],
    "objective": "nse"
})
```

## 🔌 接口对比

| 特性 | REST API | CLI | Python SDK |
|------|----------|-----|------------|
| 易用性 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 灵活性 | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| 远程调用 | ✅ | ❌ | ❌ |
| 脚本集成 | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| 大模型调用 | ✅ (MCP) | ❌ | ✅ |

## 🛠️ 开发扩展

### 添加自定义模块

```python
from dataclasses import dataclass
from hydrosis.modules.base import Module, ModuleInput, ModuleOutput, ModuleMetadata, register_module

@dataclass
class MyInput(ModuleInput):
    param1: str

@dataclass
class MyOutput(ModuleOutput):
    result: str

@register_module
class MyCustomModule(Module):
    @classmethod
    def module_id(cls) -> str:
        return "my_custom_module"
    
    @classmethod
    def metadata(cls) -> ModuleMetadata:
        return ModuleMetadata(
            module_id="my_custom_module",
            name="我的自定义模块",
            description="自定义功能",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
    
    def execute(self, inputs: MyInput, context=None) -> MyOutput:
        # 实现你的逻辑
        return MyOutput(result="处理完成")
```

模块会自动注册，立即可用！

## 💡 最佳实践

1. **模块化思维** - 将复杂流程拆分成独立步骤
2. **配置驱动** - 使用YAML配置而不是硬编码
3. **工作流复用** - 创建可重用的工作流模板
4. **版本管理** - 为工作流和配置指定版本号
5. **错误处理** - 利用模块的验证和错误处理机制

## 🔄 从旧版本迁移

旧代码：
```python
from hydrosis import HydroSISModel, ModelConfig
config = ModelConfig.from_yaml("config.yaml")
model = HydroSISModel.from_config(config)
results = model.run(forcing)
```

新代码（推荐）：
```python
from hydrosis.workflow_engine import WorkflowEngine, WorkflowDefinition
from pathlib import Path

engine = WorkflowEngine()
workflow = WorkflowDefinition.from_yaml(Path("workflow.yaml"))
run = engine.execute(workflow)
```

## 🤝 贡献

欢迎贡献！可以：

- 添加新的功能模块
- 创建工作流模板
- 改进文档
- 报告问题

## 📄 许可证

[项目许可证]

## 🙋 支持

- 文档: [docs/](docs/)
- 示例: [examples/](examples/)
- Issues: [GitHub Issues]

---

**HydroSIS Team** | 版本 1.0.0 | 更新于 2025-10-26
