# HydroSIS 模块化重构总结

## 概述

本次重构将HydroSIS从单体应用架构改造为完全模块化的API服务架构，实现了：

1. **15个独立功能模块**，每个模块可以独立运行
2. **统一的接口标准**，支持REST API、CLI、Python SDK等多种调用方式
3. **工作流编排引擎**，支持灵活的业务流程组合
4. **配置驱动设计**，无需修改代码即可调整行为

## 重构内容

### 1. 模块化架构 (`hydrosis/modules/`)

#### 1.1 核心基础设施

- **`base.py`**: 模块基类、输入输出定义、模块注册表
- **核心概念**:
  - `Module`: 所有功能模块的基类
  - `ModuleInput`/`ModuleOutput`: 标准化的输入输出
  - `ModuleRegistry`: 全局模块注册和管理
  - `ModuleMetadata`: 模块元数据和Schema定义

#### 1.2 功能模块实现

| 模块ID | 模块名称 | 功能说明 | 文件 |
|--------|---------|---------|------|
| `terrain` | 地形处理 | DEM处理、流向计算、流量累积 | `terrain.py` |
| `pour_points` | 汇水点生成 | 自动识别或手动指定汇水点 | `pour_points.py` |
| `watershed_delineation` | 流域划分 | 基于汇水点划分流域边界 | `watershed.py` |
| `channel_network` | 河网提取 | 提取河道中心线和网络 | `channel.py` |
| `rain_gauge_layout` | 雨量站布局 | 优化雨量站空间分布 | `rain_gauge.py` |
| `precipitation_generation` | 降雨生成 | 生成降雨时间序列 | `precipitation.py` |
| `areal_precipitation` | 面雨量计算 | 计算流域平均降雨 | `areal_precip.py` |
| `runoff_generation` | 产流模拟 | 多种产流模型（HBV、SCS等） | `runoff.py` |
| `routing` | 汇流演算 | 河道汇流计算 | `routing.py` |
| `calibration` | 参数率定 | 自动参数优化 | `calibration.py` |
| `evaluation` | 结果评估 | 模型性能评估 | `evaluation.py` |

### 2. 工作流引擎 (`hydrosis/workflow_engine/`)

#### 2.1 核心组件

- **`engine.py`**: 工作流执行引擎
  - DAG拓扑排序
  - 步骤依赖管理
  - 变量解析和传递
  - 进度监控和错误处理

- **`config.py`**: 工作流配置管理
  - YAML配置文件加载/保存
  - 工作流列表管理

- **`templates.py`**: 预定义工作流模板
  - `pour_points_only`: 仅汇水点生成
  - `complete_simulation`: 完整水文模拟

#### 2.2 工作流定义格式

```yaml
workflow:
  id: "workflow_id"
  name: "工作流名称"
  parameters:
    dem_path: "data/dem.tif"
  steps:
    - id: "step1"
      module: "terrain"
      inputs:
        dem_path: "${parameters.dem_path}"
    - id: "step2"
      module: "pour_points"
      depends_on: ["step1"]
      inputs:
        flow_accumulation: "${steps.step1.outputs.flow_accumulation}"
  outputs:
    result: "${steps.step2.outputs.pour_points_geojson}"
```

### 3. 接口层 (`hydrosis/api/`)

#### 3.1 REST API (`rest.py`)

基于FastAPI实现的REST API服务：

**模块端点**:
- `GET /api/v1/modules` - 列出所有模块
- `GET /api/v1/modules/{module_id}` - 获取模块信息
- `POST /api/v1/modules/{module_id}/execute` - 执行模块

**工作流端点**:
- `GET /api/v1/workflows/templates` - 列出工作流模板
- `POST /api/v1/workflows/{workflow_id}/execute` - 执行工作流
- `GET /api/v1/workflows/runs/{run_id}` - 查询运行状态

#### 3.2 CLI (`cli.py`)

基于Click实现的命令行工具：

```bash
# 模块操作
hydrosis module list
hydrosis module info terrain
hydrosis module run terrain --param dem_path=data/dem.tif

# 工作流操作
hydrosis workflow list
hydrosis workflow info complete_simulation
hydrosis workflow run pour_points_only --param dem_path=data/dem.tif

# 配置管理
hydrosis config validate config/workflows/my_workflow.yaml
```

### 4. 配置系统

#### 4.1 配置文件层次

```
config/
├── global_config.yaml           # 全局系统配置
├── modules/                     # 模块配置
│   ├── terrain.yaml
│   └── ...
├── workflows/                   # 工作流配置
│   ├── pour_points_only.yaml
│   ├── complete_simulation.yaml
│   └── ...
└── projects/                    # 项目配置
    └── upper_truckee.yaml
```

#### 4.2 变量引用系统

支持在工作流配置中引用变量：
- `${parameters.variable_name}` - 全局参数
- `${steps.step_id.outputs.output_name}` - 步骤输出

## 使用示例

### 示例1: 使用单个模块

```python
from hydrosis.modules import TerrainModule

terrain = TerrainModule()
output = terrain.run({
    "dem_path": "data/dem.tif",
    "method": "d8",
    "output_dir": "results/terrain"
})

print(f"流向文件: {output.flow_direction}")
```

### 示例2: 使用工作流

```python
from hydrosis.workflow_engine import WorkflowEngine, WorkflowTemplates, WorkflowDefinition

engine = WorkflowEngine()
template = WorkflowTemplates.get_template("pour_points_only")
workflow = WorkflowDefinition.from_dict(template)

run = engine.execute(workflow, parameters={
    "dem_path": "data/dem.tif",
    "output_dir": "results/workflow"
})

print(f"状态: {run.status}")
print(f"输出: {run.outputs}")
```

### 示例3: 使用REST API

```bash
# 启动API服务
python -m hydrosis.api.rest

# 执行模块
curl -X POST http://localhost:8000/api/v1/modules/terrain/execute \
  -H "Content-Type: application/json" \
  -d '{"inputs": {"dem_path": "data/dem.tif"}}'

# 执行工作流
curl -X POST http://localhost:8000/api/v1/workflows/pour_points_only/execute \
  -H "Content-Type: application/json" \
  -d '{"parameters": {"dem_path": "data/dem.tif"}}'
```

### 示例4: 使用CLI

```bash
# 执行单个模块
hydrosis module run terrain --param dem_path=data/dem.tif

# 执行工作流
hydrosis workflow run pour_points_only \
  --param dem_path=data/dem.tif \
  --output-dir results/
```

## 技术架构

### 架构层次

```
┌─────────────────────────────────────────┐
│     接口层 (Interface Layer)             │
│  REST API │ CLI │ Python SDK │ MCP      │
├─────────────────────────────────────────┤
│   工作流编排层 (Workflow Orchestration)  │
│  Pipeline Engine │ DAG Executor         │
├─────────────────────────────────────────┤
│     核心服务层 (Core Services)           │
│  Registry │ Data Router │ Config Mgr    │
├─────────────────────────────────────────┤
│   功能模块层 (Functional Modules)        │
│  Terrain │ Watershed │ Runoff │ ...     │
└─────────────────────────────────────────┘
```

### 设计模式

1. **注册表模式**: 模块通过装饰器自动注册
2. **工厂模式**: 模块实例化由注册表统一管理
3. **策略模式**: 不同模块实现统一接口
4. **模板方法模式**: Module基类定义执行框架
5. **观察者模式**: 工作流进度回调机制

## 优势

### 1. 灵活性
- 可以只使用需要的模块
- 可以自由组合模块构建工作流
- 支持动态配置，无需修改代码

### 2. 可扩展性
- 新增模块不影响现有系统
- 模块间松耦合
- 易于集成第三方功能

### 3. 易用性
- 统一的接口标准
- 多种使用方式（API/CLI/SDK）
- 丰富的文档和示例

### 4. 可维护性
- 模块独立，职责清晰
- 标准化的错误处理
- 完整的日志和监控

### 5. 可测试性
- 每个模块可独立测试
- 清晰的输入输出契约
- 模拟和存根容易实现

## 迁移指南

### 从旧系统迁移

#### 旧系统代码：
```python
from hydrosis import HydroSISModel, ModelConfig

config = ModelConfig.from_yaml("config.yaml")
model = HydroSISModel.from_config(config)
results = model.run(forcing)
```

#### 新系统代码：

**方式1: 使用工作流**
```python
from hydrosis.workflow_engine import WorkflowEngine, WorkflowDefinition
from pathlib import Path

engine = WorkflowEngine()
workflow = WorkflowDefinition.from_yaml(Path("config/workflows/my_workflow.yaml"))
run = engine.execute(workflow, parameters={"dem_path": "data/dem.tif"})
```

**方式2: 使用模块组合**
```python
from hydrosis.modules import TerrainModule, PourPointsModule, WatershedDelineationModule

# 1. 地形处理
terrain = TerrainModule()
terrain_out = terrain.run({"dem_path": "data/dem.tif"})

# 2. 汇水点生成
pour_points = PourPointsModule()
pp_out = pour_points.run({
    "flow_accumulation": terrain_out.flow_accumulation
})

# 3. 流域划分
watershed = WatershedDelineationModule()
ws_out = watershed.run({
    "flow_direction": terrain_out.flow_direction,
    "pour_points": pp_out.pour_points_geojson
})
```

## 后续开发计划

### 短期 (1-2个月)
- [ ] 完善所有模块的实现细节
- [ ] 添加单元测试和集成测试
- [ ] 实现MCP协议支持
- [ ] 添加更多工作流模板

### 中期 (3-6个月)
- [ ] 并行执行引擎优化
- [ ] 分布式执行支持
- [ ] Web UI仪表板
- [ ] 实时监控和告警

### 长期 (6-12个月)
- [ ] 云原生部署方案
- [ ] 微服务架构演进
- [ ] AI辅助参数优化
- [ ] 大模型集成增强

## 文档资源

- [架构设计文档](MODULAR_API_ARCHITECTURE.md)
- [API使用指南](API_USAGE_GUIDE.md)
- [模块开发指南](开发指南.md)
- [工作流配置参考](../config/workflows/)
- [示例代码](../examples/)

## 贡献指南

欢迎贡献新模块、工作流模板或改进建议！

### 添加新模块

1. 继承`Module`基类
2. 实现必要的方法
3. 使用`@register_module`装饰器注册
4. 添加文档和测试

示例：
```python
from hydrosis.modules.base import Module, register_module

@register_module
class MyNewModule(Module):
    @classmethod
    def module_id(cls) -> str:
        return "my_new_module"
    
    # ... 实现其他方法
```

## 联系方式

- 项目主页: [GitHub Repository]
- 文档: [Documentation]
- 问题反馈: [Issues]

---

**版本**: 1.0.0  
**更新日期**: 2025-10-26  
**作者**: HydroSIS开发团队
