# HydroSIS 模块化API使用指南

## 目录

1. [快速开始](#快速开始)
2. [模块使用](#模块使用)
3. [工作流使用](#工作流使用)
4. [REST API使用](#rest-api使用)
5. [CLI使用](#cli使用)
6. [Python SDK使用](#python-sdk使用)
7. [配置文件说明](#配置文件说明)

## 快速开始

### 安装

```bash
# 克隆仓库
git clone <repository-url>
cd HydroSIS

# 安装依赖
pip install -r requirements.txt

# 安装HydroSIS
pip install -e .
```

### 最简单的例子

```python
from hydrosis.modules import TerrainModule

# 创建模块实例
terrain = TerrainModule()

# 执行
output = terrain.run({
    "dem_path": "data/dem.tif",
    "method": "d8",
    "output_dir": "results/terrain"
})

print(f"流向文件: {output.flow_direction}")
print(f"流量累积: {output.flow_accumulation}")
```

## 模块使用

### 1. 地形处理模块 (Terrain)

```python
from hydrosis.modules import TerrainModule

terrain = TerrainModule()

# 执行地形处理
output = terrain.run({
    "dem_path": "data/upper_truckee/dem.tif",
    "method": "d8",  # 或 "dinf"
    "fill_depressions": True,
    "compute_slope": True,
    "output_dir": "results/terrain"
})

# 输出结果
print(f"流向: {output.flow_direction}")
print(f"流量累积: {output.flow_accumulation}")
print(f"填充DEM: {output.filled_dem}")
print(f"坡度: {output.slope}")
```

### 2. 汇水点生成模块 (Pour Points)

```python
from hydrosis.modules import PourPointsModule

pour_points = PourPointsModule()

# 自动识别汇水点
output = pour_points.run({
    "flow_accumulation": "results/terrain/flow_accumulation.tif",
    "method": "auto",
    "threshold": 1000.0,
    "output_dir": "results/pour_points"
})

# 手动指定汇水点
output = pour_points.run({
    "flow_accumulation": "results/terrain/flow_accumulation.tif",
    "method": "manual",
    "points": [
        {"id": "P1", "lon": -120.0, "lat": 39.0},
        {"id": "P2", "lon": -119.9, "lat": 39.1}
    ],
    "snap_distance": 500.0,
    "output_dir": "results/pour_points"
})

print(f"汇水点: {output.pour_points_geojson}")
print(f"识别到 {len(output.points)} 个汇水点")
```

### 3. 流域划分模块 (Watershed Delineation)

```python
from hydrosis.modules import WatershedDelineationModule

watershed = WatershedDelineationModule()

output = watershed.run({
    "flow_direction": "results/terrain/flow_direction.tif",
    "pour_points": "results/pour_points/pour_points.geojson",
    "output_format": "geojson",
    "compute_topology": True,
    "output_dir": "results/watersheds"
})

print(f"流域边界: {output.watersheds}")
print(f"流域拓扑: {output.topology}")
print(f"流域面积: {output.areas_km2}")
```

### 4. 产流模拟模块 (Runoff Generation)

```python
from hydrosis.modules import RunoffGenerationModule

runoff = RunoffGenerationModule()

# 使用HBV模型
output = runoff.run({
    "precipitation": "results/areal_precip/areal_precip.csv",
    "watersheds": "results/watersheds/watersheds.geojson",
    "model": "hbv",
    "parameters": {
        "field_capacity": 258.12,
        "beta": 3.0,
        "k0": 0.05,
        "k1": 0.01,
        "k2": 0.017,
        "percolation": 5.0
    },
    "output_dir": "results/runoff"
})

print(f"径流序列: {output.runoff_timeseries}")
```

### 5. 汇流演算模块 (Routing)

```python
from hydrosis.modules import RoutingModule

routing = RoutingModule()

output = routing.run({
    "runoff": "results/runoff/runoff.csv",
    "watersheds": "results/watersheds/watersheds.geojson",
    "method": "muskingum",
    "parameters": {
        "travel_time": 10.0,
        "weighting_factor": 0.1
    },
    "output_dir": "results/routing"
})

print(f"流量序列: {output.discharge_timeseries}")
```

## 工作流使用

### 方式1: 使用预定义模板

```python
from hydrosis.workflow_engine import WorkflowEngine, WorkflowTemplates

# 创建引擎
engine = WorkflowEngine()

# 加载模板
template = WorkflowTemplates.get_template("pour_points_only")
from hydrosis.workflow_engine import WorkflowDefinition
workflow = WorkflowDefinition.from_dict(template)

# 执行工作流
run = engine.execute(workflow, parameters={
    "dem_path": "data/upper_truckee/dem.tif",
    "output_dir": "results/my_workflow",
    "threshold": 1500.0
})

print(f"工作流状态: {run.status}")
print(f"进度: {run.progress_percent()}%")
print(f"输出: {run.outputs}")
```

### 方式2: 从YAML文件加载

```python
from pathlib import Path
from hydrosis.workflow_engine import WorkflowEngine, WorkflowDefinition

engine = WorkflowEngine()

# 从文件加载
workflow = WorkflowDefinition.from_yaml(
    Path("config/workflows/complete_simulation.yaml")
)

# 执行
run = engine.execute(workflow, parameters={
    "dem_path": "data/dem.tif",
    "output_dir": "results/run_001"
})
```

### 方式3: 动态构建工作流

```python
from hydrosis.workflow_engine import WorkflowDefinition, WorkflowStep

workflow = WorkflowDefinition(
    id="custom_workflow",
    name="自定义工作流",
    version="1.0",
    parameters={"dem_path": "data/dem.tif"},
    steps=[
        WorkflowStep(
            id="terrain",
            module="terrain",
            inputs={"dem_path": "${parameters.dem_path}"}
        ),
        WorkflowStep(
            id="pour_points",
            module="pour_points",
            depends_on=["terrain"],
            inputs={
                "flow_accumulation": "${steps.terrain.outputs.flow_accumulation}"
            }
        )
    ]
)

run = engine.execute(workflow)
```

## REST API使用

### 启动API服务器

```bash
# 方式1: 使用uvicorn
uvicorn hydrosis.api.rest:app --host 0.0.0.0 --port 8000

# 方式2: 使用Python
python -m hydrosis.api.rest

# 访问API文档
# http://localhost:8000/docs
```

### API端点示例

#### 列出所有模块

```bash
curl http://localhost:8000/api/v1/modules
```

#### 获取模块信息

```bash
curl http://localhost:8000/api/v1/modules/terrain
```

#### 执行模块

```bash
curl -X POST http://localhost:8000/api/v1/modules/terrain/execute \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": {
      "dem_path": "data/dem.tif",
      "method": "d8",
      "output_dir": "results/terrain"
    }
  }'
```

#### 执行工作流

```bash
curl -X POST http://localhost:8000/api/v1/workflows/pour_points_only/execute \
  -H "Content-Type: application/json" \
  -d '{
    "parameters": {
      "dem_path": "data/dem.tif",
      "output_dir": "results/workflow"
    }
  }'
```

#### 查询工作流状态

```bash
curl http://localhost:8000/api/v1/workflows/runs/{run_id}
```

## CLI使用

### 模块命令

```bash
# 列出所有模块
hydrosis module list

# 查看模块信息
hydrosis module info terrain

# 执行模块
hydrosis module run terrain \
  --param dem_path=data/dem.tif \
  --param method=d8 \
  --output-dir results/terrain

# 使用JSON配置文件
hydrosis module run terrain --input config/terrain_input.json
```

### 工作流命令

```bash
# 列出所有工作流模板
hydrosis workflow list

# 查看工作流信息
hydrosis workflow info complete_simulation

# 执行工作流
hydrosis workflow run pour_points_only \
  --param dem_path=data/dem.tif \
  --param output_dir=results/workflow

# 使用配置文件
hydrosis workflow run complete_simulation \
  --config config/workflows/complete_simulation.yaml \
  --param dem_path=data/my_dem.tif
```

### 配置验证

```bash
# 验证工作流配置文件
hydrosis config validate config/workflows/my_workflow.yaml
```

## Python SDK使用

### 完整示例

```python
from pathlib import Path
from hydrosis.modules.base import get_registry
from hydrosis.workflow_engine import WorkflowEngine, WorkflowDefinition

# 1. 获取模块注册表
registry = get_registry()

# 2. 列出所有可用模块
print("可用模块:", registry.list_modules())

# 3. 创建并执行单个模块
terrain_module = registry.create_module("terrain")
terrain_output = terrain_module.run({
    "dem_path": "data/dem.tif",
    "output_dir": "results/terrain"
})

# 4. 创建工作流引擎
engine = WorkflowEngine(registry)

# 5. 加载并执行工作流
workflow = WorkflowDefinition.from_yaml(
    Path("config/workflows/complete_simulation.yaml")
)

# 添加进度回调
def progress_callback(run, step_result):
    print(f"[{run.progress_percent():.1f}%] {step_result.step_id}: {step_result.status}")

run = engine.execute(
    workflow,
    parameters={"dem_path": "data/dem.tif"},
    progress_callback=progress_callback
)

# 6. 检查结果
print(f"工作流状态: {run.status}")
print(f"总耗时: {run.duration_seconds():.2f} 秒")
print(f"输出: {run.outputs}")

# 7. 检查每个步骤的结果
for step_id, result in run.step_results.items():
    print(f"  {step_id}: {result.status} ({result.duration_seconds():.2f}s)")
```

## 配置文件说明

### 工作流配置文件结构

```yaml
workflow:
  id: "workflow_id"
  name: "工作流名称"
  version: "1.0"
  description: "工作流描述"
  
  # 全局参数
  parameters:
    dem_path: "data/dem.tif"
    output_dir: "results/"
  
  # 工作流步骤
  steps:
    - id: "step1"
      module: "module_name"
      depends_on: []  # 依赖的步骤
      inputs:
        param1: "${parameters.dem_path}"  # 引用参数
        param2: "${steps.step0.outputs.result}"  # 引用其他步骤输出
    
    - id: "step2"
      module: "another_module"
      depends_on: ["step1"]
      inputs:
        input1: "${steps.step1.outputs.output1}"
  
  # 工作流输出
  outputs:
    final_result: "${steps.step2.outputs.result}"
```

### 变量引用语法

- `${parameters.variable_name}` - 引用全局参数
- `${steps.step_id.outputs.output_name}` - 引用步骤输出

### 模块输入配置

每个模块都有自己的输入参数schema，可以通过以下方式查看：

```bash
# CLI方式
hydrosis module info terrain

# API方式
curl http://localhost:8000/api/v1/modules/terrain/metadata

# Python方式
from hydrosis.modules import TerrainModule
metadata = TerrainModule.metadata()
print(metadata.input_schema)
```

## 高级用法

### 自定义模块

```python
from dataclasses import dataclass
from hydrosis.modules.base import Module, ModuleInput, ModuleOutput, ModuleMetadata, register_module

@dataclass
class MyModuleInput(ModuleInput):
    input_param: str

@dataclass
class MyModuleOutput(ModuleOutput):
    output_result: str

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
            description="自定义功能模块",
            version="1.0.0",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
    
    def execute(self, inputs: MyModuleInput, context=None) -> MyModuleOutput:
        # 实现模块逻辑
        result = f"处理 {inputs.input_param}"
        return MyModuleOutput(output_result=result)
```

### 并行执行工作流

工作流引擎会自动识别无依赖关系的步骤并进行并行执行（未来版本支持）。

### 错误处理和重试

```yaml
steps:
  - id: "unstable_step"
    module: "some_module"
    retry: 3  # 失败后重试3次
    timeout: 3600  # 超时时间（秒）
```

## 常见问题

### Q: 如何查看支持的所有模块？
A: 使用 `hydrosis module list` 或访问 `/api/v1/modules` 端点

### Q: 如何调试工作流？
A: 可以添加进度回调函数，或者查看日志文件

### Q: 如何扩展新模块？
A: 继承 `Module` 基类并使用 `@register_module` 装饰器注册

### Q: 配置文件在哪里？
A: 默认在 `config/workflows/` 目录下

## 更多资源

- [架构设计文档](MODULAR_API_ARCHITECTURE.md)
- [API参考文档](api/)
- [示例代码](../examples/)
- [开发指南](开发指南.md)
