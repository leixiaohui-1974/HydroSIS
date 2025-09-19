# HydroSIS

HydroSIS 是一个面向多情景建模和调度分析的分布式水文模拟框架。框架结构化地组织了流域划分、产流和汇流模块，并提供基于水文站、水库等控制单元的参数分区功能，便于统一调参和情景模拟。

## 核心特性

- **DEM 流域划分**：集成 `rasterio` 与 `richdem`，可直接读取 DEM 和汇水点自动完成子流域划分，同时仍支持注入预处理结果以兼容轻量环境。
- **多种产流算法**：内置 SCS 曲线数法、线性水库、XinAnJiang、WETSPA、HYMOD、VIC、HBV 等概念性产流模块，并保持统一接口便于扩展。
- **多种汇流算法**：提供滞后、Muskingum 及动力波路由，实现从简单桶模型到波动路由的多级组合。
- **参数分区管理**：基于控制点（如水文站、骨干水库）自动划分参数区，确保下游区段仅包含未被上游参数区覆盖的子流域。
- **多目标调参与不确定性分析**：提供参数分区优化器与蒙特卡洛不确定性评估接口，可将 NSE、RMSE 等多指标按权重统筹调参。
- **情景与输入输出配置**：通过 YAML 配置统一描述输入、模型组件、参数区以及情景修改，便于与自然语言建模接口集成。
- **精度评价与多模型对比**：内置 NSE、RMSE、MAE、百分比偏差等指标及模型对比器，可对多参数分区、多子流域情景的结果进行统一评价。
- **结果可视化与报告生成**：提供指标柱状图、径流过程对比图、模板化 Markdown 报告生成，并支持接入大模型生成中文自然语言说明。

## 目录结构

```
hydrosis/
  config.py            # 配置读取与场景应用
  model.py             # 模型核心调度
  delineation/         # DEM 流域划分配置
  runoff/              # 产流模块
  routing/             # 汇流模块
  parameters/          # 参数分区逻辑
  io/                  # 输入输出工具
config/
  example_model.yaml   # 示例配置文件
```

## 快速开始

1. **准备配置文件**：参考 `config/example_model.yaml`，补充 DEM、汇水点、子流域及参数区定义。
2. **加载配置并实例化模型**：

```python
from pathlib import Path
from hydrosis import HydroSISModel, ModelConfig
from hydrosis.io.inputs import load_forcing
from hydrosis.io.outputs import write_simulation_results

config = ModelConfig.from_yaml(Path("config/example_model.yaml"))
model = HydroSISModel.from_config(config)
forcing = load_forcing(Path("data/forcing/precipitation"))

local_results = model.run(forcing)
aggregated = model.accumulate_discharge(local_results)
write_simulation_results(config.io.results_directory, aggregated)

# 可以直接提取参数分区控制点的流量序列：
zone_flows = model.parameter_zone_discharge(local_results)
```

3. **应用情景参数调整**：

```python
config.apply_scenario("reservoir_reoperation", model.subbasins.values())
 results = model.accumulate_discharge(model.run(forcing))
```

4. **开展精度评价、可视化与报告生成**：

```python
from hydrosis import run_workflow

observed = {...}  # 例如由水文站径流观测整理得到

result = run_workflow(
    config,
    forcing,
    observations=observed,
    scenario_ids=["reservoir_reoperation"],
    persist_outputs=True,
    generate_report=True,
)

print(result.overall_scores[0].aggregated)  # 输出基准情景的指标
for outcome in result.evaluation_outcomes:
    print(outcome.plan.id, [score.model_id for score in outcome.ranking])
```

5. **结合报告模板生成自然语言摘要**：

```python
from hydrosis import default_evaluation_template

workflow = run_workflow(
    config,
    forcing,
    observations=observed,
    persist_outputs=True,
    generate_report=True,
    report_template=default_evaluation_template(),
    narrative_callback=lambda prompt: f"（示例 LLM 输出）{prompt}",
)
print("Markdown 报告：", workflow.report_path)
```

### 示例运行与输出

仓库在 `config/example_model.yaml` 中提供了一份可直接运行的配置，并在
`data/sample/forcing` 与 `data/sample/observations` 下准备了合成的降雨、径流观测
数据。执行以下命令即可运行完整流程、生成精度评估与 Markdown 报告：

```bash
python examples/run_sample_workflow.py
```

脚本会自动：

- 根据示例配置加载模型、运行基准情景与水库调度情景；
- 在 `results/example_run/baseline` 与 `results/example_run/reservoir_reoperation`
  下保存累积径流结果；
- 输出各参数分区控制点的汇流过程，并在 `results/example_run/reports`
  中生成 Markdown 格式的评估报告；
- 若缺少观测数据，将以基准情景结果为基础生成轻微扰动的合成观测，写入
  `data/sample/observations` 以便复现实验。

若运行环境未安装 PyYAML，脚本会自动改用 `config/example_model.json`
加载同等配置，因此无需额外依赖即可复现示例流程。

运行结束后可直接查看 `results/example_run` 目录下的 CSV、图表与报告，
用于快速了解模型参数、情景配置及准确性分析的效果。


## 自然语言门户原型

仓库新增了基于最小依赖实现的 Web 门户（`hydrosis.portal`），包含

- `hydrosis/portal/main.py`：提供 REST 风格 API，支持会话解析、项目配置、情景管理与模型运行；
- `hydrosis/portal/static/index.html`：轻量化的前端页面，可直接通过表单与 API 交互；
- `fastapi/`：内置的极简 FastAPI 兼容层，在离线环境也能运行同样的接口定义。

快速体验步骤：

```bash
python -m http.server 8000 --directory hydrosis/portal/static  # 也可自定义部署方式
```

或使用任意 ASGI/WGI 兼容方案加载 `hydrosis.portal.create_app()` 创建的应用。
API 支持的关键资源包括：

- `GET /projects`：列出当前门户中注册的所有项目；
- `POST /projects/{project_id}/config`：保存或更新模型配置；
- `POST /projects/{project_id}/inputs` 与 `GET /projects/{project_id}/inputs`：集中管理项目的入流与观测数据；
- `POST /projects/{project_id}/scenarios` / `PUT` / `DELETE`：创建、更新或删除情景；
- `GET /projects/{project_id}/scenarios`：查看情景清单；
- `POST /projects/{project_id}/runs`：触发建模计算；
- `GET /projects/{project_id}/runs` 与 `GET /runs`：查看运行历史与最新状态；
- `GET /runs/{run_id}`、`/runs/{run_id}/report`、`/runs/{run_id}/figures`：查询结果详情与输出。
- `GET /runs/{run_id}/summary`：提炼基准与情景运行的主要统计量与差异摘要。

静态页面中的“水文输入管理”区域可集中维护降雨/观测序列，之后在“触发模拟”区域提交运行时即可复用，无需重复粘贴时间序列。运行完成后页面会自动填充运行 ID，并可直接查看新增的摘要输出，便于快速理解情景相对于基准的变化。

单元测试 `tests/test_portal_api.py` 展示了如何以编程方式驱动该门户：

```python
from hydrosis.portal import create_app
from fastapi.testclient import TestClient

app = create_app()
client = TestClient(app)
client.post('/projects/demo/config', json={'model': {...}})
client.post('/projects/demo/inputs', json={'forcing': {...}, 'observations': {...}})
client.post('/projects/demo/runs', json={'scenario_ids': ['alternate_routing']})
```

这样即可在无外部依赖的环境下完成自然语言解析、建模运行和结果查询的端到端链路。

## 参数区多目标优化与不确定性分析

`hydrosis.parameters` 包提供 `ParameterZoneOptimizer` 与 `UncertaintyAnalyzer`
接口，可针对配置中定义的参数分区执行多目标搜索与蒙特卡洛分析。

```python
from hydrosis.parameters import (
    ObjectiveDefinition,
    ParameterZoneOptimizer,
    UncertaintyAnalyzer,
)

optimizer = ParameterZoneOptimizer(
    model.parameter_zones,
    evaluation=lambda params: simulate_metrics(model, params),
    objectives=[ObjectiveDefinition(id="rmse", weight=1.0)],
)
best = optimizer.optimise({"Z1": latin_hypercube_sampler}, max_iterations=100)

analyzer = UncertaintyAnalyzer(
    model.parameter_zones,
    evaluation=lambda params: simulate_metrics(model, params),
)
summary = analyzer.analyse({"Z1": bootstrap_sampler}, draws=200)
```

> `simulate_metrics`、`latin_hypercube_sampler` 等函数可按项目需求实现，
> 返回的 `summary` 包含均值、标准差等统计信息，便于评估参数不确定性。

## 报告模板与大模型说明

- `hydrosis.reporting.templates` 提供默认的“模型运行概述—关键发现—后续建议”结构，可按需自定义。
- 将 `narrative_callback` 指向大模型调用函数，即可把模板提示词转换为自然语言段落，实现自动撰写摘要。
- `template_context` 支持预填固定文本，结合模型指标自动生成简明结论，再用大模型补充细节。

## 与大模型集成

- 所有模型组件均以结构化 YAML/JSON 配置描述，便于通过自然语言解析或生成配置。
- 产流、汇流以及参数分区在配置中显式命名，可通过大模型对指定区域进行参数修改、情景设置与报告生成。
- 评估模块提供默认的报告模板与自然语言提示词，可结合大模型生成“模型运行概述”“关键发现”“后续建议”等段落，实现端到端的分析说明。
- 参数优化和不确定性分析接口均接受函数式回调，适合在大模型编排的工作流中根据需求动态定义目标函数与采样策略。
