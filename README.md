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

---

# HydroSIS 项目全景文档

HydroSIS 是一个面向多情景建模与调度分析的分布式水文模拟框架。项目以结构化配置驱动，涵盖流域划分、产流与汇流模块、参数分区、情景模拟、精度评价、可视化与自然语言报告生成，并提供轻量的 Web 门户与大模型接入能力。本文档从原理、代码实现逻辑、接口形式、调用方式、前端界面和大模型入口等方面，对各功能模块进行逐一说明。

---

## 1. 总体架构

- **核心包 `hydrosis/`**：包含模型配置 (`config.py`)、核心模型 (`model.py`)、工作流调度 (`workflow.py`)、产流 (`runoff/`)、汇流 (`routing/`)、参数分区 (`parameters/`)、评价 (`evaluation/`)、报告 (`reporting/`)、输入输出 (`io/`)、工具 (`utils/`) 等模块。
- **门户子系统 `hydrosis/portal/`**：提供基于类 FastAPI 的 REST API、事件流、运行调度、会话管理等功能，并配备静态网页前端。
- **快速体验脚本 `examples/`**：演示端到端建模、情景模拟、GIS 报告生成等流程。
- **轻量 FastAPI 兼容层 `fastapi/`**：为离线环境提供最小化的 Web 框架接口，使门户和测试在无三方依赖时也能运行。

各模块以 `ModelConfig` 为纽带：通过 YAML/JSON 配置描述流域划分、产流/汇流模型、参数分区、情景与 I/O 路径，再由 `HydroSISModel` 按需实例化各组件并执行仿真；`workflow.run_workflow` 则在此基础上封装了基准与情景运行、输出持久化、评价与报告生成的完整链路。

---

## 2. 配置系统 (`hydrosis/config.py`)

### 原理与结构

- `ModelConfig` 聚合 `DelineationConfig`（流域划分）、`RunoffModelConfig`、`RoutingModelConfig`、`ParameterZoneConfig`、`IOConfig`、`ScenarioConfig`、`EvaluationConfig` 等子配置。
- `from_yaml` / `from_dict` 方法负责解析 YAML/JSON，填充 dataclass 实例；`to_dict` 反向导出便于持久化或门户存储。
- `apply_scenario` 根据情景的 `modifications`，遍历模型中的 `Subbasin`，调用 `update_parameters` 写入参数调整（如替换产流/汇流模型、调节参数等）。

### 使用方式

```python
from hydrosis import ModelConfig
config = ModelConfig.from_yaml(Path("config/example_model.yaml"))
config.apply_scenario("reservoir_reoperation", model.subbasins.values())
```

---

## 3. 流域划分 (`hydrosis/delineation/`)

### 3.1 DEM 自动划分 (`dem_delineator.py`)

- **原理**：优先读取 `precomputed_subbasins`，否则依据 DEM 栅格、汇水点（CSV/GeoJSON）、可选的烧蚀河网，通过 `rasterio + richdem + numpy` 执行填洼 (`FillDepressions`)、流向 (`FlowDirD8`)、汇流 (`FlowAccumulation`)，再利用富水点生成子流域。
- **核心流程**：
  1. `_load_pour_points`：将经纬度映射到栅格索引。
  2. `_watershed_mask`：对每个控制点生成流域掩膜。
  3. `_infer_downstream_relationships`：根据 D8 方向推断上下游关系。
  4. 生成 `Subbasin` 实例列表，记录面积、下游链接。
  5. 输出派生的 GeoJSON（子流域、多边形、累积流量）供 GIS 报告使用。
- **轻量回退**：当缺少上述依赖或配置为 JSON DEM 时，调用 `simple_grid.py` 中的纯 Python 算法（D8 方向、累积、子流域分配、边界框多边形）完成划分。

### 3.2 `SimpleGridDEM`

- 数据结构包括 `GridTransform`（仿射坐标转换）、`SimpleGridDEM`（二维高程数组），用于无依赖环境。
- 提供 `compute_flow_directions`、`delineate_watersheds`、`downstream_relations` 等辅助函数。

---

## 4. 产流模块 (`hydrosis/runoff/`)

所有产流模型继承自 `RunoffModel`，接受 `parameters` 构造，在 `simulate(subbasin, precipitation)` 中返回子流域的日（或时间步）产流序列。`RunoffModelConfig.REGISTRY` 持续注册可用模型，配置文件通过 `model_type` 指向具体实现。

### 4.1 SCS 曲线数法 (`scs_curve_number.py`)

- **原理**：根据曲线数 CN 计算最大滞留量 `S`，初损比例 `λ`，对降雨序列逐日判断是否产生径流，符合条件时按 SCS 公式计算。
- **参数**：`curve_number`、`initial_abstraction_ratio`。
- **应用**：适合中小流域快速产流估算。

### 4.2 线性水库 (`linear_reservoir.py`)

- **原理**：单一线性水库状态方程 `state = state * recession + precipitation * conversion`；直达径流为 `(1 - recession) * state`。
- **参数**：`recession` 衰减系数、`conversion` 降雨转化率、`initial_storage` 初始状态。
- **特点**：结构简单，适用于入流-出流之间近似线性关系的场景。

### 4.3 HYMOD (`hymod.py`)

- **原理**：包含土壤非线性蓄水、快速串联水库链、慢速水库等组成，与原 HyMOD 概念模型一致。
- **关键函数**：
  - `_effective_rain`：按土壤蓄水曲线计算有效雨。
  - `_route_quickflow`：多级线性水库路由快速流。
- **参数**：最大土壤水 `max_storage`、曲线指数 `beta`、快慢流分配与衰减系数、快速水库数量等。

### 4.4 XinAnJiang (`xinanjiang.py`)

- **原理**：采用张力水容量曲线、非线性入渗、地表不透水比例、线性基流衰减，模拟华南典型流域产流。
- **参数**：`wm` 最大张力水、`b` 曲线系数、`imp` 不透水率、`recession` 地下水衰减。

### 4.5 WETSPA (`wetspa.py`)

- **原理**：分层水量平衡（表层-土壤-地下），根据入渗系数、土壤储量、渗漏、基流系数等，模拟降雨产流、渗透与地下回补。
- **参数**：`soil_storage_max`、`infiltration_coefficient`、`surface_runoff_coefficient`、`percolation_coefficient`、`baseflow_constant` 等。

### 4.6 VIC (`vic.py`)

- **原理**：三层土壤蓄水 + ARNO 可变入渗曲线，计算快速径流与地下水补给，再以基流系数输出。
- **参数**：`infiltration_shape`、`max_soil_moisture`、`baseflow_coefficient`、`recession` 及各层初始水量。

### 4.7 HBV (`hbv.py`)

- **原理**：仿 HBV 模型的雪蓄、土壤、地下水多仓结构，支持度日融雪、土壤蓄水曲线、两段线性地下水泄流。
- **参数**：`degree_day_factor`、`field_capacity`、`beta`、三段水库的系数与初值等。

---

## 5. 汇流模块 (`hydrosis/routing/`)

与产流类似，所有汇流模型继承 `RoutingModel`，`route(subbasin, inflow)` 输出局地汇流过程。

### 5.1 滞后模型 (`lag.py`)

- **原理**：仅对入流序列延迟 `lag_steps` 个时间步，实现纯平移。
- **使用场景**：简单延迟效应。

### 5.2 Muskingum (`muskingum.py`)

- **原理**：经典的 Muskingum-K 法，计算系数 `C0、C1、C2`，迭代更新输出。
- **参数**：`travel_time`、`weighting_factor`、`time_step`。

### 5.3 动力波 (`dynamic_wave.py`)

- **原理**：显式差分方案近似动力波 / 动力-运动波混合，利用 Courant 条件及扩散系数进行稳定性控制。
- **参数**：`time_step`、`reach_length`、`wave_celerity`、`diffusivity`。

---

## 6. 参数分区 (`hydrosis/parameters/`)

### 6.1 `ParameterZoneBuilder` (`zone.py`)

- **原理**：根据控制点（往往是水文站或水库）向上游追踪子流域（DFS/BFS），并排除已被上游分区覆盖的单元，实现 **自上而下、不重叠** 的参数区划。
- **结果**：每个 `ParameterZone` 包含 `controllers`、`controlled_subbasins`、`parameters`（如指定产流/汇流模型 ID 及参数值）。

### 6.2 优化与不确定性 (`optimization.py`)

- `ParameterZoneOptimizer`：
  - 接受多个目标 `ObjectiveDefinition(id, weight, sense)`，并依赖外部传入的采样器与评价函数 `evaluation(candidate_params)`。
  - `_draw_candidate` 迭代 `samplers` 拉丁超立方等采样器；`_composite_score` 按目标权重与取向（min/max/minabs）合成分数。
- `UncertaintyAnalyzer`：
  - 重复抽样、聚合 `evaluation` 的指标，计算均值、标准差、最小/最大值。
- **接口**：`optimise(samplers, max_iterations)` 返回 `OptimizationResult`（最优参数、目标分数、历史轨迹）；`analyse(samplers, draws)` 返回各指标统计量。
- **大模型应用**：可由 LLM 生成采样策略或自定义目标函数，再通过回调执行。

---

## 7. 核心模型 (`hydrosis/model.py`)

### 7.1 `Subbasin` 数据结构

- 保存 `id`, `area_km2`, `downstream`, `parameters`（包含产流/汇流模型 ID 以及其他参数）。
- `update_parameters` 简单合并字典，支持情景调整。

### 7.2 `HydroSISModel`

- **构造**：`__init__(subbasins, parameter_zones, runoff_models, routing_models)`。
  - `_assign_zone_parameters`：遍历参数区，将区级参数写入对应子流域。
- **工厂方法**：`from_config(ModelConfig)` 调用配置的 `to_subbasins()`、`ParameterZoneBuilder`、各模型 `build()` 注册。
- **运行流程**：
  1. `run(forcing)`：逐子流域复制相应的产流模型（防止状态共享），读取 `forcing[sub_id]`，生成局地产流；再依次调用 `routing_model.route` 计算局地汇流。
  2. `accumulate_discharge(routed)`：调用 `utils.accumulate_subbasin_flows` 沿网络自上而下累积上游贡献。
  3. `parameter_zone_discharge(routed)`：在累积结果基础上，抽取每个参数区控制点的时序，用于后续分析或报告。
- **异常处理**：若某子流域缺少 `runoff_model` 或 `routing_model` 参数，抛出 `ValueError`。

---

## 8. 网络累积工具 (`hydrosis/utils/network.py`)

- `_ensure_lengths` 验证所有序列长度一致。
- `accumulate_subbasin_flows(subbasins, local_flows)`：
  - 根据 `downstream` 构建入度，使用拓扑排序（队列）从上游到下游累积流量。
  - 遇到未知下游或环路时抛出 `KeyError` / `ValueError`。

---

## 9. 输入输出 (`hydrosis/io/`)

### 9.1 输入 (`inputs.py`)

- `load_time_series(csv_path)`：逐行读取 CSV 最后一列转为浮点数。
- `load_forcing(directory)`：遍历目录下所有 `.csv` 文件，以文件名为子流域 ID，返回 `{sub_id: [values...]}`。

### 9.2 输出 (`outputs.py`)

- `write_time_series(path, values)`：写入索引+值的 CSV。
- `write_simulation_results(directory, results)`：批量写出各子流域的时序。
- `write_markdown(path, content)`：保存 Markdown 报告。

### 9.3 GIS 报告 (`gis_report.py`)

- `LeafletReportBuilder`：构建嵌入 Leaflet 地图的单页 HTML，支持多图层（GeoJSON）、样式控制、弹窗字段、章节化 HTML。
- `gather_gis_layers(config, repo_root)`：
  - 自动读取 DEM、土壤、土地利用、水文站、水库、河网等 GeoJSON。
  - 若 DEM 为 JSON（简化 DEM），调用 `simple_grid` 转换。
  - 自动寻找派生目录 `derived/` 下的 `subbasins.geojson`、`flow_accumulation.geojson`。
- `build_parameter_zone_geojson(subbasins_geojson, zones)`：将控制区内子流域多边形聚合为 MultiPolygon。
- `summarise_paths`、`build_html_table`、`build_card`、`build_bullet_list`、`embed_image` 等辅助函数用于生成富文本报告。
- `accumulation_statistics`：统计流量累积栅格的 min/max/mean。

---

## 10. 评价与报告 (`hydrosis/evaluation/`, `hydrosis/reporting/`)

### 10.1 指标计算 (`evaluation/metrics.py`)

- 提供 `rmse`、`mae`、`percent_bias` (`pbias`)、`nash_sutcliffe_efficiency` (`nse`)；`DEFAULT_METRICS` / `DEFAULT_ORIENTATION` 指明默认指标及目标方向。
- `_validate_lengths` 保证模拟与观测长度一致。

### 10.2 评价管线 (`evaluation/comparison.py`)

- `SimulationEvaluator`：
  - 构造时可传入自定义指标及方向。
  - `evaluate_series`、`evaluate_catchment` 分别针对单条/全流域序列评估。
- `ModelComparator`：
  - `compare(simulations, observations)`：对多套模拟（字典 `{model_id: {subbasin: series}}`）与观测对比，返回 `ModelScore` 列表（包含分子流域与聚合指标）。
  - `_aggregate_metrics` 默认按均值聚合，`minabs` 指标取绝对值。
  - `rank(scores, metric)` 按指定指标排序。

### 10.3 Markdown 报告 (`reporting/markdown.py`)

- `MarkdownReportBuilder`：封装标题、段落、列表、表格、图片、水平线等添加方法，`to_markdown()` 输出最终文本，`write(path)` 落盘。
- `summarise_aggregated_metrics`：汇总 `ModelScore` 为表格。
- `_generate_metric_figures` / `_generate_hydrograph_figures`：优先使用 Matplotlib 生成 PNG，若缺失则在 `charts.py` 中生成 SVG。
- `generate_evaluation_report`：
  - 撰写“总体指标”“指标图表”“子流域径流对比”等章节。
  - 使用 `ModelComparator` 和 `ranking_metric` 生成排序列表。
  - 当未安装 Matplotlib 时给出提示。
  - 支持传入 `template` (`EvaluationReportTemplate`) 与 `narrative_callback`（大模型接口）生成自然语言段落。

### 10.4 模板与 LLM 接入 (`reporting/templates.py`)

- `EvaluationReportTemplate` 包含三段：`overview`（模型运行概述）、`highlights`（关键发现）、`next_steps`（后续建议）。
- `render_template(builder, template, context, narrator)`：
  - 优先使用 `context` 中同名段落（用于预填确定性文字）。
  - 若缺失且提供 `narrator`，则将 `prompt` 交由大模型生成文本。
- **大模型入口**：任何支持函数 `narrative_callback(prompt: str) -> str` 均可插入，如 `lambda prompt: llm.generate(prompt)`。

### 10.5 图表回退 (`reporting/charts.py`)

- 仅当安装 `matplotlib` 时使用其绘图；否则输出 SVG 字符串并落盘。
- `plot_hydrograph` / `plot_metric_bars` 内建调色板、网格、图例、数值标签。

---

## 11. 工作流 (`hydrosis/workflow.py`)

### 11.1 数据结构

- `ScenarioRun`：包含 `scenario_id`、局地/累积/参数区流量。
- `EvaluationOutcome`：比较方案结果（`ComparisonPlanConfig`、`scores`、`ranking`、`ranking_metric`）。
- `WorkflowResult`：基准情景、情景字典、整体指标、比较方案结果、报告路径。

### 11.2 执行流程 `run_workflow`

1. **准备阶段**：
   - `_instantiate_model`：基于配置实例化 `HydroSISModel`。
   - 若传入 `scenario_ids`，按顺序遍历；否则执行全部配置中的情景。
   - `progress_callback`（可选）向外部通报阶段（start/complete）及附加信息。

2. **基准运行**：
   - `baseline_model.run(forcing)` -> `accumulate_discharge` -> `parameter_zone_discharge`。
   - 结果打包为 `ScenarioRun`。

3. **情景运行**：
   - 每个情景复制配置 (`copy.deepcopy(config)`)，重新实例化模型，调用 `apply_scenario` 更新子流域参数，再运行。

4. **输出持久化**（可选 `persist_outputs`）：
   - `write_simulation_results` 将基准与情景的累积流量保存至 `config.io.results_directory / scenario_id`。

5. **评价与报告**（传入 `observations` 时执行）：
   - 构建 `SimulationEvaluator`（如自定义指标需在 `EvaluationConfig.metrics` 中声明）。
   - `ModelComparator.compare` 得到 `overall_scores`（基于观测）。
   - 对每个 `ComparisonPlanConfig`（例如基准 vs 情景）调用 `_evaluate_plan`，可筛选子流域、指定参考（观测或其他模型）。
   - 构建 `report_context`（如模型数量、指标列表、排序结果等），若 `generate_report=True` 则生成 Markdown 报告，路径为 `reports_directory/evaluation.md`。
   - `narrative_callback`、`report_template`、`template_context` 结合报告模板生成自然语言段落。

6. **返回**：封装为 `WorkflowResult`。

### 11.3 大模型结合点

- `run_workflow(..., narrative_callback=llm_function, template_context={...})`：可直接连接 LLM 生成报告段落。
- `progress_callback` 可驱动门户的实时进度事件流。

---

## 12. 门户系统 (`hydrosis/portal/`)

### 12.1 总体架构

- `create_app` (`main.py`) 返回 FastAPI 兼容应用，内部依赖：
  - `PortalState`（内存或 SQLAlchemy 后端）保存项目、输入、情景、运行、会话。
  - `IntentParser`（规则式自然语言解析）。
  - `RunEventBroker` + `RunExecutor`：后台线程执行工作流、发布事件。
- 支持环境变量/配置文件设置数据库 URL，自动回退到内存存储。

### 12.2 REST 接口

| 接口 | 说明 | 主要逻辑 |
|------|------|----------|
| `POST /conversations/{id}/messages` | 对话消息 | 保存用户消息，调用 `IntentParser.parse` 得到意图，生成内置回复（可替换为 LLM），返回会话全量消息。 |
| `GET /projects` / `GET /projects/{id}` | 列出项目 / 获取详情 | 直接访问 `PortalState`。 |
| `POST /projects/{id}/config` | 上传 `ModelConfig` | JSON -> `ModelConfig.from_dict` -> `PortalState.upsert_project`。 |
| `GET/POST /projects/{id}/inputs` | 维护 forcing/observations | 校验 JSON，存储为 `ProjectInputs`，返回更新时间及统计。 |
| `GET /projects/{id}/overview` | 项目总览 | 汇总情景数量、输入统计 (`_series_summary`)、最近运行及摘要。 |
| `GET/POST/PUT/DELETE /projects/{id}/scenarios` | 情景 CRUD | 调用 `PortalState.add/update/remove_scenario`。 |
| `POST /projects/{id}/runs` | 创建运行 | 若请求中包含 forcing/observations 则更新输入；否则使用已存储数据。调用 `RunExecutor.submit` 异步执行。 |
| `GET /projects/{id}/runs`, `GET /runs`, `GET /runs/{id}` | 查看运行记录 | 返回状态、错误、结果摘要。 |
| `GET /runs/{id}/stream` | 事件流 (SSE) | 订阅 `RunEventBroker`，实时推送阶段消息、百分比、摘要。 |
| `GET /runs/{id}/report` | 下载 Markdown 报告 | 读取文件内容返回 JSON。 |
| `GET /runs/{id}/figures` | 列出图表文件 | 查找报告目录邻近的 `figures/`。 |
| `GET /runs/{id}/summary` | 调度结果摘要 | `summarise_workflow_result`：包含基准/情景统计、与基准差异、指标、文字总结。 |
| `GET /runs/{id}/timeseries` | 提取局地/累积/参数区时序 | 通过 `serialize_scenario_run` 序列化。 |
| `GET /runs/{id}/evaluation` | 评价指标与排名 | 序列化 `ModelScore`、`EvaluationOutcome`。 |

### 12.3 状态管理 (`state.py`)

- `PortalState` 协议定义对话、项目、输入、情景、运行等操作。
- `InMemoryPortalState`：使用 Python 字典+锁实现；运行结果直接存储 `WorkflowResult`。
- 序列化/反序列化函数：`serialize_workflow_result`、`deserialize_workflow_result` 等，将 `WorkflowResult` 与 `ModelScore` 等转为 JSON 兼容结构。
- `RunRecord`：保存运行 ID、项目 ID、情景列表、创建时间、状态、结果/错误。

### 12.4 SQLAlchemy 后端 (`storage/sqlalchemy.py`)

- 定义数据库模型 `ProjectModel`、`ScenarioModel`、`ProjectInputModel`、`RunModel`。
- `create_sqlalchemy_state(database_url)` 构造引擎、建表、返回 `SQLAlchemyPortalState`。
- `upsert_project` 会将 `ModelConfig` 的基础配置存入 JSON，并分离情景存表；`list_projects`、`set_inputs`、`add_scenario` 等操作均通过 ORM 实现。
- `RunModel.result` 字段保存序列化的 `WorkflowResult`，取出时调用 `deserialize_workflow_result` 恢复对象。
- `migrations.py` 提供 CLI：`python -m hydrosis.portal.storage.migrations --database-url sqlite:///portal.db`。

### 12.5 运行执行 (`executor.py`)

- `RunExecutor.submit`：
  - 立即发布 `queue` 状态事件。
  - 后台线程中调用 `_workflow_runner`（默认为 `_execute_workflow`）执行；执行过程中 `progress_callback` 将 `run_workflow` 阶段事件（baseline/scenario/evaluation/persistence/report/workflow）转换为 `RunProgressEvent` 并推送。
  - 完成后生成摘要 `summarise_workflow_result`，更新状态（completed/failed）。
- `ProgressTracker`：跟踪基准、情景、评价阶段完成数量，计算百分比。

### 12.6 自然语言解析 (`llm.py`)

- `IntentParser`：规则匹配 `run`/`simulate`/`运行`、`create scenario`、`list scenario`、`report` 等关键字，提取情景 ID 或名称，输出结构化意图 `{action, parameters, confidence}`。
- 可被大模型替换：在 `create_app` 中注入自定义解析器即可。

### 12.7 分析摘要 (`analytics.py`)

- `_summarise_scenario_run`：计算子流域/参数区累积量、平均值、峰值及出现位置。
- `_compute_deltas`：与基准相比的体积/均值/峰值变化。
- `_build_narrative`：根据差异生成段落，供门户前端展示。

### 12.8 前端界面 (`portal/static/index.html`)

- **布局**：多个 `section` 分别用于对话入口、项目配置、水文输入、情景管理、触发模拟与实时进度/结果查看。
- **主要交互**：
  - 通过 `fetch` 与 API 通信，自动维护项目列表、情景列表、输入、运行记录。
  - 运行表单支持指定情景、是否持久化、是否生成报告；发起后自动连接 `/runs/{id}/stream` 监听进度。
  - 运行完成时自动刷新概览、摘要、时序、评估结果。
- **实时进度**：使用 SSE (`EventSource`)，展示进度条、文字状态、滚动日志。
- **数据展示**：多个 `pre` 标签显示 JSON，便于调试；可拓展为更丰富的 UI。

---

## 13. 快速示例 (`examples/`)

### 13.1 `run_sample_workflow.py`

- 演示如何加载示例配置（若缺少 PyYAML 则使用 JSON 备份），自动生成/读取观测数据，运行 `run_workflow`，输出基准与情景汇总。
- 构建 GIS 汇总报告：统计子流域面积、参数区覆盖、流量峰值等，调用 `LeafletReportBuilder` 生成 HTML，并根据是否安装 Matplotlib 绘制条形图/情景对比图。
- 演示 LLM 接入：`narrative_callback=lambda prompt: f"（示例 LLM 输出）{prompt}"`。

### 13.2 `run_gis_demo.py`

- 专注于 GIS 报告生成：加载 `config/gis_demo.json`，运行基准情景，将结果导入 Leaflet 报告。

---

## 14. 轻量 FastAPI 兼容层 (`fastapi/`)

- `app.py`：实现 `FastAPI` 类（记录路由表 `_Route`），以及 `HTTPException`、`Response`。`handle_request` 使用正则匹配路径参数，并支持 dataclass / Pydantic 请求体实例化。用于测试环境模拟最小 Web 框架行为。
- `responses.py`：定义 `HTMLResponse`、`StreamingResponse`。
- `staticfiles.py`：`StaticFiles(directory)` 仅保存静态目录路径。
- `testclient.py`：实现简单的 `TestClient`，提供 `.get/.post/.put/.delete` 方法，返回 `_ClientResponse`。
- **用途**：使门户及其测试在无真实 FastAPI 依赖时依然可运行，便于离线 CI/单元测试。

---

## 15. 测试与样例数据 (`tests/`, `data/`)

- `tests/test_workflow.py`、`test_reporting.py`、`test_portal_api.py` 等验证各模块功能与 API 行为。
- `data/sample/` 提供降雨/观测/GIS 样例，用于快速体验与报告生成。

---

## 16. 关键调用示例与建议

1. **纯 Python 运行示例**

```python
from pathlib import Path
from hydrosis import ModelConfig, HydroSISModel

config = ModelConfig.from_yaml(Path("config/example_model.yaml"))
model = HydroSISModel.from_config(config)

from hydrosis.io.inputs import load_forcing
forcing = load_forcing(config.io.precipitation)
local = model.run(forcing)
aggregated = model.accumulate_discharge(local)
zone_series = model.parameter_zone_discharge(local)
```

2. **完整工作流（含评价与报告）**

```python
from hydrosis import run_workflow, default_evaluation_template

result = run_workflow(
    config,
    forcing,
    observations=load_forcing(config.io.discharge_observations),
    persist_outputs=True,
    generate_report=True,
    report_template=default_evaluation_template(),
    narrative_callback=lambda prompt: llm.generate(prompt),
)
print(result.report_path)
```

3. **门户部署**

```bash
# 启动静态前端
python -m http.server 8000 --directory hydrosis/portal/static

# 在另一个终端启动 API（使用内存态）
python - <<'PY'
from hydrosis.portal import create_app
from fastapi.testclient import TestClient

app = create_app()
client = TestClient(app)
print(client.get("/projects").json())
PY
```

（生产环境可将 `create_app(database_url="sqlite:///portal.db")` 与真实 FastAPI/Uvicorn 搭配。）

---

## 17. 大模型集成要点

- **报告生成**：在 `run_workflow` 或 `generate_evaluation_report` 传入 `narrative_callback`，即可把模板 prompt 交给大模型生成中文摘要。
- **自然语言门户**：`IntentParser` 可被大模型替换，以获得更智能的对话解析；`Conversation` 机制支持保存大模型产生的意图与回复。
- **参数优化/不确定性**：通过大模型设计采样策略（如多目标权重、拉丁超立方采样方案），再调用 `ParameterZoneOptimizer` / `UncertaintyAnalyzer` 运行。
- **配置生成**：`ModelConfig` 的 YAML/JSON 结构友好，方便 LLM 根据自然语言描述自动构建/修改情景、参数区等配置。

---

## 18. 界面与交互总结

- **门户前端**：
  - 支持通过表单粘贴 JSON、触发模拟、查看实时进度/日志、获取摘要/时序/评价等。
  - 对话区域展示意图解析结果，提示用户调用相应 API。
  - 可扩展为更丰富的仪表盘、图表展示或集成大模型聊天窗口。

- **GIS 报告**：
  - 结合 Leaflet 渲染 DEM/土壤/土地利用/站点/水库/河网/子流域/参数区，配合 HTML 卡片、表格、图片，形成完整的空间分析页面。

---

## 19. 扩展建议

- 如需引入真实 FastAPI，可在 `requirements` 中添加并替换兼容层。
- 大模型集成可扩展为：
  - 将 `IntentParser` 替换为调用 LLM 的服务。
  - 在报告模板中加入更多业务段落。
  - 通过 LLM 生成参数优化策略（如自动设定目标权重、采样范围）。
- GIS 层可与真实 DEM/河网数据集对接，结合 `rasterio`/`richdem` 的自动划分能力。

---

通过以上模块化设计，HydroSIS 可以在多种环境下（离线/在线、轻量/丰富依赖）完成分布式水文建模的全流程，且易于与大模型、Web 门户、GIS 可视化等生态组件联动。
