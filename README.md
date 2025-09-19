# HydroSIS

HydroSIS 是一个面向多情景建模和调度分析的分布式水文模拟框架。框架结构化地组织了流域划分、产流和汇流模块，并提供基于水文站、水库等控制单元的参数分区功能，便于统一调参和情景模拟。

## 核心特性

- **DEM 流域划分**：支持读取 DEM、汇水点等数据，在缺少外部库的情况下也可以通过预处理的子流域信息进行建模。
- **多种产流算法**：内置 SCS 曲线数法、线性水库、XinAnJiang、WETSPA、HYMOD 等概念性产流模块，并保持统一接口便于扩展。
- **多种汇流算法**：提供 Muskingum 与滞后汇流等算法，能够按需组合路由单元。
- **参数分区管理**：基于控制点（如水文站、骨干水库）自动划分参数区，确保下游区段仅包含未被上游参数区覆盖的子流域。
- **情景与输入输出配置**：通过 YAML 配置统一描述输入、模型组件、参数区以及情景修改，便于与自然语言建模接口集成。
- **精度评价与多模型对比**：内置 NSE、RMSE、MAE、百分比偏差等指标及模型对比器，可对多参数分区、多子流域情景的结果进行统一评价。
- **结果可视化与报告生成**：提供指标柱状图、径流过程对比图以及 Markdown 报告生成功能，便于开展模型准确性分析与自动化汇报。

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

## 与大模型集成

- 所有模型组件均以结构化 YAML 配置描述，便于通过自然语言解析或生成配置。
- 产流、汇流以及参数分区在配置中显式命名，可通过大模型对指定区域进行参数修改、情景设置与报告生成。
- 示例配置包含 `evaluation` 节，可指示需要关注的指标、子流域及情景对比，为自动报告生成提供结构化输入。

## 下一步扩展建议

- 集成 `rasterio`、`richdem` 等库实现自动 DEM 划分流程。
- 增加更多产流/汇流模块，例如 VIC、HBV、动力波路由等。
- 对参数分区添加多目标优化与不确定性分析接口。
- 提供可视化与报告模板，结合大模型生成自然语言说明。
