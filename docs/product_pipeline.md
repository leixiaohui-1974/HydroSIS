# HydroSIS 三阶段产品体系说明

HydroSIS 的端到端流程被拆分为三个核心阶段：

1. **前处理（Preprocessing）**：准备地形、雨量、河道几何与参数等模型输入；
2. **模型计算（Modeling）**：执行水文产流、河道/湖泊水动力以及项目情景仿真；
3. **后处理（Postprocessing）**：对模拟结果进行可视化、指标统计与成果发布。

`hydrosis.pipeline` 模块提供了轻量级调度器，可以将以上阶段的任务串联执行，并通过上下文共享数据。下文给出每个阶段的主要功能、关键代码与参考文档（示例文档位于 `docs/examples/` 目录）。

---

## 阶段一：前处理

| 功能 | 关键模块 | 说明 | 参考文档 |
| --- | --- | --- | --- |
| 流域划分、河网提取 | `hydrosis.delineation.dem_delineator`、`hydrosis.delineation.simple_grid` | 处理 DEM、提取河网/子流域、建议汇流阈值，输出 GeoJSON、栅格及统计 | `docs/examples/delineation_diagnostics.md` |
| 雨量站空间权重与时序 | `hydrosis.precipitation.thiessen` 及相关脚本 | 构建泰森多边形、插值站点时序、生成面雨量与子流域面平均降雨 | `docs/examples/channel_preprocessing.md`（河道部分）、`docs/examples/upper_truckee_project.md` |
| 河道断面与中心线修正 | `hydrosis.analysis.channel_profile` | `preprocess_cross_sections` + `run_channel_profile_model`，为水动力模型准备单调中心线与修正断面 | `docs/examples/channel_preprocessing.md`、`docs/examples/channel_profile_model.md` |
| 参数赋值 | `hydrosis.parameters.*`、项目脚本 | 依据土地利用、土壤与分区生成水文/水力参数表 | `docs/examples/upper_truckee_project.md` |

> 中心线修正属于前处理阶段，其输出直接作为水动力建模的河床输入。

示例 StageTask 配置：

```python
preprocessing_tasks = [
    StageTask("delineate_watershed", delineate_watershed, args=(delineation_cfg,), store_output_as="delineation"),
    StageTask("build_precip_weights", compute_thiessen, use_context=True, store_output_as="rain_weights"),
    StageTask(
        "prepare_channel_profiles",
        _pipeline_preprocess_cross_sections,
        args=(cross_sections_path, segments_path, ["P3", "P4"], ChannelProfileConfig()),
        store_output_as="channel_preprocess",
    ),
    StageTask("apply_channel_corrections", _pipeline_run_channel_profile, use_context=True, store_output_as="channel_profile"),
]
```

---

## 阶段二：模型计算

| 功能 | 关键模块 | 说明 | 参考文档 |
| --- | --- | --- | --- |
| 水文产流模型 | `hydrosis.runoff`（HBV、SCS Curve Number、VIC、Linear Reservoir 等） | 依据参数与降雨时序计算子流域径流过程 | `docs/examples/extended_runoff_models.md` |
| 河道/湖泊水动力 | `hydrosis.routing`、`hydrosis.model`、`hydrosis.workflow` | Muskingum、Lag、Dynamic Wave 等河道演进，支持项目/情景批量运行 | `docs/examples/upper_truckee_project.md` |
| 情景评估与对比 | `hydrosis.workflow.run_workflow`、`hydrosis.evaluation` | 生成情景结果、计算 RMSE/MAE/NSE 等指标、输出排序 | `docs/examples/multi_model_storm_comparison.md` |

流水线 StageTask 示例：

```python
modeling_tasks = [
    StageTask("run_hydrologic_models", run_hydrologic_model, use_context=True, store_output_as="runoff"),
    StageTask("run_channel_routing", run_hydraulic_model, use_context=True, store_output_as="hydraulic"),
    StageTask("run_workflow", _pipeline_execute_workflow, use_context=True, kwargs={"scenario_ids": ["alternate_routing"]}, store_output_as="workflow_result"),
]
```

---

## 阶段三：后处理

| 功能 | 关键模块 | 说明 | 参考文档 |
| --- | --- | --- | --- |
| 河道成果可视化 | `examples/plot_channel_terrain.py`、`hydrosis.visualization` | 基于中心线与断面生成 3D 网格、热力图、纵剖 | `docs/examples/channel_postprocessing.md` |
| 指标分析与报告 | `hydrosis.analysis`、`hydrosis.reporting.markdown` | 汇总模型指标、生成 Markdown/HTML 报告 | `docs/examples/flood_model_validation.md` |
| 成果发布 | `hydrosis.portal` | 将模拟结果与图表推送至 Web 门户或 API | `docs/examples/upper_truckee_project.md` |

StageTask 示例：

```python
postprocessing_tasks = [
    StageTask("visualise_channel", render_channel_plots, use_context=True),
    StageTask("export_markdown_report", create_markdown_report, use_context=True),
    StageTask("publish_to_portal", publish_to_portal, use_context=True),
]
```

---

## 统筹调度与测试

`hydrosis.testing.full_feature_runner` 提供了一个综合测试程序，会：

1. 创建合成降雨；
2. 运行示例模型与情景评估；
3. 生成 Markdown 测试报告；
4. 执行上述三阶段流水线验证（新增的 “三阶段系统流水线验证” 章节）。

运行命令：

```bash
python -m hydrosis.testing.full_feature_runner --output docs/test_documentation.md
```

生成的报告详见 `docs/test_documentation.md`。

---

## 与示例文档的关联

- **前处理**：详见 `docs/examples/channel_preprocessing.md`（河道断面）、`docs/examples/channel_profile_model.md`、`docs/examples/upper_truckee_project.md`；
- **模型计算**：可参考 `docs/examples/upper_truckee_project.md`、`docs/examples/multi_model_storm_comparison.md`；
- **后处理**：详见 `docs/examples/channel_postprocessing.md` 以及测试生成的报告；
- **流水线**：本页示例的 StageTask 配置可与 `hydrosis/pipeline/stages.py` 配合使用。

通过以上三阶段文档与测试程序，HydroSIS 产品的整体流程与质量保证机制得以完整呈现。
