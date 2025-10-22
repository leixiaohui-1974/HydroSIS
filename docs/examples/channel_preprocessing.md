# 河道断面前处理模块（前处理阶段的一部分）

本节聚焦于前处理阶段中的“河道断面与中心线准备”任务，利用 `hydrosis.analysis.channel_profile`
中的工具将 DEM 断面数据转换为模型可用的几何输入。该成果被 `docs/product_pipeline.md` 归入前处理阶段，
并由全功能测试 `hydrosis.testing.full_feature_runner` 的“断面前处理”“中心线修正”任务覆盖验证。

## 输入数据

| 数据源 | 说明 |
| --- | --- |
| `main_channel_cross_sections.csv` | 含站距 `station_m`、横向偏移 `distance_from_center_m`、高程 `elevation_m`、分区 `zone_id` 等原始采样点 |
| `main_channel_segments.csv` | 主干分段长度、累计长度、坡度等信息，用于计算全局沿程 |

## 核心步骤

1. **噪声剔除 (`drop_flat_sections`)**：对同一 `zone_id + station_m` 组合，若高程极差小于 `min_variation`（默认 0.1 m），判定为 DEM 台阶或无效采样并剔除，消除锯齿噪声。
2. **全局沿程映射 (`attach_global_station`)**：利用分段表的 `length_m / cumulative_length_m` 计算每段起点，将 `station_m` 转换为跨分区连续的 `global_station_m`，所有后续热力图与纵剖均使用该沿程。
3. **配置记录 (`ChannelProfileConfig`)**：将断面最小变幅、中心线带宽、网格步长等参数封装在 `PreprocessResult.config`，供模型阶段与后处理共享。

## 输出

`preprocess_cross_sections` 返回 `PreprocessResult`：

```python
PreprocessResult(
    raw_sections=<原始 DataFrame>,
    cross_sections=<清洗并生成 global_station_m 的 DataFrame>,
    segments=<分段 DataFrame>,
    zones=<分区列表>,
    config=<ChannelProfileConfig>
)
```

其中 `cross_sections` 已持有 `global_station_m`、`raw_elevation_m` 等字段，是 `run_channel_profile_model`
的直接输入。若需了解模型阶段的单调修正流程，请参见 `docs/examples/channel_profile_model.md`。

**相关测试**：`pytest tests/test_full_feature_runner.py -k pipeline`（间接通过综合测试执行）。
