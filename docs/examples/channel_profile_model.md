# 河道中心线模型（前处理阶段的延伸）

在完成断面前处理后，通过 `hydrosis.analysis.channel_profile.run_channel_profile_model`
可对 DEM 导出的河道中心线进行单调修正，使之满足水动力建模对河床逐渐下降的物理要求。
该步骤仍属于前处理阶段，并在 `docs/product_pipeline.md` 与全功能测试中占据一席。

## 处理流程

1. **提取中心线 (`extract_centerline`)**：在指定带宽（默认 40 m）内选取最接近河心的采样点，按 `global_station_m`
排序，并通过滚动中位数 + 均值平滑得到 `smoothed_elevation_m`。
2. **单调约束 (`enforce_downhill_trend`)**：施加单调不升约束生成 `local_corrected_elevation_m`。
   参数 `monotonic_tolerance` 控制允许的最小降幅，0 表示严格下降，>0 可保留平台。
3. **跨分区衔接**：依分区顺序遍历，将下游入口压到上游末端，生成 `global_corrected_elevation_m` 与挖槽量
 `global_adjustment_m`。
4. **回写断面**：按横向距离权重将调整量作用于原始断面，得到最终 `elevation_m`（用于水动力模型或可视化）。

## 输出数据结构

`run_channel_profile_model` 返回 `ChannelProfileResult`：

- `centerlines[zone_id]`：包含 `global_station_m`、`base_elevation_m`、`local_corrected_elevation_m`、
  `global_corrected_elevation_m`、`global_adjustment_m` 等列；
- `cross_sections`：在原断面基础上增加 `raw_elevation_m`、`applied_adjustment_m`、`elevation_m` 等列；
- `config`：更新后的 `ChannelProfileConfig`。

## 关联内容

- 后处理示例：`docs/examples/channel_postprocessing.md` 展示如何使用中心线结果生成 3D 网格与纵剖；
- 流水线调用：`hydrosis.testing.full_feature_runner` 的“三阶段系统流水线验证”会自动执行中心线修正并将输出写入报告；
- 产品总览：请参见 `docs/product_pipeline.md` 的“前处理”部分。
