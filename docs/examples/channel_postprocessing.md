# 河道后处理与可视化（后处理阶段示例）

当水文水动力计算完成后，需要将结果转化为直观的图件与数据产出。
`examples/plot_channel_terrain.py` 演示了如何对中心线、断面进行可视化，核心功能归入 `docs/product_pipeline.md` 的后处理阶段。

## 主要成果

| 类型 | 文件 | 说明 |
| --- | --- | --- |
| 3D 点云 | `{zone}_channel_terrain.html` | Plotly 交互式散点图，可旋转观察断面分布 |
| 3D 网格 | `{zone}_channel_surface.html` | 将断面插值成河床表面，可供水动力模型或 GIS 使用 |
| 静态图 | `{zone}_channel_static.png` | 左侧沿程×横向热力图，右侧中心线纵剖（沿程使用 `global_station_m`）|
| 联合纵剖 | `combined_centerline_profile.(png|csv)` | 对比原始与修正中心线，高度衔接多分区 |
| 子分区热力图 | `{zone}_{subzone}_section.png` | 诊断 DEM 异常或局部河槽形态 |

## 生成流程

1. 调用 `preprocess_cross_sections`/`run_channel_profile_model` 获取修正后的断面与中心线；
2. 使用脚本或 API 函数（例如 `make_zone_surface`、`make_static_visual`、`make_combined_centerline_plot`）生成图件与 CSV；
3. 将成果纳入报告或推送至 `hydrosis.portal`。

## 关联测试

全功能测试 `hydrosis.testing.full_feature_runner` 在“输入输出与报告生成”与“三阶段系统流水线验证”两节中，会检查可视化与报告的生成情况（详见 `docs/test_documentation.md`）。
