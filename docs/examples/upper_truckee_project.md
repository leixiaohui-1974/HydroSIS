# Upper Truckee 项目工作流（阶段化版）

`examples/run_upper_truckee_project.py` 已通过阶段化流程串联分区、降雨与诊断，可直接运行：

```bash
python examples/run_upper_truckee_project.py \
  --config config/upper_truckee_project.yml \
  --base-precipitation results/upper_truckee_channel_demo/intermediate/parameter_subbasin_areal_precipitation.csv \
  --station-count 12 \
  --rng-seed 42 \
  --persist-report
```

关键参数说明：

- `--base-precipitation`：指定合成雨量站使用的基准降雨序列；缺省时读取配置中 `io.precipitation`。
- `--station-series` / `--thiessen-polygons`：输入已有雨量站时序与 Thiessen 多边形，可直接插值至参数子分区。
- `--station-count`、`--rng-seed`：控制合成站点数量与随机种子，便于重复测试。
- `--skip-diagnostics`：只生成流程结果时可跳过通道诊断；若基线缺少流量输出也会自动跳过。

仓库已提供 DEM、流向栅格，以及 `results/upper_truckee_channel_demo/intermediate/parameter_subbasin_areal_precipitation.csv`，可作为初始降雨输入。如需替换，可使用 `examples/upper_truckee_channel_workflow.py` 或外部数据重新生成。

## Muskingum 与动态波对比

配置中新增了 `dynamic_wave_p3p4` 路由模型与 `hydraulic_p3p4` 场景，可快速比较水文学与水力学汇流：

```bash
# 基线（默认 Muskingum）
python examples/run_upper_truckee_project.py --config config/upper_truckee_project.yml --skip-diagnostics

# 动态波场景
python examples/run_upper_truckee_project.py \
  --config config/upper_truckee_project.yml \
  --scenario hydraulic_p3p4 \
  --skip-diagnostics \
  --persist-report
```

该场景会将 P3_sub51、P3_sub13、P4_sub2、P4_sub5 等主干控制段切换为动态波路由，其余分区保持 Muskingum。用户可在 `config/upper_truckee_project.yml` 的 `routing_models` 与 `scenarios` 节继续调整波速、扩散系数或扩展到更多河段。

若需进一步验证水力学模型，可配合下列脚本抽取所需空间信息：

```bash
# 纵剖
python examples/extract_channel_profiles.py \
  --geojson results/upper_truckee_channel_demo/parameters/parameter_channels.geojson \
  --dem data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif \
  --zones P3 P4 \
  --output-dir results/upper_truckee_channel_demo/intermediate/channel_profiles

# 横断面
python examples/extract_channel_cross_sections.py \
  --geojson results/upper_truckee_channel_demo/parameters/parameter_channels.geojson \
  --dem data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif \
  --zones P3 P4 \
  --spacing 2000 \
  --half-width 200 \
  --points 31 \
  --output-dir results/upper_truckee_channel_demo/intermediate/channel_cross_sections
```

生成的 CSV 包含链长、平面坐标、DEM 高程以及断面采样点到河心的距离，可直接用于估算面积—水深与湿周—水深关系、设置水力学边界条件。

## 尚待迁移的旧版功能

旧版 `examples/upper_truckee_channel_workflow.py` 中仍包含以下能力，后续可视优先级迁移至阶段化框架：

1. 多方案（scenario）组合实验与性能对比（`_run_channel_experiments`、`_terminal_subbasin_id`）。
2. 参数区局地 / 上游流量诊断图（`_plot_zone_timeseries` 及 `zone_runoff_coefficients_*` 统计输出）。
3. 降雨空间可视化素材（Thiessen 分布、雨量 GIF：`_render_rain_gauge_visuals`、`_animate_subbasin_precipitation`）。
4. 多模型汇总报告与历史指标追踪（`multi_model_report`、`_write_metrics_summary` 等）。

上述特性在阶段化脚本中暂未实现，可根据需要逐项迁移或重新设计。
