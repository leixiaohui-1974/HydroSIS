# Upper Truckee Ten-Step Pipeline (Publish Edition)

> 发布日期：2025-10-12 09:25 UTC（`results/upper_truckee_project/reports/final_pipeline_report.md`）

## Run Metadata

- 配置文件：[`config/upper_truckee_project.yml`](../../config/upper_truckee_project.yml)
- 输入要素：106 m DEM、复核后汇流点、实测降雨基准序列
- 运行步骤：Step 01 – Step 10 全部校验完成，日志均位于 `results/upper_truckee_project/logs/`
- 配置调整：
  - `project.pour_points.source_geojson` 指向 Step 02 生成的汇流点
  - `partition.source_directory` 更新为 Step 03 分区成果
  - `io.*` 路径同步至 Step 08–10 产出，便于后续复算与发布

## Step Snapshot

| Step | 目标 | 关键成果 |
| --- | --- | --- |
| 01 DEM 预处理 | 生成流向/汇流栅格与 DEM 统计 | [`flow_direction.tif`](../../results/upper_truckee_project/01_dem_processing/flow_direction.tif)、[`dem_summary.csv`](../../results/upper_truckee_project/01_dem_processing/dem_summary.csv) |
| 02 汇流点筛选 | 建立主干与支流汇流点 | [`pour_points.geojson`](../../results/upper_truckee_project/02_pour_points/pour_points.geojson)、[`pour_points_map.png`](../../results/upper_truckee_project/02_pour_points/pour_points_map.png) |
| 03 参数分区 | 拆分区块与子流域、输出参数骨架 | [`parameter_zones.geojson`](../../results/upper_truckee_project/03_partitioning/parameter_zones.geojson)、[`overview_map.png`](../../results/upper_truckee_project/03_partitioning/overview_map.png) |
| 04 河道断面 | 生成主干剖面及校正断面 | [`combined_centerline_profile.png`](../../results/upper_truckee_project/04_channel_profile/combined_centerline_profile.png)、[`channel_cross_sections_corrected.csv`](../../results/upper_truckee_project/04_channel_profile/channel_cross_sections_corrected.csv) |
| 05 雨量站布设 | 随机化选取 10 个站点并计算覆盖 | [`rain_gauge_locations.geojson`](../../results/upper_truckee_project/05_rain_gauge_layout/rain_gauge_locations.geojson)、[`rain_gauge_layout_map.png`](../../results/upper_truckee_project/05_rain_gauge_layout/rain_gauge_layout_map.png) |
| 06 降雨时序 | 合成站点/面平均降雨 | [`rain_gauge_forcing.csv`](../../results/upper_truckee_project/06_rain_sequence/rain_gauge_forcing.csv)、[`storm_profile.png`](../../results/upper_truckee_project/06_rain_sequence/storm_profile.png) |
| 07 Thiessen 权重 | 生成 Thiessen 多边形与权重表 | [`rain_gauge_weights.json`](../../results/upper_truckee_project/07_thiessen_weights/rain_gauge_weights.json)、[`thiessen_map.png`](../../results/upper_truckee_project/07_thiessen_weights/thiessen_map.png) |
| 08 面雨量插值 | 计算参数子流域面雨量 | [`parameter_subbasin_areal_precipitation.csv`](../../results/upper_truckee_project/08_areal_precipitation/parameter_subbasin_areal_precipitation.csv)、[`subbasin_precipitation_heatmap.png`](../../results/upper_truckee_project/08_areal_precipitation/subbasin_precipitation_heatmap.png) |
| 09 水文基线 | HBV+Muskingum 基线模拟与汇流 | [`hydrograph_baseline.png`](../../results/upper_truckee_project/09_hydrologic_run/hydrograph_baseline.png)、[`channel_flow_timeseries.csv`](../../results/upper_truckee_project/09_hydrologic_run/channel_flow_timeseries.csv) |
| 10 水动力情景 | 动态波情景对比与差值统计 | [`hydrograph_hydraulic_p3p4.png`](../../results/upper_truckee_project/10_hydrodynamic_run/hydrograph_hydraulic_p3p4.png)、[`hydro_difference_stats.csv`](../../results/upper_truckee_project/10_hydrodynamic_run/hydro_difference_stats.csv) |

## Detailed Step Notes

### Step 01 – DEM 预处理
- 报告：[step01_dem_processing.md](../../results/upper_truckee_project/reports/step01_dem_processing.md)
- 产出：流向/汇流/坡度栅格、阴影图 (`dem_hillshade.png`)、统计汇总
- 校验：GDAL/RichDEM 日志无异常；最小/最大海拔与源 DEM 一致

### Step 02 – 汇流点提取
- 报告：[step02_pour_points.md](../../results/upper_truckee_project/reports/step02_pour_points.md)
- 产出：8 个主干/支流汇流点、`pour_points_table.csv`
- 校验：最小距离 30 m、主干比例 0.5 均满足约束

### Step 03 – 参数分区
- 报告：[step03_partitioning.md](../../results/upper_truckee_project/reports/step03_partitioning.md)
- 产出：参数区/子流域 GeoJSON 及栅格概览、控制子区蒙版
- 配置：`partition.source_directory` 已切换至该目录，用于后续参数继承

### Step 04 – 河道剖面
- 报告：[step04_channel_profile.md](../../results/upper_truckee_project/reports/step04_channel_profile.md)
- 产出：校正后剖面 (`channel_cross_sections_corrected.csv`)、主干剖面图、HTML 交互剖面
- 注意：P3/P4 中心线缺失全局高程列，静态图以英文提示跳过

### Step 05 – 雨量站布设
- 报告：[step05_rain_gauge_layout.md](../../results/upper_truckee_project/reports/step05_rain_gauge_layout.md)
- 产出：站点 GeoJSON、Thiessen 多边形、覆盖统计表
- 审核：站点数=10，随机种子 42，可复现

### Step 06 – 降雨时序
- 报告：[step06_rain_sequence.md](../../results/upper_truckee_project/reports/step06_rain_sequence.md)
- 产出：站点时序、合成面雨量、GIF 动画与概要表
- 校验：时间步长 1 小时，序列长度 120；基准列 `降雨强度_毫米每小时` 成功映射

### Step 07 – Thiessen 权重
- 报告：[step07_thiessen_weights.md](../../results/upper_truckee_project/reports/step07_thiessen_weights.md)
- 产出：权重 JSON/CSV、可视化图件
- 校验：权重总和=1.0，局部多边形覆盖与 Step 05 输出一致

### Step 08 – 面雨量插值
- 报告：[step08_areal_precipitation.md](../../results/upper_truckee_project/reports/step08_areal_precipitation.md)
- 产出：参数子流域面雨量 CSV、热力图、累积曲线、动画
- 配置：`io.precipitation` 指向 `parameter_subbasin_areal_precipitation.csv`

### Step 09 – 水文基线模拟
- 报告：[step09_hydrologic_run.md](../../results/upper_truckee_project/reports/step09_hydrologic_run.md)
- 产出：基线汇流时序、区域汇流图、baseline/hydraulic 结果目录
- 日志：`Forcing coverage lengths: [120]`，不存在缺测子流域

### Step 10 – 水动力情景分析
- 报告：[step10_hydrodynamic_run.md](../../results/upper_truckee_project/reports/step10_hydrodynamic_run.md)
- 产出：P3/P4 动态波情景对比图、差值栅格统计、差值热图
- 配置：`scenarios` 中 `hydraulic_p3p4` 成功覆盖 4 个子流域的 routing_model

## Final Deliverables

- 最终汇总报告：[final_pipeline_report.md](../../results/upper_truckee_project/reports/final_pipeline_report.md)
- Hydrologic/Hydrodynamic 结果包：`results/upper_truckee_project/09_hydrologic_run/hydro/`、`results/upper_truckee_project/10_hydrodynamic_run/hydro/`
- 影片与动画：`06_rain_sequence/rainfall_timeseries.png`、`08_areal_precipitation/areal_precip_animation.gif`
- 推荐后续发布步骤：
  1. 将 `results/upper_truckee_project` 目录压缩并归档至数据交付库
  2. 按需筛选 `reports/` 目录至门户或文档站点
  3. 如需对外发布，请附带本发布版摘要文档及配置文件快照

> 若后续需要重跑：保持 `config/upper_truckee_project.yml` 当前路径配置，直接调用 `scripts/pipeline/run_*` 即可复现本次成果。 
