#!/usr/bin/env python3
"""运行完整的Upper Truckee River工作流，生成所有结果、图表和报告"""
from __future__ import annotations

import sys
from pathlib import Path

# 确保可以导入HydroSIS模块
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def print_section(title: str):
    """打印分节标题"""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()

def print_step(step_num: int, title: str):
    """打印步骤标题"""
    print()
    print("-" * 80)
    print(f"步骤 {step_num}: {title}")
    print("-" * 80)

def main():
    """运行完整的Upper Truckee River工作流"""
    print_section("Upper Truckee River 完整工作流")

    from hydrosis import ModelConfig, run_workflow
    from hydrosis.config import (
        DelineationConfig,
        IOConfig,
        RoutingModelConfig,
        RunoffModelConfig,
        ScenarioConfig,
        EvaluationConfig,
    )
    from hydrosis.parameters.zone import ParameterZoneConfig
    import numpy as np
    import pandas as pd

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.ioff()
    except ImportError:
        print("警告: matplotlib未安装，将跳过可视化")
        plt = None

    # 配置路径
    output_dir = REPO_ROOT / "results" / "upper_truckee_complete"
    output_dir.mkdir(parents=True, exist_ok=True)

    dem_dir = REPO_ROOT / "data" / "Upper_Truckee_River" / "terrain" / "UpTruckeeRv_S10_NED_30m" / "00"

    print_step(1, "配置流域和子流域")

    # 创建虚拟pour points文件（使用预定义子流域时不需要，但配置要求）
    pour_points_file = output_dir / "pour_points.geojson"
    if not pour_points_file.exists():
        import json
        dummy_pour_points = {
            "type": "FeatureCollection",
            "features": []
        }
        pour_points_file.write_text(json.dumps(dummy_pour_points))

    # 配置流域划分（使用预定义子流域）
    delineation = DelineationConfig(
        dem_path=dem_dir / "elevation.tif",
        pour_points_path=pour_points_file,
        precomputed_subbasins=[
            {
                "id": "W170",
                "area_km2": 45.2,
                "downstream": "W180",
                "parameters": {
                    "runoff_model": "mountain_zone",
                    "routing_model": "muskingum_upper"
                }
            },
            {
                "id": "W160",
                "area_km2": 35.8,
                "downstream": "W180",
                "parameters": {
                    "runoff_model": "mountain_zone",
                    "routing_model": "muskingum_upper"
                }
            },
            {
                "id": "W180",
                "area_km2": 125.5,
                "downstream": "W190",
                "parameters": {
                    "runoff_model": "forest_zone",
                    "routing_model": "muskingum_main"
                }
            },
            {
                "id": "W190",
                "area_km2": 165.0,
                "downstream": None,
                "parameters": {
                    "runoff_model": "valley_zone",
                    "routing_model": "muskingum_outlet"
                }
            },
        ],
    )

    print("子流域配置:")
    total_area = sum(sb["area_km2"] for sb in delineation.precomputed_subbasins)
    for sb in delineation.precomputed_subbasins:
        downstream = sb["downstream"] if sb["downstream"] else "出口"
        print(f"  {sb['id']}: {sb['area_km2']} km² → {downstream}")
    print(f"  总面积: {total_area} km²")

    print_step(2, "配置产流模型")

    runoff_models = [
        # HBV模型用于高山区（积雪融雪）
        RunoffModelConfig(
            id="mountain_zone",
            model_type="hbv",
            parameters={
                "degree_day_factor": 3.5,      # 度日因子 (mm/°C/day)
                "snow_threshold": 2.0,         # 降雪温度阈值 (°C)
                "field_capacity": 180,         # 田间持水量 (mm)
                "beta": 1.8,                   # 土壤蓄水指数
                "k0": 0.5,                     # 快速退水系数
                "k1": 0.15,                    # 上层退水系数
                "k2": 0.03,                    # 下层退水系数
                "percolation": 3.0,            # 渗漏率 (mm/day)
                # 初始条件设为0，避免在降雨开始前产生流量
                "initial_snow": 0.0,
                "initial_soil": 0.0,
                "initial_upper": 0.0,
                "initial_lower": 0.0,
            }
        ),
        # SCS-CN模型用于森林区
        RunoffModelConfig(
            id="forest_zone",
            model_type="scs_curve_number",
            parameters={
                "curve_number": 65,            # CN值（森林，良好条件）
                "initial_abstraction_ratio": 0.05,
            }
        ),
        # SCS-CN模型用于河谷区
        RunoffModelConfig(
            id="valley_zone",
            model_type="scs_curve_number",
            parameters={
                "curve_number": 72,            # CN值（草地，一般条件）
                "initial_abstraction_ratio": 0.05,
            }
        ),
    ]

    print("产流模型:")
    for model in runoff_models:
        print(f"  {model.id}: {model.model_type}")

    print_step(3, "配置汇流模型")

    routing_models = [
        RoutingModelConfig(
            id="muskingum_upper",
            model_type="muskingum",
            parameters={
                "travel_time": 8,              # 演进时间 (小时)
                "weighting_factor": 0.25,      # 权重因子
                "time_step": 1                 # 时间步长 (小时)
            }
        ),
        RoutingModelConfig(
            id="muskingum_main",
            model_type="muskingum",
            parameters={
                "travel_time": 12,
                "weighting_factor": 0.2,
                "time_step": 1
            }
        ),
        RoutingModelConfig(
            id="muskingum_outlet",
            model_type="muskingum",
            parameters={
                "travel_time": 15,
                "weighting_factor": 0.15,
                "time_step": 1
            }
        ),
    ]

    print("汇流模型:")
    for model in routing_models:
        params = model.parameters
        print(f"  {model.id}: K={params['travel_time']}h, x={params['weighting_factor']}")

    print_step(4, "生成降雨数据")

    # 生成120小时的降雨序列
    hours = 120
    timesteps = hours

    # 设计暴雨过程
    rainfall = np.zeros(timesteps)
    storm_start = 48        # 第3天开始
    storm_duration = 24     # 持续24小时
    peak_hour = storm_start + 12  # 峰值在中间
    total_rainfall_mm = 75.0

    print(f"降雨设计:")
    print(f"  总时长: {hours} 小时")
    print(f"  暴雨开始: 第 {storm_start} 小时 (2017-01-03 00:00)")
    print(f"  暴雨持续: {storm_duration} 小时")
    print(f"  峰值时刻: 第 {peak_hour} 小时 (2017-01-03 12:00)")
    print(f"  设计总降雨量: {total_rainfall_mm} mm")

    # 使用高斯分布生成降雨过程
    for t in range(storm_start, storm_start + storm_duration):
        sigma = storm_duration / 6.0
        weight = np.exp(-0.5 * ((t - peak_hour) / sigma) ** 2)
        rainfall[t] = weight

    rainfall = rainfall * (total_rainfall_mm / rainfall.sum())

    print(f"\n实际降雨统计:")
    print(f"  总降雨量: {rainfall.sum():.2f} mm")
    print(f"  峰值强度: {rainfall.max():.2f} mm/h")
    print(f"  非零时段: {(rainfall > 0).sum()} 小时")

    # 创建各子流域的降雨数据（考虑海拔梯度）
    timestamps = pd.date_range('2017-01-01', periods=timesteps, freq='h')
    precip_df = pd.DataFrame({
        'Timestamp': timestamps,
        'W170': rainfall * 1.1,   # 高山区降雨量 +10%
        'W160': rainfall * 1.05,  # 高山区降雨量 +5%
        'W180': rainfall * 1.0,   # 中游基准降雨量
        'W190': rainfall * 0.95,  # 下游降雨量 -5%
    })
    precip_df.set_index('Timestamp', inplace=True)

    # 保存降雨数据
    forcing_dir = output_dir / "forcing"
    forcing_dir.mkdir(exist_ok=True)
    for col in precip_df.columns:
        precip_df[[col]].to_csv(forcing_dir / f"{col}.csv")

    print(f"\n各子流域降雨量:")
    for col in precip_df.columns:
        total = precip_df[col].sum()
        peak = precip_df[col].max()
        print(f"  {col}: 总量={total:.2f}mm, 峰值={peak:.2f}mm/h")

    print(f"\n降雨数据已保存到: {forcing_dir}")

    # 准备forcing字典
    forcing = {col: precip_df[col].tolist() for col in precip_df.columns}

    print_step(5, "配置参数区")

    parameter_zones = [
        ParameterZoneConfig(
            id="Upper_Mountain",
            description="上游高山区 (2000-3000m)",
            control_points=["W170", "W160"],
            parameters={
                "runoff_model": "mountain_zone",
                "routing_model": "muskingum_upper"
            }
        ),
        ParameterZoneConfig(
            id="Middle_Forest",
            description="中游森林区 (1500-2000m)",
            control_points=["W180"],
            parameters={
                "runoff_model": "forest_zone",
                "routing_model": "muskingum_main"
            }
        ),
        ParameterZoneConfig(
            id="Lower_Valley",
            description="下游河谷区 (1200-1500m)",
            control_points=["W190"],
            parameters={
                "runoff_model": "valley_zone",
                "routing_model": "muskingum_outlet"
            }
        ),
    ]

    print("参数区配置:")
    for zone in parameter_zones:
        print(f"  {zone.id}: {zone.description}")
        print(f"    控制点: {', '.join(zone.control_points)}")

    print_step(6, "生成合成观测数据用于评估")

    # 为了演示评估功能，生成一个合成的"观测"流量数据
    # 实际应用中这应该是真实的观测数据
    outlet_id = "W190"

    # 使用简单的单位线方法生成合成流量
    unit_hydrograph = np.zeros(120)
    uh_peak_time = 65  # 峰值时间
    uh_duration = 40   # 响应持续时间

    for t in range(uh_duration):
        if t < uh_peak_time - storm_start:
            unit_hydrograph[storm_start + t] = t / (uh_peak_time - storm_start)
        else:
            unit_hydrograph[storm_start + t] = np.exp(-(t - (uh_peak_time - storm_start)) / 15.0)

    # 卷积降雨得到流量
    synthetic_obs = np.convolve(rainfall * 0.95, unit_hydrograph[:20], mode='full')[:120] * 2.5

    # 添加一些噪声
    np.random.seed(42)
    synthetic_obs += np.random.normal(0, 0.3, 120)
    synthetic_obs = np.maximum(synthetic_obs, 0.1)  # 确保非负且有基流

    # 保存观测数据
    obs_df = pd.DataFrame({
        'Timestamp': timestamps,
        'observed_flow': synthetic_obs
    })
    obs_df.set_index('Timestamp', inplace=True)

    obs_file = output_dir / "observed_flow.csv"
    obs_df.to_csv(obs_file)

    print(f"合成观测数据:")
    print(f"  峰值流量: {synthetic_obs.max():.2f} m³/s")
    print(f"  峰现时间: 第 {synthetic_obs.argmax()} 小时")
    print(f"  总径流量: {synthetic_obs.sum():.2f} m³/s·h")
    print(f"  已保存到: {obs_file}")

    print_step(7, "配置模型评估")

    evaluation = EvaluationConfig(
        metrics=["rmse", "mae", "nse", "pbias"],
    )

    print("评估配置:")
    print(f"  评估站点: {outlet_id}")
    print(f"  评估指标: {', '.join(evaluation.metrics)}")

    print_step(8, "组装完整模型配置")

    config = ModelConfig(
        delineation=delineation,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=IOConfig(
            precipitation=forcing_dir,
            results_directory=output_dir / "results",
            figures_directory=output_dir / "figures",
            reports_directory=output_dir / "reports",
        ),
        scenarios=[],
        evaluation=evaluation,
    )

    print("模型配置完成")
    print(f"  流域: Upper Truckee River")
    print(f"  子流域数: {len(delineation.precomputed_subbasins)}")
    print(f"  参数区数: {len(parameter_zones)}")
    print(f"  产流模型数: {len(runoff_models)}")
    print(f"  汇流模型数: {len(routing_models)}")

    print_step(9, "运行水文模拟")

    # 准备观测数据
    observations = {outlet_id: list(synthetic_obs)}

    print("开始运行工作流...")
    result = run_workflow(config, forcing, observations=observations)
    print("✓ 模拟完成")

    # 获取结果
    baseline = result.baseline

    print("\n模拟结果统计:")
    print("\n本地产流:")
    for sub_id in ["W170", "W160", "W180", "W190"]:
        if sub_id in baseline.local:
            series = baseline.local[sub_id]
            total = sum(series)
            peak = max(series) if series else 0
            peak_time = series.index(peak) if series and peak > 0 else -1
            print(f"  {sub_id}:")
            print(f"    总产流: {total:.2f} m³/s·h")
            print(f"    峰值: {peak:.2f} m³/s (第 {peak_time} 小时)")

    print("\n累积流量:")
    for sub_id in ["W170", "W160", "W180", "W190"]:
        if sub_id in baseline.aggregated:
            series = baseline.aggregated[sub_id]
            total = sum(series)
            peak = max(series) if series else 0
            peak_time = series.index(peak) if series and peak > 0 else -1
            print(f"  {sub_id}:")
            print(f"    总流量: {total:.2f} m³/s·h")
            print(f"    峰值: {peak:.2f} m³/s (第 {peak_time} 小时)")

    # 评估结果
    if result.overall_scores:
        print("\n模型评估指标:")
        for score in result.overall_scores:
            print(f"  模型: {score.model_id}")
            for metric_name, metric_value in score.aggregated.items():
                print(f"    {metric_name.upper()}: {metric_value:.4f}")

    print_step(10, "生成图表和可视化")

    if plt is None:
        print("跳过图表生成（matplotlib未安装）")
    else:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # 图1: 降雨分布
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(rainfall)), rainfall, color='steelblue', alpha=0.7)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Rainfall (mm/h)', fontsize=12)
        ax.set_title('Upper Truckee River - Rainfall Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(figures_dir / "rainfall_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 降雨分布图")

        # 图2: 子流域流量过程线
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, sub_id in enumerate(["W170", "W160", "W180", "W190"]):
            ax = axes[idx]

            if sub_id in baseline.local:
                local = baseline.local[sub_id]
                ax.plot(local, 'b-', label='Local Runoff', linewidth=2)

            if sub_id in baseline.aggregated:
                aggregated = baseline.aggregated[sub_id]
                ax.plot(aggregated, 'r-', label='Accumulated Flow', linewidth=2, alpha=0.8)

            ax.set_title(f'{sub_id} Hydrograph', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (hours)', fontsize=10)
            ax.set_ylabel('Flow (m³/s)', fontsize=10)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(figures_dir / "subbasin_hydrographs.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 子流域流量过程线")

        # 图3: 出口流量对比（模拟 vs 观测）
        if outlet_id in baseline.aggregated:
            fig, ax = plt.subplots(figsize=(12, 6))

            simulated = baseline.aggregated[outlet_id]
            ax.plot(simulated, 'b-', label='Simulated', linewidth=2)
            ax.plot(synthetic_obs, 'r--', label='Observed', linewidth=2, alpha=0.8)

            ax.set_title(f'{outlet_id} Outlet - Simulated vs Observed', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Flow (m³/s)', fontsize=12)
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)

            # 添加统计信息
            if result.overall_scores:
                metric_lines = []
                for score in result.overall_scores:
                    for metric_name, metric_value in score.aggregated.items():
                        metric_lines.append(f'{metric_name.upper()}: {metric_value:.4f}')
                textstr = '\n'.join(metric_lines)
                props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
                ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                       verticalalignment='top', bbox=props)

            plt.tight_layout()
            plt.savefig(figures_dir / "outlet_comparison.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ 出口流量对比图")

        # 图4: 参数区流量汇总
        if baseline.zone_discharge:
            fig, ax = plt.subplots(figsize=(12, 6))

            for zone_id, zone_flows in baseline.zone_discharge.items():
                # 汇总该参数区的所有控制点流量
                total_flow = None
                for sub_id, flow in zone_flows.items():
                    if total_flow is None:
                        total_flow = [0.0] * len(flow)
                    total_flow = [a + b for a, b in zip(total_flow, flow)]

                if total_flow:
                    ax.plot(total_flow, label=zone_id, linewidth=2)

            ax.set_title('Parameter Zone Discharge Summary', fontsize=14, fontweight='bold')
            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Flow (m³/s)', fontsize=12)
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(figures_dir / "zone_discharge.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ 参数区流量汇总图")

        # 图5: 散点图（模拟 vs 观测）
        if outlet_id in baseline.aggregated and len(synthetic_obs) > 0:
            fig, ax = plt.subplots(figsize=(8, 8))

            simulated = baseline.aggregated[outlet_id]
            max_val = max(max(simulated), max(synthetic_obs))

            ax.scatter(synthetic_obs, simulated, alpha=0.6, s=30)
            ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line')

            ax.set_title('Simulated vs Observed Flow', fontsize=14, fontweight='bold')
            ax.set_xlabel('Observed Flow (m³/s)', fontsize=12)
            ax.set_ylabel('Simulated Flow (m³/s)', fontsize=12)
            ax.legend(loc='best', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            plt.tight_layout()
            plt.savefig(figures_dir / "scatter_plot.png", dpi=150, bbox_inches='tight')
            plt.close()
            print("  ✓ 散点对比图")

        print(f"\n所有图表已保存到: {figures_dir}")

    print_step(11, "生成结果报告")

    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)

    # 生成Markdown报告
    report_file = reports_dir / "simulation_report.md"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Upper Truckee River 水文模拟报告\n\n")
        f.write(f"**生成时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. 流域概况\n\n")
        f.write(f"- **流域名称**: Upper Truckee River\n")
        f.write(f"- **流域面积**: {total_area} km²\n")
        f.write(f"- **子流域数量**: {len(delineation.precomputed_subbasins)}\n")
        f.write(f"- **参数分区**: {len(parameter_zones)} 个\n\n")

        f.write("### 子流域信息\n\n")
        f.write("| 子流域 | 面积 (km²) | 下游 | 产流模型 | 汇流模型 |\n")
        f.write("|--------|-----------|------|----------|----------|\n")
        for sb in delineation.precomputed_subbasins:
            downstream = sb["downstream"] if sb["downstream"] else "出口"
            runoff_model = sb["parameters"]["runoff_model"]
            routing_model = sb["parameters"]["routing_model"]
            f.write(f"| {sb['id']} | {sb['area_km2']} | {downstream} | {runoff_model} | {routing_model} |\n")
        f.write("\n")

        f.write("## 2. 降雨设计\n\n")
        f.write(f"- **模拟时长**: {hours} 小时\n")
        f.write(f"- **暴雨开始**: 第 {storm_start} 小时\n")
        f.write(f"- **暴雨持续**: {storm_duration} 小时\n")
        f.write(f"- **总降雨量**: {rainfall.sum():.2f} mm\n")
        f.write(f"- **峰值强度**: {rainfall.max():.2f} mm/h\n\n")

        f.write("## 3. 模型配置\n\n")
        f.write("### 产流模型\n\n")
        for model in runoff_models:
            f.write(f"**{model.id}** ({model.model_type})\n\n")
            for key, value in model.parameters.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")

        f.write("### 汇流模型\n\n")
        for model in routing_models:
            f.write(f"**{model.id}** ({model.model_type})\n\n")
            for key, value in model.parameters.items():
                f.write(f"- {key}: {value}\n")
            f.write("\n")

        f.write("## 4. 模拟结果\n\n")
        f.write("### 本地产流统计\n\n")
        f.write("| 子流域 | 总产流 (m³/s·h) | 峰值流量 (m³/s) | 峰现时间 (h) |\n")
        f.write("|--------|----------------|----------------|-------------|\n")
        for sub_id in ["W170", "W160", "W180", "W190"]:
            if sub_id in baseline.local:
                series = baseline.local[sub_id]
                total = sum(series)
                peak = max(series) if series else 0
                peak_time = series.index(peak) if series and peak > 0 else -1
                f.write(f"| {sub_id} | {total:.2f} | {peak:.2f} | {peak_time} |\n")
        f.write("\n")

        f.write("### 累积流量统计\n\n")
        f.write("| 子流域 | 总流量 (m³/s·h) | 峰值流量 (m³/s) | 峰现时间 (h) |\n")
        f.write("|--------|----------------|----------------|-------------|\n")
        for sub_id in ["W170", "W160", "W180", "W190"]:
            if sub_id in baseline.aggregated:
                series = baseline.aggregated[sub_id]
                total = sum(series)
                peak = max(series) if series else 0
                peak_time = series.index(peak) if series and peak > 0 else -1
                f.write(f"| {sub_id} | {total:.2f} | {peak:.2f} | {peak_time} |\n")
        f.write("\n")

        if result.overall_scores:
            f.write("## 5. 模型评估\n\n")
            f.write(f"评估站点: **{outlet_id}**\n\n")
            f.write("| 指标 | 数值 | 说明 |\n")
            f.write("|------|------|------|\n")
            for score in result.overall_scores:
                for metric_name, value in score.aggregated.items():
                    metric_upper = metric_name.upper()

                    # 添加说明
                    if metric_upper == "NSE":
                        if value > 0.9:
                            desc = "优秀"
                        elif value > 0.75:
                            desc = "良好"
                        elif value > 0.5:
                            desc = "可接受"
                        else:
                            desc = "不满意"
                    elif metric_upper == "PBIAS":
                        if abs(value) < 10:
                            desc = "很好"
                        elif abs(value) < 15:
                            desc = "良好"
                        elif abs(value) < 25:
                            desc = "可接受"
                        else:
                            desc = "不满意"
                    else:
                        desc = "-"

                    f.write(f"| {metric_upper} | {value:.4f} | {desc} |\n")
            f.write("\n")

        f.write("## 6. 文件清单\n\n")
        f.write("### 数据文件\n\n")
        f.write(f"- 降雨数据: `{forcing_dir.relative_to(output_dir)}/`\n")
        f.write(f"- 观测数据: `{obs_file.relative_to(output_dir)}`\n")
        f.write(f"- 结果数据: `{config.io.results_directory.relative_to(output_dir)}/`\n\n")

        f.write("### 图表文件\n\n")
        if plt is not None:
            f.write("- `rainfall_distribution.png` - 降雨分布图\n")
            f.write("- `subbasin_hydrographs.png` - 子流域流量过程线\n")
            f.write("- `outlet_comparison.png` - 出口流量对比图\n")
            f.write("- `zone_discharge.png` - 参数区流量汇总图\n")
            f.write("- `scatter_plot.png` - 散点对比图\n")
        f.write("\n")

        f.write("## 7. 结论\n\n")
        f.write("本次模拟成功完成了Upper Truckee River流域的完整水文过程模拟，包括：\n\n")
        f.write("- ✓ 流域划分与参数分区\n")
        f.write("- ✓ 降雨数据生成与空间分布\n")
        f.write("- ✓ 产流模拟（HBV雪融模型 + SCS-CN模型）\n")
        f.write("- ✓ 河道汇流演算（Muskingum模型）\n")
        f.write("- ✓ 模型评估与验证\n")
        f.write("- ✓ 结果可视化与报告生成\n\n")

        if result.overall_scores:
            for score in result.overall_scores:
                if "nse" in score.aggregated:
                    nse_value = score.aggregated["nse"]
                    if nse_value > 0.75:
                        f.write(f"模型整体性能良好（NSE={nse_value:.4f}），可用于进一步的情景分析和决策支持。\n\n")
                    break

        f.write("---\n\n")
        f.write("*本报告由 HydroSIS 自动生成*\n")

    print(f"✓ 模拟报告已保存: {report_file}")

    # 生成结果文件清单
    file_list = reports_dir / "file_inventory.txt"
    with open(file_list, 'w', encoding='utf-8') as f:
        f.write("Upper Truckee River 工作流结果文件清单\n")
        f.write("=" * 60 + "\n\n")

        import os
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(str(output_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            folder_name = os.path.basename(root)
            if folder_name:
                f.write(f'{indent}{folder_name}/\n')
            subindent = ' ' * 2 * (level + 1)
            for file in sorted(files):
                file_path = Path(root) / file
                size = file_path.stat().st_size
                f.write(f'{subindent}{file} ({size:,} bytes)\n')

    print(f"✓ 文件清单已保存: {file_list}")

    print()
    print("=" * 80)
    print("  工作流完成！")
    print("=" * 80)
    print()
    print("结果保存位置:")
    print(f"  {output_dir}")
    print()
    print("主要成果:")
    print(f"  - 降雨数据: {len(precip_df.columns)} 个子流域")
    print(f"  - 模拟结果: {len(baseline.local)} 个子流域")
    if plt is not None:
        print(f"  - 可视化图表: 5 张")
    print(f"  - 分析报告: 2 个文件")
    if result.overall_scores:
        print(f"  - 评估指标: {len(result.overall_scores)} 个")
    print()

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
