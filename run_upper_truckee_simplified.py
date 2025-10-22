#!/usr/bin/env python3
"""
运行 Upper Truckee River 简化工作流
使用预处理的真实DEM数据
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 确保可以导入HydroSIS模块
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    """运行 Upper Truckee River 简化工作流"""
    print("=" * 80)
    print("Upper Truckee River 完整水文模拟工作流")
    print("=" * 80)
    print()
    print("使用真实DEM数据和预处理的流域数据")
    print("  • DEM: data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/")
    print("  • 流域配置: config/upper_truckee_project.yml")
    print()
    print("=" * 80)
    print()

    from hydrosis import (
        ModelConfig,
        run_workflow,
    )
    from hydrosis.config import (
        DelineationConfig,
        IOConfig,
        RoutingModelConfig,
        RunoffModelConfig,
    )
    from hydrosis.parameters.zone import ParameterZoneConfig
    from hydrosis.io.outputs import write_simulation_results

    try:
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False
        print("⚠️  Matplotlib未安装，将跳过图表生成")

    # 配置路径
    dem_dir = REPO_ROOT / "data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00"
    output_dir = REPO_ROOT / "results/upper_truckee_run"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: 配置流域划分...")
    print("-" * 80)

    # 使用预定义的 Upper Truckee 流域配置
    # 基于real-world Upper Truckee River 流域
    delineation = DelineationConfig(
        dem_path=dem_dir / "elevation.tif",
        pour_points_path=None,  # 将使用预定义的子流域
        accumulation_threshold=15000,
        precomputed_subbasins=[
            # 上游子流域
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
            # 中游子流域
            {
                "id": "W180",
                "area_km2": 125.5,
                "downstream": "W190",
                "parameters": {
                    "runoff_model": "forest_zone",
                    "routing_model": "muskingum_main"
                }
            },
            # 下游子流域（出口）
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

    print("  ✓ 配置了 4 个子流域")
    print("  ✓ 总流域面积: 371.5 km²")
    print()

    print("Step 2-4: 配置产流模型...")
    print("-" * 80)

    # 产流模型 - 使用HBV和SCS-CN
    runoff_models = [
        RunoffModelConfig(
            id="mountain_zone",
            model_type="hbv",
            parameters={
                "degree_day_factor": 3.5,  # 山区积雪融化快
                "snow_threshold": 2.0,
                "field_capacity": 180,
                "beta": 1.8,
                "k0": 0.5,
                "k1": 0.15,
                "k2": 0.03,
                "percolation": 3.0,
            }
        ),
        RunoffModelConfig(
            id="forest_zone",
            model_type="scs_curve_number",
            parameters={
                "curve_number": 65,  # 森林区渗透好
                "initial_abstraction_ratio": 0.05,
            }
        ),
        RunoffModelConfig(
            id="valley_zone",
            model_type="scs_curve_number",
            parameters={
                "curve_number": 72,  # 河谷区域
                "initial_abstraction_ratio": 0.05,
            }
        ),
    ]

    print("  ✓ HBV模型 (mountain_zone)")
    print("  ✓ SCS-CN模型 (forest_zone, valley_zone)")
    print()

    print("Step 5-10: 配置汇流模型...")
    print("-" * 80)

    # 汇流模型 - Muskingum
    routing_models = [
        RoutingModelConfig(
            id="muskingum_upper",
            model_type="muskingum",
            parameters={
                "travel_time": 8,
                "weighting_factor": 0.25,
                "time_step": 1,
            }
        ),
        RoutingModelConfig(
            id="muskingum_main",
            model_type="muskingum",
            parameters={
                "travel_time": 12,
                "weighting_factor": 0.20,
                "time_step": 1,
            }
        ),
        RoutingModelConfig(
            id="muskingum_outlet",
            model_type="muskingum",
            parameters={
                "travel_time": 15,
                "weighting_factor": 0.15,
                "time_step": 1,
            }
        ),
    ]

    print("  ✓ Muskingum河道演算 (3种配置)")
    print()

    print("Step 6-8: 生成降雨数据...")
    print("-" * 80)

    # 生成合成降雨数据 - 24小时暴雨
    import numpy as np
    import pandas as pd

    # 创建120小时的模拟（5天）
    hours = 120
    timesteps = hours

    # 设计暴雨过程（类似SCS Type II设计暴雨）
    rainfall = np.zeros(timesteps)
    storm_start = 48  # 第3天开始
    storm_duration = 24  # 持续24小时
    peak_hour = storm_start + 12  # 峰值在暴雨中期

    total_rainfall_mm = 75.0  # 总降雨量75mm

    # 使用正态分布生成暴雨过程
    for t in range(storm_start, storm_start + storm_duration):
        # 正态分布权重
        sigma = storm_duration / 6.0
        weight = np.exp(-0.5 * ((t - peak_hour) / sigma) ** 2)
        rainfall[t] = weight

    # 归一化到总雨量
    rainfall = rainfall * (total_rainfall_mm / rainfall.sum())

    # 创建降雨DataFrame
    timestamps = pd.date_range('2017-01-01', periods=timesteps, freq='H')
    precip_df = pd.DataFrame({
        'Timestamp': timestamps,
        'W170': rainfall * 1.1,  # 上游降雨稍多（地形效应）
        'W160': rainfall * 1.05,
        'W180': rainfall * 1.0,
        'W190': rainfall * 0.95,  # 下游降雨稍少
    })
    precip_df.set_index('Timestamp', inplace=True)

    # 保存降雨数据
    forcing_dir = output_dir / "forcing"
    forcing_dir.mkdir(exist_ok=True)

    for col in precip_df.columns:
        precip_df[[col]].to_csv(forcing_dir / f"{col}.csv")

    print(f"  ✓ 生成了 {storm_duration}小时暴雨过程")
    print(f"  ✓ 总降雨量: {total_rainfall_mm:.1f} mm")
    print(f"  ✓ 峰值降雨强度: {rainfall.max():.2f} mm/h")
    print()

    print("Step 9-10: 运行水文模拟...")
    print("-" * 80)

    # 参数区配置
    parameter_zones = [
        ParameterZoneConfig(
            id="Upper_Mountain",
            description="上游高山区",
            control_points=["W170", "W160"],
            parameters={
                "runoff_model": "mountain_zone",
                "routing_model": "muskingum_upper",
            }
        ),
        ParameterZoneConfig(
            id="Middle_Forest",
            description="中游森林区",
            control_points=["W180"],
            parameters={
                "runoff_model": "forest_zone",
                "routing_model": "muskingum_main",
            }
        ),
        ParameterZoneConfig(
            id="Lower_Valley",
            description="下游河谷区",
            control_points=["W190"],
            parameters={
                "runoff_model": "valley_zone",
                "routing_model": "muskingum_outlet",
            }
        ),
    ]

    # 组装模型配置
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
        evaluation=None,
    )

    # 准备降雨强迫数据
    forcing = {
        'W170': precip_df['W170'].tolist(),
        'W160': precip_df['W160'].tolist(),
        'W180': precip_df['W180'].tolist(),
        'W190': precip_df['W190'].tolist(),
    }

    # 运行工作流
    print("  正在运行水文模拟...")
    result = run_workflow(config, forcing, persist_outputs=True)

    print("  ✓ 模拟完成")
    print()

    # 输出结果统计
    print("=" * 80)
    print("模拟结果摘要")
    print("=" * 80)
    print()

    print("各子流域累积出流（前5个时间步，m³/s）:")
    for sub_id, series in result.baseline.aggregated.items():
        preview = ", ".join(f"{value:.2f}" for value in series[:5])
        print(f"  {sub_id}: {preview}")
    print()

    # 计算峰值流量
    print("峰值流量:")
    for sub_id, series in result.baseline.aggregated.items():
        peak = max(series)
        peak_time = series.index(peak)
        print(f"  {sub_id}: {peak:.2f} m³/s (时间步 {peak_time})")
    print()

    # 绘制流量过程线
    if HAS_MPL:
        print("生成图表...")
        print("-" * 80)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for idx, (sub_id, series) in enumerate(result.baseline.aggregated.items()):
            ax = axes[idx]
            ax.plot(range(len(series)), series, 'b-', linewidth=2)
            ax.set_title(f'Subbasin {sub_id} Hydrograph', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step (hours)', fontsize=10)
            ax.set_ylabel('Discharge (m³/s)', fontsize=10)
            ax.grid(True, alpha=0.3)

            # 标注峰值
            peak = max(series)
            peak_idx = series.index(peak)
            ax.plot(peak_idx, peak, 'ro', markersize=8)
            ax.annotate(f'Peak: {peak:.1f} m³/s',
                       xy=(peak_idx, peak),
                       xytext=(peak_idx + 5, peak * 0.9),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=9)

        plt.tight_layout()
        fig_path = output_dir / "upper_truckee_hydrographs.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ 流量过程线图: {fig_path}")
        print()

    # 生成总结报告
    report_path = output_dir / "simulation_summary.md"
    with open(report_path, 'w') as f:
        f.write("# Upper Truckee River 水文模拟结果\n\n")
        f.write("## 流域概况\n\n")
        f.write("| 子流域 | 面积 (km²) | 下游 | 产流模型 | 汇流模型 |\n")
        f.write("|--------|-----------|------|----------|----------|\n")

        for sub in config.delineation.precomputed_subbasins:
            downstream = sub.get('downstream', 'None') or '出口'
            runoff = sub['parameters']['runoff_model']
            routing = sub['parameters']['routing_model']
            f.write(f"| {sub['id']} | {sub['area_km2']} | {downstream} | {runoff} | {routing} |\n")

        f.write(f"\n**总流域面积**: 371.5 km²\n\n")

        f.write("## 降雨设计\n\n")
        f.write(f"- **暴雨历时**: {storm_duration} 小时\n")
        f.write(f"- **总降雨量**: {total_rainfall_mm:.1f} mm\n")
        f.write(f"- **峰值降雨强度**: {rainfall.max():.2f} mm/h\n")
        f.write(f"- **暴雨类型**: 设计暴雨 (类似SCS Type II)\n\n")

        f.write("## 模拟结果\n\n")
        f.write("### 峰值流量\n\n")
        f.write("| 子流域 | 峰值流量 (m³/s) | 峰现时间 (h) |\n")
        f.write("|--------|----------------|-------------|\n")

        for sub_id, series in result.baseline.aggregated.items():
            peak = max(series)
            peak_time = series.index(peak)
            f.write(f"| {sub_id} | {peak:.2f} | {peak_time} |\n")

        f.write("\n### 流量过程\n\n")
        if HAS_MPL:
            f.write(f"![流量过程线](upper_truckee_hydrographs.png)\n\n")

        f.write("## 模型配置\n\n")
        f.write("### 产流模型\n\n")
        f.write("- **mountain_zone**: HBV模型（雪融径流）\n")
        f.write("- **forest_zone**: SCS-CN模型（CN=65）\n")
        f.write("- **valley_zone**: SCS-CN模型（CN=72）\n\n")

        f.write("### 汇流模型\n\n")
        f.write("- **muskingum_upper**: K=8h, x=0.25\n")
        f.write("- **muskingum_main**: K=12h, x=0.20\n")
        f.write("- **muskingum_outlet**: K=15h, x=0.15\n\n")

        f.write("---\n\n")
        f.write("*Generated by HydroSIS - Upper Truckee River Workflow*\n")

    print(f"  ✓ 总结报告: {report_path}")
    print()

    print("=" * 80)
    print("✅ 工作流运行完成！")
    print("=" * 80)
    print()
    print(f"所有结果保存在: {output_dir}")
    print()
    print("生成的文件:")
    print(f"  • forcing/ - 降雨数据 ({len(list(forcing_dir.glob('*.csv')))} files)")
    print(f"  • results/ - 流量结果")
    if HAS_MPL:
        print(f"  • upper_truckee_hydrographs.png - 流量过程线图")
    print(f"  • simulation_summary.md - 模拟总结报告")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
