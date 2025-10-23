#!/usr/bin/env python3
"""
Upper Truckee River 详细诊断版本工作流
每一步都输出详细的检查信息
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def print_section(title):
    """打印分节标题"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def print_step(step_num, title):
    """打印步骤标题"""
    print(f"\n{'─' * 80}")
    print(f"步骤 {step_num}: {title}")
    print(f"{'─' * 80}")

def check_rainfall_data(precip_df, forcing):
    """检查降雨数据"""
    print("\n【降雨数据诊断】")
    print(f"总时间步数: {len(precip_df)}")
    print(f"时间范围: {precip_df.index[0]} 到 {precip_df.index[-1]}")
    print(f"\n各子流域降雨统计:")
    for col in precip_df.columns:
        total = precip_df[col].sum()
        max_val = precip_df[col].max()
        nonzero = (precip_df[col] > 0).sum()
        peak_idx = precip_df[col].idxmax()
        print(f"  {col}:")
        print(f"    总降雨量: {total:.2f} mm")
        print(f"    峰值强度: {max_val:.2f} mm/h (时间步: {peak_idx})")
        print(f"    降雨时段: {nonzero} 个时间步")

    # 检查forcing字典
    print(f"\n【Forcing数据检查】")
    for sub_id, values in forcing.items():
        print(f"  {sub_id}: {len(values)} 个值, 总计={sum(values):.2f}mm")

def check_model_results(result, label=""):
    """检查模型结果"""
    print(f"\n【{label}模型结果诊断】")

    # 检查本地产流
    if hasattr(result, 'baseline') and hasattr(result.baseline, 'local'):
        print("\n本地产流:")
        for sub_id, series in result.baseline.local.items():
            if series:
                total = sum(series)
                max_val = max(series) if series else 0
                max_idx = series.index(max(series)) if series and max(series) > 0 else -1
                print(f"  {sub_id}:")
                print(f"    总产流: {total:.2f} m³/s·时间步")
                print(f"    峰值: {max_val:.2f} m³/s (时间步: {max_idx})")

    # 检查累积流量
    if hasattr(result, 'baseline') and hasattr(result.baseline, 'aggregated'):
        print("\n累积流量:")
        for sub_id, series in result.baseline.aggregated.items():
            if series:
                total = sum(series)
                max_val = max(series) if series else 0
                max_idx = series.index(max(series)) if series and max(series) > 0 else -1
                print(f"  {sub_id}:")
                print(f"    总流量: {total:.2f} m³/s·时间步")
                print(f"    峰值: {max_val:.2f} m³/s (时间步: {max_idx})")
                print(f"    前5个时间步: {[f'{v:.2f}' for v in series[:5]]}")

def main():
    """运行详细诊断版本的Upper Truckee River工作流"""
    print_section("Upper Truckee River 详细诊断工作流")

    from hydrosis import ModelConfig, run_workflow
    from hydrosis.config import DelineationConfig, IOConfig, RoutingModelConfig, RunoffModelConfig
    from hydrosis.parameters.zone import ParameterZoneConfig
    import numpy as np
    import pandas as pd

    try:
        import matplotlib.pyplot as plt
        HAS_MPL = True
    except ImportError:
        HAS_MPL = False

    output_dir = REPO_ROOT / "results/upper_truckee_diagnostic"
    output_dir.mkdir(parents=True, exist_ok=True)

    print_step(1, "配置流域和DEM路径")
    dem_dir = REPO_ROOT / "data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00"
    print(f"DEM目录: {dem_dir}")
    print(f"DEM文件: {dem_dir / 'elevation.tif'}")
    print(f"流向文件: {dem_dir / 'flowdir.tif'}")
    print(f"汇流累积: {dem_dir / 'flowaccum.tif'}")

    # 检查文件存在
    for f in ['elevation.tif', 'flowdir.tif', 'flowaccum.tif']:
        path = dem_dir / f
        if path.exists():
            print(f"  ✓ {f} 存在 ({path.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  ✗ {f} 不存在!")
            return 1

    print_step(2, "配置子流域")
    # 使用Upper Truckee River的真实子流域配置
    delineation = DelineationConfig(
        dem_path=dem_dir / "elevation.tif",
        pour_points_path=None,
        accumulation_threshold=15000,
        precomputed_subbasins=[
            {"id": "W170", "area_km2": 45.2, "downstream": "W180",
             "parameters": {"runoff_model": "mountain_zone", "routing_model": "muskingum_upper"}},
            {"id": "W160", "area_km2": 35.8, "downstream": "W180",
             "parameters": {"runoff_model": "mountain_zone", "routing_model": "muskingum_upper"}},
            {"id": "W180", "area_km2": 125.5, "downstream": "W190",
             "parameters": {"runoff_model": "forest_zone", "routing_model": "muskingum_main"}},
            {"id": "W190", "area_km2": 165.0, "downstream": None,
             "parameters": {"runoff_model": "valley_zone", "routing_model": "muskingum_outlet"}},
        ],
    )

    print("配置的子流域:")
    for sub in delineation.precomputed_subbasins:
        print(f"  {sub['id']}: {sub['area_km2']} km², 下游→{sub['downstream'] or '出口'}")

    print_step(3, "配置产流模型")
    runoff_models = [
        RunoffModelConfig(
            id="mountain_zone",
            model_type="hbv",
            parameters={
                "degree_day_factor": 3.5,
                "snow_threshold": 2.0,
                "field_capacity": 180,
                "beta": 1.8,
                "k0": 0.5,
                "k1": 0.15,
                "k2": 0.03,
                "percolation": 3.0,
                # 初始条件设为0，避免在降雨开始前产生流量
                "initial_snow": 0.0,
                "initial_soil": 0.0,
                "initial_upper": 0.0,
                "initial_lower": 0.0,
            }
        ),
        RunoffModelConfig(
            id="forest_zone",
            model_type="scs_curve_number",
            parameters={
                "curve_number": 65,
                "initial_abstraction_ratio": 0.05,
            }
        ),
        RunoffModelConfig(
            id="valley_zone",
            model_type="scs_curve_number",
            parameters={
                "curve_number": 72,
                "initial_abstraction_ratio": 0.05,
            }
        ),
    ]

    print("产流模型配置:")
    for model in runoff_models:
        print(f"  {model.id}: {model.model_type}")

    print_step(4, "配置汇流模型")
    routing_models = [
        RoutingModelConfig(
            id="muskingum_upper",
            model_type="muskingum",
            parameters={"travel_time": 8, "weighting_factor": 0.25, "time_step": 1}
        ),
        RoutingModelConfig(
            id="muskingum_main",
            model_type="muskingum",
            parameters={"travel_time": 12, "weighting_factor": 0.20, "time_step": 1}
        ),
        RoutingModelConfig(
            id="muskingum_outlet",
            model_type="muskingum",
            parameters={"travel_time": 15, "weighting_factor": 0.15, "time_step": 1}
        ),
    ]

    print("汇流模型配置:")
    for model in routing_models:
        print(f"  {model.id}: K={model.parameters['travel_time']}h, x={model.parameters['weighting_factor']}")

    print_step(5, "生成降雨数据")
    hours = 120
    timesteps = hours

    # 生成降雨数据
    rainfall = np.zeros(timesteps)
    storm_start = 48  # 第3天开始
    storm_duration = 24
    peak_hour = storm_start + 12

    print(f"暴雨设计:")
    print(f"  总时长: {hours} 小时")
    print(f"  暴雨开始: 第 {storm_start} 小时")
    print(f"  暴雨持续: {storm_duration} 小时")
    print(f"  峰值时刻: 第 {peak_hour} 小时")

    total_rainfall_mm = 75.0

    for t in range(storm_start, storm_start + storm_duration):
        sigma = storm_duration / 6.0
        weight = np.exp(-0.5 * ((t - peak_hour) / sigma) ** 2)
        rainfall[t] = weight

    rainfall = rainfall * (total_rainfall_mm / rainfall.sum())

    print(f"\n降雨统计:")
    print(f"  总降雨量: {rainfall.sum():.2f} mm")
    print(f"  峰值强度: {rainfall.max():.2f} mm/h")
    print(f"  峰值时刻: 第 {rainfall.argmax()} 小时")
    print(f"  非零时段: {(rainfall > 0).sum()} 小时")

    # 创建DataFrame
    timestamps = pd.date_range('2017-01-01', periods=timesteps, freq='h')
    precip_df = pd.DataFrame({
        'Timestamp': timestamps,
        'W170': rainfall * 1.1,
        'W160': rainfall * 1.05,
        'W180': rainfall * 1.0,
        'W190': rainfall * 0.95,
    })
    precip_df.set_index('Timestamp', inplace=True)

    # 保存降雨数据
    forcing_dir = output_dir / "forcing"
    forcing_dir.mkdir(exist_ok=True)
    for col in precip_df.columns:
        precip_df[[col]].to_csv(forcing_dir / f"{col}.csv")

    print(f"\n降雨数据已保存到: {forcing_dir}")

    # 检查降雨数据
    forcing = {col: precip_df[col].tolist() for col in precip_df.columns}
    check_rainfall_data(precip_df, forcing)

    print_step(6, "配置参数区")
    parameter_zones = [
        ParameterZoneConfig(
            id="Upper_Mountain",
            description="上游高山区",
            control_points=["W170", "W160"],
            parameters={"runoff_model": "mountain_zone", "routing_model": "muskingum_upper"}
        ),
        ParameterZoneConfig(
            id="Middle_Forest",
            description="中游森林区",
            control_points=["W180"],
            parameters={"runoff_model": "forest_zone", "routing_model": "muskingum_main"}
        ),
        ParameterZoneConfig(
            id="Lower_Valley",
            description="下游河谷区",
            control_points=["W190"],
            parameters={"runoff_model": "valley_zone", "routing_model": "muskingum_outlet"}
        ),
    ]

    print("参数区配置:")
    for zone in parameter_zones:
        print(f"  {zone.id}: 控制 {zone.control_points}")

    print_step(7, "组装模型配置")
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

    print("模型配置完成")

    print_step(8, "运行水文模拟")
    print("开始模拟...")
    result = run_workflow(config, forcing, persist_outputs=True)
    print("模拟完成!")

    # 详细检查结果
    check_model_results(result, "水文模拟")

    print_step(9, "生成图表和报告")

    # 绘制详细的诊断图
    if HAS_MPL:
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))

        # 1. 降雨过程线
        ax = axes[0, 0]
        for col in precip_df.columns:
            ax.plot(range(len(precip_df)), precip_df[col], label=col, linewidth=2)
        ax.set_title('降雨过程线', fontsize=12, fontweight='bold')
        ax.set_xlabel('时间步 (小时)')
        ax.set_ylabel('降雨强度 (mm/h)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. 本地产流
        ax = axes[0, 1]
        if hasattr(result.baseline, 'local'):
            for sub_id, series in result.baseline.local.items():
                if series:
                    ax.plot(range(len(series)), series, label=sub_id, linewidth=2)
        ax.set_title('本地产流', fontsize=12, fontweight='bold')
        ax.set_xlabel('时间步 (小时)')
        ax.set_ylabel('产流量 (m³/s)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3-6. 各子流域累积流量
        subbasins = ['W170', 'W160', 'W180', 'W190']
        for idx, sub_id in enumerate(subbasins):
            row = (idx + 2) // 2
            col = (idx + 2) % 2
            ax = axes[row, col]

            series = result.baseline.aggregated.get(sub_id, [])
            if series:
                ax.plot(range(len(series)), series, 'b-', linewidth=2)
                peak = max(series)
                peak_idx = series.index(peak)
                ax.plot(peak_idx, peak, 'ro', markersize=10)
                ax.annotate(f'峰值: {peak:.1f} m³/s\\n时间步: {peak_idx}',
                           xy=(peak_idx, peak),
                           xytext=(peak_idx + 10, peak * 0.85),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=9,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                ax.set_title(f'{sub_id} 累积流量', fontsize=12, fontweight='bold')
                ax.set_xlabel('时间步 (小时)')
                ax.set_ylabel('流量 (m³/s)')
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = output_dir / "diagnostic_hydrographs.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 诊断图已保存: {fig_path}")
        plt.close()

    # 生成详细报告
    report_path = output_dir / "diagnostic_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Upper Truckee River 详细诊断报告\n\n")
        f.write("## 降雨数据\n\n")
        f.write(f"- 总时长: {hours} 小时\n")
        f.write(f"- 暴雨开始: 第 {storm_start} 小时\n")
        f.write(f"- 暴雨持续: {storm_duration} 小时\n")
        f.write(f"- 峰值时刻: 第 {peak_hour} 小时\n")
        f.write(f"- 总降雨量: {rainfall.sum():.2f} mm\n\n")

        f.write("## 各子流域降雨\n\n")
        f.write("| 子流域 | 总降雨(mm) | 峰值强度(mm/h) | 峰值时刻 |\n")
        f.write("|--------|-----------|--------------|----------|\n")
        for col in precip_df.columns:
            total = precip_df[col].sum()
            max_val = precip_df[col].max()
            max_idx = precip_df[col].idxmax()
            f.write(f"| {col} | {total:.2f} | {max_val:.2f} | {max_idx} |\n")
        f.write("\n")

        f.write("## 模拟结果\n\n")
        f.write("| 子流域 | 峰值流量(m³/s) | 峰值时刻 | 产流系数 |\n")
        f.write("|--------|---------------|----------|----------|\n")

        for sub_id, series in result.baseline.aggregated.items():
            if series:
                peak = max(series)
                peak_idx = series.index(peak)
                # 计算径流系数
                area = next(s['area_km2'] for s in delineation.precomputed_subbasins if s['id'] == sub_id)
                rainfall_sub = precip_df[sub_id].sum()
                # 流量转径流深: Q(m³/s) * t(s) / A(m²) * 1000 (转mm)
                total_flow = sum(series) * 3600  # m³
                runoff_depth = total_flow / (area * 1e6) * 1000  # mm
                runoff_coef = runoff_depth / rainfall_sub if rainfall_sub > 0 else 0

                f.write(f"| {sub_id} | {peak:.2f} | {peak_idx} | {runoff_coef:.2%} |\n")

        f.write("\n## 诊断图\n\n")
        f.write("![诊断图](diagnostic_hydrographs.png)\n")

    print(f"  ✓ 诊断报告已保存: {report_path}")

    print_section("诊断工作流完成!")
    print(f"\n所有结果保存在: {output_dir}")
    print("\n生成的文件:")
    print(f"  • forcing/ - 降雨数据")
    print(f"  • results/ - 流量结果")
    print(f"  • diagnostic_hydrographs.png - 诊断图")
    print(f"  • diagnostic_report.md - 诊断报告")

    return 0

if __name__ == "__main__":
    sys.exit(main())
