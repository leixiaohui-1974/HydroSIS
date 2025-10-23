#!/usr/bin/env python3
"""
生成每个参数分区的降雨径流过程图
包括：
1. 区间降雨径流过程（该分区自身）
2. 上游累积降雨径流过程（该分区汇水点上游全流域）
3. 径流系数等水文指标分析
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_zone_upstream_relationships():
    """
    加载分区的上下游关系
    基于深度编码和汇水点位置
    """
    # 读取汇水点信息
    pour_points_file = Path("results/upper_truckee_complete_11steps/step_02_pour_points/2.2_pour_points_table.csv")
    df = pd.read_csv(pour_points_file)

    # 只考虑主流汇水点
    main_points = df[df['type'] == 'main_stream'].copy()
    main_points = main_points.sort_values('depth', ascending=False)  # 从上游到下游

    # 构建上下游关系
    # Zone的上游关系基于累积面积
    upstream_zones = {}
    for idx, row in main_points.iterrows():
        zone_id = int(row['zone_id'])
        depth = int(row['depth'])

        # 找到该分区的所有上游分区（depth更大的）
        upstream = []
        for _, other_row in main_points.iterrows():
            other_zone = int(other_row['zone_id'])
            other_depth = int(other_row['depth'])
            if other_depth > depth:  # 更上游
                upstream.append(other_zone)

        upstream_zones[zone_id] = sorted(upstream)

    return upstream_zones


def get_subbasin_ids_for_zone(zone_id):
    """获取指定分区的所有子流域ID"""
    df = pd.read_csv('results/upper_truckee_complete_11steps/parameters/parameter_subbasins.csv')
    zone_subbasins = df[df['subzone_id'].astype(str).str[0] == str(zone_id)]
    return zone_subbasins['subzone_id'].astype(str).tolist()


def aggregate_precipitation_for_zone(precip_df, zone_id, upstream_zones=None):
    """
    汇总指定分区的面雨量

    Args:
        precip_df: 子流域面雨量DataFrame
        zone_id: 分区ID
        upstream_zones: 如果提供，则包含上游分区（累积）

    Returns:
        Series: 该分区的面雨量时间序列
    """
    # 获取子流域ID列表
    subbasin_ids = get_subbasin_ids_for_zone(zone_id)

    # 如果需要包含上游
    if upstream_zones:
        for upstream_zone in upstream_zones:
            subbasin_ids.extend(get_subbasin_ids_for_zone(upstream_zone))

    # 提取这些子流域的数据
    available_ids = [sid for sid in subbasin_ids if sid in precip_df.columns]

    if not available_ids:
        return pd.Series(0, index=precip_df.index)

    # 计算面积加权平均
    # 读取子流域面积
    subbasin_df = pd.read_csv('results/upper_truckee_complete_11steps/parameters/parameter_subbasins.csv')
    subbasin_df['subzone_id_str'] = subbasin_df['subzone_id'].astype(str)

    total_area = 0
    weighted_precip = pd.Series(0, index=precip_df.index)

    for sid in available_ids:
        area = subbasin_df[subbasin_df['subzone_id_str'] == sid]['area_km2'].values
        if len(area) > 0:
            area = area[0]
            weighted_precip += precip_df[sid] * area
            total_area += area

    if total_area > 0:
        return weighted_precip / total_area
    else:
        return pd.Series(0, index=precip_df.index)


def aggregate_discharge_for_zone(discharge_df, zone_id, upstream_zones=None):
    """
    汇总指定分区的径流

    对于区间径流：使用该分区所有子流域的径流总和
    对于累积径流：使用汇水点的径流（已经包含上游）
    """
    if upstream_zones is None:
        # 区间径流：汇总该分区的所有子流域
        subbasin_ids = get_subbasin_ids_for_zone(zone_id)
        available_ids = [int(sid) for sid in subbasin_ids if int(sid) in discharge_df.columns]

        if not available_ids:
            return pd.Series(0, index=discharge_df.index)

        # 径流直接相加
        return discharge_df[available_ids].sum(axis=1)
    else:
        # 累积径流：查找该分区汇水点断面的径流
        # 汇水点通常是分区出口，断面ID可能不同
        # 我们需要找到该分区的最下游断面

        # 简化处理：汇总所有相关分区的子流域
        all_zones = [zone_id] + upstream_zones
        all_subbasin_ids = []
        for zid in all_zones:
            all_subbasin_ids.extend(get_subbasin_ids_for_zone(zid))

        available_ids = [int(sid) for sid in all_subbasin_ids if int(sid) in discharge_df.columns]

        if not available_ids:
            return pd.Series(0, index=discharge_df.index)

        return discharge_df[available_ids].sum(axis=1)


def calculate_runoff_coefficient(precip_series, discharge_series, area_km2, dt_hours=1):
    """
    计算径流系数

    径流系数 = 径流深度 / 降雨深度

    Args:
        precip_series: 降雨强度序列 (mm/hr)
        discharge_series: 径流流量序列 (m³/s)
        area_km2: 流域面积 (km²)
        dt_hours: 时间步长 (小时)

    Returns:
        float: 径流系数
    """
    # 总降雨深度 (mm)
    total_precip_depth = precip_series.sum() * dt_hours

    # 总径流深度 (mm)
    # 流量 m³/s * 时间步长(秒) = 体积 m³
    # 体积 / 面积 = 深度
    total_discharge_volume_m3 = discharge_series.sum() * dt_hours * 3600  # m³
    total_discharge_depth_mm = (total_discharge_volume_m3 / (area_km2 * 1e6)) * 1000

    # 径流系数
    if total_precip_depth > 0:
        runoff_coeff = total_discharge_depth_mm / total_precip_depth
    else:
        runoff_coeff = 0

    return runoff_coeff, total_precip_depth, total_discharge_depth_mm


def plot_rainfall_runoff_hydrograph(
    precip_series,
    discharge_series,
    zone_id,
    plot_type,
    area_km2,
    output_file,
    upstream_zones=None
):
    """
    绘制标准的降雨径流过程图

    Args:
        precip_series: 降雨序列
        discharge_series: 径流序列
        zone_id: 分区ID
        plot_type: 'incremental' 或 'cumulative'
        area_km2: 流域面积
        output_file: 输出文件路径
        upstream_zones: 上游分区列表（用于标题）
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, height_ratios=[1, 2], hspace=0.05)

    ax_precip = fig.add_subplot(gs[0])
    ax_discharge = fig.add_subplot(gs[1], sharex=ax_precip)

    # 时间轴
    time_hours = np.arange(len(precip_series))

    # ============ 上图：降雨（倒置柱状图）============
    # 倒置：将降雨值取负，y轴反向
    ax_precip.bar(time_hours, -precip_series.values, width=1.0,
                  color='steelblue', alpha=0.7, edgecolor='none')

    ax_precip.set_ylabel('降雨强度 (mm/hr)', fontsize=12)
    ax_precip.invert_yaxis()  # 倒置y轴
    ax_precip.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_precip.set_xlim(0, len(time_hours))

    # 隐藏x轴标签
    plt.setp(ax_precip.get_xticklabels(), visible=False)

    # ============ 下图：径流过程线 ============
    ax_discharge.plot(time_hours, discharge_series.values,
                     color='red', linewidth=2, label='径流过程线')
    ax_discharge.fill_between(time_hours, 0, discharge_series.values,
                              color='red', alpha=0.2)

    ax_discharge.set_xlabel('时间 (小时)', fontsize=12)
    ax_discharge.set_ylabel('流量 (m³/s)', fontsize=12)
    ax_discharge.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_discharge.legend(loc='upper right', fontsize=10)
    ax_discharge.set_xlim(0, len(time_hours))

    # ============ 计算水文指标 ============
    runoff_coeff, total_precip, total_runoff = calculate_runoff_coefficient(
        precip_series, discharge_series, area_km2
    )

    peak_discharge = discharge_series.max()
    peak_time = discharge_series.idxmax()

    # ============ 标题和统计信息 ============
    if plot_type == 'incremental':
        title = f'参数分区 {zone_id} - 区间降雨径流过程\n'
        title += f'(分区面积: {area_km2:.2f} km²)'
    else:
        if upstream_zones:
            zone_list = ', '.join([f'Zone {z}' for z in sorted(upstream_zones + [zone_id])])
            title = f'参数分区 {zone_id} - 上游全流域降雨径流过程\n'
            title += f'(包含: {zone_list}, 总面积: {area_km2:.2f} km²)'
        else:
            title = f'参数分区 {zone_id} - 累积降雨径流过程\n'
            title += f'(面积: {area_km2:.2f} km²)'

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # 在右上角添加统计信息文本框
    stats_text = f'''水文指标：
总降雨量：{total_precip:.1f} mm
总径流量：{total_runoff:.1f} mm
径流系数：{runoff_coeff:.3f}
峰值流量：{peak_discharge:.2f} m³/s
峰现时间：{peak_time} 小时'''

    ax_precip.text(0.98, 0.05, stats_text, transform=ax_precip.transAxes,
                  fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()

    return {
        'runoff_coefficient': runoff_coeff,
        'total_precipitation_mm': total_precip,
        'total_runoff_mm': total_runoff,
        'peak_discharge_m3s': peak_discharge,
        'peak_time_hours': peak_time
    }


def main():
    """主函数"""
    print("="*80)
    print("生成参数分区降雨径流过程图")
    print("="*80)

    # 输出目录
    output_dir = Path("results/upper_truckee_complete_11steps/zone_rainfall_runoff_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\n⚙ 加载数据...")
    precip_file = Path("results/upper_truckee_complete_11steps/step_08_areal_rainfall/8.2_subbasin_areal_precipitation.csv")
    discharge_file = Path("results/upper_truckee_complete_11steps/step_10_routing/10.1_discharge_timeseries.csv")

    precip_df = pd.read_csv(precip_file, index_col=0, parse_dates=True)
    discharge_df = pd.read_csv(discharge_file, index_col=0)

    # 统一列名为字符串
    precip_df.columns = precip_df.columns.astype(str)
    discharge_df.columns = discharge_df.columns.astype(int)

    # 加载上下游关系
    upstream_zones = load_zone_upstream_relationships()
    print(f"  ✓ 分区上下游关系: {upstream_zones}")

    # 加载分区面积
    subbasin_df = pd.read_csv('results/upper_truckee_complete_11steps/parameters/parameter_subbasins.csv')
    subbasin_df['zone_id'] = subbasin_df['subzone_id'].astype(str).str[0].astype(int)
    zone_areas = subbasin_df.groupby('zone_id')['area_km2'].sum().to_dict()

    # 存储所有分区的统计结果
    all_stats = []

    # 对每个分区生成两个图
    for zone_id in sorted(upstream_zones.keys()):
        print(f"\n{'='*80}")
        print(f"处理分区 {zone_id}")
        print(f"{'='*80}")

        zone_area = zone_areas[zone_id]
        upstream = upstream_zones[zone_id]

        # 计算累积面积
        cumulative_area = zone_area
        for uz in upstream:
            cumulative_area += zone_areas[uz]

        print(f"  分区面积: {zone_area:.2f} km²")
        print(f"  上游分区: {upstream}")
        print(f"  累积面积: {cumulative_area:.2f} km²")

        # ===== 图1：区间降雨径流过程 =====
        print(f"\n  ⚙ 生成区间降雨径流过程图...")

        # 区间降雨
        precip_incremental = aggregate_precipitation_for_zone(precip_df, zone_id, upstream_zones=None)

        # 区间径流
        discharge_incremental = aggregate_discharge_for_zone(discharge_df, zone_id, upstream_zones=None)

        # 绘图
        output_file1 = output_dir / f"zone{zone_id}_incremental_rainfall_runoff.png"
        stats1 = plot_rainfall_runoff_hydrograph(
            precip_incremental,
            discharge_incremental,
            zone_id,
            'incremental',
            zone_area,
            output_file1
        )

        print(f"    ✓ 保存: {output_file1.name}")
        print(f"    - 径流系数: {stats1['runoff_coefficient']:.3f}")
        print(f"    - 峰值流量: {stats1['peak_discharge_m3s']:.2f} m³/s")

        stats1['zone_id'] = zone_id
        stats1['type'] = 'incremental'
        stats1['area_km2'] = zone_area
        all_stats.append(stats1)

        # ===== 图2：累积降雨径流过程 =====
        print(f"\n  ⚙ 生成累积降雨径流过程图...")

        # 累积降雨（包含上游）
        precip_cumulative = aggregate_precipitation_for_zone(precip_df, zone_id, upstream_zones=upstream)

        # 累积径流（包含上游）
        discharge_cumulative = aggregate_discharge_for_zone(discharge_df, zone_id, upstream_zones=upstream)

        # 绘图
        output_file2 = output_dir / f"zone{zone_id}_cumulative_rainfall_runoff.png"
        stats2 = plot_rainfall_runoff_hydrograph(
            precip_cumulative,
            discharge_cumulative,
            zone_id,
            'cumulative',
            cumulative_area,
            output_file2,
            upstream_zones=upstream
        )

        print(f"    ✓ 保存: {output_file2.name}")
        print(f"    - 径流系数: {stats2['runoff_coefficient']:.3f}")
        print(f"    - 峰值流量: {stats2['peak_discharge_m3s']:.2f} m³/s")

        stats2['zone_id'] = zone_id
        stats2['type'] = 'cumulative'
        stats2['area_km2'] = cumulative_area
        all_stats.append(stats2)

    # 保存统计结果
    print(f"\n{'='*80}")
    print("保存统计结果...")
    print(f"{'='*80}")

    stats_df = pd.DataFrame(all_stats)
    stats_file = output_dir / "zone_hydrological_statistics.csv"
    stats_df.to_csv(stats_file, index=False)
    print(f"  ✓ 保存统计表: {stats_file.name}")

    # 生成分析报告
    print(f"\n{'='*80}")
    print("水文指标分析")
    print(f"{'='*80}")

    print("\n区间径流系数（各分区自身）：")
    incremental_stats = stats_df[stats_df['type'] == 'incremental'].sort_values('zone_id')
    for _, row in incremental_stats.iterrows():
        print(f"  Zone {int(row['zone_id'])}: Rc = {row['runoff_coefficient']:.3f} "
              f"(降雨{row['total_precipitation_mm']:.1f}mm, 径流{row['total_runoff_mm']:.1f}mm)")

    print("\n累积径流系数（包含上游）：")
    cumulative_stats = stats_df[stats_df['type'] == 'cumulative'].sort_values('zone_id')
    for _, row in cumulative_stats.iterrows():
        print(f"  Zone {int(row['zone_id'])}: Rc = {row['runoff_coefficient']:.3f} "
              f"(降雨{row['total_precipitation_mm']:.1f}mm, 径流{row['total_runoff_mm']:.1f}mm)")

    # 合理性分析
    print(f"\n{'='*80}")
    print("模拟结果合理性分析")
    print(f"{'='*80}")

    avg_rc = incremental_stats['runoff_coefficient'].mean()
    print(f"\n平均径流系数: {avg_rc:.3f}")

    if 0.1 <= avg_rc <= 0.8:
        print("✓ 径流系数在合理范围内 (0.1-0.8)")
    else:
        print("⚠ 径流系数可能不合理，建议检查模型参数")

    # 检查峰值时间
    avg_peak_time = incremental_stats['peak_time_hours'].mean()
    print(f"\n平均峰现时间: {avg_peak_time:.1f} 小时")

    # 降雨峰值在60小时
    if 55 <= avg_peak_time <= 70:
        print("✓ 峰现时间合理（接近降雨峰值时间60小时）")
    else:
        print("⚠ 峰现时间可能异常")

    # 检查径流系数的空间分布
    rc_std = incremental_stats['runoff_coefficient'].std()
    print(f"\n径流系数标准差: {rc_std:.3f}")
    if rc_std < 0.2:
        print("✓ 各分区径流系数相对一致")
    else:
        print("⚠ 各分区径流系数差异较大，可能反映了空间异质性")

    print(f"\n{'='*80}")
    print("✓ 所有分析完成！")
    print(f"共生成 {len(all_stats)} 个图表")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
