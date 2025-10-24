"""增强Step 9输出：径流累积曲线、径流系数评价

根据用户需求生成：
1. 各参数分区的径流累积时间序列图
2. 各参数分区的径流系数评价
3. 径流过程可视化
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_discharge_data(csv_path):
    """加载径流时间序列数据"""
    df = pd.read_csv(csv_path, index_col=0)
    return df

def load_precipitation_data(csv_path):
    """加载降雨数据"""
    df = pd.read_csv(csv_path)
    # 将Timestamp设为索引
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    return df

def aggregate_precip_to_zones(precip_df, subbasins_csv):
    """将子流域降雨聚合到分区级别（面积加权平均）

    Args:
        precip_df: 子流域降雨数据，列名为subzone_id
        subbasins_csv: 子流域CSV文件路径

    Returns:
        zone_precip_df: 分区级别降雨数据，列名为zone_id
    """
    # 读取子流域信息
    subbasins = pd.read_csv(subbasins_csv)

    # 创建分区降雨DataFrame
    zone_precip = {}

    # 按分区聚合
    for zone_id in subbasins['zone_id'].unique():
        zone_subs = subbasins[subbasins['zone_id'] == zone_id]

        # 计算面积加权平均降雨
        total_area = 0
        weighted_precip = pd.Series(0.0, index=precip_df.index)

        for _, sub in zone_subs.iterrows():
            # Convert to int first to remove .0, then to string
            subzone_id = str(int(sub['subzone_id']))
            area_km2 = sub['area_km2']

            if subzone_id in precip_df.columns:
                weighted_precip += precip_df[subzone_id] * area_km2
                total_area += area_km2

        # 计算平均值
        if total_area > 0:
            zone_precip[str(zone_id)] = weighted_precip / total_area
        else:
            zone_precip[str(zone_id)] = pd.Series(0.0, index=precip_df.index)

    return pd.DataFrame(zone_precip)

def load_zone_info(geojson_path):
    """加载参数分区信息"""
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    zones = []
    for feature in data['features']:
        props = feature['properties']
        zones.append({
            'zone_id': props.get('zone_id', props.get('id')),
            'area_km2': props.get('area_km2', 0)
        })
    return zones

def get_zone_subbasins(subbasins_csv, zone_id):
    """获取指定分区的所有子流域ID"""
    df = pd.read_csv(subbasins_csv)
    # Convert zone_id to int to match CSV data type
    zone_id_int = int(zone_id) if isinstance(zone_id, str) else zone_id
    zone_subbasins = df[df['zone_id'] == zone_id_int]['subzone_id'].tolist()
    # Convert to int first to remove .0, then to string
    return [str(int(sid)) for sid in zone_subbasins]

def plot_zone_cumulative_runoff(discharge_df, zone_precip_df, zones, subbasins_csv, output_dir):
    """绘制各分区径流累积曲线

    Args:
        discharge_df: 径流数据（子流域级别）
        zone_precip_df: 降雨数据（分区级别，已聚合）
        zones: 分区信息列表
        subbasins_csv: 子流域CSV文件路径
        output_dir: 输出目录
    """

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    zone_results = []

    for idx, zone in enumerate(sorted(zones, key=lambda x: x['zone_id'])):
        zone_id = zone['zone_id']
        ax = axes[idx]

        # 获取该分区的所有子流域
        subbasin_ids = get_zone_subbasins(subbasins_csv, zone_id)

        # 计算分区总径流 (m³/s)
        zone_discharge = pd.Series(0.0, index=discharge_df.index)
        found_count = 0
        for sid in subbasin_ids:
            if sid in discharge_df.columns:
                zone_discharge += discharge_df[sid]
                found_count += 1

        if found_count == 0:
            print(f"  ⚠️  警告: 分区{zone_id}在径流数据中找不到任何子流域")

        # 计算累积径流量 (转换为mm)
        # Q (m³/s) * 3600 (s/h) = m³/h
        # (m³/h) / (area_km2 * 1e6 m²) * 1000 mm/m = mm/h
        area_m2 = zone['area_km2'] * 1e6
        runoff_depth_mm = zone_discharge * 3600 / area_m2 * 1000  # mm/h
        cumulative_runoff = runoff_depth_mm.cumsum()

        # 获取该分区的降雨（已聚合到分区级别）
        if str(zone_id) in zone_precip_df.columns:
            zone_precip = zone_precip_df[str(zone_id)]
        else:
            print(f"  ⚠️  警告: 分区{zone_id}没有降雨数据")
            zone_precip = pd.Series(0, index=zone_precip_df.index)
        cumulative_precip = zone_precip.cumsum()

        # 计算径流系数
        total_precip = cumulative_precip.iloc[-1] if len(cumulative_precip) > 0 else 0
        total_runoff = cumulative_runoff.iloc[-1] if len(cumulative_runoff) > 0 else 0
        runoff_coeff = total_runoff / total_precip if total_precip > 0 else 0

        zone_results.append({
            'zone_id': zone_id,
            'area_km2': zone['area_km2'],
            'total_precip_mm': total_precip,
            'total_runoff_mm': total_runoff,
            'runoff_coefficient': runoff_coeff,
            'subbasin_count': len(subbasin_ids)
        })

        # 绘图
        time_hours = np.arange(len(cumulative_runoff))

        ax2 = ax.twinx()

        # 降雨累积曲线（倒置在上方）
        ax.plot(time_hours, cumulative_precip, 'b-', linewidth=2, label='Cumulative Precipitation')
        ax.fill_between(time_hours, cumulative_precip, alpha=0.3, color='blue')

        # 径流累积曲线
        ax2.plot(time_hours, cumulative_runoff, 'r-', linewidth=2, label='Cumulative Runoff')
        ax2.fill_between(time_hours, cumulative_runoff, alpha=0.3, color='red')

        # 设置标题和标签
        ax.set_title(f'Zone {zone_id} - Runoff Coeff = {runoff_coeff:.3f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('Cumulative Precip (mm)', fontsize=10, color='blue')
        ax2.set_ylabel('Cumulative Runoff (mm)', fontsize=10, color='red')

        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加统计信息
        info_text = (f'Area: {zone["area_km2"]:.1f} km²\n'
                    f'Subbasins: {len(subbasin_ids)}\n'
                    f'P: {total_precip:.1f} mm\n'
                    f'R: {total_runoff:.1f} mm')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    output_path = output_dir / "9.1_zone_cumulative_runoff.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 生成分区径流累积曲线: {output_path.name}")
    return zone_results

def evaluate_runoff_coefficients(zone_results, output_dir):
    """评价各分区径流系数"""

    # 保存CSV
    csv_path = output_dir / "9.2_runoff_coefficient_evaluation.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('分区ID,面积(km²),子流域数,累积降雨(mm),累积径流(mm),径流系数,等级\n')

        for result in sorted(zone_results, key=lambda x: x['zone_id']):
            coeff = result['runoff_coefficient']

            # 径流系数等级评价
            if coeff >= 0.7:
                grade = "极高"
            elif coeff >= 0.5:
                grade = "高"
            elif coeff >= 0.3:
                grade = "中等"
            elif coeff >= 0.1:
                grade = "偏低"
            else:
                grade = "很低"

            f.write(f"{result['zone_id']},{result['area_km2']:.2f},"
                   f"{result['subbasin_count']},{result['total_precip_mm']:.2f},"
                   f"{result['total_runoff_mm']:.2f},{coeff:.4f},{grade}\n")

    print(f"  ✓ 生成径流系数评价: {csv_path.name}")

    # 打印摘要
    print(f"\n  径流系数评价摘要:")
    total_area = sum(r['area_km2'] for r in zone_results)
    weighted_coeff = sum(r['runoff_coefficient'] * r['area_km2'] for r in zone_results) / total_area
    print(f"    - 流域总面积: {total_area:.2f} km²")
    print(f"    - 面积加权平均径流系数: {weighted_coeff:.4f}")
    print(f"    - 分区详情:")
    for r in sorted(zone_results, key=lambda x: x['zone_id']):
        print(f"      Zone {r['zone_id']}: 系数={r['runoff_coefficient']:.4f}, "
              f"降雨={r['total_precip_mm']:.1f}mm, 径流={r['total_runoff_mm']:.1f}mm")

    return zone_results

def validate_runoff_results(zone_results):
    """闭环验证：检查径流系数是否合理

    验证规则：
    1. 径流系数应在0-1之间
    2. 典型流域径流系数应在0.1-0.8之间
    3. 降雨量应>0
    4. 径流量应<=降雨量

    Returns:
        validation_passed (bool): 验证是否通过
        errors (list): 错误列表
        warnings (list): 警告列表
    """
    errors = []
    warnings = []

    print("\n" + "="*80)
    print("⚙ 闭环验证：径流系数合理性检查")
    print("="*80)

    for result in zone_results:
        zone_id = result['zone_id']
        coeff = result['runoff_coefficient']
        precip = result['total_precip_mm']
        runoff = result['total_runoff_mm']

        # 1. 检查径流系数范围
        if coeff < 0:
            errors.append(f"Zone {zone_id}: 径流系数为负值 ({coeff:.4f})，这是不合理的")
        elif coeff > 1.0:
            errors.append(f"Zone {zone_id}: 径流系数>1 ({coeff:.4f})，径流不能超过降雨")

        # 2. 检查是否在典型范围内
        if 0 <= coeff < 0.05:
            warnings.append(f"Zone {zone_id}: 径流系数过低 ({coeff:.4f})，可能存在数据问题或下渗极强")
        elif coeff > 0.9:
            warnings.append(f"Zone {zone_id}: 径流系数过高 ({coeff:.4f})，接近不透水表面")

        # 3. 检查降雨量
        if precip <= 0:
            errors.append(f"Zone {zone_id}: 累积降雨量为0或负值 ({precip:.2f}mm)，无法计算径流系数")

        # 4. 检查径流<=降雨
        if runoff > precip + 0.01:  # 允许0.01mm的数值误差
            errors.append(f"Zone {zone_id}: 径流量({runoff:.2f}mm)大于降雨量({precip:.2f}mm)，违反水量平衡")

    # 打印验证结果
    validation_passed = len(errors) == 0

    if errors:
        print("\n❌ 发现错误:")
        for err in errors:
            print(f"  • {err}")

    if warnings:
        print("\n⚠️  警告:")
        for warn in warnings:
            print(f"  • {warn}")

    if validation_passed and not warnings:
        print("\n✅ 验证通过：所有径流系数均在合理范围内")
    elif validation_passed:
        print("\n✅ 验证通过：无错误，但有警告需要注意")
    else:
        print("\n❌ 验证失败：发现严重错误，请检查输入数据和计算逻辑")

    print("="*80 + "\n")

    return validation_passed, errors, warnings

def plot_zone_runoff_timeseries(discharge_df, zones, subbasins_csv, output_dir):
    """绘制各分区径流时间序列"""

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(zones)))

    for idx, zone in enumerate(sorted(zones, key=lambda x: x['zone_id'])):
        zone_id = zone['zone_id']

        # 获取该分区的所有子流域
        subbasin_ids = get_zone_subbasins(subbasins_csv, zone_id)

        # 计算分区总径流 (m³/s)
        zone_discharge = pd.Series(0.0, index=discharge_df.index)
        for sid in subbasin_ids:
            if str(sid) in discharge_df.columns:
                zone_discharge += discharge_df[str(sid)]

        # 绘制
        time_hours = np.arange(len(zone_discharge))
        ax.plot(time_hours, zone_discharge, linewidth=2,
               color=colors[idx], label=f'Zone {zone_id}', alpha=0.8)

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax.set_title('Runoff Hydrograph by Parameter Zone', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    output_path = output_dir / "9.3_zone_runoff_timeseries.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 生成分区径流过程线: {output_path.name}")

def main():
    """主函数"""
    print("\n" + "="*80)
    print("Step 9 输出增强：径流累积曲线与径流系数评价")
    print("="*80)

    # 路径设置
    base_dir = Path("results/upper_truckee_complete_11steps")
    output_dir = base_dir / "step_09_runoff"
    output_dir.mkdir(exist_ok=True)

    # 输入文件
    discharge_path = base_dir / "step_10_routing" / "10.1_discharge_timeseries.csv"
    precip_path = base_dir / "step_08_areal_rainfall" / "8.1_parameter_areal_precipitation.csv"
    zones_path = base_dir / "parameters" / "parameter_zones.geojson"
    subbasins_csv = base_dir / "parameters" / "parameter_subbasins.csv"

    # 检查文件存在性
    for path in [discharge_path, precip_path, zones_path, subbasins_csv]:
        if not path.exists():
            print(f"  ❌ 文件不存在: {path}")
            return

    # 1. 加载数据
    print("\n⚙ 加载数据...")
    discharge_df = load_discharge_data(discharge_path)
    precip_df = load_precipitation_data(precip_path)
    zones = load_zone_info(zones_path)

    print(f"  ✓ 加载{len(discharge_df)}小时径流数据")
    print(f"  ✓ 加载{len(precip_df)}小时降雨数据 (子流域级别)")
    print(f"  ✓ 加载{len(zones)}个参数分区")

    # 2. 聚合降雨数据到分区级别
    print("\n⚙ 聚合降雨数据到分区级别...")
    zone_precip_df = aggregate_precip_to_zones(precip_df, subbasins_csv)
    print(f"  ✓ 完成聚合，生成{len(zone_precip_df.columns)}个分区的降雨数据")

    # 3. 绘制分区径流累积曲线
    print("\n⚙ 生成分区径流累积曲线...")
    zone_results = plot_zone_cumulative_runoff(discharge_df, zone_precip_df, zones,
                                               subbasins_csv, output_dir)

    # 4. 评价径流系数
    print("\n⚙ 评价径流系数...")
    evaluate_runoff_coefficients(zone_results, output_dir)

    # 5. 闭环验证径流系数
    validation_passed, errors, warnings = validate_runoff_results(zone_results)

    # 6. 绘制分区径流时间序列
    print("\n⚙ 生成分区径流过程线...")
    plot_zone_runoff_timeseries(discharge_df, zones, subbasins_csv, output_dir)

    # 7. 生成结果报告
    print("\n⚙ 生成结果报告...")
    report_path = output_dir / "9.4_enhancement_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Step 9 增强输出报告：径流分析与评价\n")
        f.write("="*80 + "\n\n")

        f.write("生成文件清单:\n")
        f.write("  1. 9.1_zone_cumulative_runoff.png - 各分区径流累积曲线\n")
        f.write("  2. 9.2_runoff_coefficient_evaluation.csv - 径流系数评价\n")
        f.write("  3. 9.3_zone_runoff_timeseries.png - 各分区径流过程线\n")
        f.write("  4. 9.4_enhancement_report.txt - 本报告\n\n")

        f.write("径流系数评价结果:\n")
        total_area = sum(r['area_km2'] for r in zone_results)
        weighted_coeff = sum(r['runoff_coefficient'] * r['area_km2']
                           for r in zone_results) / total_area
        f.write(f"  - 流域总面积: {total_area:.2f} km²\n")
        f.write(f"  - 面积加权平均径流系数: {weighted_coeff:.4f}\n\n")

        f.write("各分区径流系数详情:\n")
        for r in sorted(zone_results, key=lambda x: x['zone_id']):
            f.write(f"  - Zone {r['zone_id']}: {r['runoff_coefficient']:.4f}, "
                   f"降雨={r['total_precip_mm']:.1f}mm, 径流={r['total_runoff_mm']:.1f}mm\n")

        # 添加验证结果
        f.write("\n" + "="*80 + "\n")
        f.write("闭环验证结果:\n")
        if validation_passed and not warnings:
            f.write("  ✅ 验证状态: 通过 (无错误，无警告)\n")
        elif validation_passed:
            f.write("  ✅ 验证状态: 通过 (无错误，但有警告)\n")
        else:
            f.write("  ❌ 验证状态: 失败 (发现错误)\n")

        if errors:
            f.write("\n  错误列表:\n")
            for err in errors:
                f.write(f"    • {err}\n")

        if warnings:
            f.write("\n  警告列表:\n")
            for warn in warnings:
                f.write(f"    • {warn}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("径流系数等级标准:\n")
        f.write("  - 极高: ≥0.7 (湿润区或不透水地表)\n")
        f.write("  - 高: 0.5-0.7 (较湿润或土壤饱和)\n")
        f.write("  - 中等: 0.3-0.5 (正常产流)\n")
        f.write("  - 偏低: 0.1-0.3 (干旱或高渗透)\n")
        f.write("  - 很低: <0.1 (极干旱或高下渗)\n")
        f.write("="*80 + "\n")

    print(f"  ✓ 生成结果报告: {report_path.name}")

    print("\n" + "="*80)
    print(f"✅ Step 9输出增强完成！共生成4个文件")
    print(f"📁 输出目录: {output_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
