#!/usr/bin/env python3
"""诊断分区2降雨异常问题

分析为什么分区2的降雨量只有344mm，而其他分区是600-900mm
"""
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_zone_info(geojson_path):
    """加载分区信息"""
    with open(geojson_path) as f:
        data = json.load(f)

    zones = []
    for feature in data['features']:
        props = feature['properties']
        zones.append({
            'zone_id': props['zone_id'],
            'area_km2': props['area_km2']
        })
    return zones

def load_subbasin_info(csv_path):
    """加载子流域信息"""
    return pd.read_csv(csv_path)

def analyze_zone_precipitation(zone_id, precip_df, subbasins_df, output_dir):
    """分析指定分区的降雨分布"""
    print(f"\n{'='*80}")
    print(f"分区 {zone_id} 降雨诊断分析")
    print(f"{'='*80}")

    # 找到该分区的所有子流域
    zone_subbasins = subbasins_df[subbasins_df['zone_id'] == zone_id]

    print(f"\n1. 基本信息")
    print(f"  - 子流域数量: {len(zone_subbasins)}")
    print(f"  - 总面积: {zone_subbasins['area_km2'].sum():.2f} km²")

    # 分析每个子流域的降雨
    print(f"\n2. 子流域降雨统计")
    print(f"  {'子流域ID':>10} {'面积(km²)':>12} {'总降雨(mm)':>14} {'平均雨强(mm/h)':>18} {'权重':>8}")
    print(f"  {'-'*10} {'-'*12} {'-'*14} {'-'*18} {'-'*8}")

    total_area = zone_subbasins['area_km2'].sum()
    subbasin_precip = []

    for _, subbasin in zone_subbasins.iterrows():
        subbasin_id = str(int(subbasin['subzone_id']))
        area = subbasin['area_km2']
        weight = area / total_area

        if subbasin_id in precip_df.columns:
            precip_series = precip_df[subbasin_id]
            total_precip = precip_series.sum()
            mean_intensity = precip_series.mean()

            subbasin_precip.append({
                'subbasin_id': subbasin_id,
                'area_km2': area,
                'total_precip_mm': total_precip,
                'mean_intensity_mm_h': mean_intensity,
                'weight': weight,
                'weighted_precip': total_precip * weight
            })

            print(f"  {subbasin_id:>10} {area:>12.2f} {total_precip:>14.2f} {mean_intensity:>18.4f} {weight:>8.4f}")
        else:
            print(f"  {subbasin_id:>10} {area:>12.2f} {'N/A':>14} {'N/A':>18} {weight:>8.4f}")

    # 计算面积加权平均降雨
    if subbasin_precip:
        weighted_avg = sum(s['weighted_precip'] for s in subbasin_precip)
        min_precip = min(s['total_precip_mm'] for s in subbasin_precip)
        max_precip = max(s['total_precip_mm'] for s in subbasin_precip)

        print(f"\n3. 分区汇总")
        print(f"  - 面积加权平均降雨: {weighted_avg:.2f} mm")
        print(f"  - 最小降雨: {min_precip:.2f} mm")
        print(f"  - 最大降雨: {max_precip:.2f} mm")
        print(f"  - 变异系数: {np.std([s['total_precip_mm'] for s in subbasin_precip]) / np.mean([s['total_precip_mm'] for s in subbasin_precip]):.4f}")

        # 可视化
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 子图1: 子流域降雨柱状图
        ax1 = axes[0, 0]
        subbasin_ids = [s['subbasin_id'] for s in subbasin_precip]
        precip_values = [s['total_precip_mm'] for s in subbasin_precip]
        colors = ['red' if p < 400 else 'orange' if p < 600 else 'green' for p in precip_values]

        ax1.bar(range(len(subbasin_ids)), precip_values, color=colors)
        ax1.axhline(y=weighted_avg, color='blue', linestyle='--', linewidth=2, label=f'加权平均: {weighted_avg:.1f}mm')
        ax1.set_xlabel('子流域索引')
        ax1.set_ylabel('总降雨 (mm)')
        ax1.set_title(f'分区{zone_id}各子流域降雨分布')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 降雨-面积权重散点图
        ax2 = axes[0, 1]
        areas = [s['area_km2'] for s in subbasin_precip]
        weights = [s['weight'] for s in subbasin_precip]
        scatter = ax2.scatter(precip_values, weights, s=[a*10 for a in areas], c=precip_values, cmap='RdYlGn', alpha=0.6)
        ax2.set_xlabel('总降雨 (mm)')
        ax2.set_ylabel('面积权重')
        ax2.set_title(f'分区{zone_id}降雨与权重关系 (气泡大小=面积)')
        plt.colorbar(scatter, ax=ax2, label='降雨量(mm)')
        ax2.grid(True, alpha=0.3)

        # 子图3: 时间序列
        ax3 = axes[1, 0]
        for s in subbasin_precip[:5]:  # 只显示前5个避免拥挤
            subbasin_id = s['subbasin_id']
            ax3.plot(precip_df[subbasin_id].values, label=f'子流域{subbasin_id}', alpha=0.7)
        ax3.set_xlabel('时间步 (小时)')
        ax3.set_ylabel('降雨强度 (mm/h)')
        ax3.set_title(f'分区{zone_id}降雨时间序列 (前5个子流域)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)

        # 子图4: 降雨分布直方图
        ax4 = axes[1, 1]
        ax4.hist(precip_values, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        ax4.axvline(x=weighted_avg, color='red', linestyle='--', linewidth=2, label=f'加权平均: {weighted_avg:.1f}mm')
        ax4.set_xlabel('总降雨 (mm)')
        ax4.set_ylabel('子流域数量')
        ax4.set_title(f'分区{zone_id}降雨分布直方图')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f'zone_{zone_id}_precipitation_diagnosis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\n4. 可视化已保存: {output_path}")

        return {
            'zone_id': zone_id,
            'weighted_avg_precip': weighted_avg,
            'min_precip': min_precip,
            'max_precip': max_precip,
            'num_subbasins': len(subbasin_precip),
            'subbasin_details': subbasin_precip
        }

    return None

def compare_all_zones(precip_df, subbasins_df, zones):
    """比较所有分区的降雨"""
    print(f"\n{'='*80}")
    print("所有分区降雨对比")
    print(f"{'='*80}")

    print(f"\n{'分区ID':>8} {'子流域数':>10} {'总面积(km²)':>14} {'加权平均降雨(mm)':>20} {'最小(mm)':>12} {'最大(mm)':>12}")
    print(f"{'-'*8} {'-'*10} {'-'*14} {'-'*20} {'-'*12} {'-'*12}")

    zone_stats = []
    for zone in sorted(zones, key=lambda x: x['zone_id']):
        zone_id = zone['zone_id']
        zone_subbasins = subbasins_df[subbasins_df['zone_id'] == zone_id]
        total_area = zone_subbasins['area_km2'].sum()

        precip_values = []
        for _, subbasin in zone_subbasins.iterrows():
            subbasin_id = str(int(subbasin['subzone_id']))
            if subbasin_id in precip_df.columns:
                weight = subbasin['area_km2'] / total_area
                precip = precip_df[subbasin_id].sum()
                precip_values.append((precip, weight))

        if precip_values:
            weighted_avg = sum(p * w for p, w in precip_values)
            min_p = min(p for p, w in precip_values)
            max_p = max(p for p, w in precip_values)

            zone_stats.append({
                'zone_id': zone_id,
                'num_subbasins': len(zone_subbasins),
                'area_km2': total_area,
                'weighted_avg': weighted_avg,
                'min': min_p,
                'max': max_p
            })

            print(f"{zone_id:>8} {len(zone_subbasins):>10} {total_area:>14.2f} {weighted_avg:>20.2f} {min_p:>12.2f} {max_p:>12.2f}")

    return zone_stats

def main():
    """主函数"""
    print("\n" + "="*80)
    print("分区2降雨异常诊断工具")
    print("="*80)

    # 数据路径
    base_dir = Path("results/upper_truckee_complete_11steps")
    precip_path = base_dir / "step_08_areal_rainfall" / "8.1_parameter_areal_precipitation.csv"
    zones_path = base_dir / "parameters" / "parameter_zones.geojson"
    subbasins_path = base_dir / "parameters" / "parameter_subbasins.csv"
    output_dir = base_dir / "diagnostics"
    output_dir.mkdir(exist_ok=True)

    # 加载数据
    print("\n⚙ 加载数据...")
    precip_df = pd.read_csv(precip_path, index_col='Timestamp')
    zones = load_zone_info(zones_path)
    subbasins_df = load_subbasin_info(subbasins_path)

    print(f"  ✓ 降雨数据: {len(precip_df)} 小时, {len(precip_df.columns)} 个子流域")
    print(f"  ✓ 分区信息: {len(zones)} 个分区")
    print(f"  ✓ 子流域信息: {len(subbasins_df)} 个子流域")

    # 比较所有分区
    zone_stats = compare_all_zones(precip_df, subbasins_df, zones)

    # 详细分析分区2
    zone2_result = analyze_zone_precipitation(2, precip_df, subbasins_df, output_dir)

    # 对比分析: 分区2 vs 其他分区
    print(f"\n{'='*80}")
    print("对比分析: 分区2 vs 其他分区")
    print(f"{'='*80}")

    zone2_avg = next(z['weighted_avg'] for z in zone_stats if z['zone_id'] == 2)
    other_zones_avg = np.mean([z['weighted_avg'] for z in zone_stats if z['zone_id'] != 2])

    print(f"\n分区2平均降雨: {zone2_avg:.2f} mm")
    print(f"其他分区平均降雨: {other_zones_avg:.2f} mm")
    print(f"差异: {zone2_avg - other_zones_avg:.2f} mm ({(zone2_avg/other_zones_avg - 1)*100:.1f}%)")

    # 生成诊断报告
    report_path = output_dir / "zone2_diagnosis_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("分区2降雨异常诊断报告\n")
        f.write("="*80 + "\n\n")
        f.write("1. 问题描述\n")
        f.write(f"  - 分区2降雨量: {zone2_avg:.2f} mm\n")
        f.write(f"  - 其他分区平均: {other_zones_avg:.2f} mm\n")
        f.write(f"  - 差异: {(zone2_avg/other_zones_avg - 1)*100:.1f}%\n\n")
        f.write("2. 可能原因\n")
        f.write("  a) 雨量站分布不均 - 分区2可能缺少雨量站覆盖\n")
        f.write("  b) Thiessen多边形权重问题 - 权重分配不合理\n")
        f.write("  c) 合成降雨数据问题 - 随机生成的降雨场分布不均\n")
        f.write("  d) 子流域划分问题 - 分区2子流域面积分布异常\n\n")
        f.write("3. 建议措施\n")
        f.write("  a) 检查雨量站位置和Thiessen多边形\n")
        f.write("  b) 使用真实降雨数据替代合成数据\n")
        f.write("  c) 优化降雨插值方法(如IDW或Kriging)\n")
        f.write("  d) 增加分区2的雨量站密度\n\n")

    print(f"\n✓ 诊断报告已保存: {report_path}")
    print(f"✓ 诊断完成！请查看 {output_dir} 目录下的结果文件")

if __name__ == "__main__":
    main()
