#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断分区2降雨异常问题

使用统一的诊断框架重构版本。

分析为什么分区2的降雨量只有344mm，而其他分区是600-900mm

用法:
    python scripts/diagnostics/diagnose_zone2_precipitation.py

输入:
    - results/upper_truckee_complete_11steps/step_08_areal_rainfall/8.1_parameter_areal_precipitation.csv
    - results/upper_truckee_complete_11steps/parameters/parameter_zones.geojson
    - results/upper_truckee_complete_11steps/parameters/parameter_subbasins.csv

输出:
    - results/upper_truckee_complete_11steps/diagnostics/降雨空间分布诊断_report.txt
    - results/upper_truckee_complete_11steps/diagnostics/降雨空间分布诊断_report.json
    - results/upper_truckee_complete_11steps/diagnostics/降雨空间分布诊断_visualization.png
    - results/upper_truckee_complete_11steps/diagnostics/zone_*_precipitation_detail.png
    - results/upper_truckee_complete_11steps/diagnostics/zone2_diagnosis_report.txt (传统格式)
"""
import sys
sys.path.insert(0, '/home/user/HydroSIS')

import json
from pathlib import Path
import pandas as pd
import numpy as np

from hydrosis.diagnostics import PrecipitationDiagnostic, IssueSeverity


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


def main():
    """主函数"""
    print("=" * 80)
    print("分区2降雨异常诊断工具 (使用统一诊断框架)")
    print("=" * 80)

    # 数据路径
    base_dir = Path("results/upper_truckee_complete_11steps")
    precip_path = base_dir / "step_08_areal_rainfall" / "8.1_parameter_areal_precipitation.csv"
    zones_path = base_dir / "parameters" / "parameter_zones.geojson"
    subbasins_path = base_dir / "parameters" / "parameter_subbasins.csv"
    output_dir = base_dir / "diagnostics"
    output_dir.mkdir(exist_ok=True)

    # 加载数据
    print("\n⚙ 加载数据...")
    try:
        precip_df = pd.read_csv(precip_path, index_col='Timestamp')
        zones = load_zone_info(zones_path)
        subbasins_df = pd.read_csv(subbasins_path)

        print(f"  ✓ 降雨数据: {len(precip_df)} 小时, {len(precip_df.columns)} 个子流域")
        print(f"  ✓ 分区信息: {len(zones)} 个分区")
        print(f"  ✓ 子流域信息: {len(subbasins_df)} 个子流域")

    except FileNotFoundError as e:
        print(f"\n❌ 错误: 找不到文件")
        print(f"   {e}")
        print("\n请确保已运行完整的11步工作流")
        return

    # 创建诊断器
    diagnostic = PrecipitationDiagnostic(
        output_dir=output_dir,
        verbose=True,
        anomaly_threshold=0.3  # 30%差异视为异常
    )

    # 运行诊断
    print("\n🔍 运行降雨空间分布诊断...")
    result = diagnostic.run(
        precipitation_df=precip_df,
        subbasins_df=subbasins_df
    )

    # 显示结果摘要
    print("\n" + "=" * 80)
    print("诊断结果摘要")
    print("=" * 80)

    # 分区降雨统计
    print(f"\n📊 各分区降雨统计:")
    print(f"{'分区ID':>8} {'面积加权平均(mm)':>20} {'变异系数(CV)':>15}")
    print("-" * 50)

    zone_ids = sorted(set(subbasins_df['zone_id']))
    for zone_id in zone_ids:
        avg_key = f"zone_{zone_id}_weighted_avg_mm"
        cv_key = f"zone_{zone_id}_cv"

        if avg_key in result.metrics:
            avg = result.metrics[avg_key]
            cv = result.metrics.get(cv_key, 0)
            print(f"{zone_id:>8} {avg:>20.2f} {cv:>15.4f}")

    # 显示问题
    print(f"\n⚠️  发现的问题: {len(result.issues)}")

    # 分区降雨异常
    anomalies = [
        issue for issue in result.issues
        if issue.category == "zone_precipitation_anomaly"
    ]

    if anomalies:
        print(f"\n🔴 降雨异常分区 ({len(anomalies)}):")
        for issue in anomalies:
            zone_id = issue.details.get('zone_id', 'unknown')
            zone_precip = issue.details.get('zone_precipitation', 0)
            overall_mean = issue.details.get('overall_mean', 0)
            diff = issue.details.get('relative_difference', 0)

            print(f"  • Zone {zone_id}: {zone_precip:.1f}mm "
                  f"(平均{overall_mean:.1f}mm, 差异{diff*100:+.1f}%)")
            if issue.suggestion:
                print(f"    💡 {issue.suggestion}")

    # 数据质量问题
    quality_issues = [
        issue for issue in result.issues
        if issue.category == "data_quality"
    ]

    if quality_issues:
        print(f"\n⚠️  数据质量问题 ({len(quality_issues)}):")
        for issue in quality_issues:
            print(f"  • {issue.message}")

    # 显示修复建议
    if result.recommendations:
        print("\n" + "=" * 80)
        print("修复建议")
        print("=" * 80)
        for i, rec in enumerate(result.recommendations, 1):
            print(f"\n{i}. {rec}")

    # 生成传统格式的报告（向后兼容）
    print("\n" + "=" * 80)
    print("生成传统格式报告（向后兼容）")
    print("=" * 80)

    legacy_report_path = output_dir / "zone2_diagnosis_report.txt"
    generate_legacy_report(result, zones, subbasins_df, legacy_report_path)
    print(f"✓ 传统报告已保存: {legacy_report_path}")

    # 显示所有输出文件
    print("\n" + "=" * 80)
    print("输出文件")
    print("=" * 80)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"  1. zone2_diagnosis_report.txt (传统格式)")
    print(f"  2. 降雨空间分布诊断_report.txt (新格式)")
    print(f"  3. 降雨空间分布诊断_report.json (JSON格式)")
    print(f"  4. 降雨空间分布诊断_visualization.png (分区对比)")

    # 列出各分区详细图
    detail_figs = list(output_dir.glob("zone_*_precipitation_detail.png"))
    if detail_figs:
        print(f"\n  详细诊断图 ({len(detail_figs)}个):")
        for fig in sorted(detail_figs):
            print(f"    • {fig.name}")

    print("\n✅ 诊断完成！请查看输出目录下的结果文件")


def generate_legacy_report(result, zones, subbasins_df, output_path):
    """生成传统格式的报告（向后兼容）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("分区2降雨异常诊断报告\n")
        f.write("=" * 80 + "\n\n")

        # 获取分区2数据
        zone2_avg = result.metrics.get('zone_2_weighted_avg_mm', 0)

        # 计算其他分区平均
        other_zones_precip = []
        zone_ids = sorted(set(subbasins_df['zone_id']))
        for zone_id in zone_ids:
            if zone_id != 2:
                avg_key = f"zone_{zone_id}_weighted_avg_mm"
                if avg_key in result.metrics:
                    other_zones_precip.append(result.metrics[avg_key])

        other_zones_avg = np.mean(other_zones_precip) if other_zones_precip else 0

        # 问题描述
        f.write("1. 问题描述\n")
        f.write(f"  - 分区2降雨量: {zone2_avg:.2f} mm\n")
        f.write(f"  - 其他分区平均: {other_zones_avg:.2f} mm\n")
        if other_zones_avg > 0:
            diff_pct = (zone2_avg / other_zones_avg - 1) * 100
            f.write(f"  - 差异: {diff_pct:.1f}%\n")
        f.write("\n")

        # 可能原因
        f.write("2. 可能原因\n")
        f.write("  a) 雨量站分布不均 - 分区2可能缺少雨量站覆盖\n")
        f.write("  b) Thiessen多边形权重问题 - 权重分配不合理\n")
        f.write("  c) 合成降雨数据问题 - 随机生成的降雨场分布不均\n")
        f.write("  d) 子流域划分问题 - 分区2子流域面积分布异常\n")
        f.write("\n")

        # 建议措施
        f.write("3. 建议措施\n")
        if result.recommendations:
            for i, rec in enumerate(result.recommendations, 1):
                f.write(f"  {chr(96+i)}) {rec}\n")
        else:
            f.write("  a) 检查雨量站位置和Thiessen多边形\n")
            f.write("  b) 使用真实降雨数据替代合成数据\n")
            f.write("  c) 优化降雨插值方法(如IDW或Kriging)\n")
            f.write("  d) 增加分区2的雨量站密度\n")
        f.write("\n")

        # 说明
        f.write("说明:\n")
        f.write("  本报告由HydroSIS统一诊断框架自动生成。\n")
        f.write("  框架自动检测了降雨空间分布异常、数据质量问题，\n")
        f.write("  并生成了详细的可视化图表。\n")
        f.write("  更多详细信息请查看同目录下的其他诊断报告文件。\n")


if __name__ == "__main__":
    main()
