#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断HBV模型的水量平衡和单位转换问题

使用统一的诊断框架重构版本。

用法:
    python scripts/diagnostics/diagnose_water_balance.py

输入:
    - results/extended_timeseries_60days/timeseries_60days.csv

输出:
    - results/extended_timeseries_60days/water_balance_diagnosis.txt
    - results/extended_timeseries_60days/水量平衡诊断_report.txt
    - results/extended_timeseries_60days/水量平衡诊断_report.json
    - results/extended_timeseries_60days/水量平衡诊断_visualization.png
"""
import sys
sys.path.insert(0, '/home/user/HydroSIS')

from pathlib import Path
import pandas as pd

from hydrosis.diagnostics import WaterBalanceDiagnostic, IssueSeverity


def main():
    """诊断水量平衡"""
    print("=" * 80)
    print("HBV模型水量平衡诊断 (使用统一诊断框架)")
    print("=" * 80)

    # 读取生成的时间序列数据
    data_path = 'results/extended_timeseries_60days/timeseries_60days.csv'
    print(f"\n📊 读取数据: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {data_path}")
        print("   请先运行生成时间序列的脚本")
        return

    precipitation = df['precipitation_mm_per_hour'].values
    runoff = df['runoff_mm_per_hour'].values

    # HBV参数（从原脚本）
    k2 = 0.02
    initial_lower = 4498.9

    print(f"✓ 数据加载成功: {len(precipitation)} 小时")
    print(f"  降雨总量: {precipitation.sum():.2f} mm")
    print(f"  径流总量: {runoff.sum():.2f} mm")

    # 创建诊断器
    output_dir = Path('results/extended_timeseries_60days')
    diagnostic = WaterBalanceDiagnostic(
        output_dir=output_dir,
        verbose=True,
        target_runoff_coefficient=0.5  # 目标径流系数
    )

    # 运行诊断
    print("\n🔍 运行水量平衡诊断...")
    result = diagnostic.run(
        precipitation=precipitation,
        runoff=runoff,
        initial_lower=initial_lower,
        k2=k2
    )

    # 显示基本统计
    print("\n" + "=" * 80)
    print("诊断结果摘要")
    print("=" * 80)

    print(f"\n📊 关键指标:")
    print(f"  总降雨量: {result.metrics['total_precipitation_mm']:.2f} mm")
    print(f"  总径流量: {result.metrics['total_runoff_mm']:.2f} mm")
    print(f"  径流系数: {result.metrics['runoff_coefficient']:.4f}")
    print(f"  额外径流: {result.metrics['excess_runoff_mm']:.2f} mm")
    print(f"  初始储量: {result.metrics['initial_lower_mm']:.2f} mm")
    print(f"  初始基流: {result.metrics['initial_baseflow_mm_per_hour']:.2f} mm/h")

    # 显示问题（按严重程度分组）
    print(f"\n⚠️  发现的问题: {len(result.issues)}")

    for severity in [IssueSeverity.CRITICAL, IssueSeverity.ERROR,
                     IssueSeverity.WARNING, IssueSeverity.INFO]:
        issues = result.get_issues_by_severity(severity)
        if issues:
            severity_icon = {
                IssueSeverity.CRITICAL: "🔴",
                IssueSeverity.ERROR: "❌",
                IssueSeverity.WARNING: "⚠️",
                IssueSeverity.INFO: "ℹ️"
            }[severity]

            print(f"\n{severity_icon} {severity.value.upper()} ({len(issues)}):")
            for issue in issues:
                print(f"  • {issue.message}")
                if issue.suggestion:
                    print(f"    💡 建议: {issue.suggestion}")

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

    legacy_report_path = output_dir / 'water_balance_diagnosis.txt'
    generate_legacy_report(result, legacy_report_path)
    print(f"✓ 传统报告已保存: {legacy_report_path}")

    # 显示所有输出文件
    print("\n" + "=" * 80)
    print("输出文件")
    print("=" * 80)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"  1. {legacy_report_path.name} (传统格式)")
    print(f"  2. 水量平衡诊断_report.txt (新格式)")
    print(f"  3. 水量平衡诊断_report.json (JSON格式)")
    print(f"  4. 水量平衡诊断_visualization.png (可视化)")

    print("\n✅ 诊断完成！")


def generate_legacy_report(result, output_path):
    """生成传统格式的报告（向后兼容）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HBV模型水量平衡诊断报告\n")
        f.write("=" * 80 + "\n\n")

        # 基本信息
        rc = result.metrics['runoff_coefficient']
        total_precip = result.metrics['total_precipitation_mm']
        total_runoff = result.metrics['total_runoff_mm']
        excess_runoff = result.metrics['excess_runoff_mm']

        f.write("问题: 径流系数异常\n")
        f.write(f"  径流系数: {rc:.4f}")
        if rc > 1.0:
            f.write(" (>> 1.0, 不合理)\n")
        else:
            f.write(" (合理)\n")
        f.write(f"  总降雨量: {total_precip:.2f} mm\n")
        f.write(f"  总径流量: {total_runoff:.2f} mm\n")
        f.write(f"  额外径流: {excess_runoff:.2f} mm\n\n")

        # 原因分析
        f.write("原因分析:\n")
        initial_lower = result.metrics['initial_lower_mm']
        initial_baseflow = result.metrics['initial_baseflow_mm_per_hour']
        released_storage = result.metrics.get('released_storage_mm', 0)

        f.write(f"  initial_lower = {initial_lower:.2f} mm")
        if rc > 1.0:
            f.write(" (过大)\n")
        else:
            f.write("\n")
        f.write(f"  初始基流 = k2 * initial_lower = {initial_baseflow:.2f} mm/h\n")
        f.write(f"  预测释放储量 ≈ {released_storage:.2f} mm\n\n")

        # 建议修正
        f.write("建议修正:\n")
        for i, rec in enumerate(result.recommendations, 1):
            # 提取修正值
            if "initial_lower" in rec and "mm" in rec:
                f.write(f"  方案{i}: {rec}\n")
        f.write("\n")

        # 说明
        f.write("说明:\n")
        f.write("  本报告由HydroSIS统一诊断框架自动生成。\n")
        f.write("  框架提供了标准化的问题识别、严重程度分级和修复建议。\n")
        f.write("  更多详细信息请查看同目录下的其他诊断报告文件。\n")


if __name__ == '__main__':
    main()
