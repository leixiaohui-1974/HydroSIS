#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HBV模型配置诊断脚本

使用统一的诊断框架重构版本。

全面检查HBV模型配置的合理性：
1. 单位转换是否正确（mm/h ↔ m³/s）
2. 水量平衡是否合理（降雨总量 vs 径流总量）
3. 径流系数是否在合理范围（0.3-0.7）
4. 初始状态估计是否合理
5. HBV参数设置的合理性

用法:
    python scripts/diagnostics/diagnose_hbv_configuration.py

输入:
    - results/upper_truckee_complete_11steps/step_08_areal_rainfall/8.2_subbasin_areal_precipitation.csv
    - results/upper_truckee_complete_11steps/enhanced_observations/zone_1_enhanced_runoff.csv

输出:
    - results/upper_truckee_complete_11steps/configuration_diagnosis/HBV配置诊断_report.txt
    - results/upper_truckee_complete_11steps/configuration_diagnosis/HBV配置诊断_report.json
    - results/upper_truckee_complete_11steps/configuration_diagnosis/HBV配置诊断_visualization.png
    - results/upper_truckee_complete_11steps/configuration_diagnosis/configuration_diagnosis_report.txt (传统格式)
"""
import sys
sys.path.insert(0, '/home/user/HydroSIS')

from pathlib import Path
import numpy as np
import pandas as pd

from hydrosis.diagnostics import HBVConfigurationDiagnostic, IssueSeverity


def main():
    """主函数"""
    print("=" * 80)
    print("HBV模型配置诊断工具 (使用统一诊断框架)")
    print("=" * 80)

    # 数据路径
    results_dir = Path("results/upper_truckee_complete_11steps")
    output_dir = results_dir / "configuration_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================================
    # 步骤 1: 加载数据
    # ============================================================================
    print("\n⚙ 步骤 1: 加载数据")
    print("-" * 80)

    # 1.1 加载降雨数据
    precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"

    try:
        precip_df = pd.read_csv(precip_file, index_col=0)
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {precip_file}")
        print("   请确保已运行完整的11步工作流")
        return

    # Zone 1的子分区
    zone1_subbasins = [str(i) for i in range(10, 24)]
    zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
    precipitation_mmh = precip_df[zone1_cols].mean(axis=1).values

    print(f"✓ 降雨数据加载:")
    print(f"  文件: {precip_file.name}")
    print(f"  Zone 1子分区: {len(zone1_cols)}个")
    print(f"  时间步数: {len(precipitation_mmh)}")
    print(f"  范围: {precipitation_mmh.min():.2f} - {precipitation_mmh.max():.2f} mm/h")

    # 1.2 加载增强型观测径流
    obs_file = results_dir / "enhanced_observations" / "zone_1_enhanced_runoff.csv"

    try:
        obs_df = pd.read_csv(obs_file)
        observed_m3s = obs_df['discharge_m3s'].values
        times = pd.to_datetime(obs_df['datetime'])
    except FileNotFoundError:
        print(f"❌ 错误: 找不到文件 {obs_file}")
        print("   请确保已生成增强型观测数据")
        return

    print(f"\n✓ 观测径流数据加载:")
    print(f"  文件: {obs_file.name}")
    print(f"  时间步数: {len(observed_m3s)}")
    print(f"  范围: {observed_m3s.min():.2f} - {observed_m3s.max():.2f} m³/s")

    # 1.3 流域面积
    zone1_area_km2 = 139.995
    print(f"\n✓ Zone 1流域面积: {zone1_area_km2:.2f} km²")

    # 1.4 验证时间步长
    time_diffs = (times[1:] - times[:-1]).dt.total_seconds() / 3600
    if len(set(time_diffs)) == 1:
        timestep_hours = time_diffs[0]
        print(f"✓ 时间步长: {timestep_hours:.2f} 小时（一致）")
    else:
        print(f"⚠ 时间步长不一致: {set(time_diffs)}")
        timestep_hours = time_diffs[0]  # 使用第一个

    # ============================================================================
    # 步骤 2: 准备HBV参数
    # ============================================================================
    print("\n⚙ 步骤 2: 准备HBV参数")
    print("-" * 80)

    # 使用默认参数（注意：initial_lower将由诊断器估算）
    hbv_params = {
        'FC': 400.0,
        'BETA': 2.0,
        'K0': 0.25,
        'K1': 0.08,
        'K2': 0.02,
        'PERC': 2.0,
        'LP': 0.7,
        'MAXBAS': 3.0,
        'TT': 0.0,
        'CFMAX': 3.5,
        'CFR': 0.05,
        'CWH': 0.1,
        'initial_soil': 300.0,
        'initial_upper': 20.0,
        'initial_lower': 3000.0,  # 将与估算值对比
        'initial_snow': 0.0,
    }

    print("✓ HBV参数配置:")
    for key in ['FC', 'BETA', 'K0', 'K1', 'K2', 'PERC']:
        print(f"  {key:<10s}: {hbv_params[key]:>8.3f}")
    print("\n  初始状态:")
    for key in ['initial_soil', 'initial_upper', 'initial_lower', 'initial_snow']:
        print(f"  {key:<15s}: {hbv_params[key]:>10.2f} mm")

    # ============================================================================
    # 步骤 3: 创建诊断器并运行
    # ============================================================================
    print("\n🔍 步骤 3: 运行HBV配置诊断")
    print("-" * 80)

    diagnostic = HBVConfigurationDiagnostic(
        output_dir=output_dir,
        verbose=True,
        runoff_coeff_range=(0.3, 0.7),
        k2_estimate=0.02,
        area_km2=zone1_area_km2,
        timestep_hours=timestep_hours
    )

    # 运行诊断
    result = diagnostic.run(
        precipitation_mmh=precipitation_mmh,
        observed_m3s=observed_m3s,
        area_km2=zone1_area_km2,
        hbv_params=hbv_params,
        timestep_hours=timestep_hours
    )

    # ============================================================================
    # 步骤 4: 显示诊断结果
    # ============================================================================
    print("\n" + "=" * 80)
    print("诊断结果摘要")
    print("=" * 80)

    # 4.1 关键指标
    print("\n📊 关键指标:")
    print(f"  总降雨量:        {result.metrics['total_precipitation_mm']:>10.2f} mm")
    print(f"  总径流量(观测):  {result.metrics['total_runoff_mm']:>10.2f} mm")
    print(f"  径流系数(观测):  {result.metrics['runoff_coefficient']:>10.4f}")

    if 'hbv_total_runoff_mm' in result.metrics:
        print(f"  总径流量(HBV):   {result.metrics['hbv_total_runoff_mm']:>10.2f} mm")
        print(f"  径流系数(HBV):   {result.metrics['hbv_runoff_coefficient']:>10.4f}")
        print(f"  径流系数差异:    {result.metrics.get('runoff_coefficient_difference', 0):>10.4f}")

    print(f"\n  初始基流:        {result.metrics['initial_baseflow_m3s']:>10.2f} m³/s")
    print(f"  估算initial_lower: {result.metrics['estimated_initial_lower_mm']:>10.2f} mm")

    if 'configured_initial_lower_mm' in result.metrics:
        print(f"  配置initial_lower: {result.metrics['configured_initial_lower_mm']:>10.2f} mm")

    # 4.2 显示问题（按严重程度分组）
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
                print(f"  • [{issue.category}] {issue.message}")
                if issue.suggestion:
                    print(f"    💡 {issue.suggestion}")

    # 4.3 显示修复建议
    if result.recommendations:
        print("\n" + "=" * 80)
        print("修复建议")
        print("=" * 80)
        for i, rec in enumerate(result.recommendations, 1):
            print(f"\n{i}. {rec}")

    # ============================================================================
    # 步骤 5: 生成传统格式报告（向后兼容）
    # ============================================================================
    print("\n" + "=" * 80)
    print("生成传统格式报告（向后兼容）")
    print("=" * 80)

    legacy_report_path = output_dir / "configuration_diagnosis_report.txt"
    generate_legacy_report(result, hbv_params, legacy_report_path)
    print(f"✓ 传统报告已保存: {legacy_report_path}")

    # ============================================================================
    # 步骤 6: 显示输出文件
    # ============================================================================
    print("\n" + "=" * 80)
    print("输出文件")
    print("=" * 80)
    print(f"\n📁 输出目录: {output_dir}")
    print(f"  1. configuration_diagnosis_report.txt (传统格式)")
    print(f"  2. HBV配置诊断_report.txt (新格式)")
    print(f"  3. HBV配置诊断_report.json (JSON格式)")
    print(f"  4. HBV配置诊断_visualization.png (可视化)")

    print("\n✅ 诊断完成！请查看输出目录下的结果文件")

    # ============================================================================
    # 总结
    # ============================================================================
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)

    rc = result.metrics['runoff_coefficient']
    print(f"  径流系数(观测): {rc:.4f} {'✓ 合理' if 0.3 <= rc <= 0.7 else '⚠ 异常'}")

    if 'hbv_runoff_coefficient' in result.metrics:
        hbv_rc = result.metrics['hbv_runoff_coefficient']
        print(f"  径流系数(HBV):  {hbv_rc:.4f}")

    est_lower = result.metrics['estimated_initial_lower_mm']
    conf_lower = result.metrics.get('configured_initial_lower_mm', 0)
    print(f"  初始状态建议:   initial_lower = {est_lower:.1f} mm (当前使用{conf_lower:.0f}mm)")

    print(f"\n【下一步】")
    print(f"  1. 根据诊断结果调整HBV初始状态")
    print(f"  2. 对比简单模型/增强模型/HBV的兼容性")
    print(f"  3. 获取更长时间序列数据（当前{len(precipitation_mmh)}小时）")

    print("\n" + "=" * 80)
    print("✓ 本脚本遵循 .claude/AI_DEVELOPMENT_GUIDE.md 最佳实践")
    print("=" * 80)


def generate_legacy_report(result, hbv_params, output_path):
    """生成传统格式的报告（向后兼容）"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HBV模型配置诊断报告\n")
        f.write("=" * 80 + "\n\n")

        # 数据概览
        f.write("【数据概览】\n")
        f.write(f"  降雨数据: {len(result.metrics.get('total_precipitation_mm', 0))} 小时\n")
        f.write(f"  降雨范围: {result.metrics.get('max_precipitation_mmh', 0):.2f} mm/h (最大值)\n")
        f.write(f"  流量范围: {result.metrics.get('max_observed_m3s', 0):.2f} m³/s (最大值)\n\n")

        # 水量平衡
        f.write("【水量平衡】\n")
        total_precip = result.metrics.get('total_precipitation_mm', 0)
        total_runoff = result.metrics.get('total_runoff_mm', 0)
        rc_obs = result.metrics.get('runoff_coefficient', 0)

        f.write(f"  总降雨量:     {total_precip:.2f} mm\n")
        f.write(f"  观测总径流:   {total_runoff:.2f} mm\n")
        f.write(f"  观测径流系数: {rc_obs:.4f}\n")

        if 'hbv_total_runoff_mm' in result.metrics:
            hbv_runoff = result.metrics['hbv_total_runoff_mm']
            hbv_rc = result.metrics['hbv_runoff_coefficient']
            rc_diff = result.metrics.get('runoff_coefficient_difference', 0)

            f.write(f"  HBV总径流:    {hbv_runoff:.2f} mm\n")
            f.write(f"  HBV径流系数:  {hbv_rc:.4f}\n")
            f.write(f"  径流系数差异: {rc_diff:.4f}\n")
        f.write("\n")

        # 初始状态
        f.write("【初始状态】\n")
        initial_bf = result.metrics.get('initial_baseflow_m3s', 0)
        k2 = result.metrics.get('k2_estimate', 0.02)
        est_lower = result.metrics.get('estimated_initial_lower_mm', 0)
        conf_lower = result.metrics.get('configured_initial_lower_mm', 3000)

        f.write(f"  观测初始流量:           {initial_bf:.2f} m³/s\n")
        f.write(f"  反推初始下层储量(K2={k2}): {est_lower:.2f} mm\n")
        f.write(f"  之前使用的initial_lower:     {conf_lower:.1f} mm\n")
        f.write(f"  建议使用:                {est_lower:.1f} mm\n\n")

        # 诊断结论
        f.write("【诊断结论】\n")
        if 0.3 <= rc_obs <= 0.7:
            f.write("  ✓ 径流系数在合理范围内\n")
        else:
            f.write(f"  ✗ 径流系数异常 ({rc_obs:.3f})\n")

        if 'runoff_coefficient_difference' in result.metrics:
            rc_diff = result.metrics['runoff_coefficient_difference']
            if rc_diff < 0.1:
                f.write("  ✓ HBV与观测的径流系数接近\n")
            else:
                f.write(f"  ⚠ HBV与观测的径流系数差异较大 (Δ={rc_diff:.3f})\n")

        # 问题列表
        if result.issues:
            f.write(f"\n【发现的问题】({len(result.issues)}个)\n")
            for i, issue in enumerate(result.issues, 1):
                f.write(f"  {i}. [{issue.severity.value.upper()}] {issue.message}\n")
                if issue.suggestion:
                    f.write(f"     💡 {issue.suggestion}\n")

        # 建议
        f.write("\n【建议】\n")
        if result.recommendations:
            for i, rec in enumerate(result.recommendations, 1):
                f.write(f"  {i}. {rec}\n")
        else:
            f.write("  1. 使用反推的初始下层储量替代固定值\n")
            f.write("  2. 如果径流系数差异大，调整FC和BETA参数\n")
            f.write("  3. 获取更长时间序列（30-90天）以充分识别参数\n")
            f.write("  4. 验证增强型生成器参数与HBV参数的一致性\n")

        f.write("\n说明:\n")
        f.write("  本报告由HydroSIS统一诊断框架自动生成。\n")
        f.write("  框架提供了标准化的问题识别、严重程度分级和修复建议。\n")
        f.write("  更多详细信息请查看同目录下的其他诊断报告文件。\n")


if __name__ == "__main__":
    main()
