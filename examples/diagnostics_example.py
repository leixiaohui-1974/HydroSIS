#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断框架使用示例

展示如何使用统一诊断框架进行模型问题诊断。

使用方法:
    python examples/diagnostics_example.py

需求:
    - 降雨数据文件
    - 径流数据文件
    - HBV参数配置
"""
from pathlib import Path
import numpy as np
import pandas as pd

from hydrosis.diagnostics import (
    WaterBalanceDiagnostic,
    PrecipitationDiagnostic,
    IssueSeverity
)


def generate_sample_data():
    """生成示例数据"""
    print("📊 生成示例数据...")

    # 生成降雨数据 (30天, 每小时)
    np.random.seed(42)
    hours = 30 * 24
    dates = pd.date_range('2024-01-01', periods=hours, freq='h')

    # 模拟降雨事件
    precipitation = np.random.gamma(2, 2, size=hours) * 0.5
    precipitation[precipitation < 0.1] = 0  # 去除小雨

    # 模拟径流（径流系数约0.6，但由于初始储量会偏高）
    runoff = precipitation * 0.6 + np.random.normal(0, 0.5, hours)
    runoff[runoff < 0] = 0

    # 添加初始储量影响（造成RC > 1的问题）
    runoff[:100] += 5.0  # 前100小时有额外径流

    return {
        'dates': dates,
        'precipitation': precipitation,
        'runoff': runoff
    }


def example_water_balance_diagnostic():
    """示例1: 水量平衡诊断"""
    print("\n" + "="*80)
    print("示例1: 水量平衡诊断")
    print("="*80)

    # 生成数据
    data = generate_sample_data()

    # 创建诊断器
    diagnostic = WaterBalanceDiagnostic(
        output_dir=Path("results/diagnostic_examples"),
        verbose=True,
        target_runoff_coefficient=0.5
    )

    # 运行诊断
    print("\n🔍 开始水量平衡诊断...")
    result = diagnostic.run(
        precipitation=data['precipitation'],
        runoff=data['runoff'],
        initial_lower=3000.0,  # 过大的初始储量
        k2=0.02
    )

    # 分析结果
    print("\n📋 诊断结果:")
    print(f"  发现问题: {len(result.issues)}")

    # 按严重程度分类
    for severity in [IssueSeverity.CRITICAL, IssueSeverity.ERROR,
                     IssueSeverity.WARNING, IssueSeverity.INFO]:
        issues = result.get_issues_by_severity(severity)
        if issues:
            print(f"\n  {severity.value.upper()} ({len(issues)}):")
            for issue in issues:
                print(f"    • {issue.message}")
                if issue.suggestion:
                    print(f"      💡 建议: {issue.suggestion}")

    # 关键指标
    print("\n📊 关键指标:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # 修复建议
    if result.recommendations:
        print("\n💡 修复建议:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")

    return result


def example_precipitation_diagnostic():
    """示例2: 降雨空间分布诊断"""
    print("\n" + "="*80)
    print("示例2: 降雨空间分布诊断")
    print("="*80)

    # 生成多分区降雨数据
    print("\n📊 生成多分区降雨数据...")
    np.random.seed(42)
    hours = 30 * 24
    dates = pd.date_range('2024-01-01', periods=hours, freq='h')

    # 创建降雨数据 (5个子流域，分属2个分区)
    # Zone 1: 子流域 1, 2 - 正常降雨
    # Zone 2: 子流域 3, 4, 5 - 降雨偏低 (模拟异常)
    precip_df = pd.DataFrame({
        '1': np.random.gamma(2, 2, size=hours),
        '2': np.random.gamma(2, 2, size=hours),
        '3': np.random.gamma(2, 2, size=hours) * 0.4,  # 异常低
        '4': np.random.gamma(2, 2, size=hours) * 0.45,  # 异常低
        '5': np.random.gamma(2, 2, size=hours) * 0.35,  # 异常低
    }, index=dates)

    # 子流域信息
    subbasins_df = pd.DataFrame({
        'subzone_id': [1, 2, 3, 4, 5],
        'zone_id': [1, 1, 2, 2, 2],
        'area_km2': [100, 150, 120, 130, 110]
    })

    # 创建诊断器
    diagnostic = PrecipitationDiagnostic(
        output_dir=Path("results/diagnostic_examples"),
        verbose=True,
        anomaly_threshold=0.3  # 30%差异即视为异常
    )

    # 运行诊断
    print("\n🔍 开始降雨空间分布诊断...")
    result = diagnostic.run(
        precipitation_df=precip_df,
        subbasins_df=subbasins_df
    )

    # 分析结果
    print("\n📋 诊断结果:")
    print(f"  发现问题: {len(result.issues)}")

    # 显示异常分区
    anomalies = [
        issue for issue in result.issues
        if issue.category == "zone_precipitation_anomaly"
    ]
    if anomalies:
        print("\n  ⚠️  降雨异常分区:")
        for issue in anomalies:
            zone_id = issue.details.get('zone_id', 'unknown')
            zone_precip = issue.details.get('zone_precipitation', 0)
            overall_mean = issue.details.get('overall_mean', 0)
            diff = issue.details.get('relative_difference', 0)
            print(f"    • Zone {zone_id}: {zone_precip:.1f}mm "
                  f"(平均 {overall_mean:.1f}mm, 差异 {diff*100:.1f}%)")

    # 修复建议
    if result.recommendations:
        print("\n💡 修复建议:")
        for i, rec in enumerate(result.recommendations, 1):
            print(f"  {i}. {rec}")

    return result


def example_batch_diagnostics():
    """示例3: 批量诊断多个数据集"""
    print("\n" + "="*80)
    print("示例3: 批量诊断多个场景")
    print("="*80)

    # 模拟3个不同的场景
    scenarios = {
        'scenario_1': {'initial_lower': 1000, 'rc_expected': 'normal'},
        'scenario_2': {'initial_lower': 3000, 'rc_expected': 'high'},
        'scenario_3': {'initial_lower': 5000, 'rc_expected': 'very_high'},
    }

    results = {}

    for name, config in scenarios.items():
        print(f"\n🔍 诊断 {name} (initial_lower={config['initial_lower']}mm)...")

        # 生成数据
        data = generate_sample_data()

        # 创建诊断器
        diagnostic = WaterBalanceDiagnostic(
            output_dir=Path(f"results/diagnostic_examples/{name}"),
            verbose=False
        )

        # 运行诊断
        result = diagnostic.run(
            precipitation=data['precipitation'],
            runoff=data['runoff'],
            initial_lower=config['initial_lower'],
            k2=0.02
        )

        results[name] = result

        # 简要报告
        rc = result.metrics.get('runoff_coefficient', 0)
        error_count = len([i for i in result.issues
                          if i.severity in [IssueSeverity.ERROR, IssueSeverity.CRITICAL]])
        warning_count = len([i for i in result.issues
                            if i.severity == IssueSeverity.WARNING])

        print(f"  ✓ RC={rc:.4f}, 错误={error_count}, 警告={warning_count}")

    # 对比分析
    print("\n📊 场景对比:")
    print(f"{'场景':15} {'径流系数':12} {'总问题数':12} {'修正建议数':12}")
    print("-" * 55)
    for name, result in results.items():
        rc = result.metrics.get('runoff_coefficient', 0)
        issue_count = len(result.issues)
        rec_count = len(result.recommendations)
        print(f"{name:15} {rc:12.4f} {issue_count:12} {rec_count:12}")

    return results


def main():
    """主函数"""
    print("="*80)
    print("诊断框架使用示例")
    print("="*80)
    print("\n本示例展示如何使用 HydroSIS 诊断框架进行模型问题诊断")

    # 示例1: 水量平衡诊断
    wb_result = example_water_balance_diagnostic()

    # 示例2: 降雨空间分布诊断
    precip_result = example_precipitation_diagnostic()

    # 示例3: 批量诊断
    batch_results = example_batch_diagnostics()

    print("\n" + "="*80)
    print("✅ 所有示例运行完成！")
    print("="*80)
    print("\n📁 输出文件位置: results/diagnostic_examples/")
    print("   - 诊断报告 (txt)")
    print("   - 诊断报告 (json)")
    print("   - 可视化图表 (png)")

    print("\n💡 下一步:")
    print("   1. 查看生成的诊断报告")
    print("   2. 根据建议调整模型参数")
    print("   3. 重新运行诊断验证改进")


if __name__ == "__main__":
    main()
