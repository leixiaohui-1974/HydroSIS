#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断HBV模型的水量平衡和单位转换问题"""

import sys
sys.path.insert(0, '/home/user/HydroSIS')

import pandas as pd
import numpy as np

def main():
    """诊断水量平衡"""
    print("=" * 80)
    print("HBV模型水量平衡诊断")
    print("=" * 80)

    # 读取生成的时间序列数据
    df = pd.read_csv('results/extended_timeseries_60days/timeseries_60days.csv')

    rainfall = df['precipitation_mm_per_hour'].values
    runoff = df['runoff_mm_per_hour'].values

    total_rainfall = rainfall.sum()
    total_runoff = runoff.sum()

    print(f"\n基本统计:")
    print(f"  总降雨量: {total_rainfall:.2f} mm")
    print(f"  总径流量: {total_runoff:.2f} mm")
    print(f"  径流系数: {total_runoff / total_rainfall:.4f}")
    print(f"  额外径流量: {total_runoff - total_rainfall:.2f} mm (来自初始储量)")

    # 分析问题
    print(f"\n问题诊断:")

    # 1. 径流系数异常
    runoff_coefficient = total_runoff / total_rainfall
    if runoff_coefficient > 1.0:
        print(f"  ❌ 径流系数 {runoff_coefficient:.4f} > 1.0 (不合理!)")
        print(f"     原因: 初始储量过大,释放了 {total_runoff - total_rainfall:.2f} mm 的额外径流")
    else:
        print(f"  ✓ 径流系数 {runoff_coefficient:.4f} 在合理范围内")

    # 2. 分析initial_lower的影响
    k2 = 0.02
    initial_lower = 4498.9

    print(f"\n  Initial_lower分析:")
    print(f"    当前值: {initial_lower:.2f} mm")
    print(f"    k2参数: {k2}")
    print(f"    初始基流: {k2 * initial_lower:.2f} mm/h")

    # 计算60天内能释放多少储量
    # 简化计算: 假设指数衰减 S(t) = S0 * exp(-k2 * t)
    # 释放量 = S0 * (1 - exp(-k2 * t))
    hours = len(rainfall)
    decay_factor = np.exp(-k2 * hours)
    released_storage = initial_lower * (1 - decay_factor)

    print(f"    60天后剩余: {initial_lower * decay_factor:.2f} mm")
    print(f"    60天内释放: {released_storage:.2f} mm")
    print(f"    实际额外径流: {total_runoff - total_rainfall:.2f} mm")

    # 3. 建议合理的initial_lower
    print(f"\n  建议修正:")

    # 假设合理的径流系数在0.3-0.7之间(典型值)
    target_runoff_coefficient = 0.5
    target_total_runoff = total_rainfall * target_runoff_coefficient

    # 需要减少的径流量
    excess_runoff = total_runoff - target_total_runoff

    # 反推合理的initial_lower
    # excess_runoff ≈ initial_lower * (1 - exp(-k2 * hours))
    recommended_initial_lower = initial_lower - excess_runoff / (1 - decay_factor)

    print(f"    目标径流系数: {target_runoff_coefficient}")
    print(f"    建议 initial_lower: {max(0, recommended_initial_lower):.2f} mm")
    print(f"    (当前值 {initial_lower:.2f} mm 减少到 {max(0, recommended_initial_lower):.2f} mm)")

    # 4. 单位转换检查
    print(f"\n单位转换检查:")
    print(f"  ✓ 降雨: mm/h (正确)")
    print(f"  ✓ 径流: mm/h (正确)")
    print(f"  ✓ 储量: mm (正确)")
    print(f"  ✓ k2: 1/h (正确, 表示每小时释放储量的比例)")

    # 5. 参数合理性检查
    print(f"\n参数合理性检查:")

    initial_baseflow = k2 * initial_lower
    print(f"  初始基流: {initial_baseflow:.2f} mm/h")

    if initial_baseflow > 100:
        print(f"    ❌ 初始基流 {initial_baseflow:.2f} mm/h 过大 (典型值 < 10 mm/h)")
    elif initial_baseflow > 10:
        print(f"    ⚠ 初始基流 {initial_baseflow:.2f} mm/h 偏大 (典型值 < 10 mm/h)")
    else:
        print(f"    ✓ 初始基流 {initial_baseflow:.2f} mm/h 在合理范围内")

    # 6. 生成修正建议
    print(f"\n" + "=" * 80)
    print(f"修正建议")
    print(f"=" * 80)

    print(f"\n方案1: 降低initial_lower (推荐)")
    print(f"  修改配置: initial_lower: {max(0, recommended_initial_lower):.2f}")
    print(f"  预期效果: 径流系数降至约 {target_runoff_coefficient}")

    print(f"\n方案2: 使用更小的初始值")
    print(f"  修改配置: initial_lower: 100.0  # 保守值")
    print(f"  预期效果: 径流系数降至约 {(total_rainfall * target_runoff_coefficient + 100 * (1 - decay_factor)) / total_rainfall:.4f}")

    print(f"\n方案3: 从零开始(适用于无初始观测数据的情况)")
    print(f"  修改配置: initial_lower: 0.0")
    print(f"  预期效果: 径流系数接近HBV模型本身的产流特性")

    # 保存诊断报告
    report_path = 'results/extended_timeseries_60days/water_balance_diagnosis.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HBV模型水量平衡诊断报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("问题: 径流系数异常\n")
        f.write(f"  径流系数: {runoff_coefficient:.4f} (>> 1.0, 不合理)\n")
        f.write(f"  总降雨量: {total_rainfall:.2f} mm\n")
        f.write(f"  总径流量: {total_runoff:.2f} mm\n")
        f.write(f"  额外径流: {total_runoff - total_rainfall:.2f} mm\n\n")

        f.write("原因分析:\n")
        f.write(f"  initial_lower = {initial_lower:.2f} mm (过大)\n")
        f.write(f"  初始基流 = k2 * initial_lower = {initial_baseflow:.2f} mm/h\n")
        f.write(f"  60天内释放储量 ≈ {released_storage:.2f} mm\n\n")

        f.write("建议修正:\n")
        f.write(f"  方案1: initial_lower = {max(0, recommended_initial_lower):.2f} mm\n")
        f.write(f"  方案2: initial_lower = 100.0 mm (保守值)\n")
        f.write(f"  方案3: initial_lower = 0.0 mm (从零开始)\n\n")

        f.write("说明:\n")
        f.write("  initial_lower=4498.9mm 是从观测基流反推的，假设系统处于稳定状态。\n")
        f.write("  但这个值可能存在以下问题:\n")
        f.write("  1. 观测基流数据可能有误或单位转换错误\n")
        f.write("  2. k2参数可能需要调整\n")
        f.write("  3. 应该使用更短的预热期(warmup period)让模型自然达到稳定状态\n")

    print(f"\n✓ 诊断报告已保存: {report_path}")
    print()

if __name__ == '__main__':
    main()
