#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成60天的降雨和径流时间序列数据"""

import sys
sys.path.insert(0, '/home/user/HydroSIS')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入必要的模块
from examples.multi_model_storm_comparison import generate_storm_forcing, PRECIP_COLUMN
from hydrosis.runoff.hbv import HBVRunoff

def main():
    """生成60天的时间序列数据"""
    print("=" * 80)
    print("生成60天降雨径流时间序列")
    print("=" * 80)

    # 1. 生成60天的降雨数据
    print("\n[1/4] 生成降雨数据...")
    print("  时间范围: 60天 (1440小时)")
    print("  暴雨期: 2天 (48小时)")
    print("  前置期: 14天 (336小时)")
    print("  退水期: 44天 (1056小时)")

    synthetic_forcing = generate_storm_forcing(
        total_hours=1440,      # 60天
        storm_hours=48,        # 2天暴雨
        lead_hours=336,        # 14天前置期
        tail_hours=1056,       # 44天退水期
        time_step_minutes=60,  # 1小时步长
    )

    rainfall = synthetic_forcing[PRECIP_COLUMN].values
    timestamps = synthetic_forcing.index

    print(f"  ✓ 已生成 {len(rainfall)} 个时间步的降雨数据")
    print(f"  ✓ 降雨统计: 平均={rainfall.mean():.2f} mm/h, 最大={rainfall.max():.2f} mm/h")

    # 2. 使用HBV模型生成径流
    print("\n[2/4] 使用HBV模型生成径流...")

    # HBV参数（使用配置文件中修正后的值）
    hbv_params = {
        'degree_day_factor': 3.0,
        'snow_threshold': 0.0,
        'field_capacity': 80.0,
        'beta': 1.0,
        'k0': 0.12,
        'k1': 0.08,
        'k2': 0.02,
        'percolation': 1.0,
        'initial_snow': 0.0,
        'initial_soil': 0.0,     # 从零开始，观察模型自然产流特性
        'initial_upper': 0.0,    # 从零开始
        'initial_lower': 0.0,    # 从零开始，避免初始基流影响
    }

    print("  HBV参数:")
    for key, value in hbv_params.items():
        print(f"    {key}: {value}")

    # 初始化HBV模型状态变量
    snow = hbv_params['initial_snow']
    soil = hbv_params['initial_soil']
    upper = hbv_params['initial_upper']
    lower = hbv_params['initial_lower']

    # 提取HBV参数
    degree_day_factor = hbv_params['degree_day_factor']
    snow_threshold = hbv_params['snow_threshold']
    field_capacity = hbv_params['field_capacity']
    beta = max(1e-6, hbv_params['beta'])
    k0 = hbv_params['k0']
    k1 = hbv_params['k1']
    k2 = hbv_params['k2']
    percolation = hbv_params['percolation']

    # 模拟径流
    runoff_list = []
    temperature = 15.0  # 假设温度15°C (高于snow_threshold,无降雪)

    for i, p in enumerate(rainfall):
        # HBV模型计算（基于hydrosis/runoff/hbv.py的simulate方法）
        rainfall_step = max(0.0, p - snow_threshold)
        snowfall = max(0.0, p - rainfall_step)
        snow += snowfall

        melt = degree_day_factor * max(0.0, rainfall_step - snow_threshold)
        melt = min(melt, snow)
        snow -= melt

        effective_precip = rainfall_step + melt

        # HBV产流计算（完全按照原始代码实现）
        soil_deficit = max(0.0, field_capacity - soil)

        if soil > 0 and field_capacity > 0:
            recharge = effective_precip * ((soil / field_capacity) ** beta)
        else:
            recharge = 0.0

        # 关键：recharge不能超过土壤缺水量
        recharge = min(recharge, soil_deficit)

        # 更新土壤含水量
        soil += effective_precip - recharge

        quickflow = k0 * upper
        actual_percolation = min(percolation, max(0.0, upper + recharge - quickflow))
        upper += recharge - quickflow - actual_percolation
        upper = max(0.0, upper)

        lower += actual_percolation - k2 * lower
        lower = max(0.0, lower)

        baseflow = k1 * upper + k2 * lower
        total_runoff = quickflow + baseflow  # mm/h

        runoff_list.append(total_runoff)

        if (i + 1) % 240 == 0:  # 每10天打印一次进度
            print(f"  进度: {i+1}/{len(rainfall)} ({(i+1)/len(rainfall)*100:.1f}%) - 当前状态: upper={upper:.2f}, lower={lower:.2f}, runoff={total_runoff:.4f}")

    runoff_array = np.array(runoff_list)
    print(f"  ✓ 已生成 {len(runoff_array)} 个时间步的径流数据")
    print(f"  ✓ 径流统计: 平均={runoff_array.mean():.2f} mm/h, 最大={runoff_array.max():.2f} mm/h")

    # 3. 创建DataFrame并保存
    print("\n[3/4] 保存数据...")

    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'precipitation_mm_per_hour': rainfall,
        'runoff_mm_per_hour': runoff_array,
        'temperature_celsius': temperature,
    })

    # 创建输出目录
    output_dir = Path('results/extended_timeseries_60days')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存CSV文件
    csv_path = output_dir / 'timeseries_60days.csv'
    results_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"  ✓ CSV文件已保存: {csv_path}")

    # 保存统计信息
    stats_path = output_dir / 'statistics.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("60天降雨径流时间序列统计信息\n")
        f.write("=" * 80 + "\n\n")

        f.write("时间范围:\n")
        f.write(f"  开始时间: {timestamps[0]}\n")
        f.write(f"  结束时间: {timestamps[-1]}\n")
        f.write(f"  总时长: {len(timestamps)} 小时 ({len(timestamps)/24:.1f} 天)\n")
        f.write(f"  时间步长: 1 小时\n\n")

        f.write("降雨统计:\n")
        f.write(f"  平均强度: {rainfall.mean():.4f} mm/h\n")
        f.write(f"  最大强度: {rainfall.max():.4f} mm/h\n")
        f.write(f"  总降雨量: {rainfall.sum():.2f} mm\n")
        f.write(f"  非零时段: {(rainfall > 0).sum()} 小时\n\n")

        f.write("径流统计:\n")
        f.write(f"  平均强度: {runoff_array.mean():.4f} mm/h\n")
        f.write(f"  最大强度: {runoff_array.max():.4f} mm/h\n")
        f.write(f"  总径流量: {runoff_array.sum():.2f} mm\n")
        f.write(f"  径流系数: {runoff_array.sum() / rainfall.sum():.4f}\n\n")

        f.write("HBV模型参数:\n")
        for key, value in hbv_params.items():
            f.write(f"  {key}: {value}\n")

    print(f"  ✓ 统计信息已保存: {stats_path}")

    # 4. 生成可视化图表
    print("\n[4/4] 生成可视化图表...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # 降雨图
    axes[0].fill_between(range(len(rainfall)), rainfall, alpha=0.5, color='blue')
    axes[0].plot(rainfall, linewidth=0.8, color='blue')
    axes[0].set_ylabel('Precipitation (mm/h)', fontsize=12)
    axes[0].set_title('60-Day Rainfall and Runoff Time Series', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, len(rainfall))

    # 径流图
    axes[1].fill_between(range(len(runoff_array)), runoff_array, alpha=0.5, color='green')
    axes[1].plot(runoff_array, linewidth=0.8, color='darkgreen')
    axes[1].set_ylabel('Runoff (mm/h)', fontsize=12)
    axes[1].set_xlabel('Time (hours)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, len(runoff_array))

    # 添加天数标记
    for ax in axes:
        for day in range(0, 61, 10):
            ax.axvline(day * 24, color='red', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.tight_layout()

    plot_path = output_dir / 'timeseries_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {plot_path}")

    # 打印完成信息
    print("\n" + "=" * 80)
    print("✓ 成功生成60天降雨径流时间序列数据！")
    print("=" * 80)
    print(f"\n输出文件:")
    print(f"  1. CSV数据: {csv_path}")
    print(f"  2. 统计信息: {stats_path}")
    print(f"  3. 可视化图表: {plot_path}")
    print(f"\n数据概要:")
    print(f"  • 时间长度: 60天 (1440小时)")
    print(f"  • 总降雨量: {rainfall.sum():.2f} mm")
    print(f"  • 总径流量: {runoff_array.sum():.2f} mm")
    print(f"  • 径流系数: {runoff_array.sum() / rainfall.sum():.4f}")
    print()

if __name__ == '__main__':
    main()
