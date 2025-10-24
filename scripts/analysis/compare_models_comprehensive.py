#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""综合模型性能比较和敏感性分析

比较内容:
1. 简单模型 vs HBV模型
2. 增强模型 vs HBV模型
3. HBV参数敏感性分析
"""

import sys
sys.path.insert(0, '/home/user/HydroSIS')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import json

# 导入模型
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator


def load_rainfall_data() -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """加载60天的降雨数据"""
    df = pd.read_csv('results/extended_timeseries_60days/timeseries_60days.csv')
    rainfall = df['precipitation_mm_per_hour'].values
    timestamps = pd.to_datetime(df['timestamp'])
    return rainfall, timestamps


def run_hbv_model(rainfall: np.ndarray, params: Dict) -> np.ndarray:
    """运行HBV模型"""
    # 初始化状态
    snow = params.get('initial_snow', 0.0)
    soil = params.get('initial_soil', 0.0)
    upper = params.get('initial_upper', 0.0)
    lower = params.get('initial_lower', 0.0)

    # 参数
    degree_day_factor = params.get('degree_day_factor', 3.0)
    snow_threshold = params.get('snow_threshold', 0.0)
    field_capacity = params.get('field_capacity', 80.0)
    beta = max(1e-6, params.get('beta', 1.0))
    k0 = params.get('k0', 0.12)
    k1 = params.get('k1', 0.08)
    k2 = params.get('k2', 0.02)
    percolation = params.get('percolation', 1.0)

    runoff_list = []

    for p in rainfall:
        # HBV计算
        rainfall_step = max(0.0, p - snow_threshold)
        snowfall = max(0.0, p - rainfall_step)
        snow += snowfall

        melt = degree_day_factor * max(0.0, rainfall_step - snow_threshold)
        melt = min(melt, snow)
        snow -= melt

        effective_precip = rainfall_step + melt
        soil_deficit = max(0.0, field_capacity - soil)

        if soil > 0 and field_capacity > 0:
            recharge = effective_precip * ((soil / field_capacity) ** beta)
        else:
            recharge = 0.0

        recharge = min(recharge, soil_deficit)
        soil += effective_precip - recharge

        quickflow = k0 * upper
        actual_percolation = min(percolation, max(0.0, upper + recharge - quickflow))
        upper += recharge - quickflow - actual_percolation
        upper = max(0.0, upper)

        lower += actual_percolation - k2 * lower
        lower = max(0.0, lower)

        baseflow = k1 * upper + k2 * lower
        total_runoff = quickflow + baseflow

        runoff_list.append(total_runoff)

    return np.array(runoff_list)


def run_simple_model(rainfall: np.ndarray) -> np.ndarray:
    """运行简单模型（径流系数=1.0）"""
    return rainfall.copy()


def run_enhanced_model(rainfall: np.ndarray, params: Dict = None) -> np.ndarray:
    """运行增强模型"""
    if params is None:
        # 默认参数（尝试与HBV相似的行为）
        params = {
            'soil_capacity': 150.0,    # 类似HBV的field_capacity
            'soil_beta': 1.5,          # 非线性产流
            'fast_ratio': 0.3,
            'inter_ratio': 0.4,
            'base_ratio': 0.3,
            'k_fast': 0.15,            # 类似HBV的k0
            'k_inter': 0.08,           # 类似HBV的k1
            'k_base': 0.02,            # 类似HBV的k2
            'initial_soil': 0.0,
            'initial_fast': 0.0,
            'initial_inter': 0.0,
            'initial_base': 0.0,
        }

    generator = EnhancedRunoffGenerator(
        soil_capacity=params['soil_capacity'],
        soil_beta=params['soil_beta'],
        fast_ratio=params['fast_ratio'],
        inter_ratio=params['inter_ratio'],
        base_ratio=params['base_ratio'],
        k_fast=params['k_fast'],
        k_inter=params['k_inter'],
        k_base=params['k_base'],
        initial_soil=params['initial_soil'],
        initial_fast=params['initial_fast'],
        initial_inter=params['initial_inter'],
        initial_base=params['initial_base'],
    )

    runoff_list = []
    for p in rainfall:
        runoff = generator.step(p)
        runoff_list.append(runoff)

    return np.array(runoff_list)


def calculate_metrics(observed: np.ndarray, simulated: np.ndarray) -> Dict:
    """计算性能指标"""
    # NSE (Nash-Sutcliffe Efficiency)
    mean_obs = np.mean(observed)
    nse = 1 - np.sum((observed - simulated)**2) / np.sum((observed - mean_obs)**2)

    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((observed - simulated)**2))

    # Bias (偏差)
    bias = np.mean(simulated - observed)

    # Relative bias (%)
    rel_bias = (np.sum(simulated) - np.sum(observed)) / np.sum(observed) * 100

    # Correlation coefficient
    corr = np.corrcoef(observed, simulated)[0, 1]

    return {
        'NSE': nse,
        'RMSE': rmse,
        'Bias': bias,
        'Relative_Bias_%': rel_bias,
        'Correlation': corr,
    }


def compare_models():
    """主函数：模型比较"""
    print("=" * 80)
    print("综合模型性能比较与分析")
    print("=" * 80)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    rainfall, timestamps = load_rainfall_data()
    print(f"  ✓ 已加载 {len(rainfall)} 个时间步的降雨数据")
    print(f"  ✓ 总降雨量: {rainfall.sum():.2f} mm")

    # 2. 定义参考HBV参数
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
        'initial_soil': 0.0,
        'initial_upper': 0.0,
        'initial_lower': 0.0,
    }

    # 3. 运行所有模型
    print("\n[2/5] 运行模型...")

    print("  运行HBV模型...")
    hbv_runoff = run_hbv_model(rainfall, hbv_params)
    print(f"    HBV径流系数: {hbv_runoff.sum() / rainfall.sum():.4f}")

    print("  运行简单模型...")
    simple_runoff = run_simple_model(rainfall)
    print(f"    简单模型径流系数: {simple_runoff.sum() / rainfall.sum():.4f}")

    print("  运行增强模型...")
    try:
        enhanced_runoff = run_enhanced_model(rainfall)
        print(f"    增强模型径流系数: {enhanced_runoff.sum() / rainfall.sum():.4f}")
        enhanced_success = True
    except Exception as e:
        print(f"    ⚠ 增强模型运行失败: {e}")
        enhanced_runoff = np.zeros_like(rainfall)
        enhanced_success = False

    # 4. 性能比较（以HBV为参考）
    print("\n[3/5] 模型性能比较 (以HBV为参考)...")

    results = {}

    print("\n  简单模型 vs HBV:")
    simple_metrics = calculate_metrics(hbv_runoff, simple_runoff)
    results['Simple_vs_HBV'] = simple_metrics
    for key, value in simple_metrics.items():
        print(f"    {key}: {value:.4f}")

    if enhanced_success:
        print("\n  增强模型 vs HBV:")
        enhanced_metrics = calculate_metrics(hbv_runoff, enhanced_runoff)
        results['Enhanced_vs_HBV'] = enhanced_metrics
        for key, value in enhanced_metrics.items():
            print(f"    {key}: {value:.4f}")

    # 5. HBV参数敏感性分析
    print("\n[4/5] HBV参数敏感性分析...")

    sensitivity_params = {
        'field_capacity': [40, 60, 80, 100, 120],
        'beta': [0.5, 0.8, 1.0, 1.2, 1.5],
        'k0': [0.06, 0.09, 0.12, 0.15, 0.20],
        'k2': [0.01, 0.015, 0.02, 0.025, 0.03],
    }

    sensitivity_results = {}

    for param_name, values in sensitivity_params.items():
        print(f"\n  测试参数: {param_name}")
        param_runoff_coeffs = []

        for value in values:
            test_params = hbv_params.copy()
            test_params[param_name] = value
            test_runoff = run_hbv_model(rainfall, test_params)
            runoff_coeff = test_runoff.sum() / rainfall.sum()
            param_runoff_coeffs.append(runoff_coeff)
            print(f"    {param_name}={value}: 径流系数={runoff_coeff:.4f}")

        sensitivity_results[param_name] = {
            'values': values,
            'runoff_coefficients': param_runoff_coeffs
        }

    # 6. 保存结果
    print("\n[5/5] 保存结果...")

    output_dir = Path('results/model_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存性能指标
    metrics_path = output_dir / 'performance_metrics.json'
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ 性能指标已保存: {metrics_path}")

    # 保存敏感性分析结果
    sensitivity_path = output_dir / 'sensitivity_analysis.json'
    with open(sensitivity_path, 'w', encoding='utf-8') as f:
        json.dump(sensitivity_results, f, indent=2)
    print(f"  ✓ 敏感性分析结果已保存: {sensitivity_path}")

    # 保存时间序列数据
    comparison_df = pd.DataFrame({
        'timestamp': timestamps,
        'rainfall': rainfall,
        'HBV': hbv_runoff,
        'Simple': simple_runoff,
    })

    if enhanced_success:
        comparison_df['Enhanced'] = enhanced_runoff

    ts_path = output_dir / 'model_comparison_timeseries.csv'
    comparison_df.to_csv(ts_path, index=False)
    print(f"  ✓ 时间序列数据已保存: {ts_path}")

    # 7. 生成可视化
    print("\n  生成可视化图表...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Comparison and Sensitivity Analysis',
                 fontsize=16, fontweight='bold')

    # 子图1: 降雨
    ax = axes[0, 0]
    ax.fill_between(range(len(rainfall)), rainfall, alpha=0.5, color='blue')
    ax.set_ylabel('Rainfall (mm/h)')
    ax.set_title('Input: 60-Day Rainfall')
    ax.grid(True, alpha=0.3)

    # 子图2: HBV径流
    ax = axes[0, 1]
    ax.fill_between(range(len(hbv_runoff)), hbv_runoff, alpha=0.5, color='green')
    ax.plot(hbv_runoff, linewidth=0.8, color='darkgreen')
    ax.set_ylabel('Runoff (mm/h)')
    ax.set_title(f'HBV Model (RC={hbv_runoff.sum()/rainfall.sum():.3f})')
    ax.grid(True, alpha=0.3)

    # 子图3: 模型对比
    ax = axes[1, 0]
    ax.plot(hbv_runoff, label='HBV', linewidth=1.5, alpha=0.8)
    ax.plot(simple_runoff, label='Simple', linewidth=1.0, alpha=0.6)
    if enhanced_success:
        ax.plot(enhanced_runoff, label='Enhanced', linewidth=1.0, alpha=0.6)
    ax.set_ylabel('Runoff (mm/h)')
    ax.set_xlabel('Time (hours)')
    ax.set_title('Model Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图4: 敏感性分析 - field_capacity
    ax = axes[1, 1]
    fc_data = sensitivity_results['field_capacity']
    ax.plot(fc_data['values'], fc_data['runoff_coefficients'], 'o-', linewidth=2)
    ax.set_xlabel('Field Capacity (mm)')
    ax.set_ylabel('Runoff Coefficient')
    ax.set_title('Sensitivity: Field Capacity')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=hbv_runoff.sum()/rainfall.sum(), color='r', linestyle='--',
               label='Baseline', alpha=0.5)

    # 子图5: 敏感性分析 - beta
    ax = axes[2, 0]
    beta_data = sensitivity_results['beta']
    ax.plot(beta_data['values'], beta_data['runoff_coefficients'], 'o-', linewidth=2)
    ax.set_xlabel('Beta')
    ax.set_ylabel('Runoff Coefficient')
    ax.set_title('Sensitivity: Beta Parameter')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=hbv_runoff.sum()/rainfall.sum(), color='r', linestyle='--',
               label='Baseline', alpha=0.5)

    # 子图6: 敏感性分析 - k2
    ax = axes[2, 1]
    k2_data = sensitivity_results['k2']
    ax.plot(k2_data['values'], k2_data['runoff_coefficients'], 'o-', linewidth=2)
    ax.set_xlabel('K2 (1/h)')
    ax.set_ylabel('Runoff Coefficient')
    ax.set_title('Sensitivity: K2 (Baseflow Recession)')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=hbv_runoff.sum()/rainfall.sum(), color='r', linestyle='--',
               label='Baseline', alpha=0.5)

    plt.tight_layout()

    plot_path = output_dir / 'comprehensive_analysis.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {plot_path}")

    # 8. 生成文本报告
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("综合模型性能比较与敏感性分析报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 数据概况\n")
        f.write(f"   时间长度: 60天 (1440小时)\n")
        f.write(f"   总降雨量: {rainfall.sum():.2f} mm\n\n")

        f.write("2. 模型径流系数\n")
        f.write(f"   HBV:     {hbv_runoff.sum() / rainfall.sum():.4f}\n")
        f.write(f"   Simple:  {simple_runoff.sum() / rainfall.sum():.4f}\n")
        if enhanced_success:
            f.write(f"   Enhanced: {enhanced_runoff.sum() / rainfall.sum():.4f}\n")
        f.write("\n")

        f.write("3. 性能比较 (以HBV为参考)\n\n")

        f.write("   简单模型 vs HBV:\n")
        for key, value in simple_metrics.items():
            f.write(f"     {key}: {value:.4f}\n")
        f.write("\n")

        if enhanced_success:
            f.write("   增强模型 vs HBV:\n")
            for key, value in enhanced_metrics.items():
                f.write(f"     {key}: {value:.4f}\n")
            f.write("\n")

        f.write("4. 参数敏感性分析\n\n")
        for param_name, data in sensitivity_results.items():
            f.write(f"   {param_name}:\n")
            for val, rc in zip(data['values'], data['runoff_coefficients']):
                f.write(f"     {val}: {rc:.4f}\n")
            f.write("\n")

        f.write("5. 主要发现\n\n")
        f.write(f"   - 简单模型径流系数=1.0,明显高估径流(NSE={simple_metrics['NSE']:.3f})\n")

        if enhanced_success:
            f.write(f"   - 增强模型与HBV的相似性: NSE={enhanced_metrics['NSE']:.3f}\n")

        # 找出最敏感的参数
        sensitivities = {}
        for param_name, data in sensitivity_results.items():
            rc_range = max(data['runoff_coefficients']) - min(data['runoff_coefficients'])
            sensitivities[param_name] = rc_range

        most_sensitive = max(sensitivities, key=sensitivities.get)
        f.write(f"   - 最敏感参数: {most_sensitive} (径流系数变化范围: {sensitivities[most_sensitive]:.4f})\n")

    print(f"  ✓ 分析报告已保存: {report_path}")

    print("\n" + "=" * 80)
    print("✓ 综合分析完成！")
    print("=" * 80)
    print(f"\n输出目录: {output_dir}")
    print(f"  - performance_metrics.json")
    print(f"  - sensitivity_analysis.json")
    print(f"  - model_comparison_timeseries.csv")
    print(f"  - comprehensive_analysis.png")
    print(f"  - analysis_report.txt")
    print()


if __name__ == '__main__':
    compare_models()
