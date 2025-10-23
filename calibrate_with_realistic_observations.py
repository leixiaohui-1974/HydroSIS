#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实场景的参数率定：使用增强模型生成观测数据，用HBV模型率定

关键改进:
1. ✅ 观测数据来自增强模型（不同的模型结构）
2. ✅ 添加观测噪声（模拟真实测量误差）
3. ✅ 用HBV模型率定（测试结构不匹配情况）
4. ✅ 评估模型偏差和不确定性

这才是真实的率定场景！
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# 导入模型
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator


def run_hbv_model(rainfall: np.ndarray, params: dict) -> np.ndarray:
    """运行HBV模型"""
    snow = params.get('initial_snow', 0.0)
    soil = params.get('initial_soil', 0.0)
    upper = params.get('initial_upper', 0.0)
    lower = params.get('initial_lower', 0.0)

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


def generate_realistic_observations(rainfall: np.ndarray) -> dict:
    """
    使用增强模型生成"观测数据"并添加真实的观测噪声

    特点:
    1. 使用不同的模型结构（增强模型 vs HBV）
    2. 添加随机观测噪声（测量误差）
    3. 添加系统偏差（仪器校准误差）
    """
    print("\n  使用增强模型生成观测数据:")
    print("    模型结构: 土壤层 + 三分量线性水库")

    # 增强模型参数（"真实"的流域特性）
    generator = EnhancedRunoffGenerator(
        # 土壤参数
        soil_capacity=120.0,      # 土壤容量（不同于HBV的field_capacity）
        soil_beta=1.8,            # 产流指数（不同于HBV的beta）

        # 径流分配（HBV没有这个机制）
        fast_threshold=0.65,
        fast_ratio=0.35,
        inter_ratio=0.40,
        base_ratio=0.25,

        # 水库参数（与HBV不同）
        k_fast=0.20,              # 比HBV的k0更快
        k_inter=0.10,             # HBV没有中间水库
        k_base=0.018,             # 略小于HBV的k2

        # 初始状态
        initial_soil=50.0,        # 非零初始状态
        initial_fast=2.0,
        initial_inter=5.0,
        initial_base=15.0,

        # 蒸散发（HBV没有显式建模）
        et_rate=0.12,
    )

    # 生成"真实"径流
    runoff_true = []
    for p in rainfall:
        runoff, components = generator.step(p)
        runoff_true.append(runoff)

    runoff_true = np.array(runoff_true)

    print(f"    真实径流系数: {runoff_true.sum() / rainfall.sum():.4f}")

    # 添加观测噪声
    print("\n  添加观测噪声:")

    # 1. 随机测量误差（正态分布，相对误差）
    noise_level = 0.05  # 5%相对误差（典型的流量计误差）
    random_noise = np.random.RandomState(42).normal(0, noise_level, len(runoff_true))

    # 2. 系统偏差（仪器校准误差）
    systematic_bias = 1.03  # 3%系统高估（常见的水位-流量关系偏差）

    # 3. 添加噪声（确保非负）
    observed = runoff_true * systematic_bias * (1 + random_noise)
    observed = np.maximum(observed, 0)

    # 4. 模拟数据缺失（随机缺失5%的数据）
    missing_rate = 0.05
    missing_indices = np.random.RandomState(42).choice(
        len(observed),
        int(len(observed) * missing_rate),
        replace=False
    )

    # 对缺失数据用线性插值
    for idx in missing_indices:
        if 0 < idx < len(observed) - 1:
            observed[idx] = (observed[idx-1] + observed[idx+1]) / 2

    print(f"    随机噪声: ±{noise_level*100:.1f}% (标准差)")
    print(f"    系统偏差: +{(systematic_bias-1)*100:.1f}%")
    print(f"    数据缺失: {missing_rate*100:.1f}% (已插值)")
    print(f"    观测径流系数: {observed.sum() / rainfall.sum():.4f}")

    return {
        'observed': observed,
        'true': runoff_true,
        'noise_level': noise_level,
        'systematic_bias': systematic_bias,
    }


def calculate_metrics(observed: np.ndarray, simulated: np.ndarray) -> dict:
    """计算性能指标"""
    # NSE
    mean_obs = np.mean(observed)
    nse = 1 - np.sum((observed - simulated)**2) / np.sum((observed - mean_obs)**2)

    # RMSE
    rmse_val = np.sqrt(np.mean((observed - simulated)**2))

    # 偏差
    bias = np.mean(simulated - observed)
    rel_bias = (np.sum(simulated) - np.sum(observed)) / np.sum(observed) * 100

    # 相关系数
    corr = np.corrcoef(observed, simulated)[0, 1]

    # KGE
    mean_sim = np.mean(simulated)
    std_obs = np.std(observed)
    std_sim = np.std(simulated)

    r = corr
    alpha = std_sim / std_obs if std_obs > 0 else 0
    beta_kge = mean_sim / mean_obs if mean_obs > 0 else 0

    kge = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta_kge-1)**2)

    return {
        'NSE': nse,
        'RMSE': rmse_val,
        'Bias': bias,
        'Relative_Bias_%': rel_bias,
        'Correlation': corr,
        'KGE': kge,
    }


def main():
    """主函数"""
    print("=" * 80)
    print("真实场景的参数率定测试")
    print("=" * 80)
    print("\n策略:")
    print("  1. 用增强模型生成'观测数据'（不同结构）")
    print("  2. 添加观测噪声（5%随机误差 + 3%系统偏差）")
    print("  3. 用HBV模型率定（结构不匹配）")
    print("  4. 评估模型性能和局限性")

    # ========================================================================
    # 第1步：加载降雨数据
    # ========================================================================
    print("\n[1/6] 加载60天降雨数据...")

    data_file = Path('results/extended_timeseries_60days/timeseries_60days.csv')
    df = pd.read_csv(data_file)
    rainfall = df['precipitation_mm_per_hour'].values
    timestamps = pd.to_datetime(df['timestamp'])

    print(f"  ✓ 已加载 {len(rainfall)} 个时间步")
    print(f"  ✓ 总降雨量: {rainfall.sum():.2f} mm")

    # ========================================================================
    # 第2步：生成真实的"观测数据"
    # ========================================================================
    print("\n[2/6] 生成真实的观测数据...")

    obs_data = generate_realistic_observations(rainfall)
    observed = obs_data['observed']
    true_runoff = obs_data['true']

    print(f"\n  ✓ 观测数据统计:")
    print(f"    总径流量: {observed.sum():.2f} mm")
    print(f"    最大流量: {observed.max():.4f} mm/h")
    print(f"    平均流量: {observed.mean():.4f} mm/h")

    # ========================================================================
    # 第3步：定义HBV参数空间
    # ========================================================================
    print("\n[3/6] 定义HBV参数空间...")

    # HBV率定参数（尝试用HBV拟合增强模型生成的数据）
    param_names = ['field_capacity', 'beta', 'k0', 'k1']
    param_bounds = [
        (50, 150),     # field_capacity
        (0.5, 2.5),    # beta
        (0.05, 0.30),  # k0
        (0.03, 0.15),  # k1
    ]

    # 固定参数
    fixed_params = {
        'degree_day_factor': 3.0,
        'snow_threshold': 0.0,
        'k2': 0.02,
        'percolation': 1.0,
        'initial_snow': 0.0,
        'initial_soil': 0.0,
        'initial_upper': 0.0,
        'initial_lower': 0.0,
    }

    print(f"  待率定参数: {param_names}")

    # ========================================================================
    # 第4步：运行HBV参数率定
    # ========================================================================
    print("\n[4/6] 运行HBV参数率定...")
    print("  警告: HBV结构与观测数据的真实来源（增强模型）不同")
    print("  预期: NSE可能 < 0.9（结构误差）")

    def objective_function(params_array):
        params = fixed_params.copy()
        params['field_capacity'] = params_array[0]
        params['beta'] = params_array[1]
        params['k0'] = params_array[2]
        params['k1'] = params_array[3]

        simulated = run_hbv_model(rainfall, params)
        nse = 1 - np.sum((observed - simulated)**2) / np.sum((observed - np.mean(observed))**2)
        return -nse  # 最小化负NSE

    result = differential_evolution(
        func=objective_function,
        bounds=param_bounds,
        maxiter=150,
        popsize=15,
        seed=42,
        polish=True,
        disp=True,
    )

    best_params_array = result.x
    best_nse = -result.fun

    # 构建最优参数
    best_params = fixed_params.copy()
    best_params['field_capacity'] = best_params_array[0]
    best_params['beta'] = best_params_array[1]
    best_params['k0'] = best_params_array[2]
    best_params['k1'] = best_params_array[3]

    print(f"\n  ✓ 率定完成")
    print(f"    迭代次数: {result.nit}")
    print(f"    函数评估: {result.nfev}")
    print(f"    最优NSE: {best_nse:.6f}")

    # ========================================================================
    # 第5步：评估率定结果
    # ========================================================================
    print("\n[5/6] 评估HBV模型性能...")

    simulated = run_hbv_model(rainfall, best_params)

    # 性能指标
    metrics = calculate_metrics(observed, simulated)

    print("\n  HBV vs 观测数据:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.6f}")

    # 与真实数据比较（诊断用）
    print("\n  HBV vs 真实数据（无噪声）:")
    true_metrics = calculate_metrics(true_runoff, simulated)
    for key, value in true_metrics.items():
        print(f"    {key}: {value:.6f}")

    # 分析模型误差来源
    print("\n  误差分析:")
    total_error = np.sum((observed - simulated)**2)
    noise_error = np.sum((observed - true_runoff)**2)
    struct_error = np.sum((true_runoff - simulated)**2)

    print(f"    总误差 (MSE): {total_error:.6f}")
    print(f"    观测噪声导致: {noise_error:.6f} ({noise_error/total_error*100:.1f}%)")
    print(f"    结构误差导致: {struct_error:.6f} ({struct_error/total_error*100:.1f}%)")

    # ========================================================================
    # 第6步：保存结果
    # ========================================================================
    print("\n[6/6] 保存结果...")

    output_dir = Path('results/realistic_calibration')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存参数
    params_file = output_dir / 'hbv_calibrated_params.yaml'
    with open(params_file, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"  ✓ 参数: {params_file}")

    # 保存指标
    all_metrics = {
        'hbv_vs_observed': metrics,
        'hbv_vs_true': true_metrics,
        'observation_noise': obs_data['noise_level'],
        'systematic_bias': obs_data['systematic_bias'],
    }
    metrics_file = output_dir / 'performance_metrics.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(all_metrics, f, default_flow_style=False)
    print(f"  ✓ 指标: {metrics_file}")

    # 保存时间序列
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'rainfall': rainfall,
        'observed': observed,
        'true_runoff': true_runoff,
        'hbv_simulated': simulated,
        'observation_error': observed - true_runoff,
        'model_error': simulated - true_runoff,
        'total_residual': observed - simulated,
    })
    ts_file = output_dir / 'timeseries_comparison.csv'
    results_df.to_csv(ts_file, index=False)
    print(f"  ✓ 时间序列: {ts_file}")

    # 生成可视化
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f'Realistic Calibration: HBV vs Enhanced Model (NSE={best_nse:.4f})',
                 fontsize=14, fontweight='bold')

    # 子图1: 降雨
    ax = axes[0]
    ax.fill_between(range(len(rainfall)), rainfall, alpha=0.5, color='blue')
    ax.set_ylabel('Rainfall (mm/h)')
    ax.set_title('Input Rainfall')
    ax.grid(True, alpha=0.3)

    # 子图2: 观测 vs 模拟
    ax = axes[1]
    ax.plot(observed, label='Observed (Enhanced + Noise)', linewidth=1.2, alpha=0.8, color='black')
    ax.plot(true_runoff, label='True (Enhanced, No Noise)', linewidth=1.0, alpha=0.6, color='gray', linestyle='--')
    ax.plot(simulated, label='HBV Simulated', linewidth=1.2, alpha=0.7, color='red')
    ax.set_ylabel('Runoff (mm/h)')
    ax.set_title(f'Runoff Comparison (NSE={best_nse:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图3: 残差分解
    ax = axes[2]
    obs_error = observed - true_runoff
    model_error = simulated - true_runoff
    ax.plot(obs_error, label='Observation Error', linewidth=0.8, alpha=0.7, color='orange')
    ax.plot(model_error, label='Model Structural Error', linewidth=0.8, alpha=0.7, color='purple')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax.set_ylabel('Error (mm/h)')
    ax.set_title('Error Decomposition')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 子图4: 总残差
    ax = axes[3]
    total_residual = observed - simulated
    ax.plot(total_residual, linewidth=0.8, color='green', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.fill_between(range(len(total_residual)), total_residual, alpha=0.3, color='green')
    ax.set_ylabel('Residual (mm/h)')
    ax.set_xlabel('Time (hours)')
    ax.set_title(f'Total Residuals (Mean={np.mean(total_residual):.6f}, Std={np.std(total_residual):.6f})')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / 'calibration_diagnostic.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表: {plot_file}")

    # 散点图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # HBV vs 观测
    ax = axes[0]
    ax.scatter(observed, simulated, alpha=0.5, s=20)
    max_val = max(observed.max(), simulated.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax.set_xlabel('Observed (mm/h)')
    ax.set_ylabel('HBV Simulated (mm/h)')
    ax.set_title(f'HBV vs Observed\nNSE={metrics["NSE"]:.4f}, R²={metrics["Correlation"]**2:.4f}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # HBV vs 真实
    ax = axes[1]
    ax.scatter(true_runoff, simulated, alpha=0.5, s=20, color='orange')
    max_val = max(true_runoff.max(), simulated.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2)
    ax.set_xlabel('True (Enhanced Model, mm/h)')
    ax.set_ylabel('HBV Simulated (mm/h)')
    ax.set_title(f'HBV vs True\nNSE={true_metrics["NSE"]:.4f}, R²={true_metrics["Correlation"]**2:.4f}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    scatter_file = output_dir / 'scatter_plots.png'
    plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ 散点图: {scatter_file}")

    # 生成报告
    report_file = output_dir / 'calibration_report.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("真实场景的参数率定报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 测试设计\n")
        f.write("   观测数据来源: 增强模型（不同结构）\n")
        f.write("   观测噪声: 5% 随机误差 + 3% 系统偏差\n")
        f.write("   率定模型: HBV（结构不匹配）\n\n")

        f.write("2. 数据概况\n")
        f.write(f"   时间长度: 60天\n")
        f.write(f"   总降雨量: {rainfall.sum():.2f} mm\n")
        f.write(f"   观测径流量: {observed.sum():.2f} mm\n")
        f.write(f"   真实径流量: {true_runoff.sum():.2f} mm\n")
        f.write(f"   HBV模拟径流量: {simulated.sum():.2f} mm\n\n")

        f.write("3. HBV率定结果\n")
        f.write(f"   迭代次数: {result.nit}\n")
        f.write(f"   函数评估: {result.nfev}\n\n")

        f.write("4. 性能指标（HBV vs 观测）\n")
        for key, value in metrics.items():
            f.write(f"   {key}: {value:.6f}\n")
        f.write("\n")

        f.write("5. 性能指标（HBV vs 真实）\n")
        for key, value in true_metrics.items():
            f.write(f"   {key}: {value:.6f}\n")
        f.write("\n")

        f.write("6. 误差分析\n")
        f.write(f"   总误差: {total_error:.6f}\n")
        f.write(f"   观测噪声: {noise_error:.6f} ({noise_error/total_error*100:.1f}%)\n")
        f.write(f"   结构误差: {struct_error:.6f} ({struct_error/total_error*100:.1f}%)\n\n")

        f.write("7. 结论\n")
        if best_nse > 0.85:
            f.write("   ✓ 优秀: 尽管结构不匹配，HBV仍能很好拟合观测数据\n")
        elif best_nse > 0.7:
            f.write("   ✓ 良好: HBV基本能拟合观测数据，但存在结构误差\n")
        elif best_nse > 0.5:
            f.write("   ⚠ 可接受: HBV拟合一般，结构差异明显\n")
        else:
            f.write("   ❌ 不佳: HBV难以拟合观测数据，结构不匹配严重\n")

        f.write(f"\n   NSE差异（观测 vs 真实）: {metrics['NSE'] - true_metrics['NSE']:.4f}\n")
        f.write(f"   此差异主要由观测噪声导致\n")

    print(f"  ✓ 报告: {report_file}")

    print("\n" + "=" * 80)
    print("✓ 真实场景率定完成！")
    print("=" * 80)
    print(f"\n关键结果:")
    print(f"  • NSE (HBV vs 观测): {metrics['NSE']:.4f}")
    print(f"  • NSE (HBV vs 真实): {true_metrics['NSE']:.4f}")
    print(f"  • 观测噪声影响: {noise_error/total_error*100:.1f}%")
    print(f"  • 结构误差影响: {struct_error/total_error*100:.1f}%")
    print(f"\n输出目录: {output_dir}")
    print()


if __name__ == '__main__':
    main()
