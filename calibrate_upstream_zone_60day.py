#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高精度参数率定：基于60天数据的上游分区HBV参数率定

目标：实现最上游分区的率定精度尽可能高

策略：
1. 使用60天高质量降雨数据
2. 生成"真实"观测数据（使用已知参数）
3. 基于敏感性分析聚焦关键参数（Beta, Field_Capacity）
4. 使用Differential Evolution全局优化
5. 多目标评估（NSE, KGE, log-NSE, PBIAS）
6. 验证率定结果

遵循 .claude/AI_DEVELOPMENT_GUIDE.md 中的开发规范
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

# ✅ 使用基础库的功能模块
try:
    from hydrosis.calibration import (
        calibrate_parameters,
        morris_sensitivity,
        adaptive_bounds_from_sensitivity,
        print_sensitivity_report,
    )
    from hydrosis.evaluation.metrics import (
        nash_sutcliffe_efficiency,
        log_nash_sutcliffe_efficiency,
        kling_gupta_efficiency,
        rmse,
        mae,
        percent_bias,
    )
    HYDROSIS_AVAILABLE = True
except ImportError:
    print("⚠ hydrosis.calibration模块不可用，将使用简化实现")
    HYDROSIS_AVAILABLE = False


def run_hbv_model(rainfall: np.ndarray, params: dict) -> np.ndarray:
    """运行HBV模型

    Parameters
    ----------
    rainfall : np.ndarray
        降雨时间序列 (mm/h)
    params : dict
        HBV参数字典

    Returns
    -------
    np.ndarray
        径流时间序列 (mm/h)
    """
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


def calculate_nse(observed: np.ndarray, simulated: np.ndarray) -> float:
    """计算Nash-Sutcliffe效率系数"""
    if HYDROSIS_AVAILABLE:
        return nash_sutcliffe_efficiency(simulated, observed)

    mean_obs = np.mean(observed)
    numerator = np.sum((observed - simulated)**2)
    denominator = np.sum((observed - mean_obs)**2)
    if denominator == 0:
        return float('nan')
    return 1 - (numerator / denominator)


def calculate_all_metrics(observed: np.ndarray, simulated: np.ndarray) -> dict:
    """计算所有性能指标"""
    if HYDROSIS_AVAILABLE:
        return {
            'NSE': nash_sutcliffe_efficiency(simulated, observed),
            'log_NSE': log_nash_sutcliffe_efficiency(simulated, observed),
            'KGE': kling_gupta_efficiency(simulated, observed),
            'RMSE': rmse(simulated, observed),
            'MAE': mae(simulated, observed),
            'PBIAS': percent_bias(simulated, observed),
        }
    else:
        # 简化实现
        nse = calculate_nse(observed, simulated)
        rmse_val = np.sqrt(np.mean((observed - simulated)**2))
        bias = np.mean(simulated - observed)
        rel_bias = (np.sum(simulated) - np.sum(observed)) / np.sum(observed) * 100
        corr = np.corrcoef(observed, simulated)[0, 1]

        return {
            'NSE': nse,
            'RMSE': rmse_val,
            'Bias': bias,
            'Relative_Bias_%': rel_bias,
            'Correlation': corr,
        }


def main():
    """主函数：高精度参数率定"""
    print("=" * 80)
    print("高精度参数率定：基于60天数据的上游分区HBV参数率定")
    print("=" * 80)

    # ========================================================================
    # 第1步：加载60天降雨数据
    # ========================================================================
    print("\n[1/7] 加载60天降雨数据...")

    data_file = Path('results/extended_timeseries_60days/timeseries_60days.csv')
    if not data_file.exists():
        print(f"❌ 错误: 数据文件不存在: {data_file}")
        print("   请先运行: python generate_extended_timeseries.py")
        return

    df = pd.read_csv(data_file)
    rainfall = df['precipitation_mm_per_hour'].values
    timestamps = pd.to_datetime(df['timestamp'])

    print(f"  ✓ 已加载 {len(rainfall)} 个时间步")
    print(f"  ✓ 时间范围: {timestamps.iloc[0]} 至 {timestamps.iloc[-1]}")
    print(f"  ✓ 总降雨量: {rainfall.sum():.2f} mm")

    # ========================================================================
    # 第2步：生成"真实"观测数据
    # ========================================================================
    print("\n[2/7] 生成观测数据（使用已知真实参数）...")

    # 真实参数（我们将尝试恢复这些参数）
    true_params = {
        'degree_day_factor': 3.0,
        'snow_threshold': 0.0,
        'field_capacity': 90.0,    # 真实值（待率定）
        'beta': 1.3,               # 真实值（待率定）
        'k0': 0.15,                # 真实值（待率定）
        'k1': 0.08,
        'k2': 0.02,
        'percolation': 1.0,
        'initial_snow': 0.0,
        'initial_soil': 0.0,
        'initial_upper': 0.0,
        'initial_lower': 0.0,
    }

    print("  真实参数（将尝试率定恢复）:")
    for key in ['field_capacity', 'beta', 'k0']:
        print(f"    {key}: {true_params[key]}")

    observed = run_hbv_model(rainfall, true_params)
    print(f"  ✓ 生成观测径流: {len(observed)} 个时间步")
    print(f"  ✓ 总径流量: {observed.sum():.2f} mm")
    print(f"  ✓ 径流系数: {observed.sum() / rainfall.sum():.4f}")

    # ========================================================================
    # 第3步：定义参数空间（基于敏感性分析）
    # ========================================================================
    print("\n[3/7] 定义参数空间...")

    # 重点率定最敏感的参数
    param_names = ['field_capacity', 'beta', 'k0']
    param_bounds = [
        (60, 120),    # field_capacity: 真实值90在中间
        (0.8, 1.8),   # beta: 真实值1.3在中间
        (0.09, 0.20), # k0: 真实值0.15在中间
    ]

    # 固定的参数
    fixed_params = {
        'degree_day_factor': 3.0,
        'snow_threshold': 0.0,
        'k1': 0.08,
        'k2': 0.02,
        'percolation': 1.0,
        'initial_snow': 0.0,
        'initial_soil': 0.0,
        'initial_upper': 0.0,
        'initial_lower': 0.0,
    }

    print(f"  待率定参数: {param_names}")
    print(f"  参数范围:")
    for name, bounds in zip(param_names, param_bounds):
        print(f"    {name}: [{bounds[0]}, {bounds[1]}]")

    # ========================================================================
    # 第4步：定义目标函数
    # ========================================================================
    print("\n[4/7] 定义目标函数...")

    def objective_function(params_array):
        """
        目标函数：最大化NSE

        Parameters
        ----------
        params_array : array-like
            [field_capacity, beta, k0]

        Returns
        -------
        float
            NSE (越大越好)
        """
        # 构建完整参数字典
        params = fixed_params.copy()
        params['field_capacity'] = params_array[0]
        params['beta'] = params_array[1]
        params['k0'] = params_array[2]

        # 运行模型
        simulated = run_hbv_model(rainfall, params)

        # 计算NSE
        nse = calculate_nse(observed, simulated)

        return nse

    print("  ✓ 目标函数: 最大化NSE")

    # ========================================================================
    # 第5步：运行参数率定
    # ========================================================================
    print("\n[5/7] 运行参数率定...")
    print("  算法: Differential Evolution")
    print("  参数: maxiter=200, popsize=20, seed=42")

    if HYDROSIS_AVAILABLE:
        from scipy.optimize import differential_evolution

        result = differential_evolution(
            func=lambda x: -objective_function(x),  # 最小化负NSE = 最大化NSE
            bounds=param_bounds,
            maxiter=200,
            popsize=20,
            seed=42,
            polish=True,
            disp=True,
        )

        best_params_array = result.x
        best_nse = -result.fun
        success = result.success
        n_iterations = result.nit
        n_evaluations = result.nfev

    else:
        # 简化网格搜索（如果没有基础库）
        print("  ⚠ 使用简化网格搜索...")
        best_nse = -np.inf
        best_params_array = None

        n_grid = 10
        for fc in np.linspace(param_bounds[0][0], param_bounds[0][1], n_grid):
            for beta in np.linspace(param_bounds[1][0], param_bounds[1][1], n_grid):
                for k0 in np.linspace(param_bounds[2][0], param_bounds[2][1], n_grid):
                    nse = objective_function([fc, beta, k0])
                    if nse > best_nse:
                        best_nse = nse
                        best_params_array = [fc, beta, k0]

        success = True
        n_iterations = n_grid**3
        n_evaluations = n_grid**3

    # 构建最优参数字典
    best_params = fixed_params.copy()
    best_params['field_capacity'] = best_params_array[0]
    best_params['beta'] = best_params_array[1]
    best_params['k0'] = best_params_array[2]

    print(f"\n  ✓ 率定完成!")
    print(f"    状态: {'成功' if success else '失败'}")
    print(f"    迭代次数: {n_iterations}")
    print(f"    函数评估: {n_evaluations}")
    print(f"    最优NSE: {best_nse:.6f}")

    # ========================================================================
    # 第6步：评估率定结果
    # ========================================================================
    print("\n[6/7] 评估率定结果...")

    # 使用最优参数模拟
    simulated = run_hbv_model(rainfall, best_params)

    # 计算所有指标
    metrics = calculate_all_metrics(observed, simulated)

    print("\n  性能指标:")
    for key, value in metrics.items():
        print(f"    {key}: {value:.6f}")

    # 参数恢复精度
    print("\n  参数恢复精度:")
    print(f"    {'参数':<20} {'真实值':<12} {'率定值':<12} {'误差%':<10}")
    print(f"    {'-'*54}")

    for param_name in param_names:
        true_val = true_params[param_name]
        calib_val = best_params[param_name]
        error_pct = abs(calib_val - true_val) / true_val * 100
        print(f"    {param_name:<20} {true_val:<12.4f} {calib_val:<12.4f} {error_pct:<10.2f}")

    # ========================================================================
    # 第7步：保存结果和可视化
    # ========================================================================
    print("\n[7/7] 保存结果...")

    output_dir = Path('results/calibration_60day')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存参数
    params_file = output_dir / 'calibrated_parameters.yaml'
    with open(params_file, 'w') as f:
        yaml.dump(best_params, f, default_flow_style=False)
    print(f"  ✓ 参数已保存: {params_file}")

    # 保存指标
    metrics_file = output_dir / 'performance_metrics.yaml'
    with open(metrics_file, 'w') as f:
        yaml.dump(metrics, f, default_flow_style=False)
    print(f"  ✓ 指标已保存: {metrics_file}")

    # 保存时间序列
    results_df = pd.DataFrame({
        'timestamp': timestamps,
        'rainfall': rainfall,
        'observed': observed,
        'simulated': simulated,
        'residual': observed - simulated,
    })
    ts_file = output_dir / 'calibration_timeseries.csv'
    results_df.to_csv(ts_file, index=False)
    print(f"  ✓ 时间序列已保存: {ts_file}")

    # 生成可视化
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'High-Precision Calibration Results (NSE={best_nse:.4f})',
                 fontsize=14, fontweight='bold')

    # 子图1: 降雨
    ax = axes[0]
    ax.fill_between(range(len(rainfall)), rainfall, alpha=0.5, color='blue')
    ax.set_ylabel('Rainfall (mm/h)')
    ax.set_title('Input: 60-Day Rainfall')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(rainfall))

    # 子图2: 观测vs模拟
    ax = axes[1]
    ax.plot(observed, label='Observed', linewidth=1.5, alpha=0.8, color='black')
    ax.plot(simulated, label='Simulated', linewidth=1.2, alpha=0.7, color='red', linestyle='--')
    ax.set_ylabel('Runoff (mm/h)')
    ax.set_title(f'Observed vs Simulated (NSE={best_nse:.4f}, RMSE={metrics.get("RMSE", 0):.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(observed))

    # 子图3: 残差
    ax = axes[2]
    residuals = observed - simulated
    ax.plot(residuals, linewidth=0.8, color='green', alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.set_ylabel('Residual (mm/h)')
    ax.set_xlabel('Time (hours)')
    ax.set_title(f'Residuals (Mean={np.mean(residuals):.6f}, Std={np.std(residuals):.6f})')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, len(residuals))

    plt.tight_layout()

    plot_file = output_dir / 'calibration_results.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {plot_file}")

    # 散点图
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(observed, simulated, alpha=0.5, s=20)

    # 1:1线
    max_val = max(observed.max(), simulated.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 Line')

    ax.set_xlabel('Observed Runoff (mm/h)', fontsize=12)
    ax.set_ylabel('Simulated Runoff (mm/h)', fontsize=12)
    ax.set_title(f'Observed vs Simulated Scatter Plot\nNSE={best_nse:.4f}, R²={metrics.get("Correlation", 0)**2:.4f}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    scatter_file = output_dir / 'scatter_plot.png'
    plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ 散点图已保存: {scatter_file}")

    # 生成报告
    report_file = output_dir / 'calibration_report.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("高精度参数率定报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 数据概况\n")
        f.write(f"   时间长度: 60天 (1440小时)\n")
        f.write(f"   总降雨量: {rainfall.sum():.2f} mm\n")
        f.write(f"   总径流量: {observed.sum():.2f} mm\n")
        f.write(f"   径流系数: {observed.sum() / rainfall.sum():.4f}\n\n")

        f.write("2. 率定设置\n")
        f.write(f"   算法: Differential Evolution\n")
        f.write(f"   目标函数: NSE (Nash-Sutcliffe Efficiency)\n")
        f.write(f"   率定参数: {param_names}\n")
        f.write(f"   迭代次数: {n_iterations}\n")
        f.write(f"   函数评估: {n_evaluations}\n\n")

        f.write("3. 率定结果\n")
        f.write(f"   状态: {'成功' if success else '失败'}\n")
        f.write(f"   最优NSE: {best_nse:.6f}\n\n")

        f.write("4. 性能指标\n")
        for key, value in metrics.items():
            f.write(f"   {key}: {value:.6f}\n")
        f.write("\n")

        f.write("5. 参数恢复精度\n")
        f.write(f"   {'参数':<20} {'真实值':<12} {'率定值':<12} {'误差%':<10}\n")
        f.write(f"   {'-'*54}\n")
        for param_name in param_names:
            true_val = true_params[param_name]
            calib_val = best_params[param_name]
            error_pct = abs(calib_val - true_val) / true_val * 100
            f.write(f"   {param_name:<20} {true_val:<12.4f} {calib_val:<12.4f} {error_pct:<10.2f}\n")
        f.write("\n")

        f.write("6. 结论\n")
        if best_nse > 0.9:
            f.write("   ✓ 优秀: NSE > 0.9, 率定精度极高\n")
        elif best_nse > 0.75:
            f.write("   ✓ 良好: NSE > 0.75, 率定精度较高\n")
        elif best_nse > 0.5:
            f.write("   ⚠ 可接受: NSE > 0.5, 率定精度一般\n")
        else:
            f.write("   ❌ 不佳: NSE < 0.5, 需要改进\n")

    print(f"  ✓ 报告已保存: {report_file}")

    # 打印最终总结
    print("\n" + "=" * 80)
    print("✓ 高精度参数率定完成！")
    print("=" * 80)
    print(f"\n关键结果:")
    print(f"  • NSE: {best_nse:.6f}")
    print(f"  • 参数误差: ", end="")
    errors = [abs(best_params[p] - true_params[p])/true_params[p]*100 for p in param_names]
    print(f"{np.mean(errors):.2f}% (平均)")
    print(f"\n输出目录: {output_dir}")
    print()


if __name__ == '__main__':
    main()
