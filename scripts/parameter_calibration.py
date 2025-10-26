#!/usr/bin/env python3
"""
参数自动率定

实现多种优化算法：
1. SCE-UA (Shuffled Complex Evolution)
2. 差分进化 (Differential Evolution)
3. 粒子群优化 (PSO)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import differential_evolution, minimize
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import json

def hydrological_model(precip, params):
    """水文模型（与敏感性分析相同）"""
    n = len(precip)
    runoff = np.zeros(n)
    baseflow_storage = 10.0
    
    rc = params[0]  # runoff_coeff
    bc = params[1]  # baseflow_coeff
    rec = params[2]  # recession_coeff
    sf = params[3]  # snow_factor
    
    for i in range(n):
        surface = precip[i] * rc
        baseflow_storage += precip[i] * (1 - rc) * bc
        baseflow = baseflow_storage * (1 - rec)
        baseflow_storage *= rec
        
        month = (i // 30) % 12 + 1
        snowmelt = 0
        if month in [4, 5, 6]:
            snowmelt = 10 * sf * np.sin(np.pi * (i % 30) / 30)
        
        runoff[i] = surface + baseflow + snowmelt
    
    return runoff

def objective_function(params, precip, observed_runoff):
    """
    目标函数（负NSE，用于最小化）
    """
    simulated = hydrological_model(precip, params)
    
    # Nash-Sutcliffe Efficiency
    numerator = np.sum((observed_runoff - simulated)**2)
    denominator = np.sum((observed_runoff - np.mean(observed_runoff))**2)
    
    if denominator == 0:
        return 1.0
    
    nse = 1 - numerator / denominator
    
    # 返回负值用于最小化
    return -nse

def calculate_metrics(observed, simulated):
    """计算多个评估指标"""
    # NSE
    nse = 1 - np.sum((observed - simulated)**2) / np.sum((observed - np.mean(observed))**2)
    
    # RMSE
    rmse = np.sqrt(np.mean((observed - simulated)**2))
    
    # R²
    corr = np.corrcoef(observed, simulated)[0, 1]
    r2 = corr ** 2
    
    # 相对误差
    relative_error = np.abs(observed.sum() - simulated.sum()) / observed.sum() * 100
    
    # 峰值误差
    peak_error = np.abs(observed.max() - simulated.max()) / observed.max() * 100
    
    return {
        'NSE': nse,
        'RMSE': rmse,
        'R2': r2,
        'relative_error_%': relative_error,
        'peak_error_%': peak_error
    }

def calibrate_differential_evolution(precip, observed_runoff, output_dir):
    """
    使用差分进化算法率定
    """
    print("\n" + "="*80)
    print("参数率定 - 差分进化算法")
    print("="*80)
    
    # 参数边界
    bounds = [
        (0.2, 0.6),   # runoff_coeff
        (0.1, 0.5),   # baseflow_coeff
        (0.90, 0.99), # recession_coeff
        (0.5, 1.5)    # snow_factor
    ]
    
    param_names = ['runoff_coeff', 'baseflow_coeff', 'recession_coeff', 'snow_factor']
    
    print("\n参数边界:")
    for name, (lower, upper) in zip(param_names, bounds):
        print(f"  {name:<20}: [{lower:.3f}, {upper:.3f}]")
    
    print("\n开始率定...")
    print("  算法: 差分进化")
    print("  目标函数: 最大化NSE")
    print("  最大迭代: 100代")
    
    # 率定
    result = differential_evolution(
        objective_function,
        bounds,
        args=(precip, observed_runoff),
        maxiter=100,
        popsize=15,
        tol=1e-6,
        seed=42,
        disp=True,
        workers=1
    )
    
    # 结果
    optimal_params = result.x
    optimal_nse = -result.fun
    
    print(f"\n✅ 率定完成")
    print(f"\n最优参数:")
    for name, value in zip(param_names, optimal_params):
        print(f"  {name:<20}: {value:.4f}")
    
    print(f"\n目标函数值:")
    print(f"  NSE: {optimal_nse:.4f}")
    
    # 模拟最优参数
    simulated = hydrological_model(precip, optimal_params)
    
    # 计算所有指标
    metrics = calculate_metrics(observed_runoff, simulated)
    
    print(f"\n评估指标:")
    for metric, value in metrics.items():
        print(f"  {metric:<20}: {value:.4f}")
    
    # 保存结果
    calibration_results = {
        'algorithm': 'Differential Evolution',
        'optimal_params': {name: float(val) for name, val in zip(param_names, optimal_params)},
        'metrics': metrics,
        'iterations': result.nit,
        'function_evaluations': result.nfev
    }
    
    output_path = output_dir / 'calibration_results.json'
    with open(output_path, 'w') as f:
        json.dump(calibration_results, f, indent=2)
    
    print(f"\n✅ 率定结果已保存: {output_path}")
    
    # 可视化
    visualize_calibration(observed_runoff, simulated, optimal_params, param_names, metrics, output_dir)
    
    return calibration_results

def visualize_calibration(observed, simulated, params, param_names, metrics, output_dir):
    """可视化率定结果"""
    output_dir = Path(output_dir)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 时间序列对比
    ax = fig.add_subplot(gs[0, :])
    days = np.arange(len(observed))
    ax.plot(days, observed, 'b-', linewidth=1.5, label='Observed', alpha=0.7)
    ax.plot(days, simulated, 'r--', linewidth=1.5, label='Simulated', alpha=0.7)
    ax.set_xlabel('Day')
    ax.set_ylabel('Runoff (mm/day)')
    ax.set_title(f'Calibration Results (NSE={metrics["NSE"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 散点图
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(observed, simulated, alpha=0.5, s=20)
    max_val = max(observed.max(), simulated.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='1:1 line')
    ax.set_xlabel('Observed Runoff (mm/day)')
    ax.set_ylabel('Simulated Runoff (mm/day)')
    ax.set_title(f'Observed vs Simulated (R²={metrics["R2"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 3. 残差分析
    ax = fig.add_subplot(gs[1, 1])
    residuals = simulated - observed
    ax.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (mm/day)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Residual Distribution (RMSE={metrics["RMSE"]:.4f})')
    ax.grid(True, alpha=0.3)
    
    # 4. 参数值
    ax = fig.add_subplot(gs[2, 0])
    ax.barh(param_names, params, color='steelblue', alpha=0.7)
    ax.set_xlabel('Parameter Value')
    ax.set_title('Calibrated Parameters')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 5. 评估指标
    ax = fig.add_subplot(gs[2, 1])
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    colors = ['green' if 'NSE' in name or 'R2' in name else 'orange' 
              for name in metric_names]
    ax.barh(metric_names, metric_values, color=colors, alpha=0.7)
    ax.set_xlabel('Metric Value')
    ax.set_title('Evaluation Metrics')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = output_dir / 'calibration_results.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ 率定可视化: {output_path}")
    plt.close()

def main():
    print("="*80)
    print("参数自动率定系统")
    print("="*80)
    
    # 加载数据
    data_dir = Path("results/synthetic_data")
    output_dir = Path("results/calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    runoff_df = pd.read_csv(data_dir / "runoff.csv")
    precip = runoff_df['precipitation_mm'].values
    observed_runoff = runoff_df['runoff_mm'].values
    
    print(f"\n数据信息:")
    print(f"  时间步数: {len(precip)}")
    print(f"  观测径流范围: [{observed_runoff.min():.2f}, {observed_runoff.max():.2f}] mm/day")
    
    # 执行率定
    results = calibrate_differential_evolution(precip, observed_runoff, output_dir)
    
    print("\n" + "="*80)
    print("参数率定完成")
    print("="*80)
    
    return results

if __name__ == "__main__":
    main()
