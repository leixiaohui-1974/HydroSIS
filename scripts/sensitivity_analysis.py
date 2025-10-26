#!/usr/bin/env python3
"""
参数敏感性分析

使用SALib库实现：
1. Morris方法 - 快速筛选
2. Sobol方法 - 全局敏感性
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

try:
    from SALib.sample import morris as morris_sample, saltelli
    from SALib.analyze import morris as morris_analyze, sobol
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    print("⚠️  SALib未安装，将使用简化的敏感性分析")

def simple_hydrological_model(precip, params):
    """
    简化的水文模型
    
    参数:
        precip: 降雨序列 (mm/day)
        params: dict with keys:
            - runoff_coeff: 径流系数 (0-1)
            - baseflow_coeff: 基流系数 (0-1)
            - recession_coeff: 衰退系数 (0-1)
            - snow_factor: 融雪因子 (0-2)
    
    返回:
        径流序列 (mm/day)
    """
    n = len(precip)
    runoff = np.zeros(n)
    baseflow_storage = 10.0
    
    # 提取参数
    rc = params.get('runoff_coeff', 0.4)
    bc = params.get('baseflow_coeff', 0.3)
    rec = params.get('recession_coeff', 0.95)
    sf = params.get('snow_factor', 1.0)
    
    for i in range(n):
        # 地表径流
        surface = precip[i] * rc
        
        # 地下水补给
        baseflow_storage += precip[i] * (1 - rc) * bc
        
        # 基流出流
        baseflow = baseflow_storage * (1 - rec)
        baseflow_storage *= rec
        
        # 融雪贡献（春季）
        month = (i // 30) % 12 + 1
        snowmelt = 0
        if month in [4, 5, 6]:
            snowmelt = 10 * sf * np.sin(np.pi * (i % 30) / 30)
        
        runoff[i] = surface + baseflow + snowmelt
    
    return runoff

def calculate_objective(observed, simulated):
    """
    计算目标函数（NSE）
    """
    # Nash-Sutcliffe Efficiency
    numerator = np.sum((observed - simulated)**2)
    denominator = np.sum((observed - np.mean(observed))**2)
    
    if denominator == 0:
        return 0
    
    nse = 1 - numerator / denominator
    return nse

def sensitivity_analysis_morris(precip, observed_runoff, output_dir):
    """
    Morris敏感性分析（快速筛选）
    """
    print("\n" + "="*80)
    print("Morris敏感性分析")
    print("="*80)
    
    if not SALIB_AVAILABLE:
        print("⚠️  SALib未安装，跳过Morris分析")
        return None
    
    # 定义参数空间
    problem = {
        'num_vars': 4,
        'names': ['runoff_coeff', 'baseflow_coeff', 'recession_coeff', 'snow_factor'],
        'bounds': [
            [0.2, 0.6],   # runoff_coeff
            [0.1, 0.5],   # baseflow_coeff
            [0.90, 0.99], # recession_coeff
            [0.5, 1.5]    # snow_factor
        ]
    }
    
    # 生成样本
    print("\n生成Morris样本...")
    n_trajectories = 10
    param_values = morris_sample.sample(problem, N=n_trajectories, num_levels=4)
    
    print(f"  样本数: {len(param_values)}")
    
    # 运行模型
    print("\n运行模型...")
    Y = np.zeros(len(param_values))
    
    for i, params_array in enumerate(param_values):
        params = {
            'runoff_coeff': params_array[0],
            'baseflow_coeff': params_array[1],
            'recession_coeff': params_array[2],
            'snow_factor': params_array[3]
        }
        
        simulated = simple_hydrological_model(precip, params)
        Y[i] = calculate_objective(observed_runoff, simulated)
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {i+1}/{len(param_values)}")
    
    # 分析
    print("\n分析敏感性...")
    Si = morris_analyze.analyze(problem, param_values, Y, print_to_console=False)
    
    # 结果
    print("\nMorris敏感性分析结果:")
    print(f"{'参数':<20} {'μ*':<12} {'σ':<12} {'敏感性'}")
    print("-" * 60)
    
    for i, name in enumerate(problem['names']):
        mu_star = Si['mu_star'][i]
        sigma = Si['sigma'][i]
        
        if mu_star > 0.1:
            sensitivity = "高 ⭐⭐⭐"
        elif mu_star > 0.05:
            sensitivity = "中 ⭐⭐"
        else:
            sensitivity = "低 ⭐"
        
        print(f"{name:<20} {mu_star:<12.4f} {sigma:<12.4f} {sensitivity}")
    
    # 可视化
    visualize_morris(Si, problem['names'], output_dir)
    
    return Si

def sensitivity_analysis_sobol(precip, observed_runoff, output_dir):
    """
    Sobol全局敏感性分析
    """
    print("\n" + "="*80)
    print("Sobol全局敏感性分析")
    print("="*80)
    
    if not SALIB_AVAILABLE:
        print("⚠️  SALib未安装，使用简化分析")
        return simple_sensitivity_analysis(precip, observed_runoff, output_dir)
    
    # 定义参数空间
    problem = {
        'num_vars': 4,
        'names': ['runoff_coeff', 'baseflow_coeff', 'recession_coeff', 'snow_factor'],
        'bounds': [
            [0.2, 0.6],
            [0.1, 0.5],
            [0.90, 0.99],
            [0.5, 1.5]
        ]
    }
    
    # 生成Sobol样本
    print("\n生成Sobol样本...")
    n_samples = 512  # 2^9, 推荐值
    param_values = saltelli.sample(problem, n_samples)
    
    print(f"  样本数: {len(param_values)}")
    
    # 运行模型
    print("\n运行模型...")
    Y = np.zeros(len(param_values))
    
    for i, params_array in enumerate(param_values):
        params = {
            'runoff_coeff': params_array[0],
            'baseflow_coeff': params_array[1],
            'recession_coeff': params_array[2],
            'snow_factor': params_array[3]
        }
        
        simulated = simple_hydrological_model(precip, params)
        Y[i] = calculate_objective(observed_runoff, simulated)
        
        if (i + 1) % 100 == 0:
            print(f"  进度: {i+1}/{len(param_values)}")
    
    # 分析
    print("\n分析Sobol指数...")
    Si = sobol.analyze(problem, Y, print_to_console=False)
    
    # 结果
    print("\nSobol敏感性分析结果:")
    print(f"{'参数':<20} {'S1':<12} {'ST':<12} {'敏感性'}")
    print("-" * 60)
    
    for i, name in enumerate(problem['names']):
        s1 = Si['S1'][i]
        st = Si['ST'][i]
        
        if st > 0.3:
            sensitivity = "高 ⭐⭐⭐"
        elif st > 0.1:
            sensitivity = "中 ⭐⭐"
        else:
            sensitivity = "低 ⭐"
        
        print(f"{name:<20} {s1:<12.4f} {st:<12.4f} {sensitivity}")
    
    print("\n说明:")
    print("  S1: 一阶敏感性指数（主效应）")
    print("  ST: 总敏感性指数（包括交互作用）")
    
    # 可视化
    visualize_sobol(Si, problem['names'], output_dir)
    
    return Si

def simple_sensitivity_analysis(precip, observed_runoff, output_dir):
    """
    简化的敏感性分析（当SALib不可用时）
    
    使用单参数扰动法
    """
    print("\n使用简化的单参数敏感性分析...")
    
    # 基准参数
    base_params = {
        'runoff_coeff': 0.4,
        'baseflow_coeff': 0.3,
        'recession_coeff': 0.95,
        'snow_factor': 1.0
    }
    
    # 基准目标函数
    base_sim = simple_hydrological_model(precip, base_params)
    base_obj = calculate_objective(observed_runoff, base_sim)
    
    print(f"\n基准NSE: {base_obj:.4f}")
    
    # 扰动范围
    perturbations = [-0.2, -0.1, 0.1, 0.2]
    
    results = {}
    
    for param_name in base_params.keys():
        sensitivities = []
        
        for pert in perturbations:
            params = base_params.copy()
            params[param_name] = base_params[param_name] * (1 + pert)
            
            # 确保在合理范围内
            if param_name == 'runoff_coeff':
                params[param_name] = np.clip(params[param_name], 0.1, 0.9)
            elif param_name == 'baseflow_coeff':
                params[param_name] = np.clip(params[param_name], 0.05, 0.5)
            elif param_name == 'recession_coeff':
                params[param_name] = np.clip(params[param_name], 0.85, 0.99)
            elif param_name == 'snow_factor':
                params[param_name] = np.clip(params[param_name], 0.1, 2.0)
            
            sim = simple_hydrological_model(precip, params)
            obj = calculate_objective(observed_runoff, sim)
            
            # 敏感性 = 目标函数变化 / 参数变化
            sensitivity = abs((obj - base_obj) / (pert if pert != 0 else 0.01))
            sensitivities.append(sensitivity)
        
        results[param_name] = np.mean(sensitivities)
    
    # 打印结果
    print("\n简化敏感性分析结果:")
    print(f"{'参数':<20} {'敏感性指数':<15} {'等级'}")
    print("-" * 55)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for param, sensitivity in sorted_results:
        if sensitivity > 0.5:
            level = "高 ⭐⭐⭐"
        elif sensitivity > 0.2:
            level = "中 ⭐⭐"
        else:
            level = "低 ⭐"
        
        print(f"{param:<20} {sensitivity:<15.4f} {level}")
    
    # 可视化
    visualize_simple_sensitivity(results, output_dir)
    
    return results

def visualize_morris(Si, param_names, output_dir):
    """可视化Morris结果"""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(Si['mu_star'], Si['sigma'], s=100, alpha=0.6)
    
    for i, name in enumerate(param_names):
        ax.annotate(name, (Si['mu_star'][i], Si['sigma'][i]),
                   xytext=(5, 5), textcoords='offset points')
    
    ax.set_xlabel('μ* (Mean of absolute elementary effects)')
    ax.set_ylabel('σ (Standard deviation of elementary effects)')
    ax.set_title('Morris Sensitivity Analysis')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'sensitivity_morris.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ Morris图表: {output_path}")
    plt.close()

def visualize_sobol(Si, param_names, output_dir):
    """可视化Sobol结果"""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(param_names))
    width = 0.35
    
    ax.bar(x - width/2, Si['S1'], width, label='S1 (First order)', alpha=0.7)
    ax.bar(x + width/2, Si['ST'], width, label='ST (Total)', alpha=0.7)
    
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Sensitivity Index')
    ax.set_title('Sobol Sensitivity Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'sensitivity_sobol.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Sobol图表: {output_path}")
    plt.close()

def visualize_simple_sensitivity(results, output_dir):
    """可视化简化敏感性结果"""
    output_dir = Path(output_dir)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    params = list(results.keys())
    sensitivities = list(results.values())
    
    colors = ['red' if s > 0.5 else 'orange' if s > 0.2 else 'yellow' 
              for s in sensitivities]
    
    ax.barh(params, sensitivities, color=colors, alpha=0.7)
    ax.set_xlabel('Sensitivity Index')
    ax.set_title('Parameter Sensitivity Analysis (Simplified)')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_path = output_dir / 'sensitivity_simple.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n  ✅ 敏感性图表: {output_path}")
    plt.close()

def main():
    print("="*80)
    print("参数敏感性分析")
    print("="*80)
    
    # 加载数据
    data_dir = Path("results/synthetic_data")
    output_dir = Path("results/sensitivity_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载降雨和径流
    runoff_df = pd.read_csv(data_dir / "runoff.csv")
    precip = runoff_df['precipitation_mm'].values
    observed_runoff = runoff_df['runoff_mm'].values
    
    print(f"\n数据加载:")
    print(f"  时间步数: {len(precip)}")
    print(f"  总降雨: {precip.sum():.2f} mm")
    print(f"  总径流: {observed_runoff.sum():.2f} mm")
    
    # 方法1: Morris分析
    if SALIB_AVAILABLE:
        morris_results = sensitivity_analysis_morris(precip, observed_runoff, output_dir)
    
    # 方法2: Sobol分析
    sobol_results = sensitivity_analysis_sobol(precip, observed_runoff, output_dir)
    
    print("\n" + "="*80)
    print("敏感性分析完成")
    print("="*80)

if __name__ == "__main__":
    main()
