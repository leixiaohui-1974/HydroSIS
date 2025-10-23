#!/usr/bin/env python3
"""
完整的参数分析工作流示例

这个示例展示如何使用HydroSIS参数分析框架进行：
1. 模型性能评估
2. 参数敏感性分析
3. 参数率定
4. 不确定性分析
5. 结果可视化

这是一个集成演示，使用简单的测试模型展示完整流程。
"""
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.analysis import (
    # Metrics
    calculate_metrics,
    nash_sutcliffe_efficiency,
    # Sensitivity
    one_at_a_time_sensitivity,
    # Calibration
    calibrate_model,
    # Uncertainty
    monte_carlo_analysis,
    glue_analysis,
    # Visualization
    plot_hydrograph_comparison,
    plot_scatter,
    plot_convergence_history,
    plot_parameter_distributions,
    plot_uncertainty_envelope,
    plot_tornado_sensitivity,
    create_analysis_report_figures,
)

print("=" * 80)
print("HydroSIS 参数分析框架 - 完整工作流示例")
print("=" * 80)

# ============================================================================
# 步骤 0: 准备数据和模型
# ============================================================================
print("\n步骤 0: 准备数据和模型")
print("-" * 80)

np.random.seed(42)

# 定义"真实"参数
true_params = {
    'FC': 150.0,
    'K0': 0.30,
    'BETA': 1.0
}

print(f"真实参数: {true_params}")

# 定义简单的水文模型
def hydrological_model(FC, K0, BETA, precipitation=None, n_steps=120):
    """简化的水文模型用于演示

    Args:
        FC: 田间持水能力 (mm)
        K0: 快速响应系数
        BETA: 土壤湿度指数

    Returns:
        径流时间序列 (m³/s)
    """
    if precipitation is None:
        # 生成模拟降雨
        t = np.arange(n_steps)
        precipitation = 5 + 10 * np.sin(t / 20) + np.random.gamma(2, 2, n_steps)

    # 简化的产流计算
    soil_moisture = np.zeros(n_steps)
    runoff = np.zeros(n_steps)

    soil = FC * 0.5  # 初始土壤湿度

    for i in range(n_steps):
        # 土壤蓄水
        available_capacity = FC - soil
        infiltration = precipitation[i] * (1 - (soil / FC) ** BETA)

        # 产流
        surface_runoff = precipitation[i] - infiltration
        baseflow = K0 * soil

        # 更新土壤湿度
        soil = soil + infiltration - baseflow
        soil = np.clip(soil, 0, FC)

        # 总径流
        runoff[i] = surface_runoff + baseflow

        soil_moisture[i] = soil

    return runoff

# 生成"观测数据"
observed_runoff = hydrological_model(**true_params)
# 添加观测噪声
observed_runoff = observed_runoff + np.random.normal(0, 1.5, len(observed_runoff))
observed_runoff = np.maximum(observed_runoff, 0)

print(f"观测数据生成完成: {len(observed_runoff)} 时间步")
print(f"  平均径流: {observed_runoff.mean():.2f} m³/s")
print(f"  峰值径流: {observed_runoff.max():.2f} m³/s")

# 参数搜索范围
param_bounds = {
    'FC': [100, 200],
    'K0': [0.1, 0.5],
    'BETA': [0.5, 2.0]
}

# ============================================================================
# 步骤 1: 模型性能评估（使用真实参数）
# ============================================================================
print("\n步骤 1: 模型性能评估（基准）")
print("-" * 80)

# 使用真实参数运行模型
baseline_simulated = hydrological_model(**true_params)

# 计算性能指标
baseline_metrics = calculate_metrics(
    observed_runoff,
    baseline_simulated,
    metrics=['nse', 'rmse', 'mae', 'pbias', 'kge']
)

print("基准性能（真实参数）:")
for metric, value in baseline_metrics.items():
    if 'peak' not in metric and 'time' not in metric:
        print(f"  {metric.upper():10s}: {value:8.4f}")

# ============================================================================
# 步骤 2: 参数敏感性分析
# ============================================================================
print("\n步骤 2: 参数敏感性分析 (OAT)")
print("-" * 80)

def sensitivity_objective(FC, K0, BETA):
    """敏感性分析目标函数"""
    simulated = hydrological_model(FC, K0, BETA)
    return nash_sutcliffe_efficiency(observed_runoff, simulated)

print("运行OAT敏感性分析...")
sensitivity_results = one_at_a_time_sensitivity(
    model_function=sensitivity_objective,
    parameters=true_params.copy(),
    param_ranges=param_bounds,
    variations=[0.7, 0.85, 1.0, 1.15, 1.3],
    metric_name='NSE'
)

print("\n敏感性排序:")
for i, (param, si) in enumerate(sensitivity_results['ranked_parameters'], 1):
    print(f"  {i}. {param:5s}: SI = {si:.4f}")

# ============================================================================
# 步骤 3: 参数率定
# ============================================================================
print("\n步骤 3: 自动参数率定 (SCE-UA)")
print("-" * 80)

def calibration_objective(FC, K0, BETA):
    """率定目标函数：最大化NSE"""
    simulated = hydrological_model(FC, K0, BETA)
    return nash_sutcliffe_efficiency(observed_runoff, simulated)

print("运行SCE-UA优化...")
calibration_result = calibrate_model(
    objective_function=calibration_objective,
    param_bounds=param_bounds,
    method='sce_ua',
    maximize=True,
    n_complexes=5,
    max_iterations=50,
    patience=10,
    seed=42,
    verbose=False
)

print(f"\n率定结果:")
print(f"  最优NSE: {calibration_result.best_score:.6f}")
print(f"  迭代次数: {calibration_result.n_iterations}")
print(f"  函数评估: {calibration_result.n_evaluations}")
print(f"  计算时间: {calibration_result.computation_time:.2f}秒")

print(f"\n率定参数 vs 真实参数:")
calibrated_params = calibration_result.best_params
for param_name in ['FC', 'K0', 'BETA']:
    est = calibrated_params[param_name]
    true_val = true_params[param_name]
    error = abs(est - true_val) / true_val * 100
    print(f"  {param_name:5s}: {est:7.3f} (真值: {true_val:6.3f}, 误差: {error:5.1f}%)")

# 使用率定参数运行模型
calibrated_simulated = hydrological_model(**calibrated_params)

# 计算率定后的性能
calibrated_metrics = calculate_metrics(
    observed_runoff,
    calibrated_simulated,
    metrics=['nse', 'rmse', 'mae', 'pbias', 'kge']
)

print("\n率定后性能:")
for metric, value in calibrated_metrics.items():
    if 'peak' not in metric and 'time' not in metric:
        print(f"  {metric.upper():10s}: {value:8.4f}")

# ============================================================================
# 步骤 4: 不确定性分析
# ============================================================================
print("\n步骤 4: 不确定性分析 (Monte Carlo + GLUE)")
print("-" * 80)

# 定义参数分布（以率定参数为中心）
param_distributions = {
    'FC': {'dist': 'normal', 'mean': calibrated_params['FC'], 'std': 15, 'min': 100, 'max': 200},
    'K0': {'dist': 'normal', 'mean': calibrated_params['K0'], 'std': 0.05, 'min': 0.1, 'max': 0.5},
    'BETA': {'dist': 'normal', 'mean': calibrated_params['BETA'], 'std': 0.2, 'min': 0.5, 'max': 2.0}
}

def uncertainty_model(FC, K0, BETA):
    """不确定性分析模型函数"""
    return hydrological_model(FC, K0, BETA)

print("运行Monte Carlo不确定性分析...")
mc_results = monte_carlo_analysis(
    model_function=uncertainty_model,
    param_distributions=param_distributions,
    n_samples=200,
    sampling_method='lhs',
    observed_data=observed_runoff,
    metric_function=nash_sutcliffe_efficiency,
    seed=42
)

print(f"\nMonte Carlo结果:")
print(f"  样本数: {mc_results['n_samples']}")
print(f"  NSE均值: {mc_results['metric_mean']:.4f}")
print(f"  NSE标准差: {mc_results['metric_std']:.4f}")
print(f"  NSE范围: [{mc_results['metric_percentiles'][0]:.4f}, "
      f"{mc_results['metric_percentiles'][4]:.4f}]")

# GLUE分析
print("\n运行GLUE不确定性估计...")

# 将param_bounds转换为GLUE所需的分布格式
glue_param_distributions = {
    name: {'dist': 'uniform', 'min': bounds[0], 'max': bounds[1]}
    for name, bounds in param_bounds.items()
}

glue_results = glue_analysis(
    model_function=uncertainty_model,
    param_distributions=glue_param_distributions,  # 使用均匀分布
    observed_data=observed_runoff,
    likelihood_function=nash_sutcliffe_efficiency,
    likelihood_threshold=0.7,  # NSE > 0.7 为"行为参数"
    n_samples=500,
    sampling_method='lhs',
    confidence_levels=[0.05, 0.95],
    seed=42
)

if 'error' not in glue_results:
    print(f"\nGLUE结果:")
    print(f"  总样本数: {glue_results['n_total']}")
    print(f"  行为样本: {glue_results['n_behavioral']}")
    print(f"  接受率: {glue_results['acceptance_rate']*100:.1f}%")

    print(f"\n参数后验统计:")
    for param_name in ['FC', 'K0', 'BETA']:
        post = glue_results['posterior_param_stats'][param_name]
        true_val = true_params[param_name]
        print(f"  {param_name:5s}: {post['mean']:6.2f} ± {post['std']:5.2f} "
              f"(真值: {true_val:6.2f})")
else:
    print(f"⚠ GLUE分析: {glue_results['error']}")

# ============================================================================
# 步骤 5: 结果可视化
# ============================================================================
print("\n步骤 5: 生成分析图表")
print("-" * 80)

output_dir = Path("results/complete_analysis_workflow")
output_dir.mkdir(parents=True, exist_ok=True)

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt

    print("生成图表...")

    # 1. 水文过程对比图
    print("  1. 水文过程对比图...")
    fig = plot_hydrograph_comparison(
        observed=observed_runoff,
        simulated=calibrated_simulated,
        time_index=np.arange(len(observed_runoff)),
        metrics={'NSE': calibrated_metrics['nse'], 'PBIAS': calibrated_metrics['pbias']},
        title="率定后水文过程对比",
        save_path=output_dir / "hydrograph_comparison.png",
        show=False
    )

    # 2. 散点图
    print("  2. 观测vs模拟散点图...")
    fig = plot_scatter(
        observed=observed_runoff,
        simulated=calibrated_simulated,
        metrics={'NSE': calibrated_metrics['nse'], 'R²': calibrated_metrics['nse']},
        save_path=output_dir / "scatter_plot.png",
        show=False
    )

    # 3. 收敛历史
    print("  3. 率定收敛历史...")
    fig = plot_convergence_history(
        convergence_history=calibration_result.convergence_history,
        title="SCE-UA收敛历史",
        ylabel="NSE",
        save_path=output_dir / "convergence_history.png",
        show=False
    )

    # 4. 敏感性分析
    print("  4. 参数敏感性Tornado图...")
    fig = plot_tornado_sensitivity(
        sensitivity_indices=sensitivity_results['sensitivity_indices'],
        title="参数敏感性分析 (OAT)",
        save_path=output_dir / "sensitivity_tornado.png",
        show=False
    )

    # 5. 参数后验分布
    if 'error' not in glue_results:
        print("  5. 参数后验分布...")
        fig = plot_parameter_distributions(
            param_samples=glue_results['behavioral_params'],
            true_values=true_params,
            save_path=output_dir / "parameter_distributions.png",
            show=False
        )

    # 6. 不确定性包络
    if 'error' not in glue_results:
        print("  6. 不确定性包络...")
        fig = plot_uncertainty_envelope(
            time_index=np.arange(len(observed_runoff)),
            observed=observed_runoff,
            mean_simulation=calibrated_simulated,
            percentiles=glue_results['uncertainty_bounds'],
            title="预测不确定性包络 (GLUE)",
            save_path=output_dir / "uncertainty_envelope.png",
            show=False
        )

    print(f"\n✓ 所有图表已保存到: {output_dir}")

    # 生成完整的分析报告
    print("\n生成完整分析报告...")
    if 'error' not in glue_results:
        report_figures = create_analysis_report_figures(
            observed=observed_runoff,
            simulated=calibrated_simulated,
            metrics=calibrated_metrics,
            param_samples=glue_results['behavioral_params'],
            convergence_history=calibration_result.convergence_history,
            uncertainty_percentiles=glue_results['uncertainty_bounds'],
            output_dir=output_dir / "full_report",
            time_index=np.arange(len(observed_runoff))
        )
        print(f"✓ 分析报告生成完成: {len(report_figures)} 个图表")

except ImportError:
    print("⚠ matplotlib未安装，跳过可视化")
    print("  安装方法: pip install matplotlib")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("工作流完成！")
print("=" * 80)

print("\n分析总结:")
print(f"\n1. 模型性能:")
print(f"   - 基准NSE (真实参数): {baseline_metrics['nse']:.4f}")
print(f"   - 率定NSE: {calibrated_metrics['nse']:.4f}")
print(f"   - RMSE: {calibrated_metrics['rmse']:.4f} m³/s")

print(f"\n2. 参数敏感性 (从高到低):")
for i, (param, si) in enumerate(sensitivity_results['ranked_parameters'][:3], 1):
    print(f"   {i}. {param}: {si:.4f}")

print(f"\n3. 参数率定:")
print(f"   - 优化算法: SCE-UA")
print(f"   - 函数评估: {calibration_result.n_evaluations}次")
print(f"   - 计算时间: {calibration_result.computation_time:.2f}秒")
print(f"   - 最优NSE: {calibration_result.best_score:.4f}")

print(f"\n4. 不确定性分析:")
print(f"   - MC样本数: {mc_results['n_samples']}")
print(f"   - NSE不确定性: {mc_results['metric_mean']:.4f} ± {mc_results['metric_std']:.4f}")
if 'error' not in glue_results:
    print(f"   - GLUE行为样本: {glue_results['n_behavioral']}/{glue_results['n_total']}")
    print(f"   - 接受率: {glue_results['acceptance_rate']*100:.1f}%")

print(f"\n5. 输出文件:")
print(f"   - 率定结果: {output_dir}/")
print(f"   - 分析图表: {output_dir}/*.png")

print("\n" + "=" * 80)
print("演示完成！这个工作流展示了完整的参数分析流程：")
print("  ✓ 性能评估 → ✓ 敏感性分析 → ✓ 参数率定 → ✓ 不确定性分析 → ✓ 可视化")
print("=" * 80)
