#!/usr/bin/env python3
"""
Zone 1 HBV参数率定 + 预热期 (Warm-up Period)

根据敏感性分析的诊断结果，实施解决方案：
1. 添加48小时预热期
2. 预热期不参与率定评估
3. 验证物理参数敏感性恢复
4. 对比有无预热期的率定结果

遵循 .claude/AI_DEVELOPMENT_GUIDE.md 中的最佳实践。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import yaml

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ✅ 使用基础库的功能模块
from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.calibration import (
    calibrate_parameters,
    CalibrationResult,
    morris_sensitivity,
    print_sensitivity_report,
)
from hydrosis.evaluation.metrics import (
    nash_sutcliffe_efficiency,
    kling_gupta_efficiency,
    rmse,
    mae,
    percent_bias,
)
from hydrosis.reporting.charts import (
    plot_hydrograph,
    plot_scatter,
    plot_convergence,
)

print("=" * 80)
print("Zone 1 HBV参数率定 + 预热期修正")
print("=" * 80)

# ============================================================================
# 步骤 1: 加载数据
# ============================================================================
print("\n步骤 1: 加载Zone 1的数据")
print("-" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")

# 1.1 加载增强型观测径流
obs_file = results_dir / "enhanced_observations" / "zone_1_enhanced_runoff.csv"
if not obs_file.exists():
    print(f"错误: 观测数据不存在: {obs_file}")
    sys.exit(1)

obs_df = pd.read_csv(obs_file)
observed_runoff_full = obs_df['discharge_m3s'].values
times = pd.to_datetime(obs_df['datetime'])

# 1.2 加载降雨数据
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
precip_df = pd.read_csv(precip_file, index_col=0)

zone1_subbasins = [str(i) for i in range(10, 24)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
precipitation_full = precip_df[zone1_cols].mean(axis=1).values

# 1.3 配置预热期
WARMUP_HOURS = 48  # 2天预热期

print(f"✓ 加载完整数据:")
print(f"  总时长: {len(observed_runoff_full)} 小时")
print(f"  预热期: {WARMUP_HOURS} 小时")
print(f"  率定期: {len(observed_runoff_full) - WARMUP_HOURS} 小时")

if len(observed_runoff_full) <= WARMUP_HOURS:
    print(f"\n警告: 数据长度不足! 当前数据只有{len(observed_runoff_full)}小时")
    print(f"建议: 使用至少 {WARMUP_HOURS + 72} 小时的数据（预热期 + 率定期）")
    print(f"\n继续使用前 {WARMUP_HOURS//2} 小时作为预热期...")
    WARMUP_HOURS = len(observed_runoff_full) // 2

# 分割数据
observed_runoff = observed_runoff_full[WARMUP_HOURS:]
precipitation = precipitation_full

print(f"\n实际配置:")
print(f"  预热期: {WARMUP_HOURS} 小时")
print(f"  率定期流量范围: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")

# 1.4 生成简化温度数据
temperature = np.linspace(5, 15, len(precipitation))

# 1.5 Zone 1流域面积
zone1_area_km2 = 139.995

# ============================================================================
# 步骤 2: 定义HBV模型（带预热期）
# ============================================================================
print("\n步骤 2: 定义HBV模型（带预热期）")
print("-" * 80)

# HBV参数搜索范围
param_bounds = [
    (250, 600),     # FC: 土壤最大容量
    (1.5, 3.5),     # BETA: 土壤蓄水曲线指数
    (0.1, 0.5),     # K0: 快速径流退水系数
    (0.02, 0.15),   # K1: 中速径流退水系数
    (0.005, 0.05),  # K2: 基流退水系数
    (0.5, 4.0),     # PERC: 渗透速率
]

param_names = ['FC', 'BETA', 'K0', 'K1', 'K2', 'PERC']

# 固定参数
fixed_params = {
    'LP': 0.7,
    'MAXBAS': 3.0,
    'TT': 0.0,
    'CFMAX': 3.5,
    'CFR': 0.05,
    'CWH': 0.1,
}

# 固定初始状态（根据基流估计）
initial_baseflow = observed_runoff_full[0]  # ~85 m³/s
initial_conditions = {
    'initial_soil': 300.0,      # 中等湿度
    'initial_upper': 20.0,      # 较低初值
    'initial_lower': 3000.0,    # 根据基流反推: Q_base ≈ K2 * S_lower
    'initial_snow': 0.0,
}

print(f"率定参数: {len(param_bounds)}个")
for name, (min_val, max_val) in zip(param_names, param_bounds):
    print(f"  {name:<20s}: [{min_val:>8.3f}, {max_val:>8.3f}]")

print(f"\n固定初始状态:")
for key, value in initial_conditions.items():
    print(f"  {key:<20s}: {value:>10.2f}")

# 定义HBV模型函数
class MockSubbasin:
    def __init__(self, area_km2):
        self.area_km2 = area_km2

subbasin = MockSubbasin(zone1_area_km2)

def run_hbv_model_with_warmup(params_list):
    """
    运行HBV模型（带预热期）

    Parameters
    ----------
    params_list : list of float
        参数值列表 [FC, BETA, K0, K1, K2, PERC]

    Returns
    -------
    np.ndarray
        径流时间序列（仅率定期，不含预热期）
    """
    FC, BETA, K0, K1, K2, PERC = params_list

    params = {
        'FC': FC,
        'BETA': BETA,
        'K0': K0,
        'K1': K1,
        'K2': K2,
        'PERC': PERC,
        **fixed_params,
        **initial_conditions,
    }

    hbv = HBVRunoff(params)

    # 运行完整模拟（包含预热期）
    runoff_m3s_full = hbv.simulate(subbasin, precipitation.tolist())

    # 只返回率定期的结果
    runoff_m3s = runoff_m3s_full[WARMUP_HOURS:]

    return np.array(runoff_m3s)

# 定义目标函数
def objective_function(params):
    """目标函数：计算NSE（仅评估率定期）"""
    try:
        simulated = run_hbv_model_with_warmup(params)
        nse = nash_sutcliffe_efficiency(simulated, observed_runoff)
        return nse
    except Exception as e:
        return -999.0

# 测试HBV模型
print(f"\n测试HBV模型（带预热期）...")
default_params = [400, 2.0, 0.25, 0.08, 0.02, 2.0]
test_runoff = run_hbv_model_with_warmup(default_params)
test_nse = nash_sutcliffe_efficiency(test_runoff, observed_runoff)
print(f"  率定期输出长度: {len(test_runoff)}")
print(f"  初始NSE (默认参数): {test_nse:.4f}")

# ============================================================================
# 步骤 3: 敏感性分析（验证物理参数敏感性）
# ============================================================================
print("\n步骤 3: 敏感性分析（验证物理参数是否恢复敏感性）")
print("-" * 80)

print("\n运行Morris敏感性分析...")

sens_result = morris_sensitivity(
    model_function=objective_function,
    param_names=param_names,
    param_bounds=param_bounds,
    n_trajectories=15,
    n_levels=4,
)

print(f"\n✓ 敏感性分析完成!")
print(f"  样本数: {sens_result.n_samples}")
print(f"  计算时间: {sens_result.computation_time:.2f}秒")

print_sensitivity_report(sens_result)

# 统计敏感性分布
high_sens = [p for p in param_names if sens_result.sensitivity_indices[p] > 0.7]
med_sens = [p for p in param_names if 0.3 < sens_result.sensitivity_indices[p] <= 0.7]
low_sens = [p for p in param_names if sens_result.sensitivity_indices[p] <= 0.3]

print(f"\n敏感性分布:")
print(f"  高敏感 (>0.7): {len(high_sens)}个 - {high_sens}")
print(f"  中敏感 (0.3-0.7): {len(med_sens)}个 - {med_sens}")
print(f"  低敏感 (<0.3): {len(low_sens)}个 - {low_sens}")

# ============================================================================
# 步骤 4: 参数率定（带预热期）
# ============================================================================
print("\n步骤 4: 参数率定（带预热期）")
print("-" * 80)

print("\n运行Differential Evolution率定...")

result = calibrate_parameters(
    objective_function=objective_function,
    param_bounds=param_bounds,
    algorithm="differential_evolution",
    maximize=True,
    maxiter=150,
    popsize=20,
    seed=42,
    polish=True
)

print(f"\n✓ 完成!")
print(f"  最优NSE: {result.best_score:.6f}")
print(f"  函数评估: {result.n_evaluations}")
print(f"  计算时间: {result.computation_time:.2f}秒")

# ============================================================================
# 步骤 5: 性能评估
# ============================================================================
print("\n步骤 5: 性能评估")
print("-" * 80)

# 运行最优参数
final_simulated = run_hbv_model_with_warmup(result.best_params)

# 计算所有性能指标
final_metrics = {
    'nse': nash_sutcliffe_efficiency(final_simulated, observed_runoff),
    'rmse': rmse(final_simulated, observed_runoff),
    'mae': mae(final_simulated, observed_runoff),
    'pbias': percent_bias(final_simulated, observed_runoff),
    'kge': kling_gupta_efficiency(final_simulated, observed_runoff),
}

print(f"\n性能指标（率定期）:")
for metric, value in final_metrics.items():
    print(f"  {metric.upper():<10s}: {value:>8.4f}")

# 打印最优参数
print(f"\n最优参数（按敏感性排序）:")
for param_name in sens_result.sensitivity_rankings:
    idx = param_names.index(param_name)
    value = result.best_params[idx]
    sensitivity = sens_result.sensitivity_indices[param_name]
    min_val, max_val = param_bounds[idx]
    range_pct = (value - min_val) / (max_val - min_val) * 100

    print(f"  {param_name:<20s}: {value:>10.4f}  "
          f"(范围的{range_pct:>5.1f}%, 敏感性={sensitivity:.3f})")

# ============================================================================
# 步骤 6: 对比分析（有无预热期）
# ============================================================================
print("\n步骤 6: 对比分析（有预热期 vs 无预热期）")
print("-" * 80)

# 读取之前无预热期的结果
prev_result_file = results_dir / "calibration_with_sensitivity" / "zone1_calibrated_parameters_adaptive.yaml"
if prev_result_file.exists():
    with open(prev_result_file, 'r') as f:
        prev_results = yaml.safe_load(f)

    prev_nse = prev_results['metrics']['nse']

    print(f"\n结果对比:")
    print(f"{'方法':<20s} {'NSE':<12s} {'RMSE':<12s} {'KGE':<12s}")
    print("-" * 80)
    print(f"{'无预热期（之前）':<20s} {prev_nse:<12.6f} "
          f"{prev_results['metrics']['rmse']:<12.4f} {prev_results['metrics']['kge']:<12.6f}")
    print(f"{'有预热期（当前）':<20s} {final_metrics['nse']:<12.6f} "
          f"{final_metrics['rmse']:<12.4f} {final_metrics['kge']:<12.6f}")

    nse_improvement = final_metrics['nse'] - prev_nse
    print(f"\nNSE改善: {nse_improvement:+.6f}")

    if nse_improvement > 0:
        print("✓ 预热期策略有效！")
    else:
        print("✗ 预热期未改善结果，可能需要更长的数据或其他调整")

# ============================================================================
# 步骤 7: 保存结果
# ============================================================================
print("\n步骤 7: 保存结果")
print("-" * 80)

output_dir = results_dir / "calibration_with_warmup"
output_dir.mkdir(parents=True, exist_ok=True)

# 保存参数配置
calibration_yaml = {
    'description': f'Zone 1 HBV参数率定结果（使用预热期）',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'calibration_info': {
        'algorithm': result.algorithm,
        'warmup_hours': WARMUP_HOURS,
        'calibration_hours': len(observed_runoff),
        'n_evaluations': result.n_evaluations,
        'computation_time': result.computation_time,
    },
    'metrics': {k: float(v) for k, v in final_metrics.items()},
    'parameter_sensitivity': {
        name: float(sens_result.sensitivity_indices[name])
        for name in param_names
    },
    'zones': {
        1: {
            'runoff_model': 'HBV',
            'parameters': {
                name: float(value)
                for name, value in zip(param_names, result.best_params)
            },
            'initial_conditions': initial_conditions,
            'fixed_parameters': fixed_params,
        }
    }
}

yaml_file = output_dir / "zone1_calibrated_parameters_warmup.yaml"
with open(yaml_file, 'w') as f:
    yaml.dump(calibration_yaml, f, default_flow_style=False, sort_keys=False)
print(f"✓ 保存参数配置: {yaml_file}")

# 保存径流对比数据
comparison_df = pd.DataFrame({
    'datetime': times[WARMUP_HOURS:],
    'observed_m3s': observed_runoff,
    'simulated_m3s': final_simulated,
    'residual_m3s': observed_runoff - final_simulated
})
comparison_file = output_dir / "zone1_calibrated_runoff_warmup.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"✓ 保存径流对比数据: {comparison_file}")

# 保存敏感性分析
sensitivity_csv = output_dir / "parameter_sensitivity_warmup.csv"
sens_df = pd.DataFrame({
    'parameter': sens_result.param_names,
    'sensitivity_index': [sens_result.sensitivity_indices[p] for p in sens_result.param_names],
    'rank': range(1, len(sens_result.param_names) + 1),
})
sens_df.to_csv(sensitivity_csv, index=False)
print(f"✓ 保存敏感性分析: {sensitivity_csv}")

# ============================================================================
# 步骤 8: 生成可视化图表
# ============================================================================
print("\n步骤 8: 生成可视化图表")
print("-" * 80)

# 水文过程线
plot_hydrograph(
    output_path=output_dir / "zone1_hydrograph_warmup.png",
    simulations={"HBV (w/ Warmup)": final_simulated.tolist()},
    observed=observed_runoff.tolist(),
    title=f"Zone 1 HBV Calibration with Warmup Period (NSE={final_metrics['nse']:.4f})",
    xlabel="Time Step (hour, after warmup)",
    ylabel="Discharge (m³/s)"
)
print("  ✓ 保存: zone1_hydrograph_warmup.png")

# 散点图
plot_scatter(
    output_path=output_dir / "zone1_scatter_warmup.png",
    observed=observed_runoff.tolist(),
    simulated=final_simulated.tolist(),
    title=f"Observed vs Simulated (NSE={final_metrics['nse']:.4f})",
    xlabel="Observed (m³/s)",
    ylabel="Simulated (m³/s)",
    equal_axis=True
)
print("  ✓ 保存: zone1_scatter_warmup.png")

# 收敛历史
if result.convergence_history:
    plot_convergence(
        output_path=output_dir / "zone1_convergence_warmup.png",
        convergence_history=result.convergence_history,
        title="Calibration Convergence History (with Warmup)",
        ylabel="NSE",
        maximize=True
    )
    print("  ✓ 保存: zone1_convergence_warmup.png")

# ============================================================================
# 步骤 9: 总结
# ============================================================================
print("\n" + "=" * 80)
print("Zone 1 HBV参数率定完成（使用预热期）!")
print("=" * 80)

print(f"\n【配置】")
print(f"  预热期: {WARMUP_HOURS} 小时")
print(f"  率定期: {len(observed_runoff)} 小时")
print(f"  率定参数: {len(param_names)}个物理参数")
print(f"  固定初始状态: 根据基流估计")

print(f"\n【敏感性分析】")
print(f"  高敏感参数: {len(high_sens)}个")
print(f"  中敏感参数: {len(med_sens)}个")
print(f"  低敏感参数: {len(low_sens)}个")

if len(high_sens) + len(med_sens) > 1:
    print(f"  ✓ 物理参数敏感性已恢复！")
else:
    print(f"  ✗ 物理参数敏感性仍然较低，建议使用更长时间序列")

print(f"\n【率定结果】")
print(f"  最优NSE: {result.best_score:.6f}")
print(f"  计算时间: {result.computation_time:.2f}秒")

print(f"\n【性能指标】")
for metric, value in final_metrics.items():
    print(f"  {metric.upper()}: {value:.6f}")

print(f"\n【诊断建议】")
if final_metrics['nse'] > 0.5:
    print("  ✓ 率定成功！模型性能良好")
elif final_metrics['nse'] > 0:
    print("  ⚠ 率定部分成功，建议使用更长时间序列（7-30天）")
else:
    print("  ✗ 率定仍未成功，建议：")
    print("    1. 使用更长时间序列（当前数据太短）")
    print("    2. 调整初始状态估计方法")
    print("    3. 检查降雨和观测数据的时间对齐")

print("\n" + "=" * 80)
print("✓ 本脚本展示了如何使用预热期解决初始状态问题")
print("✓ 遵循敏感性分析诊断的建议")
print("=" * 80)
