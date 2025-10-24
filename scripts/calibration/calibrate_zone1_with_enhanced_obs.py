#!/usr/bin/env python3
"""
Zone 1 HBV参数自动率定（使用增强型观测数据 + 基础库）

本脚本展示如何正确使用HydroSIS基础库进行参数率定。
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
from hydrosis.calibration import calibrate_parameters, CalibrationResult
from hydrosis.evaluation.metrics import (
    nash_sutcliffe_efficiency,
    log_nash_sutcliffe_efficiency,
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
print("Zone 1 HBV参数自动率定（使用增强型观测 + 基础库）")
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
    print(f"请先运行: python test_enhanced_generator.py")
    sys.exit(1)

obs_df = pd.read_csv(obs_file)
observed_runoff = obs_df['discharge_m3s'].values
times = pd.to_datetime(obs_df['datetime'])

print(f"✓ 加载观测径流: {obs_file}")
print(f"  时间范围: {times.min()} 到 {times.max()}")
print(f"  数据点数: {len(observed_runoff)}")
print(f"  流量范围: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")

# 1.2 加载降雨数据
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
precip_df = pd.read_csv(precip_file, index_col=0)

# Zone 1的子分区ID (从3.4_subzone_statistics.csv获取)
zone1_subbasins = [str(i) for i in range(10, 24)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
precipitation = precip_df[zone1_cols].mean(axis=1).values

print(f"\n✓ 加载降雨数据: {precip_file}")
print(f"  Zone 1子分区: {len(zone1_cols)}个 ({zone1_subbasins[0]}-{zone1_subbasins[-1]})")
print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")
print(f"  平均降雨: {precipitation.mean():.2f} mm/h")

# 1.3 生成简化温度数据
temperature = np.linspace(5, 15, len(precipitation))

# 1.4 Zone 1流域面积
zone1_area_km2 = 139.995

print(f"\n✓ Zone 1流域面积: {zone1_area_km2:.2f} km²")

# ============================================================================
# 步骤 2: 定义HBV模型和目标函数
# ============================================================================
print("\n步骤 2: 定义HBV模型和目标函数")
print("-" * 80)

# HBV参数搜索范围
param_bounds = [
    (250, 600),     # FC: 土壤最大容量
    (1.5, 3.5),     # BETA: 土壤蓄水曲线指数
    (0.1, 0.5),     # K0: 快速径流退水系数
    (0.02, 0.15),   # K1: 中速径流退水系数
    (0.005, 0.05),  # K2: 基流退水系数
    (0.5, 4.0),     # PERC: 渗透速率
    (0.3, 0.9),     # initial_soil_ratio: 初始土壤湿度比例
    (5, 50),        # initial_upper: 初始上层储量
]

param_names = ['FC', 'BETA', 'K0', 'K1', 'K2', 'PERC', 'initial_soil_ratio', 'initial_upper']

# 固定参数
fixed_params = {
    'LP': 0.7,
    'MAXBAS': 3.0,
    'TT': 0.0,
    'CFMAX': 3.5,
    'CFR': 0.05,
    'CWH': 0.1,
}

print(f"待率定参数: {len(param_bounds)}个")
for name, (min_val, max_val) in zip(param_names, param_bounds):
    print(f"  {name:<20s}: [{min_val:>8.3f}, {max_val:>8.3f}]")

# 定义HBV模型函数
class MockSubbasin:
    def __init__(self, area_km2):
        self.area_km2 = area_km2

subbasin = MockSubbasin(zone1_area_km2)

def run_hbv_model(params_list):
    """
    运行HBV模型并返回径流时间序列

    Parameters
    ----------
    params_list : list of float
        参数值列表 [FC, BETA, K0, K1, K2, PERC, initial_soil_ratio, initial_upper]

    Returns
    -------
    np.ndarray
        径流时间序列 (m³/s)
    """
    FC, BETA, K0, K1, K2, PERC, initial_soil_ratio, initial_upper = params_list

    params = {
        'FC': FC,
        'BETA': BETA,
        'K0': K0,
        'K1': K1,
        'K2': K2,
        'PERC': PERC,
        **fixed_params,
        'initial_soil': FC * initial_soil_ratio,
        'initial_upper': initial_upper,
        'initial_lower': 30.0,
        'initial_snow': 0.0
    }

    hbv = HBVRunoff(params)
    runoff_m3s = hbv.simulate(subbasin, precipitation.tolist())

    return np.array(runoff_m3s)

# 定义目标函数（最大化NSE）
def objective_function(params):
    """
    目标函数：计算NSE

    Parameters
    ----------
    params : list of float
        参数值列表

    Returns
    -------
    float
        NSE值（用于最大化）
    """
    try:
        simulated = run_hbv_model(params)
        # ✅ 使用基础库的metrics模块
        nse = nash_sutcliffe_efficiency(simulated, observed_runoff)
        return nse
    except Exception as e:
        return -999.0  # 返回极差值表示失败

# 测试HBV模型
print("\n测试HBV模型...")
default_params = [400, 2.0, 0.25, 0.08, 0.02, 2.0, 0.6, 20]
test_runoff = run_hbv_model(default_params)
test_nse = nash_sutcliffe_efficiency(test_runoff, observed_runoff)
print(f"  模型输出长度: {len(test_runoff)}")
print(f"  初始NSE (默认参数): {test_nse:.4f}")

# ============================================================================
# 步骤 3: 参数率定
# ============================================================================
print("\n步骤 3: 参数率定")
print("-" * 80)

print("\n运行Differential Evolution率定...")
print("  这可能需要几分钟时间...")

# ✅ 使用基础库的calibration模块
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

print(f"\n  ✓ 完成!")
print(f"    最优NSE: {result.best_score:.6f}")
print(f"    函数评估: {result.n_evaluations}")
print(f"    计算时间: {result.computation_time:.2f}秒")

# ============================================================================
# 步骤 4: 使用最优参数运行HBV模型
# ============================================================================
print("\n步骤 4: 使用最优参数运行HBV模型")
print("-" * 80)

# 运行模型
final_simulated = run_hbv_model(result.best_params)

# ✅ 使用基础库计算所有性能指标
final_metrics = {
    'nse': nash_sutcliffe_efficiency(final_simulated, observed_runoff),
    'rmse': rmse(final_simulated, observed_runoff),
    'mae': mae(final_simulated, observed_runoff),
    'pbias': percent_bias(final_simulated, observed_runoff),
    'kge': kling_gupta_efficiency(final_simulated, observed_runoff),
    'log_nse': log_nash_sutcliffe_efficiency(final_simulated, observed_runoff, epsilon=1e-6),
}

print(f"\n率定后性能指标:")
for metric, value in final_metrics.items():
    print(f"  {metric.upper():<10s}: {value:>8.4f}")

# 打印最优参数
print(f"\n最优参数:")
for name, value, (min_val, max_val) in zip(param_names, result.best_params, param_bounds):
    range_pct = (value - min_val) / (max_val - min_val) * 100
    print(f"  {name:<20s}: {value:>10.4f}  (范围的{range_pct:>5.1f}%)")

# ============================================================================
# 步骤 5: 保存结果
# ============================================================================
print("\n步骤 5: 保存结果")
print("-" * 80)

output_dir = results_dir / "calibration_enhanced"
output_dir.mkdir(parents=True, exist_ok=True)

# 保存参数配置
calibration_yaml = {
    'description': f'Zone 1 HBV参数率定结果（增强型观测）',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'calibration_info': {
        'algorithm': result.algorithm,
        'observation_type': 'EnhancedRunoffGenerator',
        'n_evaluations': result.n_evaluations,
        'computation_time': result.computation_time,
    },
    'metrics': {k: float(v) for k, v in final_metrics.items()},
    'zones': {
        1: {
            'runoff_model': 'HBV',
            'parameters': {
                name: float(value)
                for name, value in zip(param_names[:6], result.best_params[:6])
            },
            'initial_conditions': {
                'initial_soil': float(result.best_params[0] * result.best_params[6]),
                'initial_upper': float(result.best_params[7]),
                'initial_lower': 30.0,
                'initial_snow': 0.0,
            },
            'fixed_parameters': fixed_params,
        }
    }
}

yaml_file = output_dir / "zone1_calibrated_parameters.yaml"
with open(yaml_file, 'w') as f:
    yaml.dump(calibration_yaml, f, default_flow_style=False, sort_keys=False)
print(f"✓ 保存参数配置: {yaml_file}")

# 保存径流对比数据
comparison_df = pd.DataFrame({
    'datetime': times,
    'observed_m3s': observed_runoff,
    'simulated_m3s': final_simulated,
    'residual_m3s': observed_runoff - final_simulated
})
comparison_file = output_dir / "zone1_calibrated_runoff.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"✓ 保存径流对比数据: {comparison_file}")

# ============================================================================
# 步骤 6: 生成可视化图表
# ============================================================================
print("\n步骤 6: 生成可视化图表")
print("-" * 80)

# ✅ 使用基础库的可视化功能
# 1. 水文过程对比图
plot_hydrograph(
    output_path=output_dir / "zone1_hydrograph.png",
    simulations={"HBV": final_simulated.tolist()},
    observed=observed_runoff.tolist(),
    title=f"Zone 1 HBV Calibration (NSE={final_metrics['nse']:.4f})",
    xlabel="Time Step (hour)",
    ylabel="Discharge (m³/s)"
)
print("  ✓ 保存: zone1_hydrograph.png")

# 2. 散点图
plot_scatter(
    output_path=output_dir / "zone1_scatter.png",
    observed=observed_runoff.tolist(),
    simulated=final_simulated.tolist(),
    title=f"Observed vs Simulated (R²={final_metrics['nse']:.4f})",
    xlabel="Observed (m³/s)",
    ylabel="Simulated (m³/s)",
    equal_axis=True
)
print("  ✓ 保存: zone1_scatter.png")

# 3. 收敛历史
if result.convergence_history:
    plot_convergence(
        output_path=output_dir / "zone1_convergence.png",
        convergence_history=result.convergence_history,
        title="Calibration Convergence History",
        ylabel="NSE",
        maximize=True
    )
    print("  ✓ 保存: zone1_convergence.png")

print(f"\n✓ 所有图表已保存到: {output_dir}")

# ============================================================================
# 步骤 7: 总结
# ============================================================================
print("\n" + "=" * 80)
print("Zone 1 HBV参数率定完成！")
print("=" * 80)

print(f"\n【率定数据】")
print(f"  观测类型: 增强型径流生成器")
print(f"  时间步数: {len(observed_runoff)}")
print(f"  平均流量: {observed_runoff.mean():.2f} m³/s")
print(f"  峰值流量: {observed_runoff.max():.2f} m³/s")

print(f"\n【率定结果】")
print(f"  算法: {result.algorithm}")
print(f"  最优NSE: {result.best_score:.6f}")
print(f"  计算时间: {result.computation_time:.2f}秒")
print(f"  函数评估: {result.n_evaluations}次")

print(f"\n【性能指标】")
for metric, value in final_metrics.items():
    print(f"  {metric.upper()}: {value:.6f}")

print(f"\n【输出文件】")
print(f"  参数配置: {yaml_file.name}")
print(f"  径流数据: {comparison_file.name}")
print(f"  可视化图表: zone1_*.png")

print("\n" + "=" * 80)
print("✓ 本脚本展示了如何正确使用HydroSIS基础库")
print("✓ 遵循 .claude/AI_DEVELOPMENT_GUIDE.md 中的最佳实践")
print("=" * 80)
