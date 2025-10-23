#!/usr/bin/env python3
"""
Zone 1参数率定 - 使用简单径流系数模型

验证率定流程的正确性：
用EstimatedRunoffGenerator生成"观测"数据，
然后用相同的模型结构反推参数，应该能达到NSE接近1.0

这个测试验证了：
1. 率定算法是否正确
2. 单位转换是否正确
3. 数据对齐是否正确
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.estimated_runoff import EstimatedRunoffGenerator
from hydrosis.analysis import (
    calculate_metrics,
    calibrate_model,
    plot_hydrograph_comparison,
    plot_convergence_history,
    plot_scatter,
)

print("=" * 80)
print("Zone 1 简单模型参数率定（验证率定流程）")
print("=" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")

# ============================================================================
# 1. 加载数据
# ============================================================================
print("\n步骤 1: 加载Zone 1的数据")
print("-" * 80)

obs_file = results_dir / "estimated_observations" / "zone_1_estimated_runoff.csv"
obs_df = pd.read_csv(obs_file)
observed_runoff = obs_df['discharge_m3s'].values
times = pd.to_datetime(obs_df['datetime'])

print(f"✓ 加载观测径流 ({len(observed_runoff)}个时间步)")
print(f"  流量范围: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")

# 加载降雨
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
precip_df = pd.read_csv(precip_file, index_col=0)
zone1_subbasins = [str(i) for i in range(101, 115)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
precipitation = precip_df[zone1_cols].mean(axis=1).values

print(f"✓ 加载降雨数据 ({len(precipitation)}个时间步)")
print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")

zone1_area_km2 = 139.995
print(f"✓ Zone 1流域面积: {zone1_area_km2:.2f} km²")

# ============================================================================
# 2. 定义率定模型
# ============================================================================
print("\n步骤 2: 定义径流系数模型和目标函数")
print("-" * 80)

# 待率定参数的搜索范围
param_bounds = {
    'runoff_coefficient': [0.1, 0.8],    # 径流系数
    'lag_hours': [0.0, 5.0],             # 滞后时间
    'attenuation_factor': [0.5, 1.0],    # 衰减系数
    'baseflow_ratio': [0.0, 0.3],        # 基流比例
}

print(f"待率定参数: {len(param_bounds)}个")
for name, bounds in param_bounds.items():
    print(f"  {name:20s}: [{bounds[0]:7.3f}, {bounds[1]:7.3f}]")

def run_simple_model(runoff_coefficient, lag_hours, attenuation_factor, baseflow_ratio):
    """运行简单径流系数模型

    Returns:
        径流序列 (m³/s)
    """
    try:
        generator = EstimatedRunoffGenerator(
            runoff_coefficient=runoff_coefficient,
            lag_hours=lag_hours,
            attenuation_factor=attenuation_factor,
            baseflow_ratio=baseflow_ratio,
            time_step_hours=1.0
        )

        runoff_m3s, stats = generator.generate(
            precipitation_series=precipitation,
            area_km2=zone1_area_km2,
            apply_lag=True,
            apply_attenuation=True,
            apply_baseflow=True
        )

        return runoff_m3s

    except Exception as e:
        print(f"模型运行错误: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(len(precipitation))

def calibration_objective(runoff_coefficient, lag_hours, attenuation_factor, baseflow_ratio):
    """率定目标函数：最大化NSE"""
    from hydrosis.analysis import nash_sutcliffe_efficiency

    simulated = run_simple_model(runoff_coefficient, lag_hours, attenuation_factor, baseflow_ratio)
    nse = nash_sutcliffe_efficiency(observed_runoff, simulated)

    return nse

# 测试模型运行
print("\n测试模型运行...")
test_params = {
    'runoff_coefficient': 0.35,
    'lag_hours': 1.5,
    'attenuation_factor': 0.85,
    'baseflow_ratio': 0.08
}
test_runoff = run_simple_model(**test_params)
print(f"  模型输出长度: {len(test_runoff)}")
print(f"  输出范围: {test_runoff.min():.2f} - {test_runoff.max():.2f} m³/s")

initial_nse = calibration_objective(**test_params)
print(f"  初始NSE (真实参数): {initial_nse:.6f}")
print(f"\n  说明: 这些是生成观测数据时使用的真实参数，")
print(f"        如果一切正确，率定应该能恢复这些参数，NSE接近1.0！")

# ============================================================================
# 3. 多算法参数率定
# ============================================================================
print("\n步骤 3: 多算法参数率定")
print("-" * 80)

calibration_configs = {
    'SCE-UA': {
        'method': 'sce_ua',
        'n_complexes': 5,
        'max_iterations': 50,
        'patience': 15,
    },
    'PSO': {
        'method': 'pso',
        'n_particles': 30,
        'max_iterations': 50,
        'w': 0.7,
        'c1': 1.5,
        'c2': 1.5,
        'patience': 15,
    },
}

calibration_results = {}

for alg_name, config in calibration_configs.items():
    print(f"\n运行{alg_name}率定...")

    result = calibrate_model(
        objective_function=calibration_objective,
        param_bounds=param_bounds,
        maximize=True,
        seed=42,
        verbose=False,
        **config
    )

    calibration_results[alg_name] = result

    print(f"  ✓ 完成!")
    print(f"    最优NSE: {result.best_score:.8f}")
    print(f"    迭代次数: {result.n_iterations}")
    print(f"    函数评估: {result.n_evaluations}")
    print(f"    计算时间: {result.computation_time:.2f}秒")

# ============================================================================
# 4. 结果对比
# ============================================================================
print("\n步骤 4: 率定结果对比")
print("-" * 80)

print(f"\n{'算法':<15s} {'NSE':>12s} {'时间(s)':>10s} {'评估次数':>10s}")
print("-" * 50)
for name, result in calibration_results.items():
    print(f"{name:<15s} {result.best_score:>12.8f} {result.computation_time:>10.2f} "
          f"{result.n_evaluations:>10d}")

best_alg = max(calibration_results.items(), key=lambda x: x[1].best_score)
best_name = best_alg[0]
best_result = best_alg[1]

print(f"\n✓ 最佳算法: {best_name} (NSE={best_result.best_score:.8f})")

# 对比真实参数和率定参数
print("\n参数对比:")
print(f"{'参数':<25s} {'真实值':>12s} {'率定值':>12s} {'差异':>12s}")
print("-" * 65)
true_params = test_params
for param, value in best_result.best_params.items():
    true_val = true_params[param]
    diff = abs(value - true_val)
    diff_pct = (diff / true_val * 100) if true_val != 0 else 0
    print(f"{param:<25s} {true_val:>12.4f} {value:>12.4f} {diff_pct:>11.2f}%")

# ============================================================================
# 5. 验证结果
# ============================================================================
print("\n步骤 5: 使用最优参数运行模型")
print("-" * 80)

calibrated_runoff = run_simple_model(**best_result.best_params)

final_metrics = calculate_metrics(
    observed_runoff,
    calibrated_runoff,
    metrics=['nse', 'rmse', 'mae', 'pbias', 'kge', 'log_nse']
)

print("\n率定后性能指标:")
for metric, value in final_metrics.items():
    if 'peak' not in metric and 'time' not in metric:
        print(f"  {metric.upper():10s}: {value:8.6f}")

# 判断率定是否成功
if final_metrics['nse'] >= 0.99:
    print(f"\n✅ 率定成功! NSE={final_metrics['nse']:.6f} >= 0.99")
    print(f"   这证明了率定流程、单位转换和数据对齐都是正确的！")
elif final_metrics['nse'] >= 0.95:
    print(f"\n✓ 率定良好! NSE={final_metrics['nse']:.6f} >= 0.95")
    print(f"   率定流程基本正确，可能有微小的数值误差")
else:
    print(f"\n⚠️  率定NSE={final_metrics['nse']:.6f} < 0.95，可能存在问题：")
    print(f"   1. 率定算法参数可能需要调整")
    print(f"   2. 参数搜索范围可能需要调整")
    print(f"   3. 存在数值精度问题")

# ============================================================================
# 6. 保存结果
# ============================================================================
print("\n步骤 6: 保存验证结果")
print("-" * 80)

output_dir = results_dir / "calibration"
output_dir.mkdir(parents=True, exist_ok=True)

# 保存对比数据
result_df = pd.DataFrame({
    'datetime': times,
    'time_step': np.arange(len(observed_runoff)),
    'observed_m3s': observed_runoff,
    'simulated_m3s': calibrated_runoff,
    'residual_m3s': observed_runoff - calibrated_runoff,
    'relative_error_pct': (calibrated_runoff - observed_runoff) / observed_runoff * 100
})
result_df.to_csv(output_dir / "zone1_simple_model_validation.csv", index=False)
print(f"✓ 保存验证数据: zone1_simple_model_validation.csv")

# 保存YAML配置（如果NSE > 0.95）
if final_metrics['nse'] >= 0.95:
    validation_yaml = {
        'description': f'Zone 1 简单模型率定验证 ({best_name})',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'validation_result': 'PASSED' if final_metrics['nse'] >= 0.99 else 'GOOD',
        'nse': float(final_metrics['nse']),
        'true_parameters': {k: float(v) for k, v in test_params.items()},
        'calibrated_parameters': {k: float(v) for k, v in best_result.best_params.items()},
        'parameter_recovery_errors': {
            k: float(abs(best_result.best_params[k] - test_params[k]) / test_params[k] * 100)
            if test_params[k] != 0 else 0
            for k in test_params.keys()
        }
    }

    yaml_file = output_dir / "zone1_simple_model_validation.yaml"
    with open(yaml_file, 'w') as f:
        yaml.dump(validation_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"✓ 保存验证配置: zone1_simple_model_validation.yaml")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("验证总结")
print("=" * 80)

print(f"\n1. 率定性能:")
print(f"   - NSE: {final_metrics['nse']:.8f}")
print(f"   - RMSE: {final_metrics['rmse']:.4f} m³/s")
print(f"   - PBIAS: {final_metrics['pbias']:.2f}%")

print(f"\n2. 参数恢复精度:")
max_error = max(
    abs(best_result.best_params[k] - test_params[k]) / test_params[k] * 100
    if test_params[k] != 0 else 0
    for k in test_params.keys()
)
print(f"   - 最大参数误差: {max_error:.2f}%")

if final_metrics['nse'] >= 0.99 and max_error < 5:
    print(f"\n✅ 结论: 率定流程完全正确!")
    print(f"   - NSE >= 0.99: 模型拟合优秀")
    print(f"   - 参数误差 < 5%: 参数恢复准确")
    print(f"   - 这证明了率定算法、单位转换、数据对齐都是正确的")
elif final_metrics['nse'] >= 0.95:
    print(f"\n✓ 结论: 率定流程基本正确")
    print(f"   - NSE >= 0.95: 模型拟合良好")
    print(f"   - 可能存在微小的数值精度误差")
else:
    print(f"\n⚠️  结论: 需要进一步调查")
    print(f"   - NSE < 0.95: 拟合精度不够")

print("\n" + "=" * 80)
