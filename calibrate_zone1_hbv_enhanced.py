#!/usr/bin/env python3
"""
Zone 1 HBV参数自动率定脚本（使用增强型观测数据）

使用增强型径流生成器生成的观测数据进行HBV参数率定。

增强型生成器特点：
- 土壤水分核算（状态依赖的产流）
- 三分量径流（快速径流、中速径流、基流）
- 线性水库汇流
- 更接近HBV的物理过程

预期：由于增强型观测数据与HBV有更相似的结构，率定精度应该显著提高（NSE > 0.7）
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

from hydrosis.runoff.hbv import HBVRunoff
from scipy.optimize import differential_evolution, minimize
import time

# 简单的性能指标计算函数
def nash_sutcliffe_efficiency(observed, simulated):
    """计算NSE"""
    obs = np.array(observed)
    sim = np.array(simulated)
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)

def rmse(observed, simulated):
    """计算RMSE"""
    return np.sqrt(np.mean((np.array(observed) - np.array(simulated))**2))

def mean_absolute_error(observed, simulated):
    """计算MAE"""
    return np.mean(np.abs(np.array(observed) - np.array(simulated)))

def percent_bias(observed, simulated):
    """计算PBIAS"""
    return 100 * np.sum(np.array(simulated) - np.array(observed)) / np.sum(np.array(observed))

def kling_gupta_efficiency(observed, simulated):
    """计算KGE"""
    obs = np.array(observed)
    sim = np.array(simulated)
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

print("=" * 80)
print("Zone 1 HBV参数自动率定（使用增强型观测数据）")
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
print(f"  平均流量: {observed_runoff.mean():.2f} m³/s")

# 1.2 加载Zone 1的面雨量数据
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
if not precip_file.exists():
    print(f"错误: 降雨数据不存在: {precip_file}")
    sys.exit(1)

precip_df = pd.read_csv(precip_file, index_col=0)

# Zone 1的子分区ID: 10-23（从3.4_subzone_statistics.csv获取）
zone1_subbasins = [str(i) for i in range(10, 24)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]

if len(zone1_cols) == 0:
    print(f"错误: 未找到Zone 1的子分区数据")
    print(f"  可用列: {precip_df.columns[:10].tolist()}")
    sys.exit(1)

# 计算Zone 1的平均面雨量
precipitation = precip_df[zone1_cols].mean(axis=1).values

print(f"\n✓ 加载降雨数据: {precip_file}")
print(f"  Zone 1子分区: {len(zone1_cols)}个")
print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")
print(f"  平均降雨: {precipitation.mean():.2f} mm/h")

# 1.3 生成简化温度数据
temperature = np.linspace(5, 15, len(precipitation))
print(f"\n✓ 生成温度数据（简化）")
print(f"  温度范围: {temperature.min():.1f} - {temperature.max():.1f} °C")

# 1.4 Zone 1流域面积
zone1_area_km2 = 139.995  # 14个子分区 × 10.00870436517134 km²
print(f"\n✓ Zone 1流域面积: {zone1_area_km2:.2f} km²")

# ============================================================================
# 步骤 2: 定义HBV模型和目标函数
# ============================================================================
print("\n步骤 2: 定义HBV模型和目标函数")
print("-" * 80)

# HBV参数搜索范围
param_bounds = {
    'FC': [250, 600],        # 土壤最大容量
    'BETA': [1.5, 3.5],      # 土壤蓄水曲线指数
    'K0': [0.1, 0.5],        # 快速径流退水系数
    'K1': [0.02, 0.15],      # 中速径流退水系数
    'K2': [0.005, 0.05],     # 基流退水系数
    'PERC': [0.5, 4.0],      # 渗透速率
    'initial_soil_ratio': [0.3, 0.9],   # 初始土壤湿度比例
    'initial_upper': [5, 50],           # 初始上层储量
}

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
for param, bounds in param_bounds.items():
    print(f"  {param:<20s}: [{bounds[0]:>8.3f}, {bounds[1]:>8.3f}]")

print(f"\n固定参数: {len(fixed_params)}个")
for param, value in fixed_params.items():
    print(f"  {param:<20s}: {value:>8.3f}")

# 定义用于率定的HBV模型函数
class MockSubbasin:
    def __init__(self, area_km2):
        self.area_km2 = area_km2

subbasin = MockSubbasin(zone1_area_km2)

def run_hbv_model(FC, BETA, K0, K1, K2, PERC, initial_soil_ratio, initial_upper):
    """
    运行HBV模型并返回径流时间序列
    """
    params = {
        'FC': FC,
        'BETA': BETA,
        'K0': K0,
        'K1': K1,
        'K2': K2,
        'PERC': PERC,
        **fixed_params,
        'initial_soil': FC * initial_soil_ratio,  # 根据FC计算初始土壤湿度
        'initial_upper': initial_upper,
        'initial_lower': 30.0,
        'initial_snow': 0.0
    }

    hbv = HBVRunoff(params)

    # 运行HBV模型（HBV.simulate已经返回m³/s，不需要额外转换！）
    runoff_m3s = hbv.simulate(subbasin, precipitation.tolist())
    return np.array(runoff_m3s)

# 测试HBV模型运行
print("\n测试HBV模型运行...")
default_params = [400, 2.0, 0.25, 0.08, 0.02, 2.0, 0.6, 20]
test_runoff = run_hbv_model(*default_params)
print(f"  模型输出长度: {len(test_runoff)}")
print(f"  输出范围: {test_runoff.min():.2f} - {test_runoff.max():.2f} m³/s")

test_nse = nash_sutcliffe_efficiency(observed_runoff, test_runoff)
print(f"  初始NSE (默认参数): {test_nse:.4f}")

# 定义目标函数（最小化负NSE）
def objective_function(params):
    """
    目标函数：返回负NSE（用于最小化）
    """
    try:
        FC, BETA, K0, K1, K2, PERC, initial_soil_ratio, initial_upper = params

        # 参数约束检查
        if not (250 <= FC <= 600): return 1e6
        if not (1.5 <= BETA <= 3.5): return 1e6
        if not (0.1 <= K0 <= 0.5): return 1e6
        if not (0.02 <= K1 <= 0.15): return 1e6
        if not (0.005 <= K2 <= 0.05): return 1e6
        if not (0.5 <= PERC <= 4.0): return 1e6
        if not (0.3 <= initial_soil_ratio <= 0.9): return 1e6
        if not (5 <= initial_upper <= 50): return 1e6

        # 运行模型
        simulated = run_hbv_model(FC, BETA, K0, K1, K2, PERC, initial_soil_ratio, initial_upper)

        # 计算NSE
        nse = nash_sutcliffe_efficiency(observed_runoff, simulated)

        # 返回负NSE（因为优化算法是最小化）
        return -nse

    except Exception as e:
        return 1e6

# ============================================================================
# 步骤 3: 参数率定（使用Differential Evolution）
# ============================================================================
print("\n步骤 3: 参数率定")
print("-" * 80)

# Differential Evolution是一种全局优化算法，类似于SCE-UA和PSO
print("\n运行Differential Evolution率定...")
print("  这可能需要几分钟时间...")

bounds_list = list(param_bounds.values())

start_time = time.time()

# 记录收敛历史
convergence_history = []

def callback(xk, convergence):
    """每次迭代后调用，记录当前最优值"""
    nse = -objective_function(xk)
    convergence_history.append(nse)
    return False  # 返回True会提前停止

result_de = differential_evolution(
    objective_function,
    bounds=bounds_list,
    strategy='best1bin',
    maxiter=150,
    popsize=20,  # 20 * 8参数 = 160 个体
    tol=1e-6,
    mutation=(0.5, 1),
    recombination=0.7,
    seed=42,
    callback=callback,
    disp=True,
    polish=True,  # 最后用局部优化refined
    workers=1  # 单线程保证可重复性
)

end_time = time.time()
computation_time = end_time - start_time

print(f"  ✓ 完成!")
print(f"    最优NSE: {-result_de.fun:.6f}")
print(f"    函数评估: {result_de.nfev}")
print(f"    计算时间: {computation_time:.2f}秒")

# ============================================================================
# 步骤 4: 率定结果
# ============================================================================
print("\n步骤 4: 率定结果")
print("-" * 80)

best_name = "Differential Evolution"
best_result = result_de
best_nse = -best_result.fun

print(f"\n算法: {best_name}")
print(f"  最优NSE: {best_nse:.6f}")
print(f"  函数评估: {result_de.nfev}")
print(f"  计算时间: {computation_time:.2f}秒")

# 最优参数
best_params = best_result.x
param_names = list(param_bounds.keys())

print(f"\n最优参数:")
for i, (name, value) in enumerate(zip(param_names, best_params)):
    bounds = param_bounds[name]
    range_pct = (value - bounds[0]) / (bounds[1] - bounds[0]) * 100
    print(f"  {name:<20s}: {value:>12.4f}  (搜索空间的{range_pct:>5.1f}%)")

# ============================================================================
# 步骤 5: 使用最优参数运行HBV模型
# ============================================================================
print("\n步骤 5: 使用最优参数运行HBV模型")
print("-" * 80)

# 运行模型
final_simulated = run_hbv_model(*best_params)

# 计算所有性能指标
final_metrics = {
    'nse': nash_sutcliffe_efficiency(observed_runoff, final_simulated),
    'rmse': rmse(observed_runoff, final_simulated),
    'mae': mean_absolute_error(observed_runoff, final_simulated),
    'pbias': percent_bias(observed_runoff, final_simulated),
    'kge': kling_gupta_efficiency(observed_runoff, final_simulated)
}

# 计算log-transformed NSE（对低流更敏感）
log_obs = np.log(observed_runoff + 1e-6)
log_sim = np.log(final_simulated + 1e-6)
final_metrics['log_nse'] = nash_sutcliffe_efficiency(log_obs, log_sim)

print(f"\n率定后性能指标:")
print(f"  NSE       : {final_metrics['nse']:>8.4f}")
print(f"  RMSE      : {final_metrics['rmse']:>8.4f}")
print(f"  MAE       : {final_metrics['mae']:>8.4f}")
print(f"  PBIAS     : {final_metrics['pbias']:>8.4f}")
print(f"  KGE       : {final_metrics['kge']:>8.4f}")
print(f"  LOG_NSE   : {final_metrics['log_nse']:>8.4f}")

# ============================================================================
# 步骤 6: 保存最优参数到YAML配置文件
# ============================================================================
print("\n步骤 6: 保存最优参数到YAML配置文件")
print("-" * 80)

output_dir = results_dir / "calibration_enhanced"
output_dir.mkdir(parents=True, exist_ok=True)

# 创建参数率定YAML配置
calibration_yaml = {
    'description': f'Zone 1 HBV参数率定结果（增强型观测）({best_name})',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'calibration_info': {
        'algorithm': best_name,
        'observation_type': 'EnhancedRunoffGenerator',
        'nse': float(final_metrics['nse']),
        'rmse': float(final_metrics['rmse']),
        'kge': float(final_metrics['kge']),
    },
    'zones': {
        1: {
            'runoff_model': 'HBV',
            'parameters': {
                'FC': float(best_params[0]),
                'BETA': float(best_params[1]),
                'K0': float(best_params[2]),
                'K1': float(best_params[3]),
                'K2': float(best_params[4]),
                'PERC': float(best_params[5]),
                'LP': float(fixed_params['LP']),
                'MAXBAS': float(fixed_params['MAXBAS']),
                'TT': float(fixed_params['TT']),
                'CFMAX': float(fixed_params['CFMAX']),
                'CFR': float(fixed_params['CFR']),
                'CWH': float(fixed_params['CWH']),
                'initial_soil': float(best_params[0] * best_params[6]),
                'initial_upper': float(best_params[7]),
                'initial_lower': 30.0,
                'initial_snow': 0.0
            }
        }
    }
}

yaml_file = output_dir / "zone1_calibrated_parameters_enhanced.yaml"
with open(yaml_file, 'w') as f:
    yaml.dump(calibration_yaml, f, default_flow_style=False, sort_keys=False)

print(f"✓ 保存参数配置: {yaml_file}")

# 保存JSON格式的详细结果
import json
json_data = {
    'algorithm': best_name,
    'nse': float(final_metrics['nse']),
    'iterations': int(best_result.nit),
    'function_evaluations': int(best_result.nfev),
    'time_seconds': float(best_result.total_time),
    'parameters': {name: float(val) for name, val in zip(param_names, best_params)},
    'metrics': {k: float(v) for k, v in final_metrics.items()},
    'convergence_history': [float(x) for x in best_result.convergence_history] if hasattr(best_result, 'convergence_history') else []
}

json_file = output_dir / f"zone1_calibration_{best_name.lower().replace('-', '_')}_enhanced.json"
with open(json_file, 'w') as f:
    json.dump(json_data, f, indent=2)

print(f"✓ 保存率定结果: {json_file}")

# ============================================================================
# 步骤 7: 生成分析图表
# ============================================================================
print("\n步骤 7: 生成分析图表")
print("-" * 80)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 1. 水文过程对比图
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(observed_runoff, 'b-', label='Enhanced Observation', linewidth=1.5)
ax.plot(final_simulated, 'r--', label='HBV Simulation', linewidth=1.5)
ax.set_xlabel('Time Step (hour)', fontsize=11)
ax.set_ylabel('Discharge (m³/s)', fontsize=11)
ax.set_title(f'Zone 1 HBV Calibration (NSE={final_metrics["nse"]:.4f})', fontsize=12, weight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / "zone1_hydrograph_enhanced.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: zone1_hydrograph_enhanced.png")

# 2. 散点图
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(observed_runoff, final_simulated, alpha=0.5, s=20)
min_val = min(observed_runoff.min(), final_simulated.min())
max_val = max(observed_runoff.max(), final_simulated.max())
ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='1:1 Line')
ax.set_xlabel('Observed (m³/s)', fontsize=11)
ax.set_ylabel('Simulated (m³/s)', fontsize=11)
ax.set_title(f'Scatter Plot (NSE={final_metrics["nse"]:.4f})', fontsize=12, weight='bold')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal', adjustable='box')
plt.tight_layout()
plt.savefig(output_dir / "zone1_scatter_enhanced.png", dpi=300, bbox_inches='tight')
plt.close()
print("  ✓ 保存: zone1_scatter_enhanced.png")

# 3. 收敛历史
if len(convergence_history) > 0:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(convergence_history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('NSE', fontsize=11)
    ax.set_title('Calibration Convergence History', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "zone1_convergence_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ 保存: zone1_convergence_enhanced.png")

print(f"\n✓ 所有图表已保存到: {output_dir}")

# ============================================================================
# 步骤 8: 保存模拟结果
# ============================================================================
print("\n步骤 8: 保存模拟结果")
print("-" * 80)

# 保存径流对比数据
comparison_df = pd.DataFrame({
    'datetime': times,
    'observed_m3s': observed_runoff,
    'simulated_m3s': final_simulated,
    'residual_m3s': observed_runoff - final_simulated
})

comparison_file = output_dir / "zone1_calibrated_runoff_enhanced.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"✓ 保存径流对比数据: {comparison_file.name}")

# ============================================================================
# 步骤 9: 总结
# ============================================================================
print("\n" + "=" * 80)
print("Zone 1 HBV参数率定完成（增强型观测）！")
print("=" * 80)

print(f"\n1. 率定数据:")
print(f"   - 观测类型: 增强型径流生成器")
print(f"   - 时间步数: {len(observed_runoff)}")
print(f"   - 平均流量: {observed_runoff.mean():.2f} m³/s")
print(f"   - 峰值流量: {observed_runoff.max():.2f} m³/s")

print(f"\n2. 率定结果:")
print(f"   - 算法: {best_name}")
print(f"   - 最优NSE: {best_nse:.6f}")
print(f"   - 计算时间: {computation_time:.2f}秒")

print(f"\n3. 性能指标:")
print(f"   - NSE: {final_metrics['nse']:.6f}")
print(f"   - RMSE: {final_metrics['rmse']:.4f} m³/s")
print(f"   - PBIAS: {final_metrics['pbias']:.2f}%")
print(f"   - KGE: {final_metrics['kge']:.6f}")

print(f"\n4. 最优参数:")
for name, value in zip(param_names, best_params):
    print(f"   - {name}: {value:.4f}")

print(f"\n5. 输出文件:")
print(f"   - 参数配置: {yaml_file.name}")
print(f"   - 率定结果: {json_file.name}")
print(f"   - 径流数据: {comparison_file.name}")
print(f"   - 分析图表: zone1_*_enhanced.png")

print(f"\n6. 对比之前的率定结果:")
print(f"   之前（简单线性观测）: NSE ≈ 0.085")
print(f"   现在（增强型观测）  : NSE = {final_metrics['nse']:.6f}")
print(f"   改进: {'显著提升' if final_metrics['nse'] > 0.5 else '有所改善' if final_metrics['nse'] > 0.2 else '仍需优化'}")

print("\n" + "=" * 80)
print("✓ Zone 1参数率定完成！")
print("=" * 80)
