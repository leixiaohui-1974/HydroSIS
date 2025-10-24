#!/usr/bin/env python3
"""
模型兼容性对比分析

对比分析：
1. 简单EstimatedRunoff生成 + HBV率定
2. 增强EnhancedRunoff生成 + HBV率定
3. HBV自模拟 + HBV率定（理论最优）

目标：理解为什么增强模型与HBV不兼容，以及如何改进

遵循 .claude/AI_DEVELOPMENT_GUIDE.md 中的最佳实践。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ✅ 使用基础库的功能模块
from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator
from hydrosis.runoff.estimated import EstimatedRunoffGenerator
from hydrosis.evaluation.water_balance import (
    calculate_water_balance,
    compare_water_balance,
)
from hydrosis.evaluation.metrics import (
    nash_sutcliffe_efficiency,
    kling_gupta_efficiency,
    rmse,
    mae,
)
from hydrosis.calibration import (
    calibrate_parameters,
    morris_sensitivity,
    print_sensitivity_report,
)

print("=" * 80)
print("模型兼容性对比分析")
print("=" * 80)

# ============================================================================
# 步骤 1: 准备测试数据
# ============================================================================
print("\n步骤 1: 准备测试数据")
print("-" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")

# 加载降雨数据
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
precip_df = pd.read_csv(precip_file, index_col=0)

zone1_subbasins = [str(i) for i in range(10, 24)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
precipitation_mmh = precip_df[zone1_cols].mean(axis=1).values

zone1_area_km2 = 139.995

print(f"✓ 降雨数据: {len(precipitation_mmh)} 小时")
print(f"  范围: {precipitation_mmh.min():.2f} - {precipitation_mmh.max():.2f} mm/h")
print(f"✓ Zone 1面积: {zone1_area_km2:.2f} km²")

class MockSubbasin:
    def __init__(self, area_km2):
        self.area_km2 = area_km2

subbasin = MockSubbasin(zone1_area_km2)

# ============================================================================
# 步骤 2: 生成不同模型的"观测"数据
# ============================================================================
print("\n步骤 2: 生成不同模型的观测数据")
print("-" * 80)

# 2.1 简单模型生成
print("\n2.1 简单EstimatedRunoff生成...")
simple_params = {
    'runoff_coef': 0.45,
    'initial_baseflow': 30.0,
    'recession_coef': 0.92,
}
simple_gen = EstimatedRunoffGenerator(simple_params)
simple_runoff_m3s = np.array(simple_gen.generate(subbasin, precipitation_mmh.tolist()))

print(f"  参数: runoff_coef={simple_params['runoff_coef']:.2f}")
print(f"  流量范围: {simple_runoff_m3s.min():.2f} - {simple_runoff_m3s.max():.2f} m³/s")

# 2.2 增强模型生成
print("\n2.2 增强EnhancedRunoff生成...")
enhanced_params = {
    'soil_capacity': 420.0,
    'soil_beta': 2.0,
    'k_fast': 0.30,
    'k_inter': 0.09,
    'k_base': 0.022,
    'initial_soil': 300.0,
}
enhanced_gen = EnhancedRunoffGenerator(enhanced_params)
enhanced_runoff_m3s = np.array(enhanced_gen.generate(subbasin, precipitation_mmh.tolist()))

print(f"  参数: soil_capacity={enhanced_params['soil_capacity']:.1f}, soil_beta={enhanced_params['soil_beta']:.1f}")
print(f"  流量范围: {enhanced_runoff_m3s.min():.2f} - {enhanced_runoff_m3s.max():.2f} m³/s")

# 2.3 HBV自模拟（作为理想对照）
print("\n2.3 HBV模型自模拟...")
hbv_true_params = {
    'FC': 400.0,
    'BETA': 2.0,
    'K0': 0.25,
    'K1': 0.08,
    'K2': 0.02,
    'PERC': 2.0,
    'LP': 0.7,
    'MAXBAS': 3.0,
    'TT': 0.0,
    'CFMAX': 3.5,
    'CFR': 0.05,
    'CWH': 0.1,
    'initial_soil': 300.0,
    'initial_upper': 20.0,
    'initial_lower': 2000.0,  # 合理的初始值
    'initial_snow': 0.0,
}
hbv_true = HBVRunoff(hbv_true_params)
hbv_self_runoff_m3s = np.array(hbv_true.simulate(subbasin, precipitation_mmh.tolist()))

print(f"  参数: FC={hbv_true_params['FC']:.1f}, BETA={hbv_true_params['BETA']:.1f}")
print(f"  流量范围: {hbv_self_runoff_m3s.min():.2f} - {hbv_self_runoff_m3s.max():.2f} m³/s")

# ============================================================================
# 步骤 3: 水量平衡对比
# ============================================================================
print("\n步骤 3: 水量平衡对比")
print("-" * 80)

# ✅ 使用基础库的水量平衡分析
balance_simple = calculate_water_balance(precipitation_mmh, simple_runoff_m3s, zone1_area_km2)
balance_enhanced = calculate_water_balance(precipitation_mmh, enhanced_runoff_m3s, zone1_area_km2)
balance_hbv_self = calculate_water_balance(precipitation_mmh, hbv_self_runoff_m3s, zone1_area_km2)

balance_results = {
    "Simple": balance_simple,
    "Enhanced": balance_enhanced,
    "HBV-Self": balance_hbv_self,
}

# ✅ 使用基础库的对比功能
comparison = compare_water_balance(balance_results)
print(comparison)

# ============================================================================
# 步骤 4: HBV率定测试（简化版）
# ============================================================================
print("\n步骤 4: HBV率定测试（各观测数据）")
print("-" * 80)

# 简化的HBV参数范围（只率定关键参数）
param_bounds = [
    (250, 600),     # FC
    (1.5, 3.5),     # BETA
    (0.1, 0.5),     # K0
    (0.02, 0.15),   # K1
]
param_names = ['FC', 'BETA', 'K0', 'K1']

fixed_hbv_params = {
    'K2': 0.02,
    'PERC': 2.0,
    'LP': 0.7,
    'MAXBAS': 3.0,
    'TT': 0.0,
    'CFMAX': 3.5,
    'CFR': 0.05,
    'CWH': 0.1,
    'initial_snow': 0.0,
}

def run_hbv_quick(params_list, initial_lower=2000.0):
    """快速运行HBV模型"""
    FC, BETA, K0, K1 = params_list
    params = {
        'FC': FC,
        'BETA': BETA,
        'K0': K0,
        'K1': K1,
        **fixed_hbv_params,
        'initial_soil': FC * 0.7,
        'initial_upper': 20.0,
        'initial_lower': initial_lower,
    }
    hbv = HBVRunoff(params)
    return np.array(hbv.simulate(subbasin, precipitation_mmh.tolist()))

# 率定结果存储
calibration_results = {}

# 4.1 对简单模型观测数据率定
print("\n4.1 简单模型观测数据 + HBV率定")

def objective_simple(params):
    try:
        simulated = run_hbv_quick(params, initial_lower=1000.0)  # 简单模型基流较低
        return nash_sutcliffe_efficiency(simulated, simple_runoff_m3s)
    except:
        return -999.0

result_simple = calibrate_parameters(
    objective_function=objective_simple,
    param_bounds=param_bounds,
    algorithm="differential_evolution",
    maximize=True,
    maxiter=50,  # 减少迭代次数加快测试
    popsize=15,
    seed=42,
)

calibration_results['Simple'] = result_simple
print(f"  最优NSE: {result_simple.best_score:.4f}")

# 4.2 对增强模型观测数据率定
print("\n4.2 增强模型观测数据 + HBV率定")

def objective_enhanced(params):
    try:
        simulated = run_hbv_quick(params, initial_lower=2000.0)
        return nash_sutcliffe_efficiency(simulated, enhanced_runoff_m3s)
    except:
        return -999.0

result_enhanced = calibrate_parameters(
    objective_function=objective_enhanced,
    param_bounds=param_bounds,
    algorithm="differential_evolution",
    maximize=True,
    maxiter=50,
    popsize=15,
    seed=42,
)

calibration_results['Enhanced'] = result_enhanced
print(f"  最优NSE: {result_enhanced.best_score:.4f}")

# 4.3 对HBV自模拟数据率定（理论最优）
print("\n4.3 HBV自模拟数据 + HBV率定（理论最优）")

def objective_hbv_self(params):
    try:
        simulated = run_hbv_quick(params, initial_lower=2000.0)
        return nash_sutcliffe_efficiency(simulated, hbv_self_runoff_m3s)
    except:
        return -999.0

result_hbv_self = calibrate_parameters(
    objective_function=objective_hbv_self,
    param_bounds=param_bounds,
    algorithm="differential_evolution",
    maximize=True,
    maxiter=50,
    popsize=15,
    seed=42,
)

calibration_results['HBV-Self'] = result_hbv_self
print(f"  最优NSE: {result_hbv_self.best_score:.4f}")

# ============================================================================
# 步骤 5: 率定结果对比
# ============================================================================
print("\n步骤 5: 率定结果对比分析")
print("-" * 80)

print(f"\n率定结果汇总:")
print(f"{'观测数据来源':<20s} {'NSE':<12s} {'评估':<30s}")
print("-" * 80)

for name, result in calibration_results.items():
    nse = result.best_score
    if nse > 0.75:
        assessment = "优秀 - 模型高度兼容"
    elif nse > 0.50:
        assessment = "良好 - 模型基本兼容"
    elif nse > 0:
        assessment = "可接受 - 模型部分兼容"
    else:
        assessment = "失败 - 模型不兼容"

    print(f"{name:<20s} {nse:<12.4f} {assessment:<30s}")

# 兼容性分析
print(f"\n兼容性分析:")
nse_simple = calibration_results['Simple'].best_score
nse_enhanced = calibration_results['Enhanced'].best_score
nse_hbv = calibration_results['HBV-Self'].best_score

print(f"  HBV自率定 (理论最优): NSE={nse_hbv:.4f}")
print(f"  简单模型 → HBV:        NSE={nse_simple:.4f} (相对理论最优: {(nse_simple/nse_hbv*100):.1f}%)")
print(f"  增强模型 → HBV:        NSE={nse_enhanced:.4f} (相对理论最优: {(nse_enhanced/nse_hbv*100):.1f}%)")

if nse_simple > nse_enhanced:
    diff = nse_simple - nse_enhanced
    print(f"\n✗ 关键发现: 简单模型比增强模型更兼容HBV (Δ NSE = {diff:.4f})")
    print(f"  可能原因:")
    print(f"    1. 增强模型的多层储量结构与HBV参数化方式不匹配")
    print(f"    2. 增强模型的线性储量vs HBV的非线性土壤水")
    print(f"    3. 两个模型的产流机制存在根本性差异")
else:
    print(f"\n✓ 增强模型与HBV兼容性更好")

# ============================================================================
# 步骤 6: 敏感性分析对比
# ============================================================================
print("\n步骤 6: 参数敏感性分析对比")
print("-" * 80)

print("\n6.1 简单模型观测数据的参数敏感性")
sens_simple = morris_sensitivity(objective_simple, param_names, param_bounds, n_trajectories=10)
print_sensitivity_report(sens_simple)

print("\n6.2 增强模型观测数据的参数敏感性")
sens_enhanced = morris_sensitivity(objective_enhanced, param_names, param_bounds, n_trajectories=10)
print_sensitivity_report(sens_enhanced)

print("\n6.3 HBV自模拟数据的参数敏感性")
sens_hbv_self = morris_sensitivity(objective_hbv_self, param_names, param_bounds, n_trajectories=10)
print_sensitivity_report(sens_hbv_self)

# ============================================================================
# 步骤 7: 保存对比报告
# ============================================================================
print("\n步骤 7: 保存对比报告")
print("-" * 80)

output_dir = results_dir / "model_compatibility"
output_dir.mkdir(parents=True, exist_ok=True)

report_file = output_dir / "compatibility_analysis.txt"

with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("模型兼容性对比分析报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("【测试设计】\n")
    f.write("  3种观测数据来源:\n")
    f.write("    1. 简单EstimatedRunoff生成\n")
    f.write("    2. 增强EnhancedRunoff生成\n")
    f.write("    3. HBV自模拟（理论最优对照）\n\n")

    f.write("【水量平衡】\n")
    for name, result in balance_results.items():
        f.write(f"  {name:<12s}: 径流系数={result.runoff_coefficient:.4f}, 质量={result.balance_quality}\n")
    f.write("\n")

    f.write("【率定结果】\n")
    for name, result in calibration_results.items():
        f.write(f"  {name:<12s}: NSE={result.best_score:.4f}\n")
    f.write("\n")

    f.write("【关键发现】\n")
    if nse_simple > nse_enhanced:
        f.write(f"  ✗ 简单模型比增强模型更兼容HBV\n")
        f.write(f"    NSE差异: {nse_simple - nse_enhanced:.4f}\n")
        f.write(f"    结论: 增强模型与HBV存在结构性不兼容\n")
    f.write(f"\n")

    f.write("【建议】\n")
    f.write(f"  1. 如果目标是HBV率定，考虑使用简单模型生成观测数据\n")
    f.write(f"  2. 或者调整增强模型参数使其更接近HBV的结构\n")
    f.write(f"  3. 获取更长时间序列数据（当前{len(precipitation_mmh)}小时不足）\n")

print(f"✓ 保存对比报告: {report_file}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("模型兼容性对比分析完成!")
print("=" * 80)

print(f"\n【核心结论】")
print(f"  简单模型 + HBV: NSE={nse_simple:.4f}")
print(f"  增强模型 + HBV: NSE={nse_enhanced:.4f}")
print(f"  HBV自率定:      NSE={nse_hbv:.4f}")

if nse_simple > nse_enhanced:
    print(f"\n  ⚠ 增强模型与HBV存在兼容性问题")
    print(f"     这解释了为什么增强观测数据的率定效果差")

print(f"\n【输出文件】")
print(f"  对比报告: {report_file}")

print("\n" + "=" * 80)
print("✓ 本脚本遵循 .claude/AI_DEVELOPMENT_GUIDE.md 最佳实践")
print("=" * 80)
