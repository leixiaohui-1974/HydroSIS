#!/usr/bin/env python3
"""
增强模型 vs HBV 兼容性分析

对比：
1. 增强EnhancedRunoff生成数据 + HBV率定
2. HBV自模拟数据 + HBV率定（理论最优）

目标：理解为什么增强模型与HBV不兼容

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
from hydrosis.evaluation.water_balance import (
    calculate_water_balance,
    compare_water_balance,
)
from hydrosis.evaluation.metrics import nash_sutcliffe_efficiency
from hydrosis.calibration import (
    calibrate_parameters,
    morris_sensitivity,
    print_sensitivity_report,
)

print("=" * 80)
print("增强模型 vs HBV 兼容性分析")
print("=" * 80)

# ============================================================================
# 步骤 1: 准备数据
# ============================================================================
print("\n步骤 1: 准备测试数据")
print("-" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
precip_df = pd.read_csv(precip_file, index_col=0)

zone1_subbasins = [str(i) for i in range(10, 24)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
precipitation_mmh = precip_df[zone1_cols].mean(axis=1).values
zone1_area_km2 = 139.995

print(f"✓ 降雨数据: {len(precipitation_mmh)} 小时")
print(f"  范围: {precipitation_mmh.min():.2f} - {precipitation_mmh.max():.2f} mm/h")

class MockSubbasin:
    def __init__(self, area_km2):
        self.area_km2 = area_km2

subbasin = MockSubbasin(zone1_area_km2)

# ============================================================================
# 步骤 2: 生成观测数据
# ============================================================================
print("\n步骤 2: 生成两种观测数据")
print("-" * 80)

# 2.1 增强模型生成
print("\n2.1 增强模型生成...")
enhanced_params = {
    'soil_capacity': 420.0,
    'soil_beta': 2.0,
    'k_fast': 0.30,
    'k_inter': 0.09,
    'k_base': 0.022,
    'initial_soil': 300.0,
}
enhanced_gen = EnhancedRunoffGenerator(enhanced_params)
enhanced_runoff_m3s, _ = enhanced_gen.generate(precipitation_mmh, zone1_area_km2)
print(f"  流量范围: {enhanced_runoff_m3s.min():.2f} - {enhanced_runoff_m3s.max():.2f} m³/s")

# 2.2 HBV自模拟（理论最优对照）
print("\n2.2 HBV自模拟（理论最优）...")
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
    'initial_lower': 1500.0,  # 合理的初始值
    'initial_snow': 0.0,
}
hbv_true = HBVRunoff(hbv_true_params)
hbv_self_runoff_m3s = np.array(hbv_true.simulate(subbasin, precipitation_mmh.tolist()))
print(f"  流量范围: {hbv_self_runoff_m3s.min():.2f} - {hbv_self_runoff_m3s.max():.2f} m³/s")

# ============================================================================
# 步骤 3: 水量平衡对比
# ============================================================================
print("\n步骤 3: 水量平衡对比")
print("-" * 80)

balance_enhanced = calculate_water_balance(precipitation_mmh, enhanced_runoff_m3s, zone1_area_km2)
balance_hbv = calculate_water_balance(precipitation_mmh, hbv_self_runoff_m3s, zone1_area_km2)

comparison = compare_water_balance({
    "Enhanced": balance_enhanced,
    "HBV-Self": balance_hbv,
})
print(comparison)

# ============================================================================
# 步骤 4: HBV率定测试
# ============================================================================
print("\n步骤 4: HBV率定测试")
print("-" * 80)

param_bounds = [
    (250, 600),     # FC
    (1.5, 3.5),     # BETA
    (0.1, 0.5),     # K0
    (0.02, 0.15),   # K1
]
param_names = ['FC', 'BETA', 'K0', 'K1']

fixed_params = {
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

def run_hbv(params_list, initial_lower=1500.0):
    FC, BETA, K0, K1 = params_list
    params = {
        'FC': FC,
        'BETA': BETA,
        'K0': K0,
        'K1': K1,
        **fixed_params,
        'initial_soil': FC * 0.7,
        'initial_upper': 20.0,
        'initial_lower': initial_lower,
    }
    hbv = HBVRunoff(params)
    return np.array(hbv.simulate(subbasin, precipitation_mmh.tolist()))

# 4.1 增强模型观测 + HBV率定
print("\n4.1 增强模型观测数据 + HBV率定...")

def objective_enhanced(params):
    try:
        simulated = run_hbv(params)
        return nash_sutcliffe_efficiency(simulated, enhanced_runoff_m3s)
    except:
        return -999.0

result_enhanced = calibrate_parameters(
    objective_function=objective_enhanced,
    param_bounds=param_bounds,
    algorithm="differential_evolution",
    maximize=True,
    maxiter=100,
    popsize=20,
    seed=42,
)
print(f"  ✓ 完成! 最优NSE: {result_enhanced.best_score:.4f}")

# 4.2 HBV自模拟 + HBV率定（理论最优）
print("\n4.2 HBV自模拟数据 + HBV率定（理论最优）...")

def objective_hbv_self(params):
    try:
        simulated = run_hbv(params)
        return nash_sutcliffe_efficiency(simulated, hbv_self_runoff_m3s)
    except:
        return -999.0

result_hbv_self = calibrate_parameters(
    objective_function=objective_hbv_self,
    param_bounds=param_bounds,
    algorithm="differential_evolution",
    maximize=True,
    maxiter=100,
    popsize=20,
    seed=42,
)
print(f"  ✓ 完成! 最优NSE: {result_hbv_self.best_score:.4f}")

# ============================================================================
# 步骤 5: 敏感性分析
# ============================================================================
print("\n步骤 5: 参数敏感性分析")
print("-" * 80)

print("\n5.1 增强模型观测的参数敏感性:")
sens_enhanced = morris_sensitivity(objective_enhanced, param_names, param_bounds, n_trajectories=15)
print_sensitivity_report(sens_enhanced)

print("\n5.2 HBV自模拟的参数敏感性:")
sens_hbv = morris_sensitivity(objective_hbv_self, param_names, param_bounds, n_trajectories=15)
print_sensitivity_report(sens_hbv)

# ============================================================================
# 步骤 6: 结果分析
# ============================================================================
print("\n步骤 6: 兼容性分析")
print("-" * 80)

nse_enhanced = result_enhanced.best_score
nse_hbv = result_hbv_self.best_score

print(f"\n率定结果对比:")
print(f"  增强模型 → HBV: NSE={nse_enhanced:.4f}")
print(f"  HBV自率定:      NSE={nse_hbv:.4f}")
print(f"  差异:           {abs(nse_hbv - nse_enhanced):.4f}")

if nse_enhanced > 0.7:
    print(f"\n✓ 增强模型与HBV兼容性良好 (NSE > 0.7)")
elif nse_enhanced > 0.5:
    print(f"\n⚠ 增强模型与HBV部分兼容 (0.5 < NSE < 0.7)")
elif nse_enhanced > 0:
    print(f"\n⚠ 增强模型与HBV兼容性较差 (0 < NSE < 0.5)")
else:
    print(f"\n✗ 增强模型与HBV不兼容 (NSE < 0)")

performance_ratio = (nse_enhanced / nse_hbv * 100) if nse_hbv > 0 else 0
print(f"\n相对理论最优性能: {performance_ratio:.1f}%")

# 参数对比
print(f"\n最优参数对比:")
enhanced_params_opt = result_enhanced.best_params
hbv_params_opt = result_hbv_self.best_params

print(f"  {'参数':<10s} {'增强→HBV':<15s} {'HBV→HBV':<15s} {'真值':<15s}")
print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
for i, name in enumerate(param_names):
    true_val = hbv_true_params[name]
    print(f"  {name:<10s} {enhanced_params_opt[i]:<15.3f} {hbv_params_opt[i]:<15.3f} {true_val:<15.3f}")

# ============================================================================
# 步骤 7: 保存报告
# ============================================================================
print("\n步骤 7: 保存分析报告")
print("-" * 80)

output_dir = results_dir / "model_compatibility"
output_dir.mkdir(parents=True, exist_ok=True)

report_file = output_dir / "enhanced_vs_hbv_analysis.txt"

with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("增强模型 vs HBV 兼容性分析报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("【测试配置】\n")
    f.write(f"  降雨数据: {len(precipitation_mmh)} 小时\n")
    f.write(f"  流域面积: {zone1_area_km2:.2f} km²\n\n")

    f.write("【水量平衡】\n")
    f.write(f"  增强模型: 径流系数={balance_enhanced.runoff_coefficient:.4f}\n")
    f.write(f"  HBV模型:  径流系数={balance_hbv.runoff_coefficient:.4f}\n\n")

    f.write("【率定结果】\n")
    f.write(f"  增强模型→HBV: NSE={nse_enhanced:.4f}\n")
    f.write(f"  HBV→HBV:      NSE={nse_hbv:.4f}\n")
    f.write(f"  性能比例:     {performance_ratio:.1f}%\n\n")

    f.write("【敏感性分析】\n")
    f.write(f"  增强模型数据 - 高敏感参数: {[p for p in param_names if sens_enhanced.sensitivity_indices[p] > 0.7]}\n")
    f.write(f"  HBV自模拟数据 - 高敏感参数: {[p for p in param_names if sens_hbv.sensitivity_indices[p] > 0.7]}\n\n")

    f.write("【结论】\n")
    if nse_enhanced < 0.5:
        f.write(f"  增强模型与HBV存在兼容性问题\n")
        f.write(f"  建议调整增强模型参数或使用其他观测数据源\n")
    else:
        f.write(f"  增强模型与HBV基本兼容\n")

print(f"✓ 保存报告: {report_file}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("增强模型 vs HBV 兼容性分析完成!")
print("=" * 80)

print(f"\n【核心结论】")
print(f"  增强模型→HBV: NSE={nse_enhanced:.4f} ({performance_ratio:.1f}% of optimal)")
print(f"  HBV自率定:    NSE={nse_hbv:.4f} (100% optimal)")

if nse_enhanced < 0.5:
    print(f"\n  ⚠ 这解释了为什么增强观测数据的率定效果差")
    print(f"     需要调整增强模型使其更接近HBV的结构")

print(f"\n【输出文件】")
print(f"  分析报告: {report_file}")

print("\n" + "=" * 80)
print("✓ 本脚本遵循 .claude/AI_DEVELOPMENT_GUIDE.md 最佳实践")
print("=" * 80)
