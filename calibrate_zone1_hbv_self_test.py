#!/usr/bin/env python3
"""
Zone 1 HBV模型自验证

正确的验证方式：
1. 用HBV模型+真实参数生成"观测"数据
2. 用HBV模型率定，恢复这些参数
3. 应该达到NSE>0.99

这验证了：
- HBV模型实现正确
- 率定算法对HBV模型有效
- 参数搜索范围合理
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

from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.analysis import (
    calculate_metrics,
    calibrate_model,
)

print("=" * 80)
print("Zone 1 HBV模型自验证")
print("=" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")

# ============================================================================
# 1. 加载降雨数据
# ============================================================================
print("\n步骤 1: 加载降雨数据")
print("-" * 80)

precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
precip_df = pd.read_csv(precip_file, index_col=0)
zone1_subbasins = [str(i) for i in range(101, 115)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
precipitation = precip_df[zone1_cols].mean(axis=1).values

zone1_area_km2 = 139.995

print(f"✓ 降雨数据: {len(precipitation)}个时间步")
print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")
print(f"✓ 流域面积: {zone1_area_km2:.2f} km²")

# ============================================================================
# 2. 用HBV生成"观测"数据（使用合理的真实参数）
# ============================================================================
print("\n步骤 2: 用HBV模型生成观测数据")
print("-" * 80)

# 真实参数（合理的HBV参数）
true_params = {
    'FC': 200.0,        # 田间持水能力
    'BETA': 2.5,        # 土壤非线性指数
    'K0': 0.15,         # 快速消退系数
    'K1': 0.08,         # 中速消退系数
    'K2': 0.02,         # 基流消退系数
    'PERC': 2.0,        # 渗漏率
    'TT': 0.0,
    'CFMAX': 3.5,
    'LP': 0.7,
    'MAXBAS': 3.0,
    'CFR': 0.05,
    'CWH': 0.1,
    'initial_soil': 60.0,
    'initial_upper': 10.0,
    'initial_lower': 30.0,
    'initial_snow': 0.0
}

print("真实HBV参数:")
for key in ['FC', 'BETA', 'K0', 'K1', 'K2', 'PERC']:
    print(f"  {key:10s}: {true_params[key]:.4f}")

# 生成观测数据
class MockSubbasin:
    def __init__(self, area_km2):
        self.area_km2 = area_km2

hbv_true = HBVRunoff(true_params)
subbasin = MockSubbasin(zone1_area_km2)
observed_runoff = np.array(hbv_true.simulate(subbasin, precipitation.tolist()))

print(f"\n生成的观测径流:")
print(f"  流量范围: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")
print(f"  平均流量: {observed_runoff.mean():.2f} m³/s")

# ============================================================================
# 3. 定义率定问题
# ============================================================================
print("\n步骤 3: 定义HBV率定问题")
print("-" * 80)

# 扩大参数搜索范围（基于真实参数调整）
param_bounds = {
    'FC': [100, 400],       # 真实200，搜索100-400
    'BETA': [1.0, 5.0],     # 真实2.5，搜索1-5
    'K0': [0.05, 0.4],      # 真实0.15，搜索0.05-0.4
    'K1': [0.01, 0.2],      # 真实0.08，搜索0.01-0.2
    'K2': [0.005, 0.05],    # 真实0.02，搜索0.005-0.05
    'PERC': [0.5, 5.0],     # 真实2.0，搜索0.5-5
}

print(f"参数搜索范围:")
for name, bounds in param_bounds.items():
    true_val = true_params[name]
    in_range = bounds[0] <= true_val <= bounds[1]
    status = "✓" if in_range else "✗"
    print(f"  {status} {name:10s}: [{bounds[0]:7.2f}, {bounds[1]:7.2f}]  真实={true_val:.2f}")

# 固定参数
fixed_params = {k: v for k, v in true_params.items() if k not in param_bounds}

def run_hbv_model(FC, BETA, K0, K1, K2, PERC):
    """运行HBV模型"""
    try:
        params = {
            'FC': FC, 'BETA': BETA, 'K0': K0,
            'K1': K1, 'K2': K2, 'PERC': PERC,
            **fixed_params
        }
        hbv = HBVRunoff(params)
        runoff = hbv.simulate(subbasin, precipitation.tolist())
        return np.array(runoff)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(len(precipitation))

def calibration_objective(FC, BETA, K0, K1, K2, PERC):
    """率定目标函数：最大化NSE"""
    from hydrosis.analysis import nash_sutcliffe_efficiency
    simulated = run_hbv_model(FC, BETA, K0, K1, K2, PERC)
    nse = nash_sutcliffe_efficiency(observed_runoff, simulated)
    return nse

# 测试
print("\n测试模型...")
test_runoff = run_hbv_model(**{k: true_params[k] for k in param_bounds.keys()})
initial_nse = calibration_objective(**{k: true_params[k] for k in param_bounds.keys()})
print(f"  用真实参数的NSE: {initial_nse:.8f}")
print(f"  (理论上应该=1.0，因为观测数据就是用这些参数生成的)")

# ============================================================================
# 4. 运行率定
# ============================================================================
print("\n步骤 4: 运行SCE-UA率定")
print("-" * 80)

result = calibrate_model(
    objective_function=calibration_objective,
    param_bounds=param_bounds,
    maximize=True,
    method='sce_ua',
    n_complexes=5,
    max_iterations=80,
    patience=20,
    seed=42,
    verbose=False,
)

print(f"✓ 率定完成!")
print(f"  最优NSE: {result.best_score:.8f}")
print(f"  迭代次数: {result.n_iterations}")
print(f"  函数评估: {result.n_evaluations}")
print(f"  计算时间: {result.computation_time:.2f}秒")

# ============================================================================
# 5. 参数对比
# ============================================================================
print("\n步骤 5: 参数对比")
print("-" * 80)

print(f"\n{'参数':<12s} {'真实值':>12s} {'率定值':>12s} {'误差%':>12s} {'状态'}")
print("-" * 60)
max_error_pct = 0
for param in param_bounds.keys():
    true_val = true_params[param]
    calib_val = result.best_params[param]
    error_pct = abs(calib_val - true_val) / true_val * 100
    max_error_pct = max(max_error_pct, error_pct)
    status = "✓" if error_pct < 10 else "⚠" if error_pct < 20 else "✗"
    print(f"{param:<12s} {true_val:>12.4f} {calib_val:>12.4f} {error_pct:>11.2f}% {status}")

# ============================================================================
# 6. 评估结果
# ============================================================================
print("\n步骤 6: 评估率定结果")
print("-" * 80)

calibrated_runoff = run_hbv_model(**result.best_params)
final_metrics = calculate_metrics(
    observed_runoff,
    calibrated_runoff,
    metrics=['nse', 'rmse', 'mae', 'pbias', 'kge']
)

print("\n性能指标:")
for metric, value in final_metrics.items():
    if 'peak' not in metric and 'time' not in metric:
        print(f"  {metric.upper():10s}: {value:8.6f}")

# ============================================================================
# 7. 判断
# ============================================================================
print("\n" + "=" * 80)
print("验证结果")
print("=" * 80)

success = True
issues = []

if final_metrics['nse'] < 0.95:
    success = False
    issues.append(f"NSE={final_metrics['nse']:.4f} < 0.95")

if max_error_pct > 20:
    success = False
    issues.append(f"最大参数误差={max_error_pct:.1f}% > 20%")

if success:
    print(f"\n✅ HBV模型验证通过!")
    print(f"  - NSE = {final_metrics['nse']:.6f} >= 0.95")
    print(f"  - 最大参数误差 = {max_error_pct:.2f}% < 20%")
    print(f"  - HBV模型实现正确")
    print(f"  - 率定算法对HBV有效")
    print(f"  - 参数搜索范围合理")
else:
    print(f"\n⚠️  HBV模型验证未通过!")
    print(f"\n问题:")
    for issue in issues:
        print(f"  - {issue}")
    print(f"\n可能原因:")
    print(f"  1. HBV模型实现有bug")
    print(f"  2. 参数搜索范围不合理")
    print(f"  3. 率定算法参数需要调整")
    print(f"  4. 数据有问题")

# ============================================================================
# 8. 保存结果
# ============================================================================
output_dir = results_dir / "calibration"
output_dir.mkdir(parents=True, exist_ok=True)

validation_yaml = {
    'description': 'Zone 1 HBV模型自验证',
    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'validation_result': 'PASSED' if success else 'FAILED',
    'nse': float(final_metrics['nse']),
    'max_parameter_error_pct': float(max_error_pct),
    'true_parameters': {k: float(v) for k, v in true_params.items() if k in param_bounds},
    'calibrated_parameters': {k: float(v) for k, v in result.best_params.items()},
    'issues': issues if not success else None,
}

yaml_file = output_dir / "zone1_hbv_self_validation.yaml"
with open(yaml_file, 'w') as f:
    yaml.dump(validation_yaml, f, default_flow_style=False, sort_keys=False)

print(f"\n✓ 保存验证结果: {yaml_file}")
print("=" * 80)
