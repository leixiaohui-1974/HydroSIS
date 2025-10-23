#!/usr/bin/env python3
"""
分析观测和模拟的径流系数

对比：
1. 估计径流的径流系数（用于生成观测数据）
2. HBV模拟的径流系数
3. 降雨时间序列是否一致
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print("=" * 80)
print("径流系数对比分析")
print("=" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")
zone1_area_km2 = 139.995
dt_hours = 1.0

# ============================================================================
# 1. 加载降雨数据
# ============================================================================
print("\n1. 加载降雨数据")
print("-" * 80)

# 用于率定的降雨数据
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
precip_df = pd.read_csv(precip_file, index_col=0)
zone1_subbasins = [str(i) for i in range(101, 115)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
precipitation_calibration = precip_df[zone1_cols].mean(axis=1).values

print(f"率定用降雨数据: {precip_file}")
print(f"  Zone 1子分区: {len(zone1_cols)}个")
print(f"  时间步数: {len(precipitation_calibration)}")
print(f"  降雨统计:")
print(f"    最小值: {precipitation_calibration.min():.2f} mm/h")
print(f"    最大值: {precipitation_calibration.max():.2f} mm/h")
print(f"    平均值: {precipitation_calibration.mean():.2f} mm/h")
print(f"    总降雨深度: {precipitation_calibration.sum() * dt_hours:.2f} mm")

# 检查是否有用于生成estimated_observations的降雨数据
estimated_precip_file = results_dir / "estimated_observations" / "zone_1_precipitation.csv"
if estimated_precip_file.exists():
    print(f"\n发现估计径流生成时的降雨数据: {estimated_precip_file}")
    estimated_precip_df = pd.read_csv(estimated_precip_file)
    if 'precipitation_mm_h' in estimated_precip_df.columns:
        precipitation_estimated = estimated_precip_df['precipitation_mm_h'].values
    elif 'precip' in estimated_precip_df.columns:
        precipitation_estimated = estimated_precip_df['precip'].values
    else:
        print(f"  ⚠️  未找到降雨列，可用列: {estimated_precip_df.columns.tolist()}")
        precipitation_estimated = None
else:
    print(f"\n未找到估计径流生成时的降雨数据")
    precipitation_estimated = None

# ============================================================================
# 2. 加载观测径流数据
# ============================================================================
print("\n2. 加载观测径流数据")
print("-" * 80)

obs_file = results_dir / "estimated_observations" / "zone_1_estimated_runoff.csv"
obs_df = pd.read_csv(obs_file)
observed_runoff = obs_df['discharge_m3s'].values
times = pd.to_datetime(obs_df['datetime'])

print(f"观测径流文件: {obs_file}")
print(f"  时间步数: {len(observed_runoff)}")
print(f"  流量统计:")
print(f"    最小值: {observed_runoff.min():.2f} m³/s")
print(f"    最大值: {observed_runoff.max():.2f} m³/s")
print(f"    平均值: {observed_runoff.mean():.2f} m³/s")

# 计算观测径流深度和径流系数
total_runoff_volume_m3 = observed_runoff.sum() * dt_hours * 3600
runoff_depth_obs_mm = (total_runoff_volume_m3 / (zone1_area_km2 * 1e6)) * 1000
runoff_coef_obs = runoff_depth_obs_mm / (precipitation_calibration.sum() * dt_hours)

print(f"\n  总径流体积: {total_runoff_volume_m3:.2e} m³")
print(f"  总径流深度: {runoff_depth_obs_mm:.2f} mm")
print(f"  径流系数: {runoff_coef_obs:.4f}")

# ============================================================================
# 3. 加载HBV模拟径流数据
# ============================================================================
print("\n3. 加载HBV模拟径流数据")
print("-" * 80)

calib_file = results_dir / "calibration" / "zone1_calibrated_runoff.csv"
if calib_file.exists():
    calib_df = pd.read_csv(calib_file)
    simulated_runoff = calib_df['simulated_m3s'].values

    print(f"HBV模拟径流文件: {calib_file}")
    print(f"  时间步数: {len(simulated_runoff)}")
    print(f"  流量统计:")
    print(f"    最小值: {simulated_runoff.min():.2f} m³/s")
    print(f"    最大值: {simulated_runoff.max():.2f} m³/s")
    print(f"    平均值: {simulated_runoff.mean():.2f} m³/s")

    # 计算HBV径流深度和径流系数
    total_runoff_volume_sim_m3 = simulated_runoff.sum() * dt_hours * 3600
    runoff_depth_sim_mm = (total_runoff_volume_sim_m3 / (zone1_area_km2 * 1e6)) * 1000
    runoff_coef_sim = runoff_depth_sim_mm / (precipitation_calibration.sum() * dt_hours)

    print(f"\n  总径流体积: {total_runoff_volume_sim_m3:.2e} m³")
    print(f"  总径流深度: {runoff_depth_sim_mm:.2f} mm")
    print(f"  径流系数: {runoff_coef_sim:.4f}")
else:
    print(f"  ✗ 未找到HBV模拟结果文件")
    simulated_runoff = None
    runoff_coef_sim = None

# ============================================================================
# 4. 对比分析
# ============================================================================
print("\n" + "=" * 80)
print("对比分析")
print("=" * 80)

print(f"\n【降雨数据】")
print(f"  流域面积: {zone1_area_km2:.2f} km²")
print(f"  时间步长: {dt_hours:.2f} 小时")
print(f"  总降雨深度: {precipitation_calibration.sum() * dt_hours:.2f} mm")
print(f"  平均降雨强度: {precipitation_calibration.mean():.2f} mm/h")

print(f"\n【观测径流】(EstimatedRunoffGenerator生成)")
print(f"  总径流深度: {runoff_depth_obs_mm:.2f} mm")
print(f"  径流系数: {runoff_coef_obs:.4f}")
print(f"  平均流量: {observed_runoff.mean():.2f} m³/s")
print(f"  峰值流量: {observed_runoff.max():.2f} m³/s")

if simulated_runoff is not None:
    print(f"\n【HBV模拟径流】(率定后)")
    print(f"  总径流深度: {runoff_depth_sim_mm:.2f} mm")
    print(f"  径流系数: {runoff_coef_sim:.4f}")
    print(f"  平均流量: {simulated_runoff.mean():.2f} m³/s")
    print(f"  峰值流量: {simulated_runoff.max():.2f} m³/s")

    print(f"\n【径流系数对比】")
    print(f"  观测: {runoff_coef_obs:.4f}")
    print(f"  模拟: {runoff_coef_sim:.4f}")
    print(f"  差异: {runoff_coef_obs - runoff_coef_sim:.4f} ({(runoff_coef_obs - runoff_coef_sim)/runoff_coef_obs*100:.1f}%)")

    if abs(runoff_coef_obs - runoff_coef_sim) / runoff_coef_obs > 0.5:
        print(f"\n  ⚠️  径流系数差异超过50%！")
        print(f"     HBV模型产流严重不足！")
    elif abs(runoff_coef_obs - runoff_coef_sim) / runoff_coef_obs > 0.2:
        print(f"\n  ⚠️  径流系数差异超过20%")
        print(f"     HBV模型产流不匹配")
    else:
        print(f"\n  ✓ 径流系数差异在合理范围内")

    print(f"\n【流量对比】")
    print(f"  平均流量比: {observed_runoff.mean() / simulated_runoff.mean():.2f}x")
    print(f"  峰值流量比: {observed_runoff.max() / simulated_runoff.max():.2f}x")

    if observed_runoff.mean() / simulated_runoff.mean() > 10:
        print(f"\n  ✗ HBV模拟流量严重偏小（平均流量只有观测的{1/(observed_runoff.mean() / simulated_runoff.mean())*100:.1f}%）")
    elif observed_runoff.mean() / simulated_runoff.mean() > 2:
        print(f"\n  ⚠️  HBV模拟流量偏小")

# ============================================================================
# 5. 检查降雨数据一致性
# ============================================================================
print(f"\n" + "=" * 80)
print("降雨数据一致性检查")
print("=" * 80)

if precipitation_estimated is not None and len(precipitation_estimated) == len(precipitation_calibration):
    diff = precipitation_estimated - precipitation_calibration
    max_diff = np.abs(diff).max()
    mean_diff = np.abs(diff).mean()

    print(f"\n比较生成estimated_observations时的降雨 vs 率定时的降雨:")
    print(f"  最大差异: {max_diff:.4f} mm/h")
    print(f"  平均差异: {mean_diff:.4f} mm/h")
    print(f"  相对差异: {mean_diff / precipitation_calibration.mean() * 100:.2f}%")

    if max_diff < 0.01:
        print(f"\n  ✓ 降雨数据完全一致")
    elif mean_diff / precipitation_calibration.mean() < 0.01:
        print(f"\n  ✓ 降雨数据基本一致（差异<1%）")
    else:
        print(f"\n  ⚠️  降雨数据不一致！")
        print(f"     这可能导致率定结果不准确")
else:
    print(f"\n  ℹ️  无法比较（未找到生成时的降雨数据或长度不匹配）")

# ============================================================================
# 6. 结论
# ============================================================================
print(f"\n" + "=" * 80)
print("问题诊断")
print("=" * 80)

if simulated_runoff is not None:
    # 诊断问题
    rc_ratio = runoff_coef_sim / runoff_coef_obs
    flow_ratio = simulated_runoff.mean() / observed_runoff.mean()

    print(f"\n1. 径流系数诊断:")
    print(f"   观测Rc = {runoff_coef_obs:.4f} (EstimatedRunoffGenerator生成)")
    print(f"   模拟Rc = {runoff_coef_sim:.4f} (HBV模型)")
    print(f"   比值   = {rc_ratio:.4f}")

    if rc_ratio < 0.3:
        print(f"\n   ✗ 严重问题: HBV产流只有观测的{rc_ratio*100:.1f}%")
        print(f"     可能原因:")
        print(f"       - FC参数太小（土壤容量不足）")
        print(f"       - 初始土壤湿度太低")
        print(f"       - PERC参数太大（渗漏太快）")
        print(f"       - K0/K1参数太小（出流太慢）")
    elif rc_ratio < 0.7:
        print(f"\n   ⚠️  问题: HBV产流偏低")
    else:
        print(f"\n   ✓ HBV产流基本正常")

    print(f"\n2. 流量尺度诊断:")
    print(f"   观测平均流量 = {observed_runoff.mean():.2f} m³/s")
    print(f"   模拟平均流量 = {simulated_runoff.mean():.2f} m³/s")
    print(f"   比值         = {flow_ratio:.4f}")

    if flow_ratio < 0.1:
        print(f"\n   ✗ 严重尺度问题: 模拟流量只有观测的{flow_ratio*100:.1f}%")
        print(f"     HBV模型可能存在结构性问题")
    elif flow_ratio < 0.5:
        print(f"\n   ⚠️  尺度偏差: 需要调整HBV参数")
    else:
        print(f"\n   ✓ 流量尺度基本合理")

    print(f"\n3. 建议:")
    if rc_ratio < 0.3 or flow_ratio < 0.1:
        print(f"   1. 检查HBV模型实现是否正确")
        print(f"   2. 扩大参数搜索范围:")
        print(f"      - FC: 100-500 (增加土壤容量)")
        print(f"      - K0: 0.1-0.5 (增大快速径流)")
        print(f"      - K1: 0.05-0.3 (增大中速径流)")
        print(f"      - PERC: 0.5-3.0 (减小渗漏)")
        print(f"   3. 调整初始状态:")
        print(f"      - initial_soil: 更高的初始湿度")
        print(f"      - initial_upper/lower: 更高的初始储量")
    elif rc_ratio < 0.7:
        print(f"   1. 微调参数搜索范围")
        print(f"   2. 增加率定迭代次数")
    else:
        print(f"   ✓ HBV参数基本合理，可以继续优化")

print(f"\n" + "=" * 80)
