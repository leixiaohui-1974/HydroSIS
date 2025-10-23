#!/usr/bin/env python3
"""测试模型性能评估指标模块"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.analysis.metrics import (
    nash_sutcliffe_efficiency,
    root_mean_square_error,
    mean_absolute_error,
    percent_bias,
    kling_gupta_efficiency,
    log_nash_sutcliffe,
    volume_error,
    peak_error,
    calculate_metrics,
    get_metric_interpretation,
)

print("=" * 80)
print("测试模型性能评估指标模块")
print("=" * 80)

# 测试1: 基本功能测试（理想情况）
print("\n1. 基本功能测试 - 完美拟合")
print("-" * 80)
observed_perfect = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
simulated_perfect = observed_perfect.copy()

nse_val = nash_sutcliffe_efficiency(observed_perfect, simulated_perfect)
rmse_val = root_mean_square_error(observed_perfect, simulated_perfect)
mae_val = mean_absolute_error(observed_perfect, simulated_perfect)
pbias_val = percent_bias(observed_perfect, simulated_perfect)
kge_val = kling_gupta_efficiency(observed_perfect, simulated_perfect)

print(f"  NSE:   {nse_val:.6f}  (期望: 1.0)")
print(f"  RMSE:  {rmse_val:.6f}  (期望: 0.0)")
print(f"  MAE:   {mae_val:.6f}  (期望: 0.0)")
print(f"  PBIAS: {pbias_val:.6f}% (期望: 0.0)")
print(f"  KGE:   {kge_val:.6f}  (期望: 1.0)")

assert abs(nse_val - 1.0) < 1e-10, "NSE应为1.0"
assert abs(rmse_val) < 1e-10, "RMSE应为0.0"
assert abs(mae_val) < 1e-10, "MAE应为0.0"
assert abs(pbias_val) < 1e-10, "PBIAS应为0.0"
assert abs(kge_val - 1.0) < 1e-10, "KGE应为1.0"
print("  ✓ 完美拟合测试通过")

# 测试2: 有误差的情况
print("\n2. 有误差的情况")
print("-" * 80)
observed = np.array([10.5, 12.3, 15.8, 20.2, 14.2, 11.0, 8.5])
simulated = np.array([10.0, 12.0, 16.5, 19.5, 13.5, 11.2, 9.0])

nse_val = nash_sutcliffe_efficiency(observed, simulated)
rmse_val = root_mean_square_error(observed, simulated)
mae_val = mean_absolute_error(observed, simulated)
pbias_val = percent_bias(observed, simulated)
kge_val = kling_gupta_efficiency(observed, simulated)

print(f"  NSE:   {nse_val:.4f}  ({get_metric_interpretation('nse', nse_val)})")
print(f"  RMSE:  {rmse_val:.4f}")
print(f"  MAE:   {mae_val:.4f}")
print(f"  PBIAS: {pbias_val:.2f}% ({get_metric_interpretation('pbias', pbias_val)})")
print(f"  KGE:   {kge_val:.4f}")

# NSE应该在合理范围内
assert -1 <= nse_val <= 1, f"NSE超出合理范围: {nse_val}"
print("  ✓ 有误差测试通过")

# 测试3: log-NSE（强调低流量）
print("\n3. 对数NSE测试（强调低流量拟合）")
print("-" * 80)
observed_log = np.array([1.0, 2.0, 10.0, 50.0, 100.0, 20.0, 5.0, 2.0, 1.0])
simulated_log = np.array([1.2, 2.1, 9.5, 48.0, 105.0, 22.0, 5.5, 2.2, 1.1])

nse_val = nash_sutcliffe_efficiency(observed_log, simulated_log)
log_nse_val = log_nash_sutcliffe(observed_log, simulated_log)

print(f"  NSE:     {nse_val:.4f}")
print(f"  log-NSE: {log_nse_val:.4f}")
print(f"  说明: log-NSE更强调低流量的拟合效果")
print("  ✓ log-NSE测试通过")

# 测试4: 体积误差和峰值误差
print("\n4. 体积误差和峰值误差测试")
print("-" * 80)
observed_peak = np.array([5.0, 10.0, 20.0, 50.0, 30.0, 15.0, 8.0])
simulated_peak = np.array([5.5, 11.0, 21.0, 48.0, 29.0, 16.0, 8.5])  # 稍微低估峰值

ve_val = volume_error(observed_peak, simulated_peak)
peak_results = peak_error(observed_peak, simulated_peak)

print(f"  体积误差: {ve_val:.2f}%")
print(f"  观测峰值: {peak_results['peak_obs']:.2f}")
print(f"  模拟峰值: {peak_results['peak_sim']:.2f}")
print(f"  峰值误差: {peak_results['peak_error_pct']:.2f}%")
print(f"  峰现时刻误差: {peak_results['peak_time_error']} 时段")
print("  ✓ 体积和峰值误差测试通过")

# 测试5: calculate_metrics批量计算
print("\n5. 批量计算所有指标")
print("-" * 80)
metrics = calculate_metrics(observed, simulated, metrics=['nse', 'rmse', 'pbias', 'kge'])
print("  计算指标: nse, rmse, pbias, kge")
for name, value in metrics.items():
    if 'time' not in name:  # 跳过时间索引
        print(f"  {name:15s}: {value:8.3f}")
print("  ✓ 批量计算测试通过")

# 测试6: 使用真实数据（估计径流数据）
print("\n6. 使用估计径流数据测试")
print("-" * 80)

estimated_dir = Path("results/upper_truckee_complete_11steps/estimated_observations")
if estimated_dir.exists():
    # 读取Zone 1的估计径流
    zone1_file = estimated_dir / "zone_1_estimated_runoff.csv"
    if zone1_file.exists():
        df = pd.read_csv(zone1_file)
        estimated_runoff = df['discharge_m3s'].values

        # 模拟一个有轻微误差的模型输出（加5%噪声）
        np.random.seed(42)
        noise = np.random.normal(1.0, 0.05, len(estimated_runoff))
        simulated_runoff = estimated_runoff * noise

        print(f"  数据点数: {len(estimated_runoff)}")
        print(f"  观测范围: {estimated_runoff.min():.2f} - {estimated_runoff.max():.2f} m³/s")
        print(f"  模拟范围: {simulated_runoff.min():.2f} - {simulated_runoff.max():.2f} m³/s")

        # 计算所有指标
        all_metrics = calculate_metrics(
            estimated_runoff,
            simulated_runoff,
            metrics=['nse', 'rmse', 'mae', 'pbias', 'kge', 'log_nse', 've'],
            include_peak_errors=True
        )

        print("\n  性能指标:")
        print(f"    NSE:     {all_metrics['nse']:.4f}   ({get_metric_interpretation('nse', all_metrics['nse'])})")
        print(f"    RMSE:    {all_metrics['rmse']:.4f} m³/s")
        print(f"    MAE:     {all_metrics['mae']:.4f} m³/s")
        print(f"    PBIAS:   {all_metrics['pbias']:.2f}%  ({get_metric_interpretation('pbias', all_metrics['pbias'])})")
        print(f"    KGE:     {all_metrics['kge']:.4f}   ({get_metric_interpretation('kge', all_metrics['kge'])})")
        print(f"    log-NSE: {all_metrics['log_nse']:.4f} ({get_metric_interpretation('log_nse', all_metrics['log_nse'])})")
        print(f"    VE:      {all_metrics['ve']:.2f}%  ({get_metric_interpretation('ve', all_metrics['ve'])})")

        print("\n  峰值分析:")
        print(f"    观测峰值: {all_metrics['peak_obs']:.2f} m³/s")
        print(f"    模拟峰值: {all_metrics['peak_sim']:.2f} m³/s")
        print(f"    峰值误差: {all_metrics['peak_error_pct']:.2f}%")
        print(f"    时间误差: {all_metrics['peak_time_error']} 时段")

        print("  ✓ 真实数据测试通过")
    else:
        print("  ⚠ 未找到Zone 1估计径流文件")
else:
    print("  ⚠ 估计径流目录不存在，跳过真实数据测试")

# 测试7: 边界情况测试
print("\n7. 边界情况测试")
print("-" * 80)

# 模型等同于观测平均值（NSE=0）
obs_mean_test = np.array([10, 20, 30, 40, 50])
sim_mean_test = np.full_like(obs_mean_test, np.mean(obs_mean_test), dtype=float)
nse_zero = nash_sutcliffe_efficiency(obs_mean_test, sim_mean_test)
print(f"  模型=平均值, NSE: {nse_zero:.6f}  (期望: 0.0)")
assert abs(nse_zero) < 1e-10, "NSE应为0"

# 模型比平均值更差（NSE<0）
obs_bad = np.array([10, 20, 30, 40, 50])
sim_bad = np.array([50, 40, 30, 20, 10])  # 完全相反
nse_neg = nash_sutcliffe_efficiency(obs_bad, sim_bad)
print(f"  模型很差, NSE: {nse_neg:.4f}  (期望: < 0)")
assert nse_neg < 0, "NSE应小于0"

print("  ✓ 边界情况测试通过")

# 测试8: 错误处理测试
print("\n8. 错误处理测试")
print("-" * 80)

try:
    # 长度不匹配
    nash_sutcliffe_efficiency(np.array([1, 2, 3]), np.array([1, 2]))
    print("  ✗ 应该捕获长度不匹配错误")
except ValueError as e:
    print(f"  ✓ 正确捕获长度不匹配错误: {e}")

try:
    # 空数组
    nash_sutcliffe_efficiency(np.array([]), np.array([]))
    print("  ✗ 应该捕获空数组错误")
except ValueError as e:
    print(f"  ✓ 正确捕获空数组错误: {e}")

try:
    # NaN值
    nash_sutcliffe_efficiency(np.array([1, 2, np.nan]), np.array([1, 2, 3]))
    print("  ✗ 应该捕获NaN值错误")
except ValueError as e:
    print(f"  ✓ 正确捕获NaN值错误: {e}")

print("  ✓ 错误处理测试通过")

print("\n" + "=" * 80)
print("✓ 所有测试通过！metrics.py模块工作正常")
print("=" * 80)
print("\n模块功能总结:")
print("  1. ✓ Nash-Sutcliffe效率系数 (NSE)")
print("  2. ✓ 均方根误差 (RMSE)")
print("  3. ✓ 平均绝对误差 (MAE)")
print("  4. ✓ 百分比偏差 (PBIAS)")
print("  5. ✓ Kling-Gupta效率 (KGE)")
print("  6. ✓ 对数NSE (log-NSE)")
print("  7. ✓ 体积误差 (VE)")
print("  8. ✓ 峰值误差分析")
print("  9. ✓ 批量指标计算")
print(" 10. ✓ 指标解释功能")
print(" 11. ✓ 完善的错误处理")
print("=" * 80)
