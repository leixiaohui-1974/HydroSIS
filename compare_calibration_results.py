#!/usr/bin/env python3
"""
对比简单观测 vs 增强观测的HBV率定结果

分析为什么增强型观测数据的率定效果仍然不理想
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.evaluation.metrics import (
    nash_sutcliffe_efficiency,
    kling_gupta_efficiency,
    pearson_correlation,
)

print("=" * 80)
print("对比简单观测 vs 增强观测的HBV率定结果")
print("=" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")

# ============================================================================
# 加载增强型观测的率定结果
# ============================================================================
print("\n加载增强型观测的率定结果...")
enhanced_file = results_dir / "calibration_enhanced" / "zone1_calibrated_runoff.csv"
if enhanced_file.exists():
    enhanced_df = pd.read_csv(enhanced_file)
    enhanced_obs = enhanced_df['observed_m3s'].values
    enhanced_sim = enhanced_df['simulated_m3s'].values
    enhanced_nse = nash_sutcliffe_efficiency(enhanced_sim, enhanced_obs)
    enhanced_kge = kling_gupta_efficiency(enhanced_sim, enhanced_obs)
    enhanced_corr = pearson_correlation(enhanced_sim, enhanced_obs)
    print(f"  ✓ NSE: {enhanced_nse:.4f}")
    print(f"  ✓ KGE: {enhanced_kge:.4f}")
    print(f"  ✓ Correlation: {enhanced_corr:.4f}")
else:
    print(f"  ✗ 未找到增强型观测率定结果")
    sys.exit(1)

# ============================================================================
# 分析不同阶段的性能
# ============================================================================
print("\n分段分析 (每30个时间步):")
print("-" * 80)

n_total = len(enhanced_obs)
segment_size = 30

for i in range(0, n_total, segment_size):
    end_idx = min(i + segment_size, n_total)
    obs_seg = enhanced_obs[i:end_idx]
    sim_seg = enhanced_sim[i:end_idx]

    if len(obs_seg) < 5:
        continue

    seg_nse = nash_sutcliffe_efficiency(sim_seg, obs_seg)
    seg_corr = pearson_correlation(sim_seg, obs_seg)
    seg_bias = (sim_seg.mean() - obs_seg.mean()) / obs_seg.mean() * 100

    print(f"\n时间步 {i:3d}-{end_idx:3d}:")
    print(f"  NSE: {seg_nse:>7.3f}")
    print(f"  相关系数: {seg_corr:>7.3f}")
    print(f"  平均偏差: {seg_bias:>7.1f}%")
    print(f"  观测均值: {obs_seg.mean():>7.2f} m³/s")
    print(f"  模拟均值: {sim_seg.mean():>7.2f} m³/s")

# ============================================================================
# 分析峰值响应
# ============================================================================
print("\n峰值响应分析:")
print("-" * 80)

obs_peak_idx = np.argmax(enhanced_obs)
sim_peak_idx = np.argmax(enhanced_sim)

print(f"观测峰值: {enhanced_obs[obs_peak_idx]:.2f} m³/s (时间步 {obs_peak_idx})")
print(f"模拟峰值: {enhanced_sim[sim_peak_idx]:.2f} m³/s (时间步 {sim_peak_idx})")
print(f"峰值延迟: {sim_peak_idx - obs_peak_idx} 小时")
print(f"峰值比率: {enhanced_sim[sim_peak_idx] / enhanced_obs[obs_peak_idx]:.2f}")

# ============================================================================
# 分析退水特性
# ============================================================================
print("\n退水特性分析 (后40个时间步):")
print("-" * 80)

recession_start = n_total - 40
obs_recession = enhanced_obs[recession_start:]
sim_recession = enhanced_sim[recession_start:]

# 计算退水系数 (ln(Q_t+1/Q_t))
def calc_recession_coef(q_series):
    """计算平均退水系数"""
    ratios = []
    for i in range(len(q_series) - 1):
        if q_series[i] > 0 and q_series[i+1] > 0:
            ratios.append(np.log(q_series[i+1] / q_series[i]))
    return np.mean(ratios) if ratios else 0

obs_recession_coef = calc_recession_coef(obs_recession)
sim_recession_coef = calc_recession_coef(sim_recession)

print(f"观测退水系数: {obs_recession_coef:.4f}")
print(f"模拟退水系数: {sim_recession_coef:.4f}")
print(f"退水差异: {(sim_recession_coef - obs_recession_coef):.4f}")

if sim_recession_coef < obs_recession_coef:
    print("  -> HBV退水过快（流量下降太快）")
else:
    print("  -> HBV退水过慢（流量下降太慢）")

# ============================================================================
# 可视化详细对比
# ============================================================================
print("\n生成详细对比图...")
print("-" * 80)

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 1. 完整时间序列
ax1 = axes[0]
ax1.plot(enhanced_obs, 'b-', label='Enhanced Observation', linewidth=2)
ax1.plot(enhanced_sim, 'r--', label='HBV Simulation', linewidth=1.5)
ax1.axvline(x=obs_peak_idx, color='blue', linestyle=':', alpha=0.5, label='Obs Peak')
ax1.axvline(x=sim_peak_idx, color='red', linestyle=':', alpha=0.5, label='Sim Peak')
ax1.set_ylabel('Discharge (m³/s)', fontsize=11)
ax1.set_title(f'Full Time Series (NSE={enhanced_nse:.4f})', fontsize=12, weight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. 残差分析
ax2 = axes[1]
residuals = enhanced_obs - enhanced_sim
ax2.plot(residuals, 'g-', linewidth=1)
ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
ax2.fill_between(range(len(residuals)), 0, residuals, alpha=0.3, color='green')
ax2.set_ylabel('Residual (Obs - Sim)', fontsize=11)
ax2.set_title(f'Residuals (Mean={residuals.mean():.2f}, Std={residuals.std():.2f})', fontsize=12)
ax2.grid(True, alpha=0.3)

# 3. 累积流量对比
ax3 = axes[2]
obs_cumsum = np.cumsum(enhanced_obs)
sim_cumsum = np.cumsum(enhanced_sim)
ax3.plot(obs_cumsum, 'b-', label='Observed Cumulative', linewidth=2)
ax3.plot(sim_cumsum, 'r--', label='Simulated Cumulative', linewidth=2)
ax3.set_xlabel('Time Step (hour)', fontsize=11)
ax3.set_ylabel('Cumulative Discharge (m³/s·h)', fontsize=11)
ax3.set_title('Cumulative Flow Comparison', fontsize=12)
ax3.legend(loc='upper left', fontsize=9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
output_file = results_dir / "calibration_enhanced" / "detailed_diagnosis.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"  ✓ 保存: {output_file}")
plt.close()

# ============================================================================
# 总结和诊断
# ============================================================================
print("\n" + "=" * 80)
print("诊断总结")
print("=" * 80)

print(f"\n【问题识别】")

# 初期响应问题
initial_bias = (enhanced_sim[:10].mean() - enhanced_obs[:10].mean()) / enhanced_obs[:10].mean() * 100
if abs(initial_bias) > 20:
    print(f"  ⚠ 初期响应偏差大: {initial_bias:+.1f}%")
    if initial_bias > 0:
        print(f"     -> HBV初期流量过高，可能是初始状态或快速径流参数问题")
    else:
        print(f"     -> HBV初期流量过低，可能是产流不足")

# 峰值问题
peak_bias = (enhanced_sim[sim_peak_idx] - enhanced_obs[obs_peak_idx]) / enhanced_obs[obs_peak_idx] * 100
if abs(peak_bias) > 15:
    print(f"  ⚠ 峰值偏差: {peak_bias:+.1f}%")

# 退水问题
if abs(sim_recession_coef - obs_recession_coef) > 0.01:
    print(f"  ⚠ 退水特性不匹配:")
    print(f"     观测退水系数: {obs_recession_coef:.4f}")
    print(f"     模拟退水系数: {sim_recession_coef:.4f}")

print(f"\n【可能原因】")
print(f"  1. 增强型生成器和HBV虽然都有土壤水分和多层水库，但参数化方式不同")
print(f"  2. 增强型生成器使用线性水库，HBV使用非线性退水")
print(f"  3. 产流机制仍有差异：增强型基于土壤饱和度，HBV有更复杂的BETA参数")
print(f"  4. 初始状态设置可能不合理")

print(f"\n【建议改进】")
print(f"  1. 调整增强型生成器的水库参数，使退水曲线更接近HBV")
print(f"  2. 添加更多非线性特性到增强型生成器")
print(f"  3. 使用更长的时间序列进行率定（当前仅120小时）")
print(f"  4. 考虑使用实际观测数据而非合成数据")

print("\n" + "=" * 80)
