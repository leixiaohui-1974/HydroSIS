#!/usr/bin/env python3
"""
HBV模型配置诊断脚本

根据敏感性分析诊断报告的建议，全面检查：
1. 单位转换是否正确（mm/h ↔ m³/s）
2. 水量平衡是否合理（降雨总量 vs 径流总量）
3. 径流系数是否在合理范围（0.3-0.7）
4. 初始状态估计是否合理
5. HBV参数设置的合理性

遵循 .claude/AI_DEVELOPMENT_GUIDE.md 中的最佳实践。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ✅ 使用基础库的功能模块
from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator

print("=" * 80)
print("HBV模型配置诊断脚本")
print("=" * 80)

# ============================================================================
# 步骤 1: 加载数据
# ============================================================================
print("\n步骤 1: 加载数据")
print("-" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")

# 1.1 加载降雨数据
precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
precip_df = pd.read_csv(precip_file, index_col=0)

# Zone 1的子分区
zone1_subbasins = [str(i) for i in range(10, 24)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
precipitation_mmh = precip_df[zone1_cols].mean(axis=1).values

print(f"✓ 降雨数据加载:")
print(f"  文件: {precip_file.name}")
print(f"  Zone 1子分区: {len(zone1_cols)}个")
print(f"  时间步数: {len(precipitation_mmh)}")
print(f"  单位: mm/h（假设）")
print(f"  范围: {precipitation_mmh.min():.2f} - {precipitation_mmh.max():.2f} mm/h")

# 1.2 加载增强型观测径流
obs_file = results_dir / "enhanced_observations" / "zone_1_enhanced_runoff.csv"
obs_df = pd.read_csv(obs_file)
observed_m3s = obs_df['discharge_m3s'].values
times = pd.to_datetime(obs_df['datetime'])

print(f"\n✓ 观测径流数据加载:")
print(f"  文件: {obs_file.name}")
print(f"  时间步数: {len(observed_m3s)}")
print(f"  单位: m³/s（从文件名推断）")
print(f"  范围: {observed_m3s.min():.2f} - {observed_m3s.max():.2f} m³/s")

# 1.3 流域面积
zone1_area_km2 = 139.995

print(f"\n✓ Zone 1流域面积: {zone1_area_km2:.2f} km²")

# ============================================================================
# 步骤 2: 单位转换检查
# ============================================================================
print("\n步骤 2: 单位转换检查")
print("-" * 80)

# 2.1 检查降雨单位：mm/h → m³/s
timestep_hours = 1.0  # 假设1小时步长

# 验证时间步长
time_diffs = (times[1:] - times[:-1]).dt.total_seconds() / 3600
if len(set(time_diffs)) == 1:
    timestep_hours = time_diffs[0]
    print(f"✓ 时间步长: {timestep_hours:.2f} 小时（一致）")
else:
    print(f"⚠ 时间步长不一致: {set(time_diffs)}")
    timestep_hours = time_diffs[0]  # 使用第一个

# 降雨单位转换
print(f"\n降雨单位转换验证:")
print(f"  假设: 降雨单位为 mm/h")

# mm/h → m³/s 转换公式
# Volume (m³) = Precip (mm) * Area (km²) * 1000 (m²/km²) / 1000 (mm/m)
#             = Precip (mm) * Area (km²)
# Flow (m³/s) = Volume (m³) / Time (s)
#             = Precip (mm) * Area (km²) / (timestep_hours * 3600)

def precip_mmh_to_m3s(precip_mmh, area_km2, timestep_hours):
    """降雨从 mm/h 转换到 m³/s"""
    # mm/h * km² → m³/s
    # 1 mm/h over 1 km² = 1000 m³/h = 1000/3600 m³/s ≈ 0.2778 m³/s
    return precip_mmh * area_km2 * 1000 / 3600

sample_precip = 10.0  # mm/h
sample_flow = precip_mmh_to_m3s(sample_precip, zone1_area_km2, timestep_hours)
print(f"  示例: {sample_precip:.1f} mm/h over {zone1_area_km2:.1f} km²")
print(f"       → {sample_flow:.2f} m³/s")

# 转换整个降雨序列
precipitation_m3s = precip_mmh_to_m3s(precipitation_mmh, zone1_area_km2, timestep_hours)

print(f"\n  转换后的降雨流量范围: {precipitation_m3s.min():.2f} - {precipitation_m3s.max():.2f} m³/s")
print(f"  观测径流范围:         {observed_m3s.min():.2f} - {observed_m3s.max():.2f} m³/s")

if precipitation_m3s.max() < observed_m3s.max():
    print(f"  ⚠ 警告: 最大降雨流量 < 最大观测流量")
    print(f"     这是合理的（考虑土壤蓄水和前期降雨）")

# ============================================================================
# 步骤 3: 水量平衡分析
# ============================================================================
print("\n步骤 3: 水量平衡分析")
print("-" * 80)

# 3.1 计算累积水量
total_precip_mm = precipitation_mmh.sum() * timestep_hours  # mm
total_precip_volume_m3 = total_precip_mm * zone1_area_km2 * 1000  # m³

total_runoff_volume_m3 = observed_m3s.sum() * timestep_hours * 3600  # m³
total_runoff_mm = total_runoff_volume_m3 / (zone1_area_km2 * 1000)  # mm

# 径流系数
runoff_coefficient = total_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0

print(f"累积水量:")
print(f"  总降雨量:     {total_precip_mm:>10.2f} mm  ({total_precip_volume_m3:>15.2f} m³)")
print(f"  总径流量:     {total_runoff_mm:>10.2f} mm  ({total_runoff_volume_m3:>15.2f} m³)")
print(f"  径流系数:     {runoff_coefficient:>10.4f} (= 径流量/降雨量)")

# 评估径流系数合理性
print(f"\n径流系数合理性评估:")
if 0.3 <= runoff_coefficient <= 0.7:
    print(f"  ✓ 径流系数在合理范围内 [0.3, 0.7]")
elif runoff_coefficient < 0.3:
    print(f"  ⚠ 径流系数偏低 (< 0.3)，可能原因:")
    print(f"     - 土壤蓄水能力强")
    print(f"     - 蒸散发量大")
    print(f"     - 观测径流低估")
elif runoff_coefficient > 0.7:
    print(f"  ⚠ 径流系数偏高 (> 0.7)，可能原因:")
    print(f"     - 土壤已饱和或不透水面积大")
    print(f"     - 观测径流高估")
    print(f"     - 单位转换可能有误")
else:
    print(f"  ✗ 径流系数 > 1.0，说明配置有严重问题！")
    print(f"     - 检查单位转换")
    print(f"     - 检查面积计算")

# 蓄水量估算
storage_change_mm = total_precip_mm - total_runoff_mm
print(f"\n蓄水量变化估算:")
print(f"  ΔS = P - Q = {storage_change_mm:>10.2f} mm")
if storage_change_mm > 0:
    print(f"  → 土壤蓄水增加（正常，因为降雨输入）")
elif storage_change_mm < 0:
    print(f"  → 土壤蓄水减少（可能前期有蓄水，或基流持续）")

# ============================================================================
# 步骤 4: 初始状态合理性检查
# ============================================================================
print("\n步骤 4: 初始状态合理性检查")
print("-" * 80)

# 4.1 从观测基流反推初始状态
initial_baseflow = observed_m3s[0]
print(f"观测初始流量: {initial_baseflow:.2f} m³/s")

# HBV基流计算: Q_base = K2 * S_lower
# 因此: S_lower = Q_base / K2
assumed_K2 = 0.02  # HBV典型K2值
estimated_initial_lower = initial_baseflow / assumed_K2

print(f"\n根据基流反推初始下层储量:")
print(f"  假设 K2 = {assumed_K2:.3f}")
print(f"  Q_base = K2 * S_lower")
print(f"  S_lower = Q_base / K2")
print(f"         = {initial_baseflow:.2f} / {assumed_K2:.3f}")
print(f"         = {estimated_initial_lower:.2f} mm")

# 4.2 检查之前使用的初始状态
print(f"\n之前率定中使用的初始状态:")
print(f"  initial_soil:  300.0 mm")
print(f"  initial_upper: 20.0 mm")
print(f"  initial_lower: 3000.0 mm  ← 注意：远高于反推值！")
print(f"  initial_snow:  0.0 mm")

print(f"\n诊断:")
if estimated_initial_lower > 1000:
    print(f"  ⚠ 反推的初始下层储量 ({estimated_initial_lower:.1f} mm) 很大")
    print(f"     可能说明初始基流很高，或K2设置不当")
else:
    print(f"  ✓ 反推的初始下层储量 ({estimated_initial_lower:.1f} mm) 在合理范围")

if abs(3000.0 - estimated_initial_lower) / estimated_initial_lower > 0.5:
    print(f"  ✗ 之前使用的initial_lower (3000 mm) 与反推值相差 > 50%")
    print(f"     建议使用: {estimated_initial_lower:.1f} mm")
else:
    print(f"  ✓ 之前使用的initial_lower与反推值接近")

# ============================================================================
# 步骤 5: HBV模型水量平衡测试
# ============================================================================
print("\n步骤 5: HBV模型水量平衡测试")
print("-" * 80)

# 5.1 使用默认参数运行HBV
print(f"\n运行HBV模型（默认参数）...")

class MockSubbasin:
    def __init__(self, area_km2):
        self.area_km2 = area_km2

subbasin = MockSubbasin(zone1_area_km2)

default_params = {
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
    'initial_lower': estimated_initial_lower,  # 使用反推值
    'initial_snow': 0.0,
}

print(f"  参数设置:")
for key in ['FC', 'BETA', 'K0', 'K1', 'K2', 'PERC']:
    print(f"    {key:<10s}: {default_params[key]:>8.3f}")
print(f"  初始状态:")
for key in ['initial_soil', 'initial_upper', 'initial_lower', 'initial_snow']:
    print(f"    {key:<15s}: {default_params[key]:>10.2f} mm")

temperature = np.linspace(5, 15, len(precipitation_mmh))

hbv = HBVRunoff(default_params)
hbv_runoff_m3s = np.array(hbv.simulate(subbasin, precipitation_mmh.tolist()))

print(f"\n✓ HBV模拟完成")
print(f"  输出长度: {len(hbv_runoff_m3s)}")
print(f"  流量范围: {hbv_runoff_m3s.min():.2f} - {hbv_runoff_m3s.max():.2f} m³/s")

# 5.2 HBV水量平衡
hbv_total_runoff_m3 = hbv_runoff_m3s.sum() * timestep_hours * 3600
hbv_total_runoff_mm = hbv_total_runoff_m3 / (zone1_area_km2 * 1000)
hbv_runoff_coefficient = hbv_total_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0

print(f"\nHBV模型水量平衡:")
print(f"  输入降雨:     {total_precip_mm:>10.2f} mm")
print(f"  HBV径流:      {hbv_total_runoff_mm:>10.2f} mm")
print(f"  HBV径流系数:  {hbv_runoff_coefficient:>10.4f}")

# 对比
print(f"\n对比观测:")
print(f"  观测径流系数: {runoff_coefficient:>10.4f}")
print(f"  HBV径流系数:  {hbv_runoff_coefficient:>10.4f}")
print(f"  差异:         {abs(hbv_runoff_coefficient - runoff_coefficient):>10.4f}")

if abs(hbv_runoff_coefficient - runoff_coefficient) < 0.1:
    print(f"  ✓ HBV与观测的径流系数接近（差异 < 0.1）")
else:
    print(f"  ⚠ HBV与观测的径流系数差异较大")
    if hbv_runoff_coefficient > runoff_coefficient:
        print(f"     HBV产流偏多，可能原因:")
        print(f"       - FC (土壤容量)设置过小")
        print(f"       - BETA设置过大（产流过快）")
        print(f"       - 初始土壤湿度过高")
    else:
        print(f"     HBV产流偏少，可能原因:")
        print(f"       - FC设置过大")
        print(f"       - BETA设置过小")
        print(f"       - 蒸散发过大（但当前模型没有蒸散发）")

# ============================================================================
# 步骤 6: 增强型生成器参数检查
# ============================================================================
print("\n步骤 6: 增强型生成器参数检查")
print("-" * 80)

# 读取增强型生成器的配置
enhanced_config_file = results_dir / "enhanced_observations" / "zone_1_config.yaml"
if enhanced_config_file.exists():
    import yaml
    with open(enhanced_config_file, 'r') as f:
        enhanced_config = yaml.safe_load(f)

    print(f"✓ 增强型生成器配置:")
    params = enhanced_config.get('parameters', {})
    for key in ['soil_capacity', 'soil_beta', 'k_fast', 'k_inter', 'k_base']:
        if key in params:
            print(f"  {key:<20s}: {params[key]:>10.4f}")

    print(f"\n增强型生成器 vs HBV参数对应:")
    print(f"  {'增强型':<25s} {'HBV':<15s} {'对比':<30s}")
    print(f"  {'-'*25} {'-'*15} {'-'*30}")
    print(f"  soil_capacity={params.get('soil_capacity', 0):<13.1f} FC={default_params['FC']:<13.1f} "
          f"{'接近' if abs(params.get('soil_capacity', 0) - default_params['FC']) < 100 else '差异较大'}")
    print(f"  soil_beta={params.get('soil_beta', 0):<17.2f} BETA={default_params['BETA']:<11.2f} "
          f"{'接近' if abs(params.get('soil_beta', 0) - default_params['BETA']) < 1 else '差异较大'}")
    print(f"  k_fast={params.get('k_fast', 0):<20.3f} K0={default_params['K0']:<13.3f} "
          f"{'接近' if abs(params.get('k_fast', 0) - default_params['K0']) < 0.1 else '差异较大'}")
    print(f"  k_inter={params.get('k_inter', 0):<19.3f} K1={default_params['K1']:<13.3f} "
          f"{'接近' if abs(params.get('k_inter', 0) - default_params['K1']) < 0.1 else '差异较大'}")
    print(f"  k_base={params.get('k_base', 0):<20.3f} K2={default_params['K2']:<13.3f} "
          f"{'接近' if abs(params.get('k_base', 0) - default_params['K2']) < 0.1 else '差异较大'}")

else:
    print(f"⚠ 未找到增强型生成器配置文件: {enhanced_config_file}")

# ============================================================================
# 步骤 7: 时间序列可视化对比
# ============================================================================
print("\n步骤 7: 生成诊断可视化图表")
print("-" * 80)

output_dir = results_dir / "configuration_diagnosis"
output_dir.mkdir(parents=True, exist_ok=True)

# 7.1 水量平衡对比图
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1: 降雨和径流对比
ax1 = axes[0]
ax1_twin = ax1.twinx()
ax1_twin.bar(range(len(precipitation_mmh)), precipitation_mmh, alpha=0.3, color='blue', label='Precipitation')
ax1.plot(observed_m3s, 'o-', label='Observed Runoff', color='orange', linewidth=1.5)
ax1.plot(hbv_runoff_m3s, 's-', label='HBV Simulated', color='green', linewidth=1.5, alpha=0.7)
ax1.set_ylabel('Discharge (m³/s)', fontsize=10)
ax1_twin.set_ylabel('Precipitation (mm/h)', fontsize=10, color='blue')
ax1_twin.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')
ax1_twin.legend(loc='upper right')
ax1.set_title(f'Precipitation and Runoff Comparison\nRunoff Coeff: Obs={runoff_coefficient:.3f}, HBV={hbv_runoff_coefficient:.3f}',
              fontsize=11)
ax1.grid(True, alpha=0.3)

# 子图2: 累积水量对比
ax2 = axes[1]
cumulative_precip = np.cumsum(precipitation_mmh) * timestep_hours
cumulative_obs = np.cumsum(observed_m3s * timestep_hours * 3600 / (zone1_area_km2 * 1000))
cumulative_hbv = np.cumsum(hbv_runoff_m3s * timestep_hours * 3600 / (zone1_area_km2 * 1000))

ax2.plot(cumulative_precip, '-', label='Cumulative Precipitation', color='blue', linewidth=2)
ax2.plot(cumulative_obs, '-', label='Cumulative Observed Runoff', color='orange', linewidth=2)
ax2.plot(cumulative_hbv, '-', label='Cumulative HBV Runoff', color='green', linewidth=2)
ax2.set_ylabel('Cumulative Depth (mm)', fontsize=10)
ax2.set_title('Cumulative Water Balance', fontsize=11)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3: 残差分析
ax3 = axes[2]
residual_obs = observed_m3s - hbv_runoff_m3s
ax3.bar(range(len(residual_obs)), residual_obs, alpha=0.6, color='red', label='Observed - HBV')
ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
ax3.axhline(y=residual_obs.mean(), color='blue', linestyle='--', linewidth=1,
            label=f'Mean Residual = {residual_obs.mean():.2f} m³/s')
ax3.set_xlabel('Time Step (hour)', fontsize=10)
ax3.set_ylabel('Residual (m³/s)', fontsize=10)
ax3.set_title(f'Residual Analysis\nRMSE = {np.sqrt((residual_obs**2).mean()):.2f} m³/s', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
diagnosis_plot_file = output_dir / "water_balance_diagnosis.png"
plt.savefig(diagnosis_plot_file, dpi=300, bbox_inches='tight')
print(f"✓ 保存诊断图表: {diagnosis_plot_file.name}")

# ============================================================================
# 步骤 8: 生成诊断报告
# ============================================================================
print("\n步骤 8: 生成诊断报告")
print("-" * 80)

report_file = output_dir / "configuration_diagnosis_report.txt"

with open(report_file, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("HBV模型配置诊断报告\n")
    f.write("=" * 80 + "\n\n")

    f.write("【数据概览】\n")
    f.write(f"  降雨数据: {len(precipitation_mmh)} 小时\n")
    f.write(f"  降雨范围: {precipitation_mmh.min():.2f} - {precipitation_mmh.max():.2f} mm/h\n")
    f.write(f"  观测流量: {len(observed_m3s)} 小时\n")
    f.write(f"  流量范围: {observed_m3s.min():.2f} - {observed_m3s.max():.2f} m³/s\n")
    f.write(f"  流域面积: {zone1_area_km2:.2f} km²\n\n")

    f.write("【水量平衡】\n")
    f.write(f"  总降雨量:     {total_precip_mm:.2f} mm\n")
    f.write(f"  观测总径流:   {total_runoff_mm:.2f} mm\n")
    f.write(f"  HBV总径流:    {hbv_total_runoff_mm:.2f} mm\n")
    f.write(f"  观测径流系数: {runoff_coefficient:.4f}\n")
    f.write(f"  HBV径流系数:  {hbv_runoff_coefficient:.4f}\n")
    f.write(f"  径流系数差异: {abs(hbv_runoff_coefficient - runoff_coefficient):.4f}\n\n")

    f.write("【初始状态】\n")
    f.write(f"  观测初始流量:           {initial_baseflow:.2f} m³/s\n")
    f.write(f"  反推初始下层储量(K2={assumed_K2}): {estimated_initial_lower:.2f} mm\n")
    f.write(f"  之前使用的initial_lower:     3000.0 mm\n")
    f.write(f"  建议使用:                {estimated_initial_lower:.1f} mm\n\n")

    f.write("【诊断结论】\n")
    if 0.3 <= runoff_coefficient <= 0.7:
        f.write("  ✓ 径流系数在合理范围内\n")
    else:
        f.write(f"  ✗ 径流系数异常 ({runoff_coefficient:.3f})\n")

    if abs(hbv_runoff_coefficient - runoff_coefficient) < 0.1:
        f.write("  ✓ HBV与观测的径流系数接近\n")
    else:
        f.write(f"  ⚠ HBV与观测的径流系数差异较大 (Δ={abs(hbv_runoff_coefficient - runoff_coefficient):.3f})\n")

    f.write("\n【建议】\n")
    f.write("  1. 使用反推的初始下层储量替代固定值3000mm\n")
    f.write("  2. 如果径流系数差异大，调整FC和BETA参数\n")
    f.write("  3. 获取更长时间序列（30-90天）以充分识别参数\n")
    f.write("  4. 验证增强型生成器参数与HBV参数的一致性\n")

print(f"✓ 保存诊断报告: {report_file.name}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 80)
print("HBV模型配置诊断完成!")
print("=" * 80)

print(f"\n【关键发现】")
print(f"  径流系数(观测): {runoff_coefficient:.4f} {'✓ 合理' if 0.3 <= runoff_coefficient <= 0.7 else '⚠ 异常'}")
print(f"  径流系数(HBV):  {hbv_runoff_coefficient:.4f}")
print(f"  初始状态建议:   initial_lower = {estimated_initial_lower:.1f} mm (当前使用3000mm)")

print(f"\n【输出文件】")
print(f"  诊断报告: {report_file}")
print(f"  诊断图表: {diagnosis_plot_file}")

print(f"\n【下一步】")
print(f"  1. 根据诊断结果调整HBV初始状态")
print(f"  2. 对比简单模型/增强模型/HBV的兼容性")
print(f"  3. 获取更长时间序列数据（当前{len(precipitation_mmh)}小时不足）")

print("\n" + "=" * 80)
print("✓ 本脚本遵循 .claude/AI_DEVELOPMENT_GUIDE.md 最佳实践")
print("=" * 80)
