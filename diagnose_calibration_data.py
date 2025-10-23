#!/usr/bin/env python3
"""
诊断Zone 1率定数据和模型问题

系统检查：
1. 降雨数据合理性
2. 观测径流数据来源和合理性
3. HBV模型水量平衡
4. 单位转换正确性
5. 时间步长和数据对齐
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.runoff.hbv import HBVRunoff

print("=" * 80)
print("Zone 1 率定数据诊断")
print("=" * 80)

results_dir = Path("results/upper_truckee_complete_11steps")

# ============================================================================
# 1. 检查观测径流数据
# ============================================================================
print("\n1. 检查观测径流数据")
print("-" * 80)

obs_file = results_dir / "estimated_observations" / "zone_1_estimated_runoff.csv"
obs_df = pd.read_csv(obs_file)
observed_runoff = obs_df['discharge_m3s'].values
times = pd.to_datetime(obs_df['datetime'])

print(f"观测数据文件: {obs_file}")
print(f"  时间范围: {times.min()} 到 {times.max()}")
print(f"  时间步数: {len(observed_runoff)}")
print(f"  时间间隔: {(times[1] - times[0]).total_seconds()/3600:.2f} 小时")
print(f"\n  流量统计:")
print(f"    最小值: {observed_runoff.min():.2f} m³/s")
print(f"    最大值: {observed_runoff.max():.2f} m³/s")
print(f"    平均值: {observed_runoff.mean():.2f} m³/s")
print(f"    标准差: {observed_runoff.std():.2f} m³/s")

# 查看前10个值
print(f"\n  前10个时间步的流量:")
for i in range(min(10, len(observed_runoff))):
    print(f"    {times[i]}: {observed_runoff[i]:.2f} m³/s")

# ============================================================================
# 2. 检查降雨数据
# ============================================================================
print("\n2. 检查降雨数据")
print("-" * 80)

precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"
precip_df = pd.read_csv(precip_file, index_col=0)

# Zone 1的子分区
zone1_subbasins = [str(i) for i in range(101, 115)]
zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
precipitation = precip_df[zone1_cols].mean(axis=1).values

print(f"降雨数据文件: {precip_file}")
print(f"  Zone 1子分区: {len(zone1_cols)}个")
print(f"  时间步数: {len(precipitation)}")
print(f"\n  降雨统计 (mm/h):")
print(f"    最小值: {precipitation.min():.2f} mm/h")
print(f"    最大值: {precipitation.max():.2f} mm/h")
print(f"    平均值: {precipitation.mean():.2f} mm/h")
print(f"    标准差: {precipitation.std():.2f} mm/h")
print(f"    总降雨深度: {precipitation.sum():.2f} mm (假设1小时时间步)")

# 查看前10个值
print(f"\n  前10个时间步的降雨:")
for i in range(min(10, len(precipitation))):
    print(f"    时间步{i}: {precipitation[i]:.2f} mm/h")

# 检查降雨合理性
total_precip_mm = precipitation.sum()  # 假设1小时步长
if total_precip_mm > 500:
    print(f"\n  ⚠️  警告: 总降雨深度 {total_precip_mm:.0f} mm 非常大!")
    print(f"       120小时内降雨超过500mm是极端暴雨事件")

# ============================================================================
# 3. 检查数据长度匹配
# ============================================================================
print("\n3. 检查数据长度匹配")
print("-" * 80)

if len(precipitation) == len(observed_runoff):
    print(f"✓ 降雨和径流数据长度匹配: {len(precipitation)}个时间步")
else:
    print(f"✗ 数据长度不匹配!")
    print(f"  降雨: {len(precipitation)}个时间步")
    print(f"  径流: {len(observed_runoff)}个时间步")

# ============================================================================
# 4. 水量平衡检查
# ============================================================================
print("\n4. 水量平衡检查")
print("-" * 80)

zone1_area_km2 = 139.995
dt_hours = 1.0

# 总降雨体积
total_precip_depth_mm = precipitation.sum() * dt_hours
total_precip_volume_m3 = total_precip_depth_mm * zone1_area_km2 * 1000

# 总径流体积
total_runoff_volume_m3 = observed_runoff.sum() * dt_hours * 3600

# 径流系数
runoff_depth_mm = (total_runoff_volume_m3 / (zone1_area_km2 * 1e6)) * 1000
runoff_coefficient = runoff_depth_mm / total_precip_depth_mm

print(f"流域面积: {zone1_area_km2:.2f} km²")
print(f"时间步长: {dt_hours:.2f} 小时")
print(f"\n降雨:")
print(f"  总降雨深度: {total_precip_depth_mm:.2f} mm")
print(f"  总降雨体积: {total_precip_volume_m3:.2e} m³")
print(f"\n径流:")
print(f"  总径流深度: {runoff_depth_mm:.2f} mm")
print(f"  总径流体积: {total_runoff_volume_m3:.2e} m³")
print(f"\n径流系数: {runoff_coefficient:.4f}")

if runoff_coefficient < 0.1 or runoff_coefficient > 0.8:
    print(f"  ⚠️  警告: 径流系数 {runoff_coefficient:.4f} 超出合理范围 (0.1-0.8)")
else:
    print(f"  ✓ 径流系数在合理范围内")

# ============================================================================
# 5. 测试HBV模型单位转换
# ============================================================================
print("\n5. 测试HBV模型单位转换")
print("-" * 80)

# 简单测试：1小时内1mm/h降雨
test_precip = [10.0]  # mm/h
test_params = {
    'FC': 200.0,
    'BETA': 2.0,
    'K0': 0.2,
    'K1': 0.05,
    'K2': 0.01,
    'PERC': 2.0,
    'TT': 0.0,
    'CFMAX': 3.5,
    'LP': 0.7,
    'MAXBAS': 3.0,
    'CFR': 0.05,
    'CWH': 0.1,
    'initial_soil': 60.0,
    'initial_upper': 5.0,
    'initial_lower': 20.0,
    'initial_snow': 0.0
}

class MockSubbasin:
    def __init__(self, area_km2):
        self.area_km2 = area_km2

hbv = HBVRunoff(test_params)
subbasin = MockSubbasin(zone1_area_km2)

# 运行模型
runoff_mm_h = hbv.simulate(subbasin, test_precip)

print(f"测试输入: 10.0 mm/h 降雨")
print(f"模型输出: {runoff_mm_h[0]:.4f} mm/h")
print(f"\n单位转换检查:")

# 方法1: 直接转换
q_m3s_method1 = runoff_mm_h[0] * zone1_area_km2 / 3.6
print(f"  方法1: Q = R × A / 3.6")
print(f"        = {runoff_mm_h[0]:.4f} × {zone1_area_km2:.2f} / 3.6")
print(f"        = {q_m3s_method1:.4f} m³/s")

# 方法2: 详细步骤
runoff_mm_s = runoff_mm_h[0] / 3600  # mm/s
runoff_m_s = runoff_mm_s / 1000  # m/s
area_m2 = zone1_area_km2 * 1e6  # m²
q_m3s_method2 = runoff_m_s * area_m2  # m³/s
print(f"\n  方法2 (详细步骤):")
print(f"    径流深度速率: {runoff_mm_s:.6f} mm/s = {runoff_m_s:.9f} m/s")
print(f"    流域面积: {area_m2:.0f} m²")
print(f"    流量: {q_m3s_method2:.4f} m³/s")

if abs(q_m3s_method1 - q_m3s_method2) < 0.001:
    print(f"\n  ✓ 两种方法结果一致，单位转换正确")
else:
    print(f"\n  ✗ 两种方法结果不一致!")

# ============================================================================
# 6. 使用真实数据测试HBV模型
# ============================================================================
print("\n6. 使用真实数据测试HBV模型")
print("-" * 80)

# 使用前20个时间步测试
n_test = 20
test_precip_full = precipitation[:n_test].tolist()

# 重新初始化模型
hbv_test = HBVRunoff(test_params)
runoff_mm_h_full = hbv_test.simulate(subbasin, test_precip_full)

# 转换为m³/s
runoff_m3s_full = np.array(runoff_mm_h_full) * zone1_area_km2 / 3.6

print(f"测试前{n_test}个时间步:")
print(f"\n{'步':<4s} {'降雨(mm/h)':>12s} {'径流(mm/h)':>12s} {'流量(m³/s)':>12s} {'观测(m³/s)':>12s}")
print("-" * 60)
for i in range(n_test):
    print(f"{i:<4d} {precipitation[i]:>12.2f} {runoff_mm_h_full[i]:>12.4f} "
          f"{runoff_m3s_full[i]:>12.2f} {observed_runoff[i]:>12.2f}")

# ============================================================================
# 7. 检查估计径流的生成配置
# ============================================================================
print("\n7. 检查估计径流的生成配置")
print("-" * 80)

config_file = results_dir / "estimated_observations" / "generation_config.yaml"
if config_file.exists():
    import yaml
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    print(f"配置文件: {config_file}")
    if 'zone_configs' in config and 1 in config['zone_configs']:
        zone1_config = config['zone_configs'][1]
        print(f"\nZone 1配置参数:")
        for key, value in zone1_config.items():
            print(f"  {key}: {value}")
else:
    print(f"配置文件不存在: {config_file}")

# ============================================================================
# 8. 诊断结论
# ============================================================================
print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)

issues = []

# 检查降雨
if total_precip_depth_mm > 500:
    issues.append("降雨深度过大 ({:.0f} mm)".format(total_precip_depth_mm))

# 检查径流系数
if runoff_coefficient < 0.1 or runoff_coefficient > 0.8:
    issues.append("径流系数异常 ({:.4f})".format(runoff_coefficient))

# 检查模型输出尺度
avg_simulated = runoff_m3s_full.mean()
avg_observed = observed_runoff[:n_test].mean()
scale_ratio = avg_simulated / avg_observed
if scale_ratio > 2.0 or scale_ratio < 0.5:
    issues.append("模型输出尺度偏差大 (模拟/观测 = {:.2f})".format(scale_ratio))

if issues:
    print("\n发现的问题:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
else:
    print("\n未发现明显问题")

print("\n建议:")
print("  1. 检查降雨数据单位是否正确（应该是mm/h，不是mm）")
print("  2. 检查观测径流数据的生成方法")
print("  3. 考虑调整HBV模型的初始状态")
print("  4. 尝试更合理的参数搜索范围")

print("\n" + "=" * 80)
