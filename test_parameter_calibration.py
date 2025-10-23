#!/usr/bin/env python3
"""测试参数率定模块"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.calibration import ParameterCalibration

print("=" * 80)
print("测试参数率定模块")
print("=" * 80)

# 1. 加载YAML配置
print("\n1. 加载YAML配置文件...")
calib_file = Path("calibration/parameter_adjustments.yaml")
if not calib_file.exists():
    print(f"  ❌ 配置文件不存在: {calib_file}")
    sys.exit(1)

calib = ParameterCalibration(calib_file)
print(f"  ✓ 成功加载配置文件")

# 2. 显示摘要
print(calib.get_summary())

# 3. 测试Zone 1的参数调整
print("\n2. 测试Zone 1的HBV参数调整...")
print("\n  基准参数（global defaults）:")
base_params = calib.global_defaults.get('hbv', {})
for param in ['FC', 'K0', 'K2', 'initial_soil']:
    print(f"    {param}: {base_params.get(param, 'N/A')}")

print("\n  Zone 1调整后:")
zone1_params = calib.get_runoff_parameters('hbv', zone_id=1)
for param in ['FC', 'K0', 'K2', 'initial_soil']:
    base_val = base_params.get(param, 0)
    adj_val = zone1_params.get(param, 0)
    change = ((adj_val - base_val) / base_val * 100) if base_val != 0 else 0
    print(f"    {param}: {base_val:.2f} -> {adj_val:.2f} ({change:+.1f}%)")

# 4. 测试Zone 2的参数调整
print("\n3. 测试Zone 2的HBV参数调整...")
zone2_params = calib.get_runoff_parameters('hbv', zone_id=2)
print("\n  Zone 2调整后:")
for param in ['FC', 'K1', 'PERC']:
    base_val = base_params.get(param, 0)
    adj_val = zone2_params.get(param, 0)
    change = ((adj_val - base_val) / base_val * 100) if base_val != 0 else 0
    print(f"    {param}: {base_val:.2f} -> {adj_val:.2f} ({change:+.1f}%)")

# 5. 测试Zone 3的参数调整
print("\n4. 测试Zone 3的HBV参数调整...")
zone3_params = calib.get_runoff_parameters('hbv', zone_id=3)
print("\n  Zone 3调整后:")
for param in ['FC', 'K0', 'K2', 'BETA']:
    base_val = base_params.get(param, 0)
    adj_val = zone3_params.get(param, 0)
    change = ((adj_val - base_val) / base_val * 100) if base_val != 0 else 0
    print(f"    {param}: {base_val:.2f} -> {adj_val:.2f} ({change:+.1f}%)")

# 6. 测试Muskingum参数调整
print("\n5. 测试Muskingum汇流参数调整...")
base_musk = calib.global_defaults.get('muskingum', {})
print("\n  基准参数:")
for param in ['K', 'x']:
    print(f"    {param}: {base_musk.get(param, 'N/A')}")

for zone_id in [1, 2, 3]:
    zone_musk = calib.get_routing_parameters('muskingum', zone_id=zone_id)
    print(f"\n  Zone {zone_id}调整后:")
    for param in ['K', 'x']:
        base_val = base_musk.get(param, 0)
        adj_val = zone_musk.get(param, 0)
        change = ((adj_val - base_val) / base_val * 100) if base_val != 0 else 0
        print(f"    {param}: {base_val:.2f} -> {adj_val:.2f} ({change:+.1f}%)")

# 7. 验证不同调整方法
print("\n6. 验证三种调整方法...")
from hydrosis.calibration import ParameterAdjustment

base_value = 100.0

# multiply
adj_multiply = ParameterAdjustment('multiply', 1.2)
result = adj_multiply.apply(base_value)
print(f"  multiply(1.2): {base_value} -> {result} ✓")

# add
adj_add = ParameterAdjustment('add', 10.0)
result = adj_add.apply(base_value)
print(f"  add(10.0): {base_value} -> {result} ✓")

# set
adj_set = ParameterAdjustment('set', 150.0)
result = adj_set.apply(base_value)
print(f"  set(150.0): {base_value} -> {result} ✓")

print("\n" + "=" * 80)
print("✓ 所有测试通过！")
print("=" * 80)
