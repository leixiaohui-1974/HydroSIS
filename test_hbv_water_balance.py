#!/usr/bin/env python3
"""测试HBV模型的水量平衡"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.runoff.hbv import HBVRunoff

# 创建简单的子流域
class DummySubbasin:
    def __init__(self, area_km2):
        self.id = "test"
        self.area_km2 = area_km2

# 简单参数
params = {
    "FC": 150.0,
    "BETA": 1.0,
    "K0": 0.30,
    "K1": 0.10,
    "K2": 0.02,
    "PERC": 0.5,
    "initial_soil": 25.0,
    "initial_upper": 2.0,
    "initial_lower": 10.0,
}

hbv = HBVRunoff(params)

# 记录初始状态
initial_storage = hbv.soil + hbv.upper + hbv.lower + hbv.snow
print("="*80)
print("HBV模型水量平衡测试")
print("="*80)
print(f"\n初始储水量:")
print(f"  soil: {hbv.soil:.2f} mm")
print(f"  upper: {hbv.upper:.2f} mm")
print(f"  lower: {hbv.lower:.2f} mm")
print(f"  snow: {hbv.snow:.2f} mm")
print(f"  总计: {initial_storage:.2f} mm")

# 测试：恒定5 mm/hr降雨，120个时间步
subbasin = DummySubbasin(area_km2=10.0)
precipitation = [5.0] * 120  # 120小时，每小时5mm

flows = hbv.simulate(subbasin, precipitation)

# 记录最终状态
final_storage = hbv.soil + hbv.upper + hbv.lower + hbv.snow
storage_change = final_storage - initial_storage

print(f"\n最终储水量:")
print(f"  soil: {hbv.soil:.2f} mm")
print(f"  upper: {hbv.upper:.2f} mm")
print(f"  lower: {hbv.lower:.2f} mm")
print(f"  snow: {hbv.snow:.2f} mm")
print(f"  总计: {final_storage:.2f} mm")
print(f"  储水变化: {storage_change:+.2f} mm")

# 计算输入输出
total_input = sum(precipitation) + initial_storage  # mm
print(f"\n总输入:")
print(f"  降雨: {sum(precipitation):.2f} mm")
print(f"  初始储水: {initial_storage:.2f} mm")
print(f"  总计: {total_input:.2f} mm")

# 计算径流深度（从m³/s转换回mm）
dt_hours = 1.0
total_flow_volume_m3 = sum(flows) * dt_hours * 3600  # m³/s * hours * 3600s/hr
runoff_depth_mm = (total_flow_volume_m3 / (subbasin.area_km2 * 1e6)) * 1000

total_output = runoff_depth_mm + final_storage
print(f"\n总输出:")
print(f"  径流深度: {runoff_depth_mm:.2f} mm")
print(f"  最终储水: {final_storage:.2f} mm")
print(f"  总计: {total_output:.2f} mm")

# 水量平衡误差
water_balance_error = total_input - total_output
error_pct = (water_balance_error / total_input) * 100 if total_input > 0 else 0

print(f"\n水量平衡:")
print(f"  输入 - 输出 = {water_balance_error:.2f} mm")
print(f"  误差百分比: {error_pct:.3f}%")

if abs(water_balance_error) < 0.01:
    print(f"  ✓ 水量平衡正确!")
else:
    print(f"  ✗ 水量平衡有误差!")

# 径流系数
runoff_coeff = runoff_depth_mm / sum(precipitation)
print(f"\n径流系数: {runoff_coeff:.3f}")

if runoff_coeff > 1.0:
    print(f"  ⚠ 径流系数>1，违反物理规律!")
    print(f"  这意味着径流深度({runoff_depth_mm:.2f}mm) > 降雨深度({sum(precipitation):.2f}mm)")
    print(f"  差值: {runoff_depth_mm - sum(precipitation):.2f} mm")
    print(f"  这个差值应该等于初始储水减少量: {initial_storage - final_storage:.2f} mm")
elif 0.1 <= runoff_coeff <= 0.8:
    print(f"  ✓ 径流系数合理!")
else:
    print(f"  ⚠ 径流系数偏低")
