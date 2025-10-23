#!/usr/bin/env python3
"""测试HBV模型输出单位"""
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
    "initial_soil": 25.0,   # 降低初始储水
    "initial_upper": 2.0,   # 降低初始储水
    "initial_lower": 10.0,  # 降低初始储水
}

hbv = HBVRunoff(params)

# 测试：10 km²流域，恒定1 mm/hr降雨，10个时间步
subbasin = DummySubbasin(area_km2=10.0)
precipitation = [1.0] * 10  # 10小时，每小时1mm

flows = hbv.simulate(subbasin, precipitation)

print("="*80)
print("HBV模型输出单位测试")
print("="*80)
print(f"\n输入:")
print(f"  子流域面积: {subbasin.area_km2} km²")
print(f"  降雨: {precipitation[:5]} ... mm/hr (共{len(precipitation)}小时)")
print(f"  总降雨深度: {sum(precipitation)} mm")

print(f"\n输出:")
print(f"  前5个时间步流量: {flows[:5]}")
print(f"  单位应该是: m³/s")
print(f"  均值: {sum(flows)/len(flows):.3f} m³/s")

# 计算径流系数
total_volume_m3 = sum(flows) * 3600  # m³/s * 3600 s/hr = m³
runoff_depth_mm = (total_volume_m3 / (subbasin.area_km2 * 1e6)) * 1000
runoff_coeff = runoff_depth_mm / sum(precipitation)

print(f"\n验证:")
print(f"  总径流体积: {total_volume_m3:.0f} m³")
print(f"  径流深度: {runoff_depth_mm:.2f} mm")
print(f"  径流系数: {runoff_coeff:.3f}")

# 预期值
print(f"\n理论检验:")
print(f"  如果径流系数应该在0.1-0.3范围:")
print(f"    预期径流深度: {sum(precipitation)*0.2:.2f} mm")
print(f"    预期平均流量: {sum(precipitation)*0.2 * 10 / 3.6 / 10:.3f} m³/s")
print(f"  实际平均流量: {sum(flows)/len(flows):.3f} m³/s")

if 0.1 <= runoff_coeff <= 0.8:
    print(f"\n✓ 径流系数合理!")
else:
    print(f"\n⚠ 径流系数异常!")
