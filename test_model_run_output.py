#!/usr/bin/env python3
"""测试model.run()的实际返回值"""
import sys
from pathlib import Path

# 添加路径
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.model import HydroSISModel
from hydrosis.runoff.hbv import HBVRunoff

# Create a simple test
from hydrosis.model import Subbasin

# Simple test subbasin
sub = Subbasin(
    id="test1",
    area_km2=10.0,
    downstream=None,
    parameters={
        "runoff_model": "hbv",
        "routing_model": "passthrough"
    }
)

# Simple runoff model
hbv = HBVRunoff({"FC": 100, "BETA": 1.0, "K0": 0.1, "K1": 0.05, "K2": 0.01, "PERC": 1.0})

# Simple routing model (identity)
class PassthroughRouting:
    def route(self, subbasin, flows):
        return flows  # No routing, just return flows as-is

# Create model
from hydrosis.parameters.partition import ParameterZone

zone = ParameterZone(zone_id="1", subbasins=["test1"], controllers=[], parameters={})

model = HydroSISModel(
    subbasins=[sub],
    parameter_zones=[zone],
    runoff_models={"hbv": hbv},
    routing_models={"passthrough": PassthroughRouting()}
)

# Test forcing
forcing = {"test1": [10, 5, 2, 1, 0.5]}

print("="*80)
print("测试 model.run() 返回值")
print("="*80)

result = model.run(forcing)

print(f"\n返回值类型: {type(result)}")
print(f"返回值: {result}")

if isinstance(result, tuple):
    print(f"\n是元组! 长度: {len(result)}")
    print(f"  result[0] 类型: {type(result[0])}")
    print(f"  result[1] 类型: {type(result[1])}")

    if isinstance(result[0], dict):
        print(f"\n  result[0] (routed) keys: {list(result[0].keys())}")
        print(f"  result[0]['test1'] 前5个值: {result[0]['test1'][:5]}")

    if isinstance(result[1], dict):
        print(f"\n  result[1] (runoff) keys: {list(result[1].keys())}")
        print(f"  result[1]['test1'] 前5个值: {result[1]['test1'][:5]}")
else:
    print(f"\n不是元组!")
    if isinstance(result, dict):
        print(f"  是字典, keys: {list(result.keys())}")
