#!/usr/bin/env python3
"""测试HBV并行化模块

快速验证parallel_hbv.py的功能
"""
import numpy as np
from pathlib import Path
import sys

# Add hydrosis to path
sys.path.insert(0, str(Path(__file__).parent))

from hydrosis.runoff.parallel_hbv import (
    run_hbv_parallel,
    ParallelHBVConfig
)

def test_parallel_hbv():
    """测试并行HBV执行"""
    print("=" * 80)
    print("HBV并行化模块测试")
    print("=" * 80)

    # 1. 创建测试数据
    print("\n⚙ 准备测试数据...")
    zones = [
        {'zone_id': 1, 'area_km2': 100},
        {'zone_id': 2, 'area_km2': 150},
        {'zone_id': 3, 'area_km2': 200},
        {'zone_id': 4, 'area_km2': 120},
    ]

    # 生成模拟降雨序列(120个时间步)
    precipitation_data = {}
    for zone in zones:
        # 随机降雨，总量在300-500mm之间
        total_precip = np.random.uniform(300, 500)
        # 生成雨型(正态分布模拟)
        time_steps = 120
        precip = np.random.gamma(2, 2, size=time_steps)
        precip = precip / precip.sum() * total_precip
        precipitation_data[zone['zone_id']] = precip

    # HBV参数(使用典型值)
    hbv_params = {
        'FC': 300,      # Field capacity
        'BETA': 2.0,    # Shape coefficient
        'LP': 0.7,      # Evapotranspiration threshold
        'K0': 0.1,      # Recession coefficient 0
        'K1': 0.05,     # Recession coefficient 1
        'K2': 0.01,     # Recession coefficient 2
        'PERC': 2.0,    # Percolation rate
        'UZL': 50.0,    # Upper zone threshold
        'TT': 0.0,      # Temperature threshold
        'CFMAX': 3.0,   # Degree-day factor
        'CFR': 0.05,    # Refreezing coefficient
        'CWH': 0.1      # Water holding capacity
    }

    print(f"  ✓ 创建 {len(zones)} 个分区")
    print(f"  ✓ 每个分区 {time_steps} 个时间步")

    # 2. 测试串行执行
    print("\n⚙ 测试串行执行...")
    config_serial = ParallelHBVConfig(
        max_workers=1,
        show_progress=False
    )

    results_serial = run_hbv_parallel(
        zones,
        precipitation_data,
        hbv_params,
        config_serial
    )

    print(f"  ✓ 串行执行完成: {len(results_serial)} 个分区")

    # 3. 测试并行执行
    print("\n⚙ 测试并行执行 (2 workers)...")
    config_parallel = ParallelHBVConfig(
        max_workers=2,
        show_progress=False
    )

    results_parallel = run_hbv_parallel(
        zones,
        precipitation_data,
        hbv_params,
        config_parallel
    )

    print(f"  ✓ 并行执行完成: {len(results_parallel)} 个分区")

    # 4. 验证结果一致性
    print("\n⚙ 验证结果一致性...")
    all_match = True

    for zone_id in results_serial:
        if zone_id not in results_parallel:
            print(f"  ❌ 分区 {zone_id} 在并行结果中缺失")
            all_match = False
            continue

        serial = results_serial[zone_id]
        parallel = results_parallel[zone_id]

        # 比较关键指标
        rc_diff = abs(serial['runoff_coefficient'] - parallel['runoff_coefficient'])
        if rc_diff > 1e-6:
            print(f"  ❌ 分区 {zone_id} RC不一致: {rc_diff}")
            all_match = False

        series_diff = np.abs(serial['runoff_series'] - parallel['runoff_series']).max()
        if series_diff > 1e-6:
            print(f"  ❌ 分区 {zone_id} 径流序列不一致: {series_diff}")
            all_match = False

    if all_match:
        print("  ✓ 串行和并行结果完全一致")

    # 5. 输出结果摘要
    print("\n⚙ 结果摘要:")
    print(f"  {'分区ID':<10} {'面积(km²)':<15} {'降雨(mm)':<15} {'径流(mm)':<15} {'RC':<10}")
    print("  " + "-" * 70)

    for zone in zones:
        zone_id = zone['zone_id']
        result = results_serial[zone_id]
        print(f"  {zone_id:<10} {zone['area_km2']:<15.2f} "
              f"{result['total_precip_mm']:<15.2f} "
              f"{result['total_runoff_mm']:<15.2f} "
              f"{result['runoff_coefficient']:<10.4f}")

    print("\n" + "=" * 80)
    if all_match:
        print("✅ HBV并行化模块测试通过！")
        return 0
    else:
        print("❌ HBV并行化模块测试失败！")
        return 1


if __name__ == "__main__":
    sys.exit(test_parallel_hbv())
