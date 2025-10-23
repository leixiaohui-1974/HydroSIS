#!/usr/bin/env python3
"""
测试前3步工作流（带YAML配置）
"""
from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到Python路径
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_upper_truckee_complete_11steps import (
    load_config,
    step01_dem_processing,
    step02_pour_point_generation,
    step03_parameter_zones_and_subbasins,
)


def main():
    """测试前3步"""
    print("\n")
    print("=" * 80)
    print("测试前3步工作流（带YAML配置）")
    print("=" * 80)
    print()

    # 加载配置文件
    config_path = Path("config_upper_truckee_11steps.yml")
    if config_path.exists():
        config = load_config(config_path)
    else:
        print(f"⚠ 配置文件不存在: {config_path}，使用默认配置")
        config = {}

    # 从配置文件读取路径
    global_config = config.get('global', {})
    input_config = global_config.get('input', {})
    output_config = global_config.get('output', {})

    # 输入数据路径
    dem_dir = Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00")
    dem_path = Path(input_config.get('dem_path', dem_dir / "elevation.tif"))
    flow_dir_path = Path(input_config.get('flow_direction_path', dem_dir / "flowdir.tif"))
    flow_acc_path = Path(input_config.get('flow_accumulation_path', dem_dir / "flowaccum.tif"))

    # 输出目录
    output_root = Path(output_config.get('root_dir', "results/upper_truckee_complete_11steps"))
    output_root.mkdir(parents=True, exist_ok=True)

    # 检查输入
    for path in [dem_path, flow_dir_path, flow_acc_path]:
        if not path.exists():
            print(f"错误：输入文件不存在: {path}")
            return 1

    print(f"✓ 输入数据检查完成")
    print(f"✓ 输出目录: {output_root}")
    print()

    try:
        # 第1步：DEM处理
        print("运行第1步...")
        result1 = step01_dem_processing(
            dem_path, flow_dir_path, flow_acc_path, output_root
        )

        # 第2步：汇水点生成
        print("\n运行第2步...")
        result2 = step02_pour_point_generation(
            flow_dir_path, flow_acc_path, output_root,
            config=config
        )
        pour_points_path = result2['pour_points_path']

        # 第3步：参数分区和子流域划分
        print("\n运行第3步...")
        result3 = step03_parameter_zones_and_subbasins(
            dem_path, flow_dir_path, flow_acc_path, pour_points_path, output_root,
            config=config
        )

        print("\n" + "=" * 80)
        print("✓ 前3步测试完成！")
        print("=" * 80)
        print(f"第1步输出: {len(result1.get('outputs', []))}个文件")
        print(f"第2步输出: {len(result2.get('outputs', []))}个文件")
        print(f"第3步输出: {len(result3.get('outputs', []))}个文件")

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
