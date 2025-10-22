#!/usr/bin/env python3
"""
运行 Upper Truckee River 完整的10步工作流
使用真实的DEM数据和流域特征
"""
from __future__ import annotations

import sys
from pathlib import Path
import warnings

# 确保可以导入HydroSIS模块
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def main():
    """运行 Upper Truckee River 完整工作流"""
    print("=" * 80)
    print("Upper Truckee River 完整10步工作流")
    print("=" * 80)
    print()
    print("使用真实DEM数据:")
    print("  • DEM: data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif")
    print("  • 流向: flowdir.tif")
    print("  • 汇流累积: flowaccum.tif")
    print()
    print("=" * 80)
    print()

    # 设置数据路径
    dem_path = REPO_ROOT / "data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/elevation.tif"
    flow_accum_path = REPO_ROOT / "data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/flowaccum.tif"
    flow_dir_path = REPO_ROOT / "data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00/flowdir.tif"
    output_dir = REPO_ROOT / "results/upper_truckee_complete_run"

    # 验证文件存在
    if not dem_path.exists():
        print(f"错误: DEM文件不存在: {dem_path}")
        return 1
    if not flow_accum_path.exists():
        print(f"错误: 流向累积文件不存在: {flow_accum_path}")
        return 1
    if not flow_dir_path.exists():
        print(f"错误: 流向文件不存在: {flow_dir_path}")
        return 1

    print("✓ 所有输入文件已验证")
    print()

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print("开始运行完整工作流...")
    print("-" * 80)
    print()

    # 运行 Upper Truckee 工作流
    from examples.upper_truckee_channel_workflow import main as run_upper_truckee

    # 设置命令行参数
    sys.argv = [
        'upper_truckee_channel_workflow.py',
        '--dem', str(dem_path),
        '--flow-accum', str(flow_accum_path),
        '--flow-dir', str(flow_dir_path),
        '--output', str(output_dir),
        '--channel-threshold', '15000',
        '--target-subzone-area', '25.0',
        '--min-subzone-area', '5.0',
        '--max-subzones', '6',
        '--dynamic-wave-length', '2000.0',
        '--pour-count', '6',
        '--use-tree-pour-points',
        '--tree-acc-threshold', '0.15',
        '--tree-min-distance', '45',
        '--tree-max-children', '3',
    ]

    try:
        run_upper_truckee()

        print()
        print("=" * 80)
        print("工作流运行完成！")
        print("=" * 80)
        print()
        print("生成的结果包括：")
        print("  ✓ 流域划分和子流域边界")
        print("  ✓ 参数区划分")
        print("  ✓ 河道网络提取")
        print("  ✓ 河道断面几何")
        print("  ✓ 雨量站点布局")
        print("  ✓ 降雨序列数据")
        print("  ✓ 泰森多边形权重")
        print("  ✓ 面雨量计算")
        print("  ✓ 产汇流模拟")
        print("  ✓ 水动力河道演算")
        print()
        print(f"结果保存在: {output_dir}")
        print()

        # 列出主要输出文件
        print("主要输出文件:")
        print("-" * 80)

        # 检查并列出生成的文件
        intermediate_dir = output_dir / "intermediate"
        parameters_dir = output_dir / "parameters"

        if intermediate_dir.exists():
            print("\n📁 流域划分结果 (intermediate/):")
            for f in sorted(intermediate_dir.glob("*.geojson")):
                print(f"  • {f.name}")
            for f in sorted(intermediate_dir.glob("*.png")):
                print(f"  • {f.name}")
            for f in sorted(intermediate_dir.glob("*.csv")):
                print(f"  • {f.name}")

        if parameters_dir.exists():
            print("\n📁 参数区结果 (parameters/):")
            for f in sorted(parameters_dir.glob("*.geojson")):
                print(f"  • {f.name}")
            for f in sorted(parameters_dir.glob("*.csv")):
                print(f"  • {f.name}")

        # 列出水文模拟结果
        hydro_project = output_dir / "hydro_project"
        if hydro_project.exists():
            print("\n📁 水文模拟结果 (hydro_project/):")
            for subdir in sorted(hydro_project.iterdir()):
                if subdir.is_dir():
                    csv_files = list(subdir.glob("*.csv"))
                    if csv_files:
                        print(f"  • {subdir.name}/ ({len(csv_files)} files)")

        # 列出图表
        print("\n📊 生成的图表:")
        for f in sorted(output_dir.glob("*.png")):
            print(f"  • {f.name}")

        print()
        return 0

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
