#!/usr/bin/env python3
"""
测试第4-8步工作流（带YAML配置）
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
    step04_channel_cross_sections,
    step06_to_08_precipitation_processing,
)


def main():
    """测试第4-8步"""
    print("\n")
    print("=" * 80)
    print("测试第4-8步工作流（带YAML配置）")
    print("="  * 80)
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
    output_config = global_config.get('output', {})

    # 输入数据路径（从第3步的输出）
    dem_dir = Path("data/Upper_Truckee_River/terrain/UpTruckeeRv_S10_NED_30m/00")
    dem_path = dem_dir / "elevation.tif"

    # 输出目录
    output_root = Path(output_config.get('root_dir', "results/upper_truckee_complete_11steps"))
    parameter_dir = output_root / "parameters"
    intermediate_dir = output_root / "intermediate"

    # 检查输入
    if not dem_path.exists():
        print(f"错误：DEM文件不存在: {dem_path}")
        return 1

    if not parameter_dir.exists():
        print(f"错误：参数目录不存在: {parameter_dir}")
        print("请先运行test_first_3_steps.py生成前3步的结果")
        return 1

    print(f"✓ 输入数据检查完成")
    print(f"✓ 输出目录: {output_root}")
    print()

    try:
        # 第4步：河道断面提取
        print("运行第4步...")
        result4 = step04_channel_cross_sections(
            dem_path, parameter_dir, output_root,
            config=config
        )
        print(f"✓ 第4步完成：生成{len(result4.get('outputs', []))}个断面文件")

        # 加载第3步的结果（用于第6-8步）
        print("\n加载第3步结果...")

        # 加载子流域几何体
        import json
        from shapely.geometry import shape as shapely_shape

        subbasin_geometries = {}
        param_subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
        if param_subbasin_geojson.exists():
            with open(param_subbasin_geojson, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            for feature in geojson_data['features']:
                sub_id = feature['properties'].get('subzone_id', feature['properties'].get('id'))
                if sub_id:
                    geom = shapely_shape(feature['geometry'])
                    subbasin_geometries[sub_id] = geom
        print(f"✓ 加载{len(subbasin_geometries)}个子流域几何体")

        # 加载分区几何体
        zone_geometries = {}
        param_zones_geojson = parameter_dir / "parameter_zones.geojson"
        if param_zones_geojson.exists():
            with open(param_zones_geojson, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            for feature in geojson_data['features']:
                zone_id = feature['properties'].get('zone_id', feature['properties'].get('id'))
                if zone_id:
                    geom = shapely_shape(feature['geometry'])
                    zone_geometries[str(zone_id)] = geom
        print(f"✓ 加载{len(zone_geometries)}个分区几何体")

        # 加载子流域信息
        import pandas as pd
        from hydrosis.model import Subbasin

        subbasins = []
        param_subbasin_csv = parameter_dir / "parameter_subbasins.csv"
        if param_subbasin_csv.exists():
            df = pd.read_csv(param_subbasin_csv)
            for _, row in df.iterrows():
                sub_id = str(row.get('subzone_id', row.get('id')))
                area = row.get('area_km2', 0.0)
                downstream = row.get('downstream_subzone_id', None)
                if downstream and pd.notna(downstream):
                    downstream = str(downstream)
                else:
                    downstream = None

                subbasin = Subbasin(
                    id=sub_id,
                    area_km2=area,
                    downstream=downstream,
                    parameters={},
                )
                subbasins.append(subbasin)
        print(f"✓ 加载{len(subbasins)}个子流域信息")

        # 第6-8步：雨量处理
        print("\n运行第6-8步...")
        result6_8 = step06_to_08_precipitation_processing(
            partition_outputs=None,  # 不需要partition_outputs
            subbasins=subbasins,
            subbasin_geometries=subbasin_geometries,
            zone_geometries=zone_geometries,
            intermediate_dir=intermediate_dir,
            output_dir=output_root,
            config=config,
        )
        print(f"✓ 第6-8步完成：生成{len(result6_8.get('outputs', []))}个输出文件")

        print("\n" + "=" * 80)
        print("✓ 第4-8步测试完成！")
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
