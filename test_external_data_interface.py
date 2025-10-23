#!/usr/bin/env python3
"""
测试外部数据接口的双模式测试脚本

测试两种模式：
1. 自动生成模式（external_data.enabled=false）
2. 外部数据模式（external_data.enabled=true）

确保两种模式都能正常工作
"""
from pathlib import Path
import yaml
import shutil

from run_upper_truckee_complete_11steps import (
    load_config,
    step02_pour_point_generation,
    step03_parameter_zones_and_subbasins,
    step06_to_08_precipitation_processing,
)


def test_pour_points_external_data():
    """测试汇水点外部数据加载"""
    print("\n" + "="*80)
    print("测试 1: 汇水点外部数据接口")
    print("="*80)

    # 加载配置
    config_path = Path("config_upper_truckee_11steps.yml")
    config = load_config(config_path)

    # 测试目录
    test_output = Path("results/test_external_data")
    test_output.mkdir(parents=True, exist_ok=True)

    # DEM路径
    dem_path = Path(config['global']['input']['dem_path'])
    flow_dir = Path(config['global']['input']['flow_direction_path'])
    flow_acc = Path(config['global']['input']['flow_accumulation_path'])

    print("\n--- 模式1: 自动生成汇水点 ---")
    # 修改配置为自动生成模式
    config['step02_pour_points']['external_data']['enabled'] = False

    output_auto = test_output / "auto_generated"
    output_auto.mkdir(parents=True, exist_ok=True)

    result_auto = step02_pour_point_generation(
        flow_dir_path=flow_dir,
        flow_acc_path=flow_acc,
        output_dir=output_auto,
        config=config,
    )

    print(f"\n✓ 自动生成模式完成，输出：{len(result_auto['outputs'])} 个文件")

    print("\n--- 模式2: 外部数据加载汇水点 ---")
    # 修改配置为外部数据模式
    config['step02_pour_points']['external_data']['enabled'] = True
    config['step02_pour_points']['external_data']['file_path'] = \
        "data/upper_truckee_external_data/pour_points.geojson"

    output_external = test_output / "external_data"
    output_external.mkdir(parents=True, exist_ok=True)

    result_external = step02_pour_point_generation(
        flow_dir_path=flow_dir,
        flow_acc_path=flow_acc,
        output_dir=output_external,
        config=config,
    )

    print(f"\n✓ 外部数据模式完成，输出：{len(result_external['outputs'])} 个文件")

    # 验证
    print("\n验证结果：")
    import json
    auto_geojson = output_auto / "step_02_pour_points" / "2.1_pour_points.geojson"
    external_geojson = output_external / "step_02_pour_points" / "2.1_pour_points.geojson"

    with open(auto_geojson) as f:
        auto_data = json.load(f)
    with open(external_geojson) as f:
        external_data = json.load(f)

    print(f"  自动生成: {len(auto_data['features'])} 个汇水点")
    print(f"  外部数据: {len(external_data['features'])} 个汇水点")

    # 外部数据应该和示例文件匹配
    external_file = Path("data/upper_truckee_external_data/pour_points.geojson")
    with open(external_file) as f:
        expected_data = json.load(f)
    print(f"  预期汇水点: {len(expected_data['features'])} 个")

    assert len(external_data['features']) == len(expected_data['features']), \
        "外部数据加载的汇水点数量不匹配!"

    print("✓ 汇水点外部数据接口测试通过")


def test_rain_gauges_external_data():
    """测试雨量站外部数据加载"""
    print("\n" + "="*80)
    print("测试 2: 雨量站外部数据接口")
    print("="*80)

    # 加载配置
    config_path = Path("config_upper_truckee_11steps.yml")
    config = load_config(config_path)

    # 测试目录
    test_output = Path("results/test_external_data")

    # 需要先运行step03来获取分区信息
    print("\n准备工作：运行第3步获取参数分区...")
    dem_path = Path(config['global']['input']['dem_path'])
    flow_dir = Path(config['global']['input']['flow_direction_path'])
    flow_acc = Path(config['global']['input']['flow_accumulation_path'])

    # 使用自动生成的汇水点
    config['step02_pour_points']['external_data']['enabled'] = False
    output_prep = test_output / "prep"
    output_prep.mkdir(parents=True, exist_ok=True)

    # 运行step02
    result02 = step02_pour_point_generation(
        flow_dir_path=flow_dir,
        flow_acc_path=flow_acc,
        output_dir=output_prep,
        config=config,
    )

    # 运行step03获取分区
    pour_points_geojson = output_prep / "step_02_pour_points" / "2.1_pour_points.geojson"

    result03 = step03_parameter_zones_and_subbasins(
        dem_path=dem_path,
        flow_dir_path=flow_dir,
        flow_acc_path=flow_acc,
        pour_points_path=pour_points_geojson,
        output_dir=output_prep,
        config=config,
    )

    # 参数分区输出路径（step03输出到parameters/子目录）
    parameter_dir = output_prep / "parameters"

    # 加载分区信息
    import json
    from shapely.geometry import shape as shapely_shape

    subbasin_file = parameter_dir / "parameter_subbasins.geojson"
    zone_file = parameter_dir / "parameter_zones.geojson"

    with open(subbasin_file) as f:
        subbasin_data = json.load(f)
    with open(zone_file) as f:
        zone_data = json.load(f)

    subbasin_geometries = {}
    for feature in subbasin_data['features']:
        # 使用subzone_id作为ID
        sub_id = feature['properties']['subzone_id']
        geom = shapely_shape(feature['geometry'])
        subbasin_geometries[sub_id] = geom

    zone_geometries = {}
    for feature in zone_data['features']:
        zone_id = feature['properties'].get('zone_id', feature['properties'].get('id'))
        geom = shapely_shape(feature['geometry'])
        zone_geometries[str(zone_id)] = geom

    # 加载子流域信息
    import pandas as pd
    from hydrosis.model import Subbasin

    subbasin_csv = parameter_dir / "parameter_subbasins.csv"
    df = pd.read_csv(subbasin_csv)
    subbasins = []
    for _, row in df.iterrows():
        sub = Subbasin(
            id=row['subzone_id'],  # 使用subzone_id
            area_km2=row['area_km2'],
            downstream=str(row['downstream_subzone_id']) if pd.notna(row.get('downstream_subzone_id')) else None,
            channel_id=None,  # CSV中没有channel_id字段
        )
        subbasins.append(sub)

    print(f"✓ 准备完成：{len(subbasin_geometries)}个子流域，{len(zone_geometries)}个分区")

    print("\n--- 模式1: 自动生成雨量站（分层采样）---")
    config['step05_rain_gauges']['external_data']['enabled'] = False
    config['step05_rain_gauges']['generation']['method'] = 'stratified'

    output_auto = test_output / "rain_auto"
    output_auto.mkdir(parents=True, exist_ok=True)

    result_auto = step06_to_08_precipitation_processing(
        partition_outputs=None,
        subbasins=subbasins,
        subbasin_geometries=subbasin_geometries,
        zone_geometries=zone_geometries,
        intermediate_dir=output_auto / "intermediate",
        output_dir=output_auto,
        config=config,
    )

    print(f"\n✓ 自动生成模式完成，输出：{len(result_auto['outputs'])} 个文件")

    print("\n--- 模式2: 外部数据加载雨量站 ---")
    config['step05_rain_gauges']['external_data']['enabled'] = True
    config['step05_rain_gauges']['external_data']['file_path'] = \
        "data/upper_truckee_external_data/rain_gauges.csv"

    output_external = test_output / "rain_external"
    output_external.mkdir(parents=True, exist_ok=True)

    result_external = step06_to_08_precipitation_processing(
        partition_outputs=None,
        subbasins=subbasins,
        subbasin_geometries=subbasin_geometries,
        zone_geometries=zone_geometries,
        intermediate_dir=output_external / "intermediate",
        output_dir=output_external,
        config=config,
    )

    print(f"\n✓ 外部数据模式完成，输出：{len(result_external['outputs'])} 个文件")

    # 验证
    print("\n验证结果：")
    auto_stations = output_auto / "step_07_thiessen" / "7.2_gauge_locations.geojson"
    external_stations = output_external / "step_07_thiessen" / "7.2_gauge_locations.geojson"

    with open(auto_stations) as f:
        auto_data = json.load(f)
    with open(external_stations) as f:
        external_data = json.load(f)

    print(f"  自动生成: {len(auto_data['features'])} 个雨量站")
    print(f"  外部数据: {len(external_data['features'])} 个雨量站")

    # 外部数据应该和示例文件匹配
    external_file = Path("data/upper_truckee_external_data/rain_gauges.csv")
    import pandas as pd
    expected_df = pd.read_csv(external_file)
    print(f"  预期雨量站: {len(expected_df)} 个")

    assert len(external_data['features']) == len(expected_df), \
        "外部数据加载的雨量站数量不匹配!"

    print("✓ 雨量站外部数据接口测试通过")


def main():
    """运行所有测试"""
    print("="*80)
    print("外部数据接口双模式测试")
    print("="*80)

    try:
        # 测试1：汇水点外部数据
        test_pour_points_external_data()

        # 测试2：雨量站外部数据
        test_rain_gauges_external_data()

        print("\n" + "="*80)
        print("✓ 所有测试通过！")
        print("="*80)

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
