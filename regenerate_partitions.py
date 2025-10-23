#!/usr/bin/env python3
"""
重新生成参数分区结果，使用新的顺序编码
"""
from pathlib import Path
import sys

# 添加项目根目录到Python路径
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.config import (
    DelineationConfig,
    ModelStructureConfig,
    OutputArtifactsConfig,
    ParameterPartitionConfig,
)
from hydrosis.parameters.partition_builder import partition_parameter_zones

def main():
    # 定义路径
    base_dir = Path("results/upper_truckee_complete_11steps")
    dem_path = Path("data/upper_truckee/upper_truckee_dem_10m.tif")

    # Step 1: 创建DelineationConfig
    delineation_cfg = DelineationConfig(
        dem_path=dem_path,
        flow_direction_path=base_dir / "intermediate" / "flow_direction.tif",
        flow_accumulation_path=base_dir / "intermediate" / "flow_accumulation.tif",
        intermediate_directory=base_dir / "intermediate",
        parameter_directory=base_dir / "parameters",
    )

    # Step 2: 创建PartitionConfig
    partition_cfg = ParameterPartitionConfig(
        pour_points_path=base_dir / "step_02_pour_points" / "2.1_pour_points.geojson",
        subzone_accumulation_threshold=None,
        target_subzone_area_km2=10.0,
        min_subzone_area_km2=5.0,
        area_balance_tolerance=0.3,
    )

    # Step 3: 创建ModelStructureConfig
    model_structure = ModelStructureConfig(
        default_runoff_model="hbv",
        default_routing_model="muskingum",
        subbasin_assignments=[],
    )

    # Step 4: 创建OutputConfig
    outputs_cfg = OutputArtifactsConfig(
        enable_figures=True,
        enable_diagnostics=True,
    )

    print("开始生成参数分区...")
    print(f"DEM路径: {dem_path}")
    print(f"流向文件: {delineation_cfg.flow_direction_path}")
    print(f"Pour Points: {partition_cfg.pour_points_path}")

    # 执行分区
    partition_outputs = partition_parameter_zones(
        delineation_cfg=delineation_cfg,
        partition_cfg=partition_cfg,
        model_structure=model_structure,
        outputs_cfg=outputs_cfg,
    )

    print(f"\n✅ 分区生成完成!")
    print(f"参数分区数量: {len(partition_outputs.zone_summaries)}")
    print(f"子流域数量: {len(partition_outputs.subzone_summaries)}")
    print(f"河道段数量: {len(partition_outputs.channel_summaries)}")

    print("\n分区编号:")
    for zone in partition_outputs.zone_summaries:
        print(f"  Zone {zone.id}: {zone.area_km2:.2f} km², downstream={zone.downstream_id or 'outlet'}")

    print(f"\n结果文件已保存到: {base_dir / 'parameters'}")

if __name__ == "__main__":
    main()
