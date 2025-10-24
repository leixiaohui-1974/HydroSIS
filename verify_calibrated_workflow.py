#!/usr/bin/env python3
"""验证使用率定后HBV参数的工作流

目标：
1. 使用率定后的HBV参数重新运行Step 9
2. 验证径流系数在合理范围内(0-1)
3. 生成验证报告

特点：
- 完全配置驱动
- 使用validation框架
- 对比率定前后的结果
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from hydrosis.config import (
    load_workflow_config,
    get_step_paths,
    create_hydrologic_criteria,
)
from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.validation import validate_runoff_coefficient


class MockSubbasin:
    """模拟Subbasin对象"""
    def __init__(self, area_km2: float):
        self.area_km2 = area_km2


def load_zone_info(geojson_path: Path) -> List[Dict]:
    """加载分区信息"""
    with open(geojson_path, 'r') as f:
        data = json.load(f)

    zones = []
    for feature in data['features']:
        props = feature['properties']
        zones.append({
            'zone_id': props.get('zone_id', props.get('id')),
            'area_km2': props.get('area_km2', 0)
        })
    return zones


def load_zone_precipitation(
    precip_df: pd.DataFrame,
    subbasins_csv: Path,
    zone_id: int
) -> np.ndarray:
    """加载分区降雨"""
    subbasins = pd.read_csv(subbasins_csv)
    zone_subs = subbasins[subbasins['zone_id'] == zone_id]

    total_area = 0
    weighted_precip = pd.Series(0.0, index=precip_df.index)

    for _, sub in zone_subs.iterrows():
        subzone_id = str(int(sub['subzone_id']))
        area_km2 = sub['area_km2']

        if subzone_id in precip_df.columns:
            weighted_precip += precip_df[subzone_id] * area_km2
            total_area += area_km2

    if total_area > 0:
        return (weighted_precip / total_area).values
    else:
        return precip_df.mean(axis=1).values


def run_hbv_simulation(
    precipitation: np.ndarray,
    area_km2: float,
    hbv_params: Dict,
    zone_id: int
) -> Dict:
    """运行HBV模拟并计算径流系数

    Args:
        precipitation: 降雨时间序列 (mm/h)
        area_km2: 流域面积 (km²)
        hbv_params: HBV参数字典
        zone_id: 分区ID

    Returns:
        包含径流系数和其他统计信息的字典
    """
    subbasin = MockSubbasin(area_km2)

    # 创建HBV模型
    model = HBVRunoff(parameters=hbv_params)

    # 运行模拟
    simulated_runoff = np.array(model.simulate(subbasin, precipitation.tolist()))

    # 计算径流系数
    total_precip_mm = precipitation.sum() * 1.0  # mm (时间步长为1小时)
    total_runoff_mm = simulated_runoff.sum() * 3600 / (area_km2 * 1e6) * 1000
    runoff_coefficient = total_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0

    return {
        'zone_id': zone_id,
        'runoff_coefficient': runoff_coefficient,
        'total_precip_mm': total_precip_mm,
        'total_runoff_mm': total_runoff_mm,
        'peak_runoff_m3s': simulated_runoff.max(),
        'mean_runoff_m3s': simulated_runoff.mean(),
        'runoff_series': simulated_runoff,
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='验证使用率定后HBV参数的工作流'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/workflow_config.yaml'),
        help='工作流配置文件'
    )
    parser.add_argument(
        '--validation-config',
        type=Path,
        default=Path('config/validation_criteria.yaml'),
        help='验证标准配置文件'
    )
    parser.add_argument(
        '--use-calibrated',
        action='store_true',
        default=True,
        help='使用率定后的参数（默认）'
    )
    parser.add_argument(
        '--use-default',
        action='store_true',
        help='使用默认参数（对比用）'
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("验证使用率定后HBV参数的工作流")
    print("="*80)
    print(f"配置文件: {args.config}")
    print(f"验证配置: {args.validation_config}")

    # 加载配置
    print("\n⚙ 加载配置...")
    workflow_config = load_workflow_config(args.config)
    hydrologic_criteria = create_hydrologic_criteria(args.validation_config)

    # 确定使用哪组参数
    if args.use_default:
        hbv_params = workflow_config['hbv_model']['default_parameters']
        print("  ⚠️  使用默认参数（未率定）")
    else:
        hbv_params = workflow_config['hbv_model']['calibrated_parameters']
        print(f"  ✓ 使用率定参数（{workflow_config['hbv_model']['calibration_date']}）")

    print("\n  HBV参数:")
    for key, value in hbv_params.items():
        print(f"    {key}: {value}")

    # 加载数据
    print("\n⚙ 加载数据...")
    base_dir = Path(workflow_config['directories']['base_results'])

    precip_path = base_dir / "step_08_areal_rainfall" / "8.1_parameter_areal_precipitation.csv"
    zones_path = base_dir / "parameters" / "parameter_zones.geojson"
    subbasins_csv = base_dir / "parameters" / "parameter_subbasins.csv"

    precip_df = pd.read_csv(precip_path)
    if 'Timestamp' in precip_df.columns:
        precip_df = precip_df.set_index('Timestamp')

    zones = load_zone_info(zones_path)

    print(f"  ✓ 加载降雨数据: {len(precip_df)} 小时")
    print(f"  ✓ 加载分区信息: {len(zones)} 个分区")

    # 为每个分区运行HBV模拟
    print("\n⚙ 运行HBV模拟...")
    results = []

    for zone in sorted(zones, key=lambda x: x['zone_id']):
        zone_id = zone['zone_id']
        area_km2 = zone['area_km2']

        # 加载分区降雨
        precipitation = load_zone_precipitation(precip_df, subbasins_csv, zone_id)

        # 运行HBV模拟
        result = run_hbv_simulation(precipitation, area_km2, hbv_params, zone_id)
        results.append(result)

        print(f"  分区 {zone_id}: RC={result['runoff_coefficient']:.4f}, "
              f"降雨={result['total_precip_mm']:.1f}mm, "
              f"径流={result['total_runoff_mm']:.1f}mm")

    # 使用验证框架验证
    print("\n" + "="*80)
    print("闭环验证")
    print("="*80)

    runoff_coefficients = {
        str(r['zone_id']): r['runoff_coefficient']
        for r in results
    }

    validation_result = validate_runoff_coefficient(
        runoff_coefficients=runoff_coefficients,
        criteria=hydrologic_criteria,
        step_name="使用率定HBV参数的径流系数验证" if not args.use_default else "使用默认HBV参数的径流系数验证"
    )

    print(validation_result.summary())

    # 保存结果
    output_dir = base_dir / "workflow_verification"
    output_dir.mkdir(parents=True, exist_ok=True)

    param_type = "calibrated" if not args.use_default else "default"
    summary_file = output_dir / f"verification_{param_type}_params.json"

    with open(summary_file, 'w') as f:
        json.dump({
            'parameter_type': param_type,
            'hbv_parameters': hbv_params,
            'results': [
                {
                    'zone_id': r['zone_id'],
                    'runoff_coefficient': r['runoff_coefficient'],
                    'total_precip_mm': r['total_precip_mm'],
                    'total_runoff_mm': r['total_runoff_mm'],
                    'peak_runoff_m3s': float(r['peak_runoff_m3s']),
                    'mean_runoff_m3s': float(r['mean_runoff_m3s']),
                }
                for r in results
            ],
            'validation': {
                'is_valid': validation_result.is_valid,
                'errors': validation_result.errors,
                'warnings': validation_result.warnings,
                'metrics': validation_result.metrics,
            }
        }, f, indent=2)

    print(f"\n✓ 验证结果保存至: {summary_file}")

    # 如果使用率定参数，保存径流时间序列
    if not args.use_default:
        print("\n⚙ 保存径流时间序列...")
        runoff_df = pd.DataFrame(
            {str(r['zone_id']): r['runoff_series'] for r in results},
            index=precip_df.index
        )
        runoff_df.index.name = 'Timestamp'

        runoff_csv = output_dir / "calibrated_runoff_timeseries.csv"
        runoff_df.to_csv(runoff_csv)
        print(f"  ✓ 径流时间序列: {runoff_csv.name}")

    # 总结
    print("\n" + "="*80)
    if validation_result.is_valid:
        print("✅ 验证通过！所有分区径流系数均在合理范围内")
    else:
        print("❌ 验证失败！部分分区径流系数异常")

    mean_rc = sum(r['runoff_coefficient'] for r in results) / len(results)
    print(f"\n关键指标:")
    print(f"  平均径流系数: {mean_rc:.4f}")
    print(f"  参数类型: {param_type}")
    print(f"  验证状态: {'✅ 通过' if validation_result.is_valid else '❌ 失败'}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
