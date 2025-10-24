#!/usr/bin/env python3
"""Step 9径流分析增强脚本 - 使用HBV直接模拟

这个版本直接运行HBV模拟来计算本地径流,而不是从汇流后的discharge读取。
这样可以正确计算径流系数(本地径流/本地降雨),避免上游累积导致的RC>1问题。

用法:
    python enhance_step_09_with_hbv_simulation.py --config config/workflow_config.yaml
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# HydroSIS模块
from hydrosis.config import (
    load_workflow_config,
    load_validation_criteria,
    create_hydrologic_criteria,
)
from hydrosis.validation import validate_runoff_coefficient
from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.model import Subbasin

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_zone_info(geojson_path: Path) -> List[Dict]:
    """加载分区信息"""
    import json
    with open(geojson_path) as f:
        data = json.load(f)

    zones = []
    for feature in data['features']:
        props = feature['properties']
        zones.append({
            'zone_id': str(props['zone_id']),
            'area_km2': props['area_km2']
        })
    return zones


def load_zone_precipitation(
    precip_df: pd.DataFrame,
    subbasins_csv: Path,
    zone_id: str
) -> np.ndarray:
    """加载指定分区的降雨数据"""
    # 加载子流域信息
    subbasins = pd.read_csv(subbasins_csv)

    # 找到属于该分区的子流域
    zone_subbasins = subbasins[subbasins['zone_id'] == int(zone_id)]

    if len(zone_subbasins) == 0:
        raise ValueError(f"未找到分区{zone_id}的子流域")

    # 计算分区的面积加权平均降雨
    precip_values = []
    total_area = zone_subbasins['area_km2'].sum()

    for _, subbasin in zone_subbasins.iterrows():
        subbasin_id = str(int(subbasin['subzone_id']))  # 转换为整数再转字符串,去掉小数点
        if subbasin_id in precip_df.columns:
            weight = subbasin['area_km2'] / total_area
            precip_values.append(precip_df[subbasin_id].values * weight)

    if not precip_values:
        print(f"  调试: 分区{zone_id}的子流域: {zone_subbasins['subzone_id'].tolist()}")
        print(f"  调试: 降雨列前10个: {list(precip_df.columns[:10])}")
        raise ValueError(f"未找到分区{zone_id}的降雨数据")

    return np.sum(precip_values, axis=0)


def run_hbv_simulation(
    precipitation: np.ndarray,
    area_km2: float,
    hbv_params: Dict,
    zone_id: int
) -> Dict:
    """运行HBV模拟并计算径流系数"""
    # 创建虚拟的Subbasin对象
    subbasin = Subbasin(
        id=str(zone_id),
        area_km2=area_km2,
        downstream=None
    )

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


def plot_cumulative_runoff(
    results: List[Dict],
    zones: List[Dict],
    output_path: Path
):
    """绘制分区累积径流曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))

    # 按照zone_id排序
    sorted_results = sorted(results, key=lambda x: int(x['zone_id']))

    for result in sorted_results:
        zone_id = result['zone_id']
        runoff_series = result['runoff_series']
        cumulative = np.cumsum(runoff_series)
        hours = np.arange(len(runoff_series))

        ax.plot(hours, cumulative, label=f'分区{zone_id}', linewidth=2)

    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('累积径流 (m³)', fontsize=12)
    ax.set_title('各参数分区累积径流曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_evaluation_csv(
    results: List[Dict],
    zones: List[Dict],
    output_path: Path,
    criteria
):
    """保存径流系数评价CSV"""
    rows = []

    for result in sorted(results, key=lambda x: int(x['zone_id'])):
        zone_id = result['zone_id']

        # 找到对应的分区信息
        zone = next((z for z in zones if z['zone_id'] == str(zone_id)), None)
        if zone is None:
            print(f"  警告: 未找到分区{zone_id}的区域信息")
            continue

        rc = result['runoff_coefficient']

        # 评级
        if rc > criteria.runoff_coefficient_max:
            grade = "异常(>1)"
        elif rc > criteria.runoff_coefficient_warning_high:
            grade = "偏高"
        elif rc < criteria.runoff_coefficient_warning_low:
            grade = "偏低"
        else:
            grade = "正常"

        # 计算子流域数（从zones信息中获取）
        # 注意：这里简化处理，实际应该从subbasins.csv读取
        num_subbasins = 0  # 占位符

        rows.append({
            '分区ID': zone_id,
            '面积(km²)': f"{zone['area_km2']:.2f}",
            '子流域数': num_subbasins,
            '累积降雨(mm)': f"{result['total_precip_mm']:.2f}",
            '累积径流(mm)': f"{result['total_runoff_mm']:.2f}",
            '径流系数': f"{rc:.4f}",
            '等级': grade
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')


def generate_report(
    results: List[Dict],
    validation_result,
    output_path: Path,
    config_file: Path,
    validation_config_file: Path,
    zones: List[Dict]
):
    """生成结果报告"""
    mean_rc = validation_result.metrics.get('mean_runoff_coefficient', 0)

    report = []
    report.append("=" * 80)
    report.append("Step 9 增强输出报告：径流分析与评价 (HBV直接模拟版)")
    report.append("=" * 80)
    report.append("")
    report.append("说明：本报告使用HBV模型直接模拟本地径流，避免汇流累积影响")
    report.append(f"  - 配置文件: {config_file}")
    report.append(f"  - 验证标准: {validation_config_file}")
    report.append(f"  - HBV模拟: 直接计算本地径流（非汇流后discharge）")
    report.append("")
    report.append("生成文件清单:")
    report.append("  - 9.1_zone_cumulative_runoff.png")
    report.append("  - 9.2_runoff_coefficient_evaluation.csv")
    report.append("")
    report.append("径流系数评价结果:")
    report.append(f"  - 面积加权平均径流系数: {mean_rc:.4f}")
    report.append("")
    report.append("各分区径流系数详情:")

    for result in sorted(results, key=lambda x: int(x['zone_id'])):
        zone_id = result['zone_id']
        rc = result['runoff_coefficient']
        precip = result['total_precip_mm']
        runoff = result['total_runoff_mm']
        report.append(f"  - 分区 {zone_id}: {rc:.4f}, 降雨={precip:.1f}mm, 径流={runoff:.1f}mm")

    report.append("")
    report.append("=" * 80)
    report.append("闭环验证结果:")
    if validation_result.is_valid:
        report.append("  验证状态: ✅ 通过")
    else:
        report.append("  验证状态: ❌ 失败")

    report.append("")

    if validation_result.errors:
        report.append(f"  错误 ({len(validation_result.errors)}):")
        for error in validation_result.errors:
            report.append(f"    • {error}")
        report.append("")

    if validation_result.warnings:
        report.append(f"  警告 ({len(validation_result.warnings)}):")
        for warning in validation_result.warnings:
            report.append(f"    • {warning}")
        report.append("")

    report.append("  验证指标:")
    for key, value in validation_result.metrics.items():
        report.append(f"    - {key}: {value:.4f}" if isinstance(value, float) else f"    - {key}: {value}")

    report.append("")
    report.append("=" * 80)
    report.append("")

    output_path.write_text('\n'.join(report), encoding='utf-8')


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Step 9增强 - 使用HBV直接模拟')
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
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Step 9 输出增强：径流分析与HBV模拟")
    print("="*80)
    print(f"配置文件: {args.config}")
    print(f"验证标准: {args.validation_config}")

    # 加载配置
    print("\n⚙ 加载配置...")
    workflow_config = load_workflow_config(args.config)
    hydrologic_criteria = create_hydrologic_criteria(args.validation_config)

    # 获取HBV参数
    hbv_config = workflow_config.get('hbv_model', {})
    use_calibrated = hbv_config.get('use_calibrated', False)

    if use_calibrated and 'calibrated_parameters' in hbv_config:
        hbv_params_source = hbv_config['calibrated_parameters']
        print("  ✓ 使用率定后的HBV参数")
    else:
        hbv_params_source = hbv_config.get('default_parameters', {})
        print("  ⚠ 使用默认HBV参数")

    # 映射HBV参数名称
    hbv_params = {
        'degree_day_factor': hbv_params_source.get('degree_day_factor', 3.0),
        'snow_threshold': hbv_params_source.get('snow_threshold', 0.0),
        'field_capacity': hbv_params_source.get('field_capacity', 100.0),
        'beta': hbv_params_source.get('beta', 1.0),
        'k0': hbv_params_source.get('k0', 0.15),
        'k1': hbv_params_source.get('k1', 0.05),
        'k2': hbv_params_source.get('k2', 0.01),
        'percolation': hbv_params_source.get('percolation', 2.0),
        'initial_snow': hbv_params_source.get('initial_snow', 0.0),
        'initial_soil': hbv_params_source.get('initial_soil', 40.0),
        'initial_upper': hbv_params_source.get('initial_upper', 5.0),
        'initial_lower': hbv_params_source.get('initial_lower', 20.0),
    }

    print(f"  ✓ HBV参数: FC={hbv_params['field_capacity']:.2f}, "
          f"BETA={hbv_params['beta']:.2f}, K0={hbv_params['k0']:.3f}")

    # 加载数据
    print("\n⚙ 加载数据...")
    base_dir = Path(workflow_config['directories']['base_results'])

    precip_path = base_dir / "step_08_areal_rainfall" / "8.1_parameter_areal_precipitation.csv"
    zones_path = base_dir / "parameters" / "parameter_zones.geojson"
    subbasins_csv = base_dir / "parameters" / "parameter_subbasins.csv"

    precip_df = pd.read_csv(precip_path, index_col='Timestamp')

    global zones  # For use in generate_report
    zones = load_zone_info(zones_path)

    print(f"  ✓ 加载降雨数据: {len(precip_df)} 小时")
    print(f"  ✓ 加载分区信息: {len(zones)} 个分区")
    print(f"  ✓ 降雨数据列数: {len(precip_df.columns)}")

    # 为每个分区运行HBV模拟
    print("\n⚙ 运行HBV模拟...")
    results = []

    for zone in sorted(zones, key=lambda x: int(x['zone_id'])):
        zone_id = zone['zone_id']
        area_km2 = zone['area_km2']

        # 加载分区降雨
        precipitation = load_zone_precipitation(precip_df, subbasins_csv, zone_id)

        # 运行HBV模拟
        result = run_hbv_simulation(precipitation, area_km2, hbv_params, int(zone_id))
        results.append(result)

        print(f"  ✓ 分区 {zone_id}: RC={result['runoff_coefficient']:.4f}, "
              f"降雨={result['total_precip_mm']:.1f}mm, "
              f"径流={result['total_runoff_mm']:.1f}mm")

    # 验证径流系数
    print("\n⚙ 闭环验证：径流系数合理性检查")
    runoff_coefficients = {str(r['zone_id']): r['runoff_coefficient'] for r in results}
    validation_result = validate_runoff_coefficient(
        runoff_coefficients,
        criteria=hydrologic_criteria,
        step_name="Step 9 径流系数验证"
    )

    print(validation_result)

    # 生成可视化
    print("\n⚙ 生成可视化输出...")
    output_dir = base_dir / "step_09_runoff"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_cumulative_runoff(results, zones, output_dir / "9.1_zone_cumulative_runoff.png")
    print("  ✓ 生成分区径流累积曲线: 9.1_zone_cumulative_runoff.png")

    save_evaluation_csv(results, zones, output_dir / "9.2_runoff_coefficient_evaluation.csv", hydrologic_criteria)
    print("  ✓ 生成径流系数评价: 9.2_runoff_coefficient_evaluation.csv")

    generate_report(
        results,
        validation_result,
        output_dir / "9.4_enhancement_report.txt",
        args.config,
        args.validation_config,
        zones
    )
    print("  ✓ 生成结果报告: 9.4_enhancement_report.txt")

    # 最终状态
    print("\n" + "="*80)
    if validation_result.is_valid:
        print("✅ Step 9输出增强完成！验证通过")
    else:
        print("⚠️  Step 9输出增强完成！但验证发现错误")
    print(f"📁 输出目录: {output_dir}")
    print("="*80)
    print()

    return 0 if validation_result.is_valid else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
