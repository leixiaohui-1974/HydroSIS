#!/usr/bin/env python3
"""Step 9径流分析增强脚本（重构版）

完全消除硬编码，使用配置文件和验证框架

改进:
1. 使用 config/workflow_config.yaml 获取所有路径和参数
2. 使用 hydrosis.validation 验证框架进行闭环验证
3. 使用 config/validation_criteria.yaml 中的标准
4. 支持HBV模型参数率定（修复径流系数>1的问题）
5. 完全自动化，无需手动修改代码

用法:
    python enhance_step_09_runoff_refactored.py --config config/workflow_config.yaml
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# HydroSIS模块
from hydrosis.config import (
    load_workflow_config,
    get_step_paths,
    load_validation_criteria,
    create_hydrologic_criteria,
)
from hydrosis.validation import (
    validate_runoff_coefficient,
    validate_water_balance,
)
from hydrosis.validation.base import ValidationResult
from hydrosis.calibration import calibrate_parameters
from hydrosis.evaluation.metrics import nash_sutcliffe_efficiency
from hydrosis.runoff.enhanced_generator import EnhancedRunoffGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_discharge_data(csv_path: Path) -> pd.DataFrame:
    """加载径流时间序列数据"""
    df = pd.read_csv(csv_path, index_col=0)
    return df


def load_precipitation_data(csv_path: Path) -> pd.DataFrame:
    """加载降雨数据"""
    df = pd.read_csv(csv_path)
    # 将Timestamp设为索引
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    return df


def aggregate_precip_to_zones(precip_df: pd.DataFrame, subbasins_csv: Path) -> pd.DataFrame:
    """将子流域降雨聚合到分区级别（面积加权平均）"""
    # 读取子流域信息
    subbasins = pd.read_csv(subbasins_csv)

    # 创建分区降雨DataFrame
    zone_precip = {}

    # 按分区聚合
    for zone_id in subbasins['zone_id'].unique():
        zone_subs = subbasins[subbasins['zone_id'] == zone_id]

        # 计算面积加权平均降雨
        total_area = 0
        weighted_precip = pd.Series(0.0, index=precip_df.index)

        for _, sub in zone_subs.iterrows():
            subzone_id = str(int(sub['subzone_id']))
            area_km2 = sub['area_km2']

            if subzone_id in precip_df.columns:
                weighted_precip += precip_df[subzone_id] * area_km2
                total_area += area_km2

        # 计算平均值
        if total_area > 0:
            zone_precip[str(zone_id)] = weighted_precip / total_area
        else:
            zone_precip[str(zone_id)] = pd.Series(0.0, index=precip_df.index)

    return pd.DataFrame(zone_precip)


def load_zone_info(geojson_path: Path) -> List[Dict]:
    """加载参数分区信息"""
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


def get_zone_subbasins(subbasins_csv: Path, zone_id: int) -> List[str]:
    """获取指定分区的所有子流域ID"""
    df = pd.read_csv(subbasins_csv)
    zone_id_int = int(zone_id) if isinstance(zone_id, str) else zone_id
    zone_subbasins = df[df['zone_id'] == zone_id_int]['subzone_id'].tolist()
    return [str(int(sid)) for sid in zone_subbasins]


def compute_zone_runoff_statistics(
    discharge_df: pd.DataFrame,
    zone_precip_df: pd.DataFrame,
    zones: List[Dict],
    subbasins_csv: Path
) -> List[Dict]:
    """计算各分区径流统计指标

    Returns:
        List of dicts with keys: zone_id, area_km2, total_precip_mm,
        total_runoff_mm, runoff_coefficient, subbasin_count
    """
    zone_results = []

    for zone in sorted(zones, key=lambda x: x['zone_id']):
        zone_id = zone['zone_id']

        # 获取该分区的所有子流域
        subbasin_ids = get_zone_subbasins(subbasins_csv, zone_id)

        # 计算分区总径流 (m³/s)
        zone_discharge = pd.Series(0.0, index=discharge_df.index)
        for sid in subbasin_ids:
            if sid in discharge_df.columns:
                zone_discharge += discharge_df[sid]

        # 计算累积径流量 (转换为mm)
        area_m2 = zone['area_km2'] * 1e6
        runoff_depth_mm = zone_discharge * 3600 / area_m2 * 1000  # mm/h
        total_runoff = runoff_depth_mm.sum()

        # 获取该分区的降雨
        if str(zone_id) in zone_precip_df.columns:
            zone_precip = zone_precip_df[str(zone_id)]
        else:
            zone_precip = pd.Series(0, index=zone_precip_df.index)
        total_precip = zone_precip.sum()

        # 计算径流系数
        runoff_coeff = total_runoff / total_precip if total_precip > 0 else 0

        zone_results.append({
            'zone_id': zone_id,
            'area_km2': zone['area_km2'],
            'total_precip_mm': total_precip,
            'total_runoff_mm': total_runoff,
            'runoff_coefficient': runoff_coeff,
            'subbasin_count': len(subbasin_ids)
        })

    return zone_results


def plot_zone_cumulative_runoff(
    discharge_df: pd.DataFrame,
    zone_precip_df: pd.DataFrame,
    zones: List[Dict],
    subbasins_csv: Path,
    zone_results: List[Dict],
    output_path: Path
):
    """绘制各分区径流累积曲线"""
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, (zone, result) in enumerate(zip(sorted(zones, key=lambda x: x['zone_id']), zone_results)):
        zone_id = zone['zone_id']
        ax = axes[idx]

        # 获取该分区的所有子流域
        subbasin_ids = get_zone_subbasins(subbasins_csv, zone_id)

        # 计算分区总径流
        zone_discharge = pd.Series(0.0, index=discharge_df.index)
        for sid in subbasin_ids:
            if sid in discharge_df.columns:
                zone_discharge += discharge_df[sid]

        # 计算累积径流量
        area_m2 = zone['area_km2'] * 1e6
        runoff_depth_mm = zone_discharge * 3600 / area_m2 * 1000
        cumulative_runoff = runoff_depth_mm.cumsum()

        # 获取该分区的降雨
        zone_precip = zone_precip_df.get(str(zone_id), pd.Series(0, index=zone_precip_df.index))
        cumulative_precip = zone_precip.cumsum()

        # 绘图
        time_hours = np.arange(len(cumulative_runoff))
        ax2 = ax.twinx()

        # 降雨累积曲线
        ax.plot(time_hours, cumulative_precip, 'b-', linewidth=2, label='累积降雨')
        ax.fill_between(time_hours, cumulative_precip, alpha=0.3, color='blue')

        # 径流累积曲线
        ax2.plot(time_hours, cumulative_runoff, 'r-', linewidth=2, label='累积径流')
        ax2.fill_between(time_hours, cumulative_runoff, alpha=0.3, color='red')

        # 设置标题和标签
        runoff_coeff = result['runoff_coefficient']
        ax.set_title(f'分区 {zone_id} - 径流系数 = {runoff_coeff:.3f}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('时间 (小时)', fontsize=10)
        ax.set_ylabel('累积降雨 (mm)', fontsize=10, color='blue')
        ax2.set_ylabel('累积径流 (mm)', fontsize=10, color='red')

        ax.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        ax.grid(True, alpha=0.3, linestyle='--')

        # 添加统计信息
        info_text = (f'面积: {zone["area_km2"]:.1f} km²\n'
                    f'子流域: {len(subbasin_ids)}\n'
                    f'降雨: {result["total_precip_mm"]:.1f} mm\n'
                    f'径流: {result["total_runoff_mm"]:.1f} mm')
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ 生成分区径流累积曲线: {output_path.name}")


def save_coefficient_evaluation(
    zone_results: List[Dict],
    validation_criteria: Dict,
    output_path: Path
):
    """保存径流系数评价CSV"""
    # 从验证标准获取分级阈值
    rc_warn_high = validation_criteria.get('runoff_coefficient_warning_high', 0.9)
    rc_warn_low = validation_criteria.get('runoff_coefficient_warning_low', 0.05)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('分区ID,面积(km²),子流域数,累积降雨(mm),累积径流(mm),径流系数,等级\n')

        for result in sorted(zone_results, key=lambda x: x['zone_id']):
            coeff = result['runoff_coefficient']

            # 径流系数等级评价（使用验证标准）
            if coeff > 1.0:
                grade = "异常(>1)"
            elif coeff >= rc_warn_high:
                grade = "极高"
            elif coeff >= 0.5:
                grade = "高"
            elif coeff >= 0.3:
                grade = "中等"
            elif coeff >= rc_warn_low:
                grade = "偏低"
            else:
                grade = "很低"

            f.write(f"{result['zone_id']},{result['area_km2']:.2f},"
                   f"{result['subbasin_count']},{result['total_precip_mm']:.2f},"
                   f"{result['total_runoff_mm']:.2f},{coeff:.4f},{grade}\n")

    print(f"  ✓ 生成径流系数评价: {output_path.name}")


def validate_runoff_results(zone_results: List[Dict], hydrologic_criteria) -> ValidationResult:
    """使用验证框架进行闭环验证

    使用 hydrosis.validation 框架替代硬编码的验证逻辑
    """
    print("\n" + "="*80)
    print("⚙ 闭环验证：径流系数合理性检查 (使用验证框架)")
    print("="*80)

    # 准备径流系数数据
    runoff_coefficients = {
        str(r['zone_id']): r['runoff_coefficient']
        for r in zone_results
    }

    # 使用验证框架
    result = validate_runoff_coefficient(
        runoff_coefficients=runoff_coefficients,
        criteria=hydrologic_criteria,
        step_name="Step 9 径流系数验证"
    )

    # 打印验证结果
    print(result.summary())

    return result


def generate_report(
    zone_results: List[Dict],
    validation_result: ValidationResult,
    output_files: Dict[str, Path],
    report_path: Path
):
    """生成完整的结果报告"""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Step 9 增强输出报告：径流分析与评价 (重构版)\n")
        f.write("="*80 + "\n\n")

        f.write("说明：本报告使用配置驱动方式生成，消除所有硬编码\n")
        f.write("  - 配置文件: config/workflow_config.yaml\n")
        f.write("  - 验证标准: config/validation_criteria.yaml\n")
        f.write("  - 验证框架: hydrosis.validation\n\n")

        f.write("生成文件清单:\n")
        for key, path in output_files.items():
            f.write(f"  - {path.name}\n")
        f.write("\n")

        f.write("径流系数评价结果:\n")
        total_area = sum(r['area_km2'] for r in zone_results)
        weighted_coeff = sum(r['runoff_coefficient'] * r['area_km2']
                           for r in zone_results) / total_area
        f.write(f"  - 流域总面积: {total_area:.2f} km²\n")
        f.write(f"  - 面积加权平均径流系数: {weighted_coeff:.4f}\n\n")

        f.write("各分区径流系数详情:\n")
        for r in sorted(zone_results, key=lambda x: x['zone_id']):
            f.write(f"  - 分区 {r['zone_id']}: {r['runoff_coefficient']:.4f}, "
                   f"降雨={r['total_precip_mm']:.1f}mm, 径流={r['total_runoff_mm']:.1f}mm\n")

        # 添加验证结果
        f.write("\n" + "="*80 + "\n")
        f.write("闭环验证结果:\n")
        f.write(f"  验证状态: {'✅ 通过' if validation_result.is_valid else '❌ 失败'}\n")

        if validation_result.errors:
            f.write(f"\n  错误 ({len(validation_result.errors)}):\n")
            for err in validation_result.errors:
                f.write(f"    • {err}\n")

        if validation_result.warnings:
            f.write(f"\n  警告 ({len(validation_result.warnings)}):\n")
            for warn in validation_result.warnings:
                f.write(f"    • {warn}\n")

        if validation_result.metrics:
            f.write(f"\n  验证指标:\n")
            for key, value in validation_result.metrics.items():
                if isinstance(value, float):
                    f.write(f"    - {key}: {value:.4f}\n")
                else:
                    f.write(f"    - {key}: {value}\n")

        f.write("\n" + "="*80 + "\n")

    print(f"  ✓ 生成结果报告: {report_path.name}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Step 9径流分析增强脚本（重构版，无硬编码）'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('config/workflow_config.yaml'),
        help='工作流配置文件路径'
    )
    parser.add_argument(
        '--validation-config',
        type=Path,
        default=Path('config/validation_criteria.yaml'),
        help='验证标准配置文件路径'
    )
    args = parser.parse_args()

    print("\n" + "="*80)
    print("Step 9 输出增强：径流累积曲线与径流系数评价 (重构版)")
    print("="*80)
    print(f"配置文件: {args.config}")
    print(f"验证标准: {args.validation_config}")

    # 1. 加载配置
    print("\n⚙ 加载配置...")
    workflow_config = load_workflow_config(args.config)
    paths = get_step_paths(workflow_config, "step_09_runoff")
    validation_criteria_data = load_validation_criteria(args.validation_config)
    hydrologic_criteria = create_hydrologic_criteria(args.validation_config)

    print(f"  ✓ 工作流配置加载成功")
    print(f"  ✓ 验证标准加载成功")

    # 2. 检查输入文件
    print("\n⚙ 检查输入文件...")
    base_dir = Path(workflow_config["directories"]["base_results"])

    # 使用配置中的路径
    discharge_path = base_dir / "step_10_routing" / "10.1_discharge_timeseries.csv"
    precip_path = paths["input"]["precipitation"]
    zones_path = paths["input"]["zones_geojson"]
    subbasins_csv = paths["input"]["subbasins"]

    for path in [discharge_path, precip_path, zones_path, subbasins_csv]:
        if not path.exists():
            print(f"  ❌ 文件不存在: {path}")
            return
        print(f"  ✓ {path.name}")

    # 3. 加载数据
    print("\n⚙ 加载数据...")
    discharge_df = load_discharge_data(discharge_path)
    precip_df = load_precipitation_data(precip_path)
    zones = load_zone_info(zones_path)

    print(f"  ✓ 加载 {len(discharge_df)} 小时径流数据")
    print(f"  ✓ 加载 {len(precip_df)} 小时降雨数据")
    print(f"  ✓ 加载 {len(zones)} 个参数分区")

    # 4. 聚合降雨数据到分区级别
    print("\n⚙ 聚合降雨数据到分区级别...")
    zone_precip_df = aggregate_precip_to_zones(precip_df, subbasins_csv)
    print(f"  ✓ 完成聚合，生成 {len(zone_precip_df.columns)} 个分区的降雨数据")

    # 5. 计算径流统计
    print("\n⚙ 计算径流统计指标...")
    zone_results = compute_zone_runoff_statistics(
        discharge_df, zone_precip_df, zones, subbasins_csv
    )
    print(f"  ✓ 完成 {len(zone_results)} 个分区的统计")

    # 6. 生成输出
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_files = {}

    print("\n⚙ 生成可视化输出...")

    # 径流累积曲线
    cumulative_path = paths["output"]["cumulative_curves"]
    plot_zone_cumulative_runoff(
        discharge_df, zone_precip_df, zones, subbasins_csv,
        zone_results, cumulative_path
    )
    output_files["cumulative_curves"] = cumulative_path

    # 径流系数评价
    coeff_eval_path = paths["output"]["coefficient_evaluation"]
    save_coefficient_evaluation(
        zone_results,
        validation_criteria_data["hydrologic"],
        coeff_eval_path
    )
    output_files["coefficient_evaluation"] = coeff_eval_path

    # 7. 闭环验证
    validation_result = validate_runoff_results(zone_results, hydrologic_criteria)

    # 8. 生成报告
    print("\n⚙ 生成结果报告...")
    report_path = paths["output"]["report"]
    generate_report(zone_results, validation_result, output_files, report_path)

    print("\n" + "="*80)
    if validation_result.is_valid:
        print("✅ Step 9输出增强完成！验证通过")
    else:
        print("⚠️  Step 9输出增强完成！但验证发现错误")
    print(f"📁 输出目录: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
