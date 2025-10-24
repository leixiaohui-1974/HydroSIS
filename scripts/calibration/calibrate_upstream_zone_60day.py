#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高精度参数率定：基于60天数据的上游分区HBV参数率定

目标：实现最上游分区的率定精度尽可能高

策略：
1. 使用60天高质量降雨数据
2. 生成"真实"观测数据（使用已知参数）
3. 基于敏感性分析聚焦关键参数（Beta, Field_Capacity, k0）
4. 使用Differential Evolution全局优化
5. 多目标评估（NSE, KGE, RMSE, PBIAS）
6. 验证参数恢复精度

重构特点：
- 使用 HBVCalibrator 统一校准框架
- 使用标准化的数据和配置接口
- 自动参数恢复验证
"""

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from hydrosis.calibration.base_calibrator import CalibrationData, CalibrationConfig
from hydrosis.calibration.hbv_calibrator import HBVCalibrator
from hydrosis.evaluation.metrics import calculate_metrics


def generate_synthetic_observations(rainfall: np.ndarray, true_params: dict) -> np.ndarray:
    """
    使用已知参数生成合成观测数据

    Parameters
    ----------
    rainfall : np.ndarray
        降雨时间序列 (mm/h)
    true_params : dict
        真实HBV参数

    Returns
    -------
    np.ndarray
        合成的观测径流数据
    """
    # 使用HBV模型生成"真实"径流
    from hydrosis.runoff.hbv import HBVRunoff

    # 创建一个模拟的子流域
    class MockSubbasin:
        def __init__(self):
            self.area_km2 = 10.0  # 假设面积

    subbasin = MockSubbasin()

    try:
        model = HBVRunoff(parameters=true_params)
        observed = np.array(model.simulate(subbasin, rainfall.tolist()))
    except Exception as e:
        print(f"  ⚠ 使用HBVRunoff失败: {e}")
        print("  使用简化的HBV模型")
        observed = run_simple_hbv(rainfall, true_params)

    return observed


def run_simple_hbv(rainfall: np.ndarray, params: dict) -> np.ndarray:
    """简化的HBV模型实现（作为备用）"""
    snow = params.get('initial_snow', 0.0)
    soil = params.get('initial_soil', 0.0)
    upper = params.get('initial_upper', 0.0)
    lower = params.get('initial_lower', 0.0)

    field_capacity = params.get('field_capacity', 80.0)
    beta = max(1e-6, params.get('beta', 1.0))
    k0 = params.get('k0', 0.12)
    k1 = params.get('k1', 0.08)
    k2 = params.get('k2', 0.02)
    percolation = params.get('percolation', 1.0)

    runoff_list = []

    for p in rainfall:
        effective_precip = p  # 简化：忽略融雪
        soil_deficit = max(0.0, field_capacity - soil)

        if soil > 0 and field_capacity > 0:
            recharge = effective_precip * ((soil / field_capacity) ** beta)
        else:
            recharge = 0.0

        recharge = min(recharge, soil_deficit)
        soil += effective_precip - recharge

        quickflow = k0 * upper
        actual_percolation = min(percolation, max(0.0, upper + recharge - quickflow))
        upper += recharge - quickflow - actual_percolation
        upper = max(0.0, upper)

        lower += actual_percolation - k2 * lower
        lower = max(0.0, lower)

        baseflow = k1 * upper + k2 * lower
        total_runoff = quickflow + baseflow

        runoff_list.append(total_runoff)

    return np.array(runoff_list)


def main():
    """主函数：高精度参数率定"""
    print("=" * 80)
    print("高精度参数率定：基于60天数据的上游分区HBV参数率定")
    print("=" * 80)

    # ========================================================================
    # 第1步：加载60天降雨数据
    # ========================================================================
    print("\n[1/5] 加载60天降雨数据...")

    data_file = Path('results/extended_timeseries_60days/timeseries_60days.csv')
    if not data_file.exists():
        print(f"❌ 错误: 数据文件不存在: {data_file}")
        print("   请先运行生成60天时间序列的脚本")
        return

    df = pd.read_csv(data_file)
    rainfall = df['precipitation_mm_per_hour'].values
    timestamps = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else None

    print(f"  ✓ 已加载 {len(rainfall)} 个时间步")
    print(f"  ✓ 总降雨量: {rainfall.sum():.2f} mm")

    # ========================================================================
    # 第2步：生成"真实"观测数据
    # ========================================================================
    print("\n[2/5] 生成观测数据（使用已知真实参数）...")

    # 真实参数（我们将尝试恢复这些参数）
    true_params = {
        'degree_day_factor': 3.0,
        'snow_threshold': 0.0,
        'field_capacity': 90.0,    # 真实值（待率定）
        'beta': 1.3,               # 真实值（待率定）
        'k0': 0.15,                # 真实值（待率定）
        'k1': 0.08,
        'k2': 0.02,
        'percolation': 1.0,
        'initial_snow': 0.0,
        'initial_soil': 0.0,
        'initial_upper': 0.0,
        'initial_lower': 0.0,
    }

    print("  真实参数（将尝试率定恢复）:")
    for key in ['field_capacity', 'beta', 'k0']:
        print(f"    {key}: {true_params[key]}")

    observed = generate_synthetic_observations(rainfall, true_params)

    print(f"  ✓ 生成观测径流: {len(observed)} 个时间步")
    print(f"  ✓ 总径流量: {observed.sum():.2f} mm")
    print(f"  ✓ 径流系数: {observed.sum() / rainfall.sum():.4f}")

    # ========================================================================
    # 第3步：准备校准数据
    # ========================================================================
    print("\n[3/5] 准备校准数据...")

    calibration_data = CalibrationData(
        rainfall=rainfall,
        observed_runoff=observed,
        timestamps=timestamps,
    )

    print(f"  ✓ 校准数据已准备")

    # ========================================================================
    # 第4步：配置校准参数
    # ========================================================================
    print("\n[4/5] 配置校准参数...")

    # 重点率定最敏感的参数
    param_bounds = {
        'field_capacity': (60, 120),   # 真实值90在中间
        'beta': (0.8, 1.8),            # 真实值1.3在中间
        'k0': (0.09, 0.20),            # 真实值0.15在中间
    }

    calibration_config = CalibrationConfig(
        param_bounds=param_bounds,
        algorithm='differential_evolution',
        max_iterations=200,
        population_size=20,
        objective='nse',
        random_seed=42,
    )

    print(f"  待率定参数: {list(param_bounds.keys())}")
    for name, bounds in param_bounds.items():
        true_val = true_params[name]
        print(f"    {name}: [{bounds[0]}, {bounds[1]}], 真实值={true_val}")

    # ========================================================================
    # 第5步：执行校准
    # ========================================================================
    print("\n[5/5] 执行参数率定...")

    calibrator = HBVCalibrator(
        data=calibration_data,
        config=calibration_config,
    )

    result = calibrator.calibrate()

    # 打印结果摘要
    print("\n" + "=" * 80)
    print("✓ 率定完成！")
    print("=" * 80)

    print(f"\n性能指标:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.6f}")

    # 参数恢复精度分析
    print("\n参数恢复精度:")
    print(f"  {'参数':<20} {'真实值':<12} {'率定值':<12} {'误差%':<10}")
    print(f"  {'-'*54}")

    param_errors = []
    for param_name in param_bounds.keys():
        true_val = true_params[param_name]
        calib_val = result.best_params[param_name]
        error_pct = abs(calib_val - true_val) / true_val * 100
        param_errors.append(error_pct)
        print(f"  {param_name:<20} {true_val:<12.4f} {calib_val:<12.4f} {error_pct:<10.2f}")

    avg_error = np.mean(param_errors)
    print(f"\n  平均参数误差: {avg_error:.2f}%")

    # 保存结果
    output_dir = Path('results/calibration_60day')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 添加真实参数到元数据
    result.metadata['true_params'] = {k: true_params[k] for k in param_bounds.keys()}
    result.metadata['parameter_recovery_errors'] = {
        k: abs(result.best_params[k] - true_params[k]) / true_params[k] * 100
        for k in param_bounds.keys()
    }

    calibrator.save_results(result, output_dir)

    print(f"\n输出目录: {output_dir}")

    # 生成详细报告
    report_file = output_dir / 'calibration_report.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("高精度参数率定报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 数据概况\n")
        f.write(f"   时间长度: 60天 ({len(rainfall)}小时)\n")
        f.write(f"   总降雨量: {rainfall.sum():.2f} mm\n")
        f.write(f"   总径流量: {observed.sum():.2f} mm\n")
        f.write(f"   径流系数: {observed.sum() / rainfall.sum():.4f}\n\n")

        f.write("2. 率定设置\n")
        f.write(f"   算法: {calibration_config.algorithm}\n")
        f.write(f"   目标函数: {calibration_config.objective}\n")
        f.write(f"   率定参数: {list(param_bounds.keys())}\n")
        f.write(f"   最大迭代数: {calibration_config.max_iterations}\n")
        f.write(f"   种群大小: {calibration_config.population_size}\n\n")

        f.write("3. 率定结果\n")
        f.write(f"   迭代次数: {result.n_iterations}\n")
        f.write(f"   函数评估: {result.n_evaluations}\n")
        f.write(f"   计算时间: {result.computation_time:.1f}秒\n\n")

        f.write("4. 性能指标\n")
        for key, value in result.metrics.items():
            f.write(f"   {key}: {value:.6f}\n")
        f.write("\n")

        f.write("5. 参数恢复精度\n")
        f.write(f"   {'参数':<20} {'真实值':<12} {'率定值':<12} {'误差%':<10}\n")
        f.write(f"   {'-'*54}\n")
        for param_name in param_bounds.keys():
            true_val = true_params[param_name]
            calib_val = result.best_params[param_name]
            error_pct = abs(calib_val - true_val) / true_val * 100
            f.write(f"   {param_name:<20} {true_val:<12.4f} {calib_val:<12.4f} {error_pct:<10.2f}\n")
        f.write(f"\n   平均误差: {avg_error:.2f}%\n\n")

        f.write("6. 结论\n")
        nse = result.metrics.get('NSE', result.metrics.get('nse', 0))
        if nse > 0.9 and avg_error < 5:
            f.write("   ✓ 优秀: NSE > 0.9 且参数恢复误差 < 5%\n")
        elif nse > 0.75 and avg_error < 10:
            f.write("   ✓ 良好: NSE > 0.75 且参数恢复误差 < 10%\n")
        elif nse > 0.5:
            f.write("   ⚠ 可接受: NSE > 0.5 但参数恢复精度一般\n")
        else:
            f.write("   ❌ 不佳: NSE < 0.5 或参数恢复精度较低\n")

    print(f"  ✓ 报告已保存: {report_file}")
    print()


if __name__ == '__main__':
    main()
