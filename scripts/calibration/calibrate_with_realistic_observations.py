#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
真实场景的参数率定：使用增强模型生成观测数据，用HBV模型率定

目标：
1. 观测数据来自增强模型（不同的模型结构）
2. 添加观测噪声（模拟真实测量误差）
3. 用HBV模型率定（测试结构不匹配情况）
4. 评估模型偏差和不确定性

重构特点：
- 使用 HBVCalibrator 统一校准框架
- 使用 CalibrationData 和 CalibrationConfig 标准接口
- 自动保存结果和生成报告
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


def load_or_generate_observations(rainfall: np.ndarray) -> tuple:
    """
    生成或加载"观测"径流数据

    注意：如果simple_runoff_generator模块可用，使用它生成观测数据
    否则使用简化的方法

    Parameters
    ----------
    rainfall : np.ndarray
        降雨时间序列 (mm/h)

    Returns
    -------
    tuple
        (observed_runoff, true_runoff, metadata)
    """
    try:
        # 尝试导入自定义简单模型
        from simple_runoff_generator import SimpleRunoffGenerator, add_observation_errors

        print("  使用 SimpleRunoffGenerator 生成观测数据:")
        print("    模型结构: 初损后损 + 阈值效应 + 饱和超渗 + 非线性退水")

        generator = SimpleRunoffGenerator(
            initial_loss=18.0,
            constant_loss=0.4,
            runoff_coefficient=0.42,
            reservoir_k=0.18,
            initial_storage=8.0,
            random_noise_level=0.15,
            random_seed=42,
            rainfall_threshold=2.5,
            saturation_capacity=30.0,
            recession_exponent=1.8,
            time_delay_std=3.0,
        )

        runoff_true, stats = generator.generate(rainfall)
        observed = add_observation_errors(runoff_true, seed=42)

        print(f"    真实径流系数: {stats['runoff_coefficient']:.4f}")
        print(f"    观测径流系数: {observed.sum() / rainfall.sum():.4f}")

        metadata = {
            'generation_method': 'SimpleRunoffGenerator',
            'true_runoff_coefficient': stats['runoff_coefficient'],
            'noise_level': 0.10,
            'systematic_bias': 1.05,
        }

    except ImportError:
        # 如果模块不可用，使用简化方法
        print("  ⚠ SimpleRunoffGenerator 不可用，使用简化方法生成观测数据")
        print("    使用固定径流系数 0.42")

        # 简单的径流生成：固定径流系数 + 滞后
        runoff_coefficient = 0.42
        runoff_true = rainfall * runoff_coefficient

        # 添加简单的滞后效应
        runoff_true = np.convolve(runoff_true, [0.3, 0.5, 0.2], mode='same')

        # 添加噪声
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, len(runoff_true))
        observed = runoff_true * (1 + noise) * 1.05  # 5%噪声 + 5%系统偏差
        observed = np.maximum(observed, 0)

        metadata = {
            'generation_method': 'simplified',
            'true_runoff_coefficient': runoff_coefficient,
            'noise_level': 0.05,
            'systematic_bias': 1.05,
        }

        print(f"    真实径流系数: {runoff_coefficient:.4f}")
        print(f"    观测径流系数: {observed.sum() / rainfall.sum():.4f}")

    return observed, runoff_true, metadata


def main():
    """主函数：真实场景的参数率定"""
    print("=" * 80)
    print("真实场景的参数率定测试")
    print("=" * 80)
    print("\n策略:")
    print("  1. 用增强模型生成'观测数据'（不同结构）")
    print("  2. 添加观测噪声（模拟真实测量误差）")
    print("  3. 用HBV模型率定（结构不匹配）")
    print("  4. 评估模型性能和局限性")

    # ========================================================================
    # 第1步：加载降雨数据
    # ========================================================================
    print("\n[1/5] 加载60天降雨数据...")

    data_file = Path('results/extended_timeseries_60days/timeseries_60days.csv')
    if not data_file.exists():
        print(f"❌ 错误: 数据文件不存在: {data_file}")
        print("   请先运行生成60天时间序列的脚本")
        return

    df = pd.read_csv(data_file)
    rainfall = df['precipitation_mm_per_hour'].values

    print(f"  ✓ 已加载 {len(rainfall)} 个时间步")
    print(f"  ✓ 总降雨量: {rainfall.sum():.2f} mm")

    # ========================================================================
    # 第2步：生成真实的"观测数据"
    # ========================================================================
    print("\n[2/5] 生成真实的观测数据...")

    observed, true_runoff, metadata = load_or_generate_observations(rainfall)

    print(f"\n  ✓ 观测数据统计:")
    print(f"    总径流量: {observed.sum():.2f} mm")
    print(f"    最大流量: {observed.max():.4f} mm/h")
    print(f"    平均流量: {observed.mean():.4f} mm/h")

    # ========================================================================
    # 第3步：准备校准数据
    # ========================================================================
    print("\n[3/5] 准备校准数据...")

    calibration_data = CalibrationData(
        rainfall=rainfall,
        observed_runoff=observed,
        timestamps=pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else None,
    )

    print(f"  ✓ 校准数据已准备")
    print(f"    降雨时间步: {len(rainfall)}")
    print(f"    观测径流时间步: {len(observed)}")

    # ========================================================================
    # 第4步：配置校准参数
    # ========================================================================
    print("\n[4/5] 配置校准参数...")

    # HBV率定参数（尝试用HBV拟合增强模型生成的数据）
    param_bounds = {
        'field_capacity': (50, 150),
        'beta': (0.5, 2.5),
        'k0': (0.05, 0.30),
        'k1': (0.03, 0.15),
    }

    calibration_config = CalibrationConfig(
        param_bounds=param_bounds,
        algorithm='differential_evolution',
        max_iterations=150,
        population_size=15,
        objective='nse',
        random_seed=42,
    )

    print(f"  待率定参数: {list(param_bounds.keys())}")
    print(f"  优化算法: {calibration_config.algorithm}")
    print(f"  最大迭代数: {calibration_config.max_iterations}")

    # ========================================================================
    # 第5步：执行校准
    # ========================================================================
    print("\n[5/5] 执行HBV参数率定...")
    print("  警告: HBV结构与观测数据的真实来源（增强模型）不同")
    print("  预期: NSE可能 < 0.9（结构误差）")

    calibrator = HBVCalibrator(
        data=calibration_data,
        config=calibration_config,
    )

    result = calibrator.calibrate()

    # 打印结果摘要
    print("\n" + "=" * 80)
    print("✓ 率定完成！")
    print("=" * 80)

    print(f"\n性能指标（HBV vs 观测）:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.6f}")

    # 与真实数据比较（诊断用）
    if true_runoff is not None:
        print("\n性能指标（HBV vs 真实，无噪声）:")
        true_metrics = calculate_metrics(true_runoff, result.simulated)
        for key, value in true_metrics.items():
            print(f"  {key}: {value:.6f}")

        # 误差分析
        total_error = np.sum((observed - result.simulated)**2)
        noise_error = np.sum((observed - true_runoff)**2)
        struct_error = np.sum((true_runoff - result.simulated)**2)

        print(f"\n误差分析:")
        print(f"  总误差 (MSE): {total_error:.6f}")
        if total_error > 0:
            print(f"  观测噪声导致: {noise_error:.6f} ({noise_error/total_error*100:.1f}%)")
            print(f"  结构误差导致: {struct_error:.6f} ({struct_error/total_error*100:.1f}%)")

    # 保存结果
    output_dir = Path('results/realistic_calibration')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存扩展信息
    result.metadata.update(metadata)
    result.metadata['observation_generation'] = metadata['generation_method']

    calibrator.save_results(result, output_dir)

    print(f"\n输出目录: {output_dir}")

    # 生成报告
    report_file = output_dir / 'calibration_report.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("真实场景的参数率定报告\n")
        f.write("=" * 80 + "\n\n")

        f.write("1. 测试设计\n")
        f.write(f"   观测数据来源: {metadata['generation_method']}\n")
        f.write(f"   观测噪声: {metadata['noise_level']*100:.0f}%\n")
        f.write(f"   系统偏差: {(metadata['systematic_bias']-1)*100:.0f}%\n")
        f.write("   率定模型: HBV（结构不匹配）\n\n")

        f.write("2. 数据概况\n")
        f.write(f"   时间长度: 60天\n")
        f.write(f"   总降雨量: {rainfall.sum():.2f} mm\n")
        f.write(f"   观测径流量: {observed.sum():.2f} mm\n")
        f.write(f"   HBV模拟径流量: {result.simulated.sum():.2f} mm\n\n")

        f.write("3. HBV率定结果\n")
        f.write(f"   迭代次数: {result.n_iterations}\n")
        f.write(f"   函数评估: {result.n_evaluations}\n")
        f.write(f"   计算时间: {result.computation_time:.1f}秒\n\n")

        f.write("4. 性能指标\n")
        for key, value in result.metrics.items():
            f.write(f"   {key}: {value:.6f}\n")
        f.write("\n")

        f.write("5. 最优参数\n")
        for key, value in result.best_params.items():
            f.write(f"   {key}: {value:.6f}\n")
        f.write("\n")

        f.write("6. 结论\n")
        nse = result.metrics.get('NSE', result.metrics.get('nse', 0))
        if nse > 0.85:
            f.write("   ✓ 优秀: 尽管结构不匹配，HBV仍能很好拟合观测数据\n")
        elif nse > 0.7:
            f.write("   ✓ 良好: HBV基本能拟合观测数据，但存在结构误差\n")
        elif nse > 0.5:
            f.write("   ⚠ 可接受: HBV拟合一般，结构差异明显\n")
        else:
            f.write("   ❌ 不佳: HBV难以拟合观测数据，结构不匹配严重\n")

    print(f"  ✓ 报告已保存: {report_file}")
    print()


if __name__ == '__main__':
    main()
