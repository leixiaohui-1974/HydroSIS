#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""参数敏感性分析示例脚本

演示如何对水文模型进行参数敏感性分析，识别关键参数。

主要功能：
1. 对任意水文模型进行敏感性分析
2. 识别高敏感性和低敏感性参数
3. 生成参数校准建议
4. 根据敏感性调整参数边界

支持的方法：
- OAT (One-At-a-Time): 局部敏感性
- Morris: 全局敏感性筛选

Author: Claude Code
Date: 2025-01-24
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from hydrosis.calibration import (
    GenericHydrologicCalibrator,
    CalibrationData,
    analyze_model_sensitivity
)


def main():
    """主函数：参数敏感性分析"""
    print("=" * 80)
    print("水文模型参数敏感性分析")
    print("=" * 80)

    # ========================================================================
    # 步骤 1: 准备数据
    # ========================================================================
    print("\n步骤 1: 准备数据")
    print("-" * 80)

    # 生成合成数据
    n_hours = 30 * 24
    np.random.seed(42)

    precipitation = np.random.gamma(0.5, 0.3, n_hours)
    for event_start in [100, 300, 500]:
        event_length = 36
        precipitation[event_start:event_start+event_length] += np.random.gamma(2, 3, event_length)

    area_km2 = 100.0
    times = pd.date_range('2024-01-01', periods=n_hours, freq='H')

    # 生成观测数据（使用XinAnJiang模型）
    from hydrosis.runoff.xinanjiang import XinAnJiangRunoff

    class MockSubbasin:
        def __init__(self, area_km2):
            self.area_km2 = area_km2

    true_params = {'wm': 150.0, 'b': 0.3, 'imp': 0.05, 'recession': 0.6}
    subbasin = MockSubbasin(area_km2)
    model = XinAnJiangRunoff(true_params)
    observed_runoff = np.array(model.simulate(subbasin, precipitation.tolist()))

    # 添加噪声
    noise = np.random.normal(0, 0.03, len(observed_runoff))
    observed_runoff = observed_runoff * (1 + noise)
    observed_runoff = np.maximum(observed_runoff, 0)

    print(f"✓ 数据准备完成")
    print(f"  时间长度: 30天")
    print(f"  流域面积: {area_km2:.2f} km²")

    # 创建校准数据
    calib_data = CalibrationData(
        precipitation=precipitation,
        observed_runoff=observed_runoff,
        area_km2=area_km2,
        times=times
    )

    # ========================================================================
    # 步骤 2: 选择模型
    # ========================================================================
    print("\n步骤 2: 选择要分析的模型")
    print("-" * 80)

    # 列出可用模型
    available = GenericHydrologicCalibrator.list_available_models()
    runoff_models = list(available['runoff_models'].keys())

    print(f"可用的产流模型:")
    for i, model in enumerate(runoff_models, 1):
        print(f"  {i}. {model}")

    # 选择模型进行分析
    models_to_analyze = ['xin_an_jiang']

    # 如果hbv可用，也分析它
    if 'hbv' in runoff_models:
        models_to_analyze.append('hbv')

    print(f"\n将对以下模型进行敏感性分析:")
    for model in models_to_analyze:
        print(f"  - {model}")

    # ========================================================================
    # 步骤 3: 敏感性分析
    # ========================================================================
    print("\n步骤 3: 执行敏感性分析")
    print("-" * 80)

    output_dir = Path('results/sensitivity_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity_results = {}

    for model_name in models_to_analyze:
        print(f"\n{'='*60}")
        print(f"分析模型: {model_name.upper()}")
        print(f"{'='*60}")

        # 创建校准器
        calibrator = GenericHydrologicCalibrator.create_for_model(
            data=calib_data,
            runoff_model_type=model_name,
            objective_metric='nse',
            maximize=True,
            seed=42
        )

        print(f"\n模型参数: {list(calibrator.config.param_bounds.keys())}")

        # 执行敏感性分析 - Morris方法
        print(f"\n运行 Morris 敏感性分析...")
        result_morris = analyze_model_sensitivity(
            calibrator=calibrator,
            method='morris',
            n_samples=20,  # Morris轨迹数
            output_dir=output_dir / model_name / 'morris',
            seed=42
        )

        sensitivity_results[f"{model_name}_morris"] = result_morris

        # 执行敏感性分析 - OAT方法
        print(f"\n运行 OAT 敏感性分析...")
        result_oat = analyze_model_sensitivity(
            calibrator=calibrator,
            method='oat',
            n_samples=10,  # 每个参数的采样点数
            output_dir=output_dir / model_name / 'oat',
            seed=42
        )

        sensitivity_results[f"{model_name}_oat"] = result_oat

    # ========================================================================
    # 步骤 4: 对比不同方法的结果
    # ========================================================================
    print("\n\n步骤 4: 对比不同敏感性分析方法")
    print("-" * 80)

    for model_name in models_to_analyze:
        print(f"\n模型: {model_name.upper()}")
        print("-" * 60)

        result_morris = sensitivity_results[f"{model_name}_morris"]
        result_oat = sensitivity_results[f"{model_name}_oat"]

        print(f"\n{'参数':<15s} {'Morris排名':<12s} {'Morris值':<12s} {'OAT排名':<12s} {'OAT值':<12s}")
        print("-" * 70)

        for param in result_morris.sensitivity_result.param_names:
            morris_rank = result_morris.sensitivity_result.sensitivity_rankings.index(param) + 1
            morris_val = result_morris.sensitivity_result.sensitivity_indices[param]

            oat_rank = result_oat.sensitivity_result.sensitivity_rankings.index(param) + 1
            oat_val = result_oat.sensitivity_result.sensitivity_indices[param]

            print(f"{param:<15s} {morris_rank:<12d} {morris_val:<12.4f} {oat_rank:<12d} {oat_val:<12.4f}")

    # ========================================================================
    # 步骤 5: 基于敏感性的校准策略
    # ========================================================================
    print("\n\n步骤 5: 基于敏感性的校准策略")
    print("-" * 80)

    for model_name in models_to_analyze:
        print(f"\n模型: {model_name.upper()}")
        print("-" * 60)

        result = sensitivity_results[f"{model_name}_morris"]

        critical_params = result.get_critical_params(threshold=0.7)
        insensitive_params = result.get_insensitive_params(threshold=0.3)

        print(f"\n关键参数 (敏感性 > 0.7): {len(critical_params)}个")
        if critical_params:
            for param in critical_params:
                sens = result.sensitivity_result.sensitivity_indices[param]
                print(f"  - {param}: {sens:.4f}")
                print(f"    建议: 优先校准，需要精确估计")

        print(f"\n中等敏感性参数: {len(result.sensitivity_result.param_names) - len(critical_params) - len(insensitive_params)}个")

        print(f"\n不敏感参数 (敏感性 < 0.3): {len(insensitive_params)}个")
        if insensitive_params:
            for param in insensitive_params:
                sens = result.sensitivity_result.sensitivity_indices[param]
                suggested = result.suggested_bounds[param]
                print(f"  - {param}: {sens:.4f}")
                print(f"    建议: 可固定为 {(suggested[0]+suggested[1])/2:.2f}")

    # ========================================================================
    # 步骤 6: 生成优化的校准配置
    # ========================================================================
    print("\n\n步骤 6: 生成优化的校准配置")
    print("-" * 80)

    for model_name in models_to_analyze:
        result = sensitivity_results[f"{model_name}_morris"]

        print(f"\n模型: {model_name.upper()}")

        # 选择敏感性 > 0.3 的参数进行校准
        params_to_calibrate = {
            param: result.suggested_bounds[param]
            for param in result.sensitivity_result.param_names
            if result.sensitivity_result.sensitivity_indices[param] > 0.3
        }

        # 固定不敏感的参数
        fixed_params = {}
        for param in result.get_insensitive_params(threshold=0.3):
            bounds = result.suggested_bounds[param]
            fixed_params[param] = (bounds[0] + bounds[1]) / 2

        print(f"\n待校准参数 ({len(params_to_calibrate)}个):")
        for param, bounds in params_to_calibrate.items():
            sens = result.sensitivity_result.sensitivity_indices[param]
            print(f"  {param:<15s}: [{bounds[0]:>8.2f}, {bounds[1]:>8.2f}]  (敏感性: {sens:.4f})")

        if fixed_params:
            print(f"\n固定参数 ({len(fixed_params)}个):")
            for param, value in fixed_params.items():
                sens = result.sensitivity_result.sensitivity_indices[param]
                print(f"  {param:<15s}: {value:>8.2f}  (敏感性: {sens:.4f})")

        # 保存优化的配置
        config_file = output_dir / model_name / 'optimized_calibration_config.json'
        import json
        with open(config_file, 'w') as f:
            json.dump({
                'model': model_name,
                'param_bounds': {k: list(v) for k, v in params_to_calibrate.items()},
                'fixed_params': fixed_params,
                'sensitivity_based': True,
                'critical_params': result.get_critical_params(),
                'insensitive_params': result.get_insensitive_params(),
            }, f, indent=2)

        print(f"\n✓ 优化的校准配置已保存: {config_file}")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("参数敏感性分析完成！")
    print("=" * 80)

    print(f"\n主要成果:")
    print(f"  1. 分析了 {len(models_to_analyze)} 个水文模型")
    print(f"  2. 使用了 Morris 和 OAT 两种方法")
    print(f"  3. 识别了关键参数和不敏感参数")
    print(f"  4. 生成了优化的校准配置")

    print(f"\n输出目录: {output_dir}")
    print(f"  每个模型包含:")
    print(f"    - sensitivity_analysis.json (敏感性分析结果)")
    print(f"    - parameter_sensitivity.csv (参数排名)")
    print(f"    - sensitivity_report.txt (详细报告)")
    print(f"    - optimized_calibration_config.json (优化的校准配置)")

    print(f"\n下一步建议:")
    print(f"  1. 使用优化的配置进行参数校准")
    print(f"  2. 重点关注高敏感性参数")
    print(f"  3. 固定不敏感参数以提高校准效率")

    print("\n✨ 敏感性分析帮助您:")
    print("  ✓ 理解哪些参数对模型影响最大")
    print("  ✓ 优化校准策略，减少计算时间")
    print("  ✓ 提高参数估计的可靠性")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
