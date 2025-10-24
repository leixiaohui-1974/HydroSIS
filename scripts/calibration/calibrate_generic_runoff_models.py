#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用产流模型校准脚本

演示如何使用GenericHydrologicCalibrator对多个产流模型进行校准和对比。

支持的模型：
- HBV
- XinAnJiang (新安江)
- VIC
- HYMOD
- WetSpa
- 以及所有已注册的产流模型

主要特点：
1. 统一的校准接口 - 无需为每个模型单独编写代码
2. 多模型对比 - 自动对比不同模型的性能
3. 自动参数边界 - 使用推荐的默认参数范围
4. 详细报告生成

Author: Claude Code
Date: 2025-01-24
"""
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from datetime import datetime
import json

from hydrosis.calibration import (
    GenericHydrologicCalibrator,
    CalibrationData,
    CalibrationConfig,
)
from hydrosis.analysis import calculate_metrics


def generate_synthetic_observations(
    rainfall: np.ndarray,
    area_km2: float,
    model_type: str = 'xin_an_jiang'
) -> np.ndarray:
    """
    生成合成观测数据（用于演示）

    Parameters
    ----------
    rainfall : np.ndarray
        降雨时间序列 (mm/h)
    area_km2 : float
        流域面积 (km²)
    model_type : str
        用于生成数据的模型类型

    Returns
    -------
    np.ndarray
        合成的观测径流数据 (m³/s)
    """
    from hydrosis.runoff.base import RunoffModelConfig

    # 真实参数（用于生成观测数据）
    true_params = {
        'xin_an_jiang': {
            'wm': 150.0,
            'b': 0.3,
            'imp': 0.05,
            'recession': 0.6,
        },
        'hbv': {
            'FC': 300.0,
            'BETA': 2.0,
            'K0': 0.15,
            'K1': 0.08,
            'K2': 0.02,
            'PERC': 2.0,
            'LP': 0.7,
            'TT': 0.0,
            'CFMAX': 3.0,
            'CFR': 0.05,
            'CWH': 0.1,
        }
    }

    params = true_params.get(model_type, true_params['xin_an_jiang'])

    # 创建模型
    model_class = RunoffModelConfig.REGISTRY[model_type]

    class MockSubbasin:
        def __init__(self, area_km2):
            self.area_km2 = area_km2

    subbasin = MockSubbasin(area_km2)
    model = model_class(parameters=params)
    observed = np.array(model.simulate(subbasin, rainfall.tolist()))

    # 添加少量观测噪声
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, len(observed))
    observed = observed * (1 + noise)
    observed = np.maximum(observed, 0)

    return observed


def main():
    """主函数：多模型校准对比"""
    print("=" * 80)
    print("通用产流模型校准 - 多模型对比")
    print("=" * 80)

    # ========================================================================
    # 步骤 1: 准备数据
    # ========================================================================
    print("\n步骤 1: 准备校准数据")
    print("-" * 80)

    # 生成60天的合成降雨数据
    n_hours = 60 * 24
    np.random.seed(42)

    # 模拟降雨模式
    precipitation = np.random.gamma(0.5, 0.3, n_hours)

    # 添加几个降雨事件
    for event_start in [200, 500, 900, 1200]:
        event_length = 48
        precipitation[event_start:event_start+event_length] += np.random.gamma(2, 3, event_length)

    area_km2 = 100.0
    times = pd.date_range('2024-01-01', periods=n_hours, freq='H')

    # 生成"观测"数据（使用新安江模型）
    observed_runoff = generate_synthetic_observations(
        precipitation, area_km2, model_type='xin_an_jiang'
    )

    print(f"✓ 数据统计:")
    print(f"  时间长度: 60天 ({n_hours}小时)")
    print(f"  总降雨量: {precipitation.sum():.2f} mm")
    print(f"  流量范围: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")
    print(f"  流域面积: {area_km2:.2f} km²")

    # 创建校准数据
    calib_data = CalibrationData(
        precipitation=precipitation,
        observed_runoff=observed_runoff,
        area_km2=area_km2,
        times=times
    )

    # ========================================================================
    # 步骤 2: 列出可用模型
    # ========================================================================
    print("\n步骤 2: 列出可用的产流模型")
    print("-" * 80)

    available_models = GenericHydrologicCalibrator.list_available_models()
    runoff_models = list(available_models['runoff_models'].keys())

    print(f"✓ 已注册的产流模型 ({len(runoff_models)}个):")
    for i, model_name in enumerate(runoff_models, 1):
        print(f"  {i}. {model_name}")

    # ========================================================================
    # 步骤 3: 选择要对比的模型
    # ========================================================================
    print("\n步骤 3: 选择要校准的模型")
    print("-" * 80)

    # 选择几个主要模型进行对比
    models_to_calibrate = ['xin_an_jiang', 'hbv']

    # 过滤掉不可用的模型
    models_to_calibrate = [m for m in models_to_calibrate if m in runoff_models]

    print(f"✓ 将对以下模型进行校准和对比:")
    for model_name in models_to_calibrate:
        print(f"  - {model_name}")

    # ========================================================================
    # 步骤 4: 多模型校准
    # ========================================================================
    print("\n步骤 4: 执行多模型校准")
    print("-" * 80)

    output_dir = Path('results/generic_runoff_calibration')
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_results = {}

    for model_name in models_to_calibrate:
        print(f"\n正在校准 {model_name.upper()} 模型...")
        print(f"  使用默认参数边界和差分进化算法")

        try:
            # 使用便捷方法创建校准器（自动使用默认参数边界）
            calibrator = GenericHydrologicCalibrator.create_for_model(
                data=calib_data,
                runoff_model_type=model_name,
                algorithm='differential_evolution',
                algorithm_options={
                    'maxiter': 50,   # 演示用，实际应用建议100+
                    'popsize': 15,
                    'seed': 42,
                    'workers': 1,
                    'polish': True
                },
                objective_metric='nse',
                maximize=True
            )

            # 运行校准
            result = calibrator.run_calibration()

            # 保存结果
            model_output_dir = output_dir / model_name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            calibrator.save_results(result, prefix=f"{model_name}_")

            calibration_results[model_name] = result

            print(f"  ✓ 完成!")
            print(f"    NSE: {result.best_score:.6f}")
            print(f"    计算时间: {result.elapsed_time:.2f}秒")

        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")
            continue

    # ========================================================================
    # 步骤 5: 结果对比
    # ========================================================================
    print("\n步骤 5: 模型性能对比")
    print("-" * 80)

    # 计算所有指标
    comparison_data = []

    for model_name, result in calibration_results.items():
        # 计算完整指标
        metrics = calculate_metrics(
            observed_runoff,
            result.simulated_runoff,
            metrics=['nse', 'log_nse', 'kge', 'rmse', 'mae', 'pbias']
        )

        comparison_data.append({
            'model': model_name,
            'nse': metrics['nse'],
            'kge': metrics['kge'],
            'rmse': metrics['rmse'],
            'pbias': metrics['pbias'],
            'n_params': len(result.best_params),
            'time_s': result.elapsed_time,
            'n_eval': result.n_evaluations
        })

    # 创建对比表
    comparison_df = pd.DataFrame(comparison_data)

    print("\n模型性能对比:")
    print(f"{'模型':<20s} {'NSE':>8s} {'KGE':>8s} {'RMSE':>8s} {'PBIAS%':>8s} {'参数数':>6s} {'时间(s)':>10s}")
    print("-" * 90)

    for row in comparison_data:
        print(f"{row['model']:<20s} "
              f"{row['nse']:>8.4f} "
              f"{row['kge']:>8.4f} "
              f"{row['rmse']:>8.4f} "
              f"{row['pbias']:>8.2f} "
              f"{row['n_params']:>6d} "
              f"{row['time_s']:>10.2f}")

    # 保存对比结果
    comparison_df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\n✓ 对比结果已保存: {output_dir / 'model_comparison.csv'}")

    # 找出最佳模型
    best_model_idx = comparison_df['nse'].idxmax()
    best_model = comparison_df.loc[best_model_idx]

    print(f"\n✓ 最佳模型: {best_model['model'].upper()}")
    print(f"  NSE: {best_model['nse']:.6f}")
    print(f"  KGE: {best_model['kge']:.6f}")

    # ========================================================================
    # 步骤 6: 生成详细报告
    # ========================================================================
    print("\n步骤 6: 生成详细报告")
    print("-" * 80)

    report_file = output_dir / 'calibration_report.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("通用产流模型校准对比报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"校准框架: GenericHydrologicCalibrator\n\n")

        f.write("1. 数据概况\n")
        f.write(f"   时间长度: 60天 ({n_hours}小时)\n")
        f.write(f"   总降雨量: {precipitation.sum():.2f} mm\n")
        f.write(f"   总径流量: {observed_runoff.sum():.2f} mm\n")
        f.write(f"   流域面积: {area_km2:.2f} km²\n\n")

        f.write("2. 校准的模型\n")
        for model_name in models_to_calibrate:
            f.write(f"   - {model_name}\n")
        f.write("\n")

        f.write("3. 性能对比\n")
        f.write(f"   {'模型':<20s} {'NSE':>8s} {'KGE':>8s} {'RMSE':>8s}\n")
        f.write("   " + "-" * 50 + "\n")
        for row in comparison_data:
            f.write(f"   {row['model']:<20s} "
                   f"{row['nse']:>8.4f} "
                   f"{row['kge']:>8.4f} "
                   f"{row['rmse']:>8.4f}\n")
        f.write("\n")

        f.write("4. 最佳模型\n")
        f.write(f"   模型名称: {best_model['model']}\n")
        f.write(f"   NSE: {best_model['nse']:.6f}\n")
        f.write(f"   KGE: {best_model['kge']:.6f}\n")
        f.write(f"   RMSE: {best_model['rmse']:.6f}\n")
        f.write(f"   PBIAS: {best_model['pbias']:.2f}%\n\n")

        best_result = calibration_results[best_model['model']]
        f.write("5. 最佳模型参数\n")
        for param, value in best_result.best_params.items():
            f.write(f"   {param:<20s}: {value:.6f}\n")

        f.write("\n6. 框架优势\n")
        f.write("   ✓ 统一接口 - 所有模型使用相同的校准代码\n")
        f.write("   ✓ 自动参数边界 - 使用专家经验的默认范围\n")
        f.write("   ✓ 模型注册表 - 动态加载任意已注册模型\n")
        f.write("   ✓ 多模型对比 - 轻松对比不同模型性能\n")

    print(f"✓ 详细报告已保存: {report_file}")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("通用产流模型校准完成！")
    print("=" * 80)

    print(f"\n主要成果:")
    print(f"  1. 成功校准 {len(calibration_results)} 个产流模型")
    print(f"  2. 最佳模型: {best_model['model'].upper()} (NSE={best_model['nse']:.4f})")
    print(f"  3. 所有结果保存在: {output_dir}")

    print(f"\n框架优势:")
    print(f"  ✓ 无需为每个模型编写单独的校准代码")
    print(f"  ✓ 自动使用推荐的参数边界")
    print(f"  ✓ 轻松添加新模型到对比中")
    print(f"  ✓ 统一的结果格式便于分析")

    print(f"\n输出文件:")
    print(f"  - model_comparison.csv  (模型对比表)")
    print(f"  - calibration_report.txt  (详细报告)")
    print(f"  - {models_to_calibrate[0]}/  (各模型详细结果)")

    print("\n✨ 这就是统一校准框架的威力！")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
