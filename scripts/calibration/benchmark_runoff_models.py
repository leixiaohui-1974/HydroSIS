#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""水文模型基准测试和对比脚本

对多个产流模型进行标准化基准测试和性能对比。

主要功能：
1. 多模型并行校准
2. 标准化性能评估
3. 生成对比报告
4. 模型选择建议

支持的模型：
- XinAnJiang (新安江)
- HBV
- VIC
- HYMOD
- 以及所有已注册的产流模型

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
    CalibrationData,
    ModelComparator,
    compare_models,
    GenericHydrologicCalibrator
)


def main():
    """主函数：水文模型基准测试"""
    print("=" * 80)
    print("水文模型基准测试和性能对比")
    print("=" * 80)

    # ========================================================================
    # 步骤 1: 准备基准测试数据
    # ========================================================================
    print("\n步骤 1: 准备基准测试数据")
    print("-" * 80)

    # 生成60天的合成数据
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

    print(f"✓ 基准测试数据准备完成")
    print(f"  时间长度: 60天 ({n_hours}小时)")
    print(f"  总降雨量: {precipitation.sum():.2f} mm")
    print(f"  总径流量: {observed_runoff.sum():.2f} m³")
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

    available = GenericHydrologicCalibrator.list_available_models()
    runoff_models = list(available['runoff_models'].keys())

    print(f"✓ 已注册的产流模型 ({len(runoff_models)}个):")
    for i, model in enumerate(runoff_models, 1):
        print(f"  {i}. {model}")

    # ========================================================================
    # 步骤 3: 选择要测试的模型
    # ========================================================================
    print("\n步骤 3: 选择要测试的模型")
    print("-" * 80)

    # 选择主要模型进行对比
    models_to_test = ['xin_an_jiang']

    # 如果HBV可用，添加它
    if 'hbv' in runoff_models:
        models_to_test.append('hbv')

    # 如果VIC可用，添加它
    if 'vic' in runoff_models:
        models_to_test.append('vic')

    # 如果HYMOD可用，添加它
    if 'hymod' in runoff_models:
        models_to_test.append('hymod')

    print(f"✓ 将对以下模型进行基准测试:")
    for model in models_to_test:
        print(f"  - {model}")

    # ========================================================================
    # 步骤 4: 快速对比（使用便捷函数）
    # ========================================================================
    print("\n步骤 4: 快速模型对比（简化版）")
    print("-" * 80)

    # 对于演示，使用较少的迭代次数
    print("使用简化配置进行快速对比...")

    from hydrosis.calibration import CalibrationConfig

    # 创建简化的校准配置
    quick_config = CalibrationConfig(
        param_bounds={},  # 使用默认边界
        algorithm='differential_evolution',
        algorithm_options={
            'maxiter': 30,  # 演示用，实际应用建议100+
            'popsize': 15,
            'seed': 42,
            'workers': 1
        },
        objective_metric='nse',
        maximize=True
    )

    output_dir = Path('results/model_benchmark')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用便捷函数进行对比
    result = compare_models(
        data=calib_data,
        models=models_to_test,
        calibration_config=quick_config,
        comparison_metrics=['nse', 'kge', 'rmse', 'pbias', 'log_nse'],
        parallel=False,  # 设为True可启用并行
        output_dir=output_dir,
        seed=42
    )

    # ========================================================================
    # 步骤 5: 详细分析最佳模型
    # ========================================================================
    print("\n\n步骤 5: 详细分析最佳模型")
    print("-" * 80)

    best_model_name = result.best_model
    best_perf = result.get_model_performance(best_model_name)

    print(f"\n最佳模型: {best_model_name.upper()}")
    print("-" * 60)

    print(f"\n性能指标:")
    for metric, value in best_perf.metrics.items():
        print(f"  {metric.upper():<10s}: {value:>10.6f}")

    print(f"\n最优参数:")
    for param, value in best_perf.calibration_result.best_params.items():
        print(f"  {param:<20s}: {value:>10.6f}")

    print(f"\n计算信息:")
    print(f"  参数数量: {best_perf.n_parameters}")
    print(f"  函数评估: {best_perf.n_evaluations}")
    print(f"  计算时间: {best_perf.computation_time:.2f}秒")

    # ========================================================================
    # 步骤 6: 模型选择建议
    # ========================================================================
    print("\n\n步骤 6: 模型选择建议")
    print("-" * 80)

    df = result.get_performance_matrix()

    # 归一化性能指标（NSE和KGE越大越好，RMSE越小越好）
    nse_scores = df['nse'].values
    kge_scores = df['kge'].values if 'kge' in df.columns else nse_scores
    rmse_scores = df['rmse'].values if 'rmse' in df.columns else np.ones_like(nse_scores)

    # 计算综合得分
    综合得分 = (nse_scores + kge_scores) / 2 - (rmse_scores / rmse_scores.max()) * 0.2

    print("\n综合评分 (NSE + KGE - normalized_RMSE):")
    print("-" * 60)

    scored_models = sorted(
        zip(df['model'].values, 综合得分, nse_scores, df['n_params'].values),
        key=lambda x: x[1],
        reverse=True
    )

    for i, (model, score, nse, n_params) in enumerate(scored_models, 1):
        print(f"  {i}. {model:<20s}  得分: {score:.4f}  NSE: {nse:.4f}  参数数: {n_params}")

    # 推荐建议
    print("\n推荐建议:")
    print("-" * 60)

    top_model = scored_models[0][0]
    top_nse = scored_models[0][2]
    top_params = scored_models[0][3]

    print(f"\n最佳性能模型: {top_model}")
    print(f"  - NSE: {top_nse:.6f}")
    print(f"  - 适用场景: 对精度要求高的应用")

    # 如果有参数较少但性能相近的模型
    for model, score, nse, n_params in scored_models[1:]:
        if nse > top_nse * 0.95 and n_params < top_params:
            print(f"\n简化模型: {model}")
            print(f"  - NSE: {nse:.6f} (仅比最佳差{(1-nse/top_nse)*100:.1f}%)")
            print(f"  - 参数少{top_params - n_params}个")
            print(f"  - 适用场景: 数据有限或需要简化的应用")
            break

    # ========================================================================
    # 步骤 7: 敏感性对比（可选）
    # ========================================================================
    print("\n\n步骤 7: 参数敏感性对比")
    print("-" * 80)

    print("\n对最佳模型进行敏感性分析...")

    from hydrosis.calibration import analyze_model_sensitivity

    # 创建最佳模型的校准器
    best_calibrator = GenericHydrologicCalibrator.create_for_model(
        data=calib_data,
        runoff_model_type=best_model_name,
        seed=42
    )

    # 执行敏感性分析
    sens_result = analyze_model_sensitivity(
        calibrator=best_calibrator,
        method='morris',
        n_samples=10,  # 演示用
        output_dir=output_dir / best_model_name / 'sensitivity'
    )

    print(f"\n{best_model_name.upper()} 参数敏感性:")
    critical_params = sens_result.get_critical_params(threshold=0.7)
    if critical_params:
        print(f"  关键参数 (敏感性 > 0.7): {', '.join(critical_params)}")

    insensitive_params = sens_result.get_insensitive_params(threshold=0.3)
    if insensitive_params:
        print(f"  不敏感参数 (敏感性 < 0.3): {', '.join(insensitive_params)}")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n\n" + "=" * 80)
    print("水文模型基准测试完成！")
    print("=" * 80)

    print(f"\n主要成果:")
    print(f"  1. 测试了 {len(models_to_test)} 个水文模型")
    print(f"  2. 最佳模型: {best_model_name} (NSE={best_perf.get_metric('nse'):.4f})")
    print(f"  3. 所有结果保存在: {output_dir}")

    print(f"\n输出文件:")
    print(f"  - model_comparison.json (完整对比数据)")
    print(f"  - performance_comparison.csv (性能表格)")
    print(f"  - comparison_report.txt (详细报告)")
    print(f"  - {best_model_name}/ (最佳模型详细结果)")

    print(f"\n下一步建议:")
    print(f"  1. 在实际数据上运行基准测试")
    print(f"  2. 使用更多迭代次数（maxiter=100+）")
    print(f"  3. 对最佳模型进行详细的敏感性分析")
    print(f"  4. 考虑数据质量和流域特征选择合适的模型")

    print("\n✨ 基准测试框架的优势:")
    print("  ✓ 标准化的对比流程")
    print("  ✓ 多个性能指标综合评估")
    print("  ✓ 自动识别最佳模型")
    print("  ✓ 考虑模型复杂度的平衡")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
