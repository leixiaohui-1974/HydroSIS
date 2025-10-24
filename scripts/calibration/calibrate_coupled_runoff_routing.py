#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""产汇流组合模型校准脚本

演示如何使用GenericHydrologicCalibrator同时校准产流和汇流模型。

支持的组合（示例）：
- HBV + Muskingum
- XinAnJiang + Muskingum
- VIC + Lag
- 任意产流模型 + 任意汇流模型

主要特点：
1. 联合校准产流和汇流参数
2. 自动参数分离和管理
3. 多种组合对比
4. 评估产流和汇流各自的贡献

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

from hydrosis.calibration import (
    GenericHydrologicCalibrator,
    CalibrationData,
    CalibrationConfig,
    ModelMode
)
from hydrosis.analysis import calculate_metrics


def main():
    """主函数：产汇流组合校准"""
    print("=" * 80)
    print("产汇流组合模型校准示例")
    print("=" * 80)

    # ========================================================================
    # 步骤 1: 准备数据
    # ========================================================================
    print("\n步骤 1: 准备校准数据")
    print("-" * 80)

    # 生成合成数据
    n_hours = 30 * 24  # 30天
    np.random.seed(42)

    precipitation = np.random.gamma(0.5, 0.3, n_hours)
    for event_start in [100, 300, 500]:
        event_length = 36
        precipitation[event_start:event_start+event_length] += np.random.gamma(2, 3, event_length)

    area_km2 = 100.0
    times = pd.date_range('2024-01-01', periods=n_hours, freq='H')

    # 生成"观测"径流（使用XinAnJiang + Muskingum）
    from hydrosis.runoff.xinanjiang import XinAnJiangRunoff
    from hydrosis.routing.muskingum import MuskingumRouting

    class MockSubbasin:
        def __init__(self, area_km2):
            self.area_km2 = area_km2

    subbasin = MockSubbasin(area_km2)

    # 真实参数
    true_runoff_params = {'wm': 150.0, 'b': 0.3, 'imp': 0.05, 'recession': 0.6}
    true_routing_params = {'K': 2.5, 'x': 0.25}

    # 生成观测数据
    runoff_model = XinAnJiangRunoff(true_runoff_params)
    runoff = np.array(runoff_model.simulate(subbasin, precipitation.tolist()))

    routing_model = MuskingumRouting(true_routing_params)
    observed_runoff = np.array(routing_model.route(subbasin, runoff.tolist()))

    # 添加噪声
    noise = np.random.normal(0, 0.03, len(observed_runoff))
    observed_runoff = observed_runoff * (1 + noise)
    observed_runoff = np.maximum(observed_runoff, 0)

    print(f"✓ 数据统计:")
    print(f"  时间长度: 30天")
    print(f"  总降雨量: {precipitation.sum():.2f} mm")
    print(f"  观测径流生成方式: XinAnJiang + Muskingum")
    print(f"  真实产流参数: wm={true_runoff_params['wm']}, b={true_runoff_params['b']}")
    print(f"  真实汇流参数: K={true_routing_params['K']}, x={true_routing_params['x']}")

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
    print("\n步骤 2: 列出可用的产流和汇流模型")
    print("-" * 80)

    available = GenericHydrologicCalibrator.list_available_models()

    print(f"✓ 产流模型: {list(available['runoff_models'].keys())}")
    print(f"✓ 汇流模型: {list(available['routing_models'].keys())}")

    # ========================================================================
    # 步骤 3: 定义要测试的组合
    # ========================================================================
    print("\n步骤 3: 定义要校准的产汇流组合")
    print("-" * 80)

    combinations = [
        {
            'name': 'XinAnJiang + Muskingum',
            'runoff': 'xin_an_jiang',
            'routing': 'muskingum',
            'description': '新安江模型 + Muskingum汇流'
        },
    ]

    # 如果HBV和Muskingum都可用，添加这个组合
    if 'hbv' in available['runoff_models'] and 'muskingum' in available['routing_models']:
        combinations.append({
            'name': 'HBV + Muskingum',
            'runoff': 'hbv',
            'routing': 'muskingum',
            'description': 'HBV模型 + Muskingum汇流'
        })

    print(f"✓ 将测试以下组合:")
    for combo in combinations:
        print(f"  - {combo['name']}: {combo['description']}")

    # ========================================================================
    # 步骤 4: 校准各个组合
    # ========================================================================
    print("\n步骤 4: 校准各个产汇流组合")
    print("-" * 80)

    output_dir = Path('results/coupled_calibration')
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_results = {}

    for combo in combinations:
        print(f"\n正在校准 {combo['name']}...")

        try:
            # 创建校准器
            calibrator = GenericHydrologicCalibrator.create_for_model(
                data=calib_data,
                runoff_model_type=combo['runoff'],
                routing_model_type=combo['routing'],
                algorithm='differential_evolution',
                algorithm_options={
                    'maxiter': 40,  # 演示用
                    'popsize': 15,
                    'seed': 42,
                    'workers': 1
                },
                objective_metric='nse',
                maximize=True
            )

            # 显示校准器信息
            print(f"  模型模式: {calibrator.mode.value}")
            print(f"  产流参数: {calibrator.runoff_param_names}")
            print(f"  汇流参数: {calibrator.routing_param_names}")
            print(f"  总参数数: {len(calibrator.config.param_bounds)}")

            # 运行校准
            result = calibrator.run_calibration()

            # 保存结果
            combo_output_dir = output_dir / combo['name'].replace(' ', '_').replace('+', 'plus')
            combo_output_dir.mkdir(parents=True, exist_ok=True)
            calibrator.save_results(result)

            calibration_results[combo['name']] = {
                'result': result,
                'calibrator': calibrator,
                'combo': combo
            }

            print(f"  ✓ 完成!")
            print(f"    NSE: {result.best_score:.6f}")
            print(f"    计算时间: {result.elapsed_time:.2f}秒")
            print(f"    参数数量: {len(result.best_params)}")

        except Exception as e:
            print(f"  ✗ 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # 步骤 5: 对比结果
    # ========================================================================
    print("\n步骤 5: 对比不同组合的性能")
    print("-" * 80)

    comparison_data = []

    for combo_name, data_dict in calibration_results.items():
        result = data_dict['result']
        combo = data_dict['combo']

        # 计算指标
        metrics = calculate_metrics(
            observed_runoff,
            result.simulated_runoff,
            metrics=['nse', 'kge', 'rmse', 'pbias']
        )

        comparison_data.append({
            'combination': combo_name,
            'runoff_model': combo['runoff'],
            'routing_model': combo['routing'],
            'nse': metrics['nse'],
            'kge': metrics['kge'],
            'rmse': metrics['rmse'],
            'pbias': metrics['pbias'],
            'n_params': len(result.best_params),
            'time_s': result.elapsed_time
        })

    # 显示对比
    print("\n产汇流组合性能对比:")
    print(f"{'组合':<30s} {'NSE':>8s} {'KGE':>8s} {'RMSE':>8s} {'参数数':>6s} {'时间(s)':>10s}")
    print("-" * 80)

    for row in comparison_data:
        print(f"{row['combination']:<30s} "
              f"{row['nse']:>8.4f} "
              f"{row['kge']:>8.4f} "
              f"{row['rmse']:>8.4f} "
              f"{row['n_params']:>6d} "
              f"{row['time_s']:>10.2f}")

    # 保存对比
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / 'combination_comparison.csv', index=False)

    # ========================================================================
    # 步骤 6: 参数恢复分析
    # ========================================================================
    print("\n步骤 6: 参数恢复精度分析")
    print("-" * 80)

    # 分析XinAnJiang + Muskingum组合（与真实参数对比）
    if 'XinAnJiang + Muskingum' in calibration_results:
        result = calibration_results['XinAnJiang + Muskingum']['result']

        print("\nXinAnJiang + Muskingum 参数恢复:")
        print(f"{'参数':<20s} {'真实值':<12s} {'率定值':<12s} {'误差%':<10s}")
        print("-" * 60)

        all_true_params = {**true_runoff_params, **true_routing_params}

        for param, calib_val in result.best_params.items():
            if param in all_true_params:
                true_val = all_true_params[param]
                error_pct = abs(calib_val - true_val) / true_val * 100
                print(f"{param:<20s} {true_val:<12.4f} {calib_val:<12.4f} {error_pct:<10.2f}")

    # ========================================================================
    # 步骤 7: 生成报告
    # ========================================================================
    print("\n步骤 7: 生成详细报告")
    print("-" * 80)

    report_file = output_dir / 'coupled_calibration_report.txt'
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("产汇流组合模型校准报告\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"校准框架: GenericHydrologicCalibrator (耦合模式)\n\n")

        f.write("1. 数据概况\n")
        f.write(f"   时间长度: 30天\n")
        f.write(f"   总降雨量: {precipitation.sum():.2f} mm\n")
        f.write(f"   流域面积: {area_km2:.2f} km²\n\n")

        f.write("2. 测试的产汇流组合\n")
        for combo in combinations:
            f.write(f"   - {combo['name']}\n")
            f.write(f"     产流: {combo['runoff']}\n")
            f.write(f"     汇流: {combo['routing']}\n")
        f.write("\n")

        f.write("3. 性能对比\n")
        f.write(f"   {'组合':<30s} {'NSE':>8s} {'KGE':>8s}\n")
        f.write("   " + "-" * 50 + "\n")
        for row in comparison_data:
            f.write(f"   {row['combination']:<30s} {row['nse']:>8.4f} {row['kge']:>8.4f}\n")

        f.write("\n4. 框架优势\n")
        f.write("   ✓ 统一接口支持产流、汇流、以及组合模型\n")
        f.write("   ✓ 自动参数分离和管理\n")
        f.write("   ✓ 轻松测试不同组合\n")
        f.write("   ✓ 无需为每个组合编写单独代码\n")

    print(f"✓ 报告已保存: {report_file}")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("产汇流组合校准完成！")
    print("=" * 80)

    print(f"\n主要成果:")
    print(f"  1. 成功校准 {len(calibration_results)} 个产汇流组合")
    print(f"  2. 所有结果保存在: {output_dir}")

    print(f"\n统一框架的威力:")
    print(f"  ✓ 同一套代码支持产流、汇流、以及组合")
    print(f"  ✓ 自动处理参数分离和组合")
    print(f"  ✓ 轻松添加新的模型组合")
    print(f"  ✓ 无需重复编写校准逻辑")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
