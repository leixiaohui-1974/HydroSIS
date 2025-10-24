#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新安江(XinAnJiang)模型基础校准脚本

演示如何使用XinAnJiangCalibrator进行参数率定。

新安江模型是中国最经典的水文模型之一，由赵人俊院士于20世纪70年代提出，
采用蓄满产流机制，适用于湿润和半湿润地区的流域水文模拟。

本脚本展示：
1. 使用XinAnJiangCalibrator统一校准框架
2. 新安江模型的参数率定
3. 率定结果的评估和保存

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

from hydrosis.calibration import XinAnJiangCalibrator, CalibrationData, CalibrationConfig
from hydrosis.analysis import calculate_metrics


def generate_synthetic_observations(rainfall: np.ndarray, true_params: dict, area_km2: float) -> np.ndarray:
    """
    生成合成观测数据（用于演示）

    在实际应用中，应该使用真实的观测径流数据。

    Parameters
    ----------
    rainfall : np.ndarray
        降雨时间序列 (mm/h)
    true_params : dict
        真实XinAnJiang参数
    area_km2 : float
        流域面积 (km²)

    Returns
    -------
    np.ndarray
        合成的观测径流数据 (m³/s)
    """
    from hydrosis.runoff.xinanjiang import XinAnJiangRunoff

    class MockSubbasin:
        def __init__(self, area_km2):
            self.area_km2 = area_km2

    subbasin = MockSubbasin(area_km2)
    model = XinAnJiangRunoff(parameters=true_params)
    observed = np.array(model.simulate(subbasin, rainfall.tolist()))

    # 添加少量观测噪声（模拟真实测量误差）
    np.random.seed(42)
    noise = np.random.normal(0, 0.03, len(observed))
    observed = observed * (1 + noise)
    observed = np.maximum(observed, 0)  # 确保非负

    return observed


def main():
    """主函数"""
    print("=" * 80)
    print("新安江(XinAnJiang)模型参数率定")
    print("=" * 80)

    # ========================================================================
    # 步骤 1: 准备数据
    # ========================================================================
    print("\n步骤 1: 准备校准数据")
    print("-" * 80)

    # 对于演示，我们生成合成数据
    # 在实际应用中，应该从文件加载真实的观测数据

    # 1.1 生成或加载降雨数据
    use_real_data = False  # 设为True使用真实数据

    if use_real_data:
        # 从真实数据文件加载
        results_dir = Path("results/upper_truckee_complete_11steps")
        obs_file = results_dir / "estimated_observations" / "zone_1_estimated_runoff.csv"
        precip_file = results_dir / "step_08_areal_rainfall" / "8.2_subbasin_areal_precipitation.csv"

        if not obs_file.exists() or not precip_file.exists():
            print(f"❌ 错误: 数据文件不存在")
            print(f"   将使用合成数据代替...")
            use_real_data = False
        else:
            # 加载观测数据
            obs_df = pd.read_csv(obs_file)
            observed_runoff = obs_df['discharge_m3s'].values
            times = pd.to_datetime(obs_df['datetime'])

            # 加载降雨数据
            precip_df = pd.read_csv(precip_file, index_col=0)
            zone1_subbasins = [str(i) for i in range(101, 115)]
            zone1_cols = [col for col in precip_df.columns if col in zone1_subbasins]
            precipitation = precip_df[zone1_cols].mean(axis=1).values

            area_km2 = 139.995  # Zone 1面积

            print(f"✓ 加载真实数据")
            print(f"  时间范围: {times.min()} 到 {times.max()}")
            print(f"  数据点数: {len(observed_runoff)}")

    if not use_real_data:
        # 生成合成数据用于演示
        print("使用合成数据进行演示...")

        # 生成60天的降雨数据
        n_hours = 60 * 24  # 60天
        np.random.seed(42)

        # 模拟降雨模式：基础降雨 + 几次降雨事件
        precipitation = np.random.gamma(0.5, 0.3, n_hours)

        # 添加几个降雨事件
        for event_start in [200, 500, 900, 1200]:
            event_length = 48
            precipitation[event_start:event_start+event_length] += np.random.gamma(2, 3, event_length)

        # 流域面积
        area_km2 = 100.0

        # 生成"真实"观测数据
        true_params = {
            'wm': 150.0,          # 张力水容量
            'b': 0.3,             # 蓄水容量曲线指数
            'imp': 0.05,          # 不透水面积比例
            'recession': 0.6,     # 地下水消退系数
        }

        observed_runoff = generate_synthetic_observations(precipitation, true_params, area_km2)
        times = pd.date_range('2024-01-01', periods=n_hours, freq='H')

        print(f"✓ 生成合成数据")
        print(f"  时间长度: 60天 ({n_hours}小时)")
        print(f"  真实参数: wm={true_params['wm']}, b={true_params['b']}, "
              f"imp={true_params['imp']}, recession={true_params['recession']}")

    print(f"\n数据统计:")
    print(f"  降雨范围: {precipitation.min():.2f} - {precipitation.max():.2f} mm/h")
    print(f"  总降雨量: {precipitation.sum():.2f} mm")
    print(f"  流量范围: {observed_runoff.min():.2f} - {observed_runoff.max():.2f} m³/s")
    print(f"  平均流量: {observed_runoff.mean():.2f} m³/s")
    print(f"  流域面积: {area_km2:.2f} km²")

    # ========================================================================
    # 步骤 2: 配置XinAnJiang校准器
    # ========================================================================
    print("\n步骤 2: 配置XinAnJiang校准器")
    print("-" * 80)

    # 新安江参数搜索范围
    # 基于文献和实践经验的推荐范围
    param_bounds = {
        'wm': [50, 250],          # 张力水容量 (mm)
        'b': [0.1, 0.5],          # 蓄水容量曲线指数
        'imp': [0.0, 0.3],        # 不透水面积比例
        'recession': [0.3, 0.9],  # 地下水消退系数
    }

    print(f"待率定参数: {len(param_bounds)}个")
    for name, (min_val, max_val) in param_bounds.items():
        print(f"  {name:<20s}: [{min_val:>8.3f}, {max_val:>8.3f}]")

    # 固定参数（初始状态）
    fixed_params = {
        'initial_tension_water': 75.0,  # 初始张力水
        'initial_groundwater': 0.0,     # 初始地下水
    }

    print(f"\n固定参数: {len(fixed_params)}个")
    for name, value in fixed_params.items():
        print(f"  {name:<20s}: {value:>8.3f}")

    # 创建校准数据
    calib_data = CalibrationData(
        precipitation=precipitation,
        observed_runoff=observed_runoff,
        area_km2=area_km2,
        times=times
    )

    # 创建校准配置
    calib_config = CalibrationConfig(
        param_bounds=param_bounds,
        fixed_params=fixed_params,
        algorithm='differential_evolution',  # 使用差分进化算法
        objective_metric='nse',
        maximize_objective=True,
        algorithm_options={
            'maxiter': 100,
            'popsize': 15,
            'seed': 42,
            'workers': 1,
            'polish': True,
            'atol': 1e-6,
            'tol': 1e-4
        }
    )

    print(f"\n校准配置:")
    print(f"  算法: {calib_config.algorithm}")
    print(f"  目标函数: {calib_config.objective_metric}")
    print(f"  最大迭代数: {calib_config.algorithm_options['maxiter']}")
    print(f"  种群大小: {calib_config.algorithm_options['popsize']}")

    # ========================================================================
    # 步骤 3: 执行参数率定
    # ========================================================================
    print("\n步骤 3: 执行参数率定")
    print("-" * 80)
    print("这可能需要几分钟时间...")

    # 创建输出目录
    output_dir = Path('results/calibration_xinanjiang')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建校准器
    calibrator = XinAnJiangCalibrator(
        data=calib_data,
        config=calib_config,
        output_dir=output_dir
    )

    # 运行校准
    result = calibrator.run_calibration()

    print(f"\n✓ 率定完成!")
    print(f"  最优NSE: {result.best_score:.6f}")
    print(f"  函数评估: {result.n_evaluations}")
    print(f"  计算时间: {result.elapsed_time:.2f}秒")

    # ========================================================================
    # 步骤 4: 分析率定结果
    # ========================================================================
    print("\n步骤 4: 分析率定结果")
    print("-" * 80)

    print("\n最优参数:")
    for param, value in result.best_params.items():
        if param in param_bounds:
            bounds = param_bounds[param]
            pct = (value - bounds[0]) / (bounds[1] - bounds[0]) * 100
            print(f"  {param:<20s}: {value:>10.4f}  (搜索空间的{pct:>5.1f}%)")

            # 如果使用合成数据，显示与真实值的对比
            if not use_real_data and param in true_params:
                true_val = true_params[param]
                error = abs(value - true_val) / true_val * 100
                print(f"    → 真实值: {true_val:.4f}, 误差: {error:.2f}%")

    # 使用最优参数运行模型
    from hydrosis.runoff.xinanjiang import XinAnJiangRunoff

    class MockSubbasin:
        def __init__(self, area_km2):
            self.area_km2 = area_km2

    final_params = {**fixed_params, **result.best_params}
    subbasin = MockSubbasin(area_km2)
    xaj_model = XinAnJiangRunoff(final_params)
    calibrated_runoff = np.array(xaj_model.simulate(subbasin, precipitation.tolist()))

    # 计算所有性能指标
    final_metrics = calculate_metrics(
        observed_runoff,
        calibrated_runoff,
        metrics=['nse', 'log_nse', 'kge', 'rmse', 'mae', 'pbias']
    )

    print("\n率定后性能指标:")
    for metric, value in final_metrics.items():
        if 'peak' not in metric and 'time' not in metric:
            print(f"  {metric.upper():10s}: {value:8.6f}")

    # ========================================================================
    # 步骤 5: 保存结果
    # ========================================================================
    print("\n步骤 5: 保存结果")
    print("-" * 80)

    # 保存校准结果
    calibrator.save_results(result)

    # 保存模拟径流对比
    result_df = pd.DataFrame({
        'datetime': times,
        'time_step': np.arange(len(observed_runoff)),
        'observed_m3s': observed_runoff,
        'simulated_m3s': calibrated_runoff,
        'residual_m3s': observed_runoff - calibrated_runoff,
        'relative_error_pct': (calibrated_runoff - observed_runoff) / (observed_runoff + 1e-6) * 100
    })
    result_df.to_csv(output_dir / "xinanjiang_calibrated_runoff.csv", index=False)
    print(f"✓ 保存径流对比数据: xinanjiang_calibrated_runoff.csv")

    # 保存参数到文本文件
    param_file = output_dir / "xinanjiang_calibrated_parameters.txt"
    with open(param_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("新安江模型率定参数\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"率定时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据长度: {len(observed_runoff)} 时间步\n")
        f.write(f"流域面积: {area_km2:.2f} km²\n\n")

        f.write("最优参数:\n")
        for param, value in result.best_params.items():
            f.write(f"  {param:<20s}: {value:.6f}\n")

        f.write("\n性能指标:\n")
        for metric, value in final_metrics.items():
            f.write(f"  {metric.upper():<10s}: {value:.6f}\n")

        f.write(f"\n率定信息:\n")
        f.write(f"  算法: {calib_config.algorithm}\n")
        f.write(f"  目标函数: {calib_config.objective_metric}\n")
        f.write(f"  函数评估: {result.n_evaluations}\n")
        f.write(f"  计算时间: {result.elapsed_time:.2f}秒\n")

    print(f"✓ 保存参数文件: xinanjiang_calibrated_parameters.txt")

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 80)
    print("新安江模型参数率定完成！")
    print("=" * 80)

    print(f"\n1. 率定数据:")
    print(f"   - 时间步数: {len(observed_runoff)}")
    print(f"   - 平均流量: {observed_runoff.mean():.2f} m³/s")
    print(f"   - 峰值流量: {observed_runoff.max():.2f} m³/s")

    print(f"\n2. 率定结果:")
    print(f"   - 算法: {calib_config.algorithm}")
    print(f"   - 最优NSE: {result.best_score:.6f}")
    print(f"   - 计算时间: {result.elapsed_time:.2f}秒")

    print(f"\n3. 性能指标:")
    for metric, value in final_metrics.items():
        if 'peak' not in metric and 'time' not in metric:
            print(f"   - {metric.upper()}: {value:.6f}")

    print(f"\n4. 最优参数:")
    for param, value in result.best_params.items():
        if param in param_bounds:
            print(f"   - {param}: {value:.4f}")

    print(f"\n5. 输出文件 (位于 {output_dir}):")
    print(f"   - xinanjiang_calibrated_parameters.txt")
    print(f"   - xinanjiang_calibrated_runoff.csv")
    print(f"   - calibration_result.json")
    print(f"   - convergence_history.csv")

    print("\n✨ 新安江模型特点:")
    print("  1. 蓄满产流机制 - 适用于湿润地区")
    print("  2. 非线性土壤蓄水过程")
    print("  3. 参数物理意义明确")
    print("  4. 中国水文界广泛应用")

    if not use_real_data:
        print("\n📝 使用真实数据:")
        print("  将 use_real_data = True 并提供真实的观测径流和降雨数据")

    print("\n" + "=" * 80)
    print("✓ 新安江模型率定完成！")
    print("✓ 使用XinAnJiangCalibrator统一校准框架")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
