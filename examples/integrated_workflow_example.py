#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""综合工作流示例

展示如何组合使用HydroSIS的多个框架完成完整的建模工作流:
1. 数据验证 (验证框架)
2. 并行模拟 (并行框架)
3. 模型诊断 (诊断框架)
4. 参数校准 (校准框架)

使用方法:
    python examples/integrated_workflow_example.py
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd

from hydrosis.parallel import (
    ParallelExecutor,
    ExecutionConfig,
    ExecutionMode
)
from hydrosis.diagnostics import (
    WaterBalanceDiagnostic,
    PrecipitationDiagnostic
)
from hydrosis.calibration import (
    HBVCalibrator,
    CalibrationConfig,
    CalibrationData
)
from hydrosis.validation import BaseValidator, ValidationResult


class HydrologicWorkflow:
    """完整的水文建模工作流"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 工作流状态
        self.data = {}
        self.validation_results = {}
        self.diagnostic_results = {}
        self.calibration_results = {}

    def step1_generate_sample_data(self):
        """步骤1: 生成示例数据"""
        print("\n" + "="*80)
        print("步骤1: 生成示例数据")
        print("="*80)

        np.random.seed(42)

        # 生成多个流域的数据
        n_basins = 3
        n_days = 90
        hours = n_days * 24

        for basin_id in range(1, n_basins + 1):
            dates = pd.date_range('2024-01-01', periods=hours, freq='h')

            # 生成降雨数据
            precipitation = np.random.gamma(2, 2, size=hours) * 0.8
            precipitation[precipitation < 0.1] = 0

            # 生成温度数据
            base_temp = 15 + 10 * np.sin(2 * np.pi * np.arange(hours) / (24 * 365))
            temperature = base_temp + np.random.normal(0, 2, hours)

            # 模拟观测径流 (用于校准)
            runoff = precipitation * (0.4 + basin_id * 0.05)
            runoff += np.random.normal(0, 0.3, hours)
            runoff[runoff < 0] = 0

            self.data[f'basin_{basin_id}'] = {
                'dates': dates,
                'precipitation': precipitation,
                'temperature': temperature,
                'runoff': runoff,
                'area_km2': 100 + basin_id * 50
            }

        print(f"✓ 生成了 {len(self.data)} 个流域的数据")
        print(f"  时长: {n_days} 天 ({hours} 小时)")
        for name, data in self.data.items():
            print(f"  {name}: 面积 {data['area_km2']:.0f} km², "
                  f"平均降雨 {data['precipitation'].mean():.2f} mm/h")

    def step2_validate_data(self):
        """步骤2: 数据验证"""
        print("\n" + "="*80)
        print("步骤2: 数据质量验证")
        print("="*80)

        class DataQualityValidator(BaseValidator):
            """数据质量验证器"""

            def validate(self, data: dict) -> ValidationResult:
                errors = []
                warnings = []
                metrics = {}

                # 检查降雨数据
                precip = data.get('precipitation')
                if precip is None:
                    errors.append("缺少降雨数据")
                else:
                    # 缺失值检查
                    missing_count = np.isnan(precip).sum()
                    if missing_count > 0:
                        errors.append(f"降雨数据存在 {missing_count} 个缺失值")

                    # 异常值检查
                    if np.any(precip < 0):
                        errors.append("降雨数据存在负值")
                    if np.any(precip > 100):
                        warnings.append("降雨数据存在极端值 (>100 mm/h)")

                    metrics['mean_precipitation'] = np.nanmean(precip)
                    metrics['max_precipitation'] = np.nanmax(precip)

                # 检查径流数据
                runoff = data.get('runoff')
                if runoff is not None:
                    if np.any(runoff < 0):
                        errors.append("径流数据存在负值")
                    metrics['mean_runoff'] = np.nanmean(runoff)

                # 计算径流系数
                if precip is not None and runoff is not None:
                    total_precip = np.nansum(precip)
                    total_runoff = np.nansum(runoff)
                    if total_precip > 0:
                        rc = total_runoff / total_precip
                        metrics['runoff_coefficient'] = rc
                        if rc > 1.0:
                            errors.append(f"径流系数异常: {rc:.2f} > 1.0")
                        elif rc > 0.9:
                            warnings.append(f"径流系数偏高: {rc:.2f}")

                is_valid = len(errors) == 0

                return ValidationResult(
                    is_valid=is_valid,
                    errors=errors,
                    warnings=warnings,
                    metrics=metrics
                )

        # 验证所有流域数据
        validator = DataQualityValidator()

        print("\n⚙️  验证数据质量...")
        for name, data in self.data.items():
            result = validator.validate(data)
            self.validation_results[name] = result

            status = "✓" if result.is_valid else "✗"
            print(f"  {status} {name}: 错误={len(result.errors)}, "
                  f"警告={len(result.warnings)}")

        # 汇总
        valid_count = sum(1 for r in self.validation_results.values() if r.is_valid)
        print(f"\n✓ 验证完成: {valid_count}/{len(self.validation_results)} 个流域数据有效")

    def step3_run_diagnostics(self):
        """步骤3: 模型诊断"""
        print("\n" + "="*80)
        print("步骤3: 运行模型诊断")
        print("="*80)

        # 选择第一个流域进行详细诊断
        basin_name = list(self.data.keys())[0]
        data = self.data[basin_name]

        print(f"\n🔍 诊断流域: {basin_name}")

        # 水量平衡诊断
        print("\n  运行水量平衡诊断...")
        wb_diagnostic = WaterBalanceDiagnostic(
            output_dir=self.output_dir / "diagnostics" / basin_name,
            verbose=False
        )

        wb_result = wb_diagnostic.run(
            precipitation=data['precipitation'],
            runoff=data['runoff'],
            initial_lower=2000.0,
            k2=0.02
        )

        self.diagnostic_results[f'{basin_name}_water_balance'] = wb_result

        # 报告关键问题
        errors = [i for i in wb_result.issues if i.severity.value in ['error', 'critical']]
        if errors:
            print(f"    ⚠️  发现 {len(errors)} 个问题")
            for issue in errors[:2]:  # 只显示前2个
                print(f"      • {issue.message}")
        else:
            print(f"    ✓ 未发现严重问题")

        print(f"    径流系数: {wb_result.metrics.get('runoff_coefficient', 0):.4f}")

    def step4_calibrate_model(self):
        """步骤4: 模型校准"""
        print("\n" + "="*80)
        print("步骤4: 模型参数校准")
        print("="*80)

        # 对每个流域进行校准
        for basin_name, data in self.data.items():
            print(f"\n⚙️  校准 {basin_name}...")

            # 准备校准数据
            calib_data = CalibrationData(
                precipitation=data['precipitation'],
                observed_runoff=data['runoff'],
                area_km2=data['area_km2'],
                temperature=data['temperature']
            )

            # 配置校准
            config = CalibrationConfig(
                param_bounds={
                    'FC': [100, 500],
                    'BETA': [1.0, 4.0],
                },
                fixed_params={
                    'LP': 0.7,
                    'K0': 0.1,
                    'K1': 0.05,
                    'K2': 0.01,
                    'PERC': 2.0,
                    'UZL': 50.0,
                    'TT': 0.0,
                    'CFMAX': 3.0,
                    'CFR': 0.05,
                    'CWH': 0.1,
                },
                algorithm='differential_evolution',
                algorithm_params={'maxiter': 5, 'popsize': 8},
                objective_metric='nse',
                warmup_steps=24,  # 1天预热
                seed=42
            )

            # 运行校准
            calibrator = HBVCalibrator(
                data=calib_data,
                config=config,
                output_dir=self.output_dir / "calibration" / basin_name
            )

            try:
                result = calibrator.run_calibration()
                self.calibration_results[basin_name] = result

                print(f"  ✓ NSE: {result.metrics['nse']:.4f}")
                print(f"    最优参数: FC={result.best_params['FC']:.1f}, "
                      f"BETA={result.best_params['BETA']:.2f}")

            except Exception as e:
                print(f"  ✗ 校准失败: {e}")

    def step5_generate_report(self):
        """步骤5: 生成工作流报告"""
        print("\n" + "="*80)
        print("步骤5: 生成综合报告")
        print("="*80)

        report_path = self.output_dir / "workflow_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HydroSIS 水文建模工作流报告\n")
            f.write("="*80 + "\n\n")

            # 数据概况
            f.write("1. 数据概况\n")
            f.write("-" * 40 + "\n")
            for name, data in self.data.items():
                f.write(f"  {name}:\n")
                f.write(f"    面积: {data['area_km2']:.0f} km²\n")
                f.write(f"    数据长度: {len(data['precipitation'])} 小时\n")
                f.write(f"    平均降雨: {data['precipitation'].mean():.2f} mm/h\n")
                f.write(f"    平均径流: {data['runoff'].mean():.2f} mm/h\n")
            f.write("\n")

            # 验证结果
            f.write("2. 数据验证结果\n")
            f.write("-" * 40 + "\n")
            for name, result in self.validation_results.items():
                status = "通过" if result.is_valid else "失败"
                f.write(f"  {name}: {status}\n")
                if result.errors:
                    for error in result.errors:
                        f.write(f"    错误: {error}\n")
                if result.warnings:
                    for warning in result.warnings:
                        f.write(f"    警告: {warning}\n")
            f.write("\n")

            # 诊断结果
            f.write("3. 模型诊断结果\n")
            f.write("-" * 40 + "\n")
            for name, result in self.diagnostic_results.items():
                f.write(f"  {name}:\n")
                f.write(f"    问题数: {len(result.issues)}\n")
                if result.recommendations:
                    f.write(f"    建议数: {len(result.recommendations)}\n")
            f.write("\n")

            # 校准结果
            f.write("4. 模型校准结果\n")
            f.write("-" * 40 + "\n")
            for name, result in self.calibration_results.items():
                f.write(f"  {name}:\n")
                f.write(f"    NSE: {result.metrics['nse']:.4f}\n")
                f.write(f"    KGE: {result.metrics['kge']:.4f}\n")
                f.write(f"    RMSE: {result.metrics['rmse']:.4f}\n")
                f.write(f"    计算时间: {result.computation_time:.2f}秒\n")
                f.write(f"    评估次数: {result.n_evaluations}\n")
            f.write("\n")

        print(f"✓ 报告已生成: {report_path}")

    def run_complete_workflow(self):
        """运行完整工作流"""
        print("="*80)
        print("HydroSIS 综合工作流示例")
        print("="*80)
        print("\n展示完整的建模流程: 数据生成 → 验证 → 诊断 → 校准 → 报告")

        start_time = time.time()

        # 执行各步骤
        self.step1_generate_sample_data()
        self.step2_validate_data()
        self.step3_run_diagnostics()
        self.step4_calibrate_model()
        self.step5_generate_report()

        total_time = time.time() - start_time

        # 最终总结
        print("\n" + "="*80)
        print("✅ 工作流执行完成！")
        print("="*80)
        print(f"\n📊 执行统计:")
        print(f"  总耗时: {total_time:.2f}秒")
        print(f"  处理流域: {len(self.data)}")
        print(f"  验证流域: {len(self.validation_results)}")
        print(f"  诊断次数: {len(self.diagnostic_results)}")
        print(f"  校准流域: {len(self.calibration_results)}")

        print(f"\n📁 输出文件:")
        print(f"  工作目录: {self.output_dir}")
        print(f"  - 诊断报告和图表: diagnostics/")
        print(f"  - 校准结果: calibration/")
        print(f"  - 综合报告: workflow_report.txt")

        print(f"\n💡 下一步:")
        print(f"  1. 查看各流域的诊断报告")
        print(f"  2. 分析校准结果和最优参数")
        print(f"  3. 根据建议优化模型配置")


def main():
    """主函数"""
    # 创建工作流
    workflow = HydrologicWorkflow(
        output_dir=Path("results/integrated_workflow_example")
    )

    # 运行完整工作流
    workflow.run_complete_workflow()


if __name__ == "__main__":
    main()
