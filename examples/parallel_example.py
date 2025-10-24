#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并行框架使用示例

展示如何使用通用并行执行框架加速计算密集型任务。

使用方法:
    python examples/parallel_example.py

特性:
    - 支持多进程、多线程、串行3种模式
    - 自动重试失败任务
    - 实时进度跟踪
    - 详细性能统计
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

from hydrosis.parallel import (
    ParallelExecutor,
    ExecutionConfig,
    ExecutionMode,
    parallel_map,
    validate_datasets_parallel
)
from hydrosis.validation import BaseValidator, ValidationResult


def example_simple_parallel_map():
    """示例1: 简单并行映射"""
    print("\n" + "="*80)
    print("示例1: 简单并行映射 - 计算平方")
    print("="*80)

    def square(x):
        """计算平方（模拟耗时操作）"""
        time.sleep(0.01)  # 模拟计算延迟
        return x ** 2

    numbers = list(range(50))

    # 串行执行
    print("\n⚙️  串行执行...")
    config_seq = ExecutionConfig(mode=ExecutionMode.SEQUENTIAL)
    start = time.time()
    results_seq = parallel_map(square, numbers, config=config_seq, verbose=False)
    time_seq = time.time() - start

    # 并行执行（多进程）
    print("\n⚙️  并行执行（多进程）...")
    config_parallel = ExecutionConfig(
        mode=ExecutionMode.MULTIPROCESS,
        max_workers=4
    )
    start = time.time()
    results_parallel = parallel_map(square, numbers, config=config_parallel, verbose=False)
    time_parallel = time.time() - start

    # 对比结果
    print("\n📊 性能对比:")
    print(f"  串行时间: {time_seq:.2f}秒")
    print(f"  并行时间: {time_parallel:.2f}秒")
    print(f"  加速比: {time_seq/time_parallel:.2f}x")
    print(f"  结果匹配: {results_seq == results_parallel}")


def example_custom_parallel_executor():
    """示例2: 自定义并行执行器"""
    print("\n" + "="*80)
    print("示例2: 自定义并行执行器 - 数据处理")
    print("="*80)

    class DataProcessor(ParallelExecutor[Dict, pd.DataFrame]):
        """自定义数据处理执行器"""

        def execute_task(self, task: Dict) -> pd.DataFrame:
            """处理单个数据任务"""
            # 模拟数据处理
            n_rows = task.get('n_rows', 100)
            n_cols = task.get('n_cols', 5)

            # 生成随机数据
            np.random.seed(task.get('seed', 0))
            data = np.random.randn(n_rows, n_cols)

            # 模拟耗时操作
            time.sleep(0.05)

            # 计算统计量
            df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(n_cols)])
            df['mean'] = df.mean(axis=1)
            df['std'] = df.std(axis=1)

            return df

        def get_task_id(self, task: Dict) -> str:
            """获取任务ID"""
            return f"dataset_{task.get('id', 'unknown')}"

    # 创建任务
    tasks = [
        {'id': 1, 'n_rows': 100, 'n_cols': 5, 'seed': 42},
        {'id': 2, 'n_rows': 200, 'n_cols': 5, 'seed': 43},
        {'id': 3, 'n_rows': 150, 'n_cols': 5, 'seed': 44},
        {'id': 4, 'n_rows': 120, 'n_cols': 5, 'seed': 45},
    ]

    # 创建并行执行器
    processor = DataProcessor(
        config=ExecutionConfig(
            mode=ExecutionMode.MULTIPROCESS,
            max_workers=2,
            show_progress=True
        )
    )

    # 执行任务
    print("\n⚙️  处理数据集...")
    results = processor.run(tasks)

    # 分析结果
    print(f"\n📊 处理结果:")
    print(f"  总任务数: {len(results)}")
    print(f"  成功: {processor._successful_tasks}")
    print(f"  失败: {processor._failed_tasks}")
    print(f"  成功率: {processor.success_rate*100:.1f}%")
    print(f"  总时间: {processor._total_time:.2f}秒")

    # 显示处理后的数据
    successful_dfs = processor.get_successful_results(results)
    for i, df in enumerate(successful_dfs):
        print(f"\n  Dataset {i+1}: {df.shape} - mean={df['mean'].mean():.4f}")


def example_parallel_validation():
    """示例3: 并行数据验证"""
    print("\n" + "="*80)
    print("示例3: 并行数据验证")
    print("="*80)

    # 创建简单验证器
    class SimpleValidator(BaseValidator):
        """简单数据验证器"""

        def validate(self, data: pd.DataFrame) -> ValidationResult:
            """验证DataFrame"""
            errors = []
            warnings = []
            metrics = {}

            # 检查是否为空
            if data.empty:
                errors.append("数据为空")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    metrics=metrics
                )

            # 检查缺失值
            missing_count = data.isnull().sum().sum()
            missing_ratio = missing_count / data.size
            metrics['missing_ratio'] = missing_ratio

            if missing_ratio > 0.1:
                errors.append(f"缺失值过多: {missing_ratio*100:.1f}%")
            elif missing_ratio > 0.05:
                warnings.append(f"存在缺失值: {missing_ratio*100:.1f}%")

            # 检查异常值
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                outliers = 0
                for col in numeric_cols:
                    q1 = data[col].quantile(0.25)
                    q3 = data[col].quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers += ((data[col] < lower) | (data[col] > upper)).sum()

                outlier_ratio = outliers / data.size
                metrics['outlier_ratio'] = outlier_ratio

                if outlier_ratio > 0.05:
                    warnings.append(f"检测到异常值: {outlier_ratio*100:.1f}%")

            # 判断是否有效
            is_valid = len(errors) == 0

            return ValidationResult(
                is_valid=is_valid,
                errors=errors,
                warnings=warnings,
                metrics=metrics
            )

    # 生成多个测试数据集
    print("\n📊 生成测试数据集...")
    np.random.seed(42)

    datasets = {}
    for year in [2020, 2021, 2022, 2023, 2024]:
        # 生成不同质量的数据
        n_rows = 365
        data = {
            'temperature': np.random.normal(20, 5, n_rows),
            'precipitation': np.random.gamma(2, 2, n_rows),
            'humidity': np.random.uniform(30, 90, n_rows),
        }

        df = pd.DataFrame(data)

        # 随机添加缺失值
        if year % 2 == 0:
            missing_indices = np.random.choice(df.index, size=int(n_rows*0.02), replace=False)
            df.loc[missing_indices, 'temperature'] = np.nan

        datasets[str(year)] = df

    # 并行验证
    print(f"\n⚙️  并行验证 {len(datasets)} 个数据集...")
    validator = SimpleValidator()

    validation_results = validate_datasets_parallel(
        validator,
        datasets,
        config=ExecutionConfig(
            mode=ExecutionMode.MULTIPROCESS,
            max_workers=3
        )
    )

    # 显示验证结果
    print("\n📋 验证结果:")
    print(f"{'数据集':10} {'有效性':8} {'错误数':8} {'警告数':8} {'缺失率':12}")
    print("-" * 60)

    for name, result in validation_results.items():
        validity = "✓" if result.is_valid else "✗"
        missing_ratio = result.metrics.get('missing_ratio', 0)
        print(f"{name:10} {validity:^8} {len(result.errors):^8} {len(result.warnings):^8} {missing_ratio*100:>10.1f}%")


def example_retry_mechanism():
    """示例4: 失败重试机制"""
    print("\n" + "="*80)
    print("示例4: 失败重试机制")
    print("="*80)

    # 模拟不稳定的任务
    attempt_counter = {}

    class FlakyExecutor(ParallelExecutor[int, int]):
        """不稳定的执行器（模拟网络请求等）"""

        def execute_task(self, task: int) -> int:
            """执行任务，前两次失败"""
            if task not in attempt_counter:
                attempt_counter[task] = 0
            attempt_counter[task] += 1

            # 前两次尝试失败
            if attempt_counter[task] <= 2:
                raise ConnectionError(f"模拟失败 (尝试 {attempt_counter[task]})")

            # 第三次成功
            return task * 10

    # 不启用重试
    print("\n⚙️  不启用重试...")
    executor_no_retry = FlakyExecutor(
        config=ExecutionConfig(
            mode=ExecutionMode.SEQUENTIAL,
            retry_on_failure=False
        ),
        verbose=False
    )

    attempt_counter.clear()
    results_no_retry = executor_no_retry.run([1, 2, 3])
    success_no_retry = executor_no_retry._successful_tasks

    # 启用重试
    print("\n⚙️  启用重试（最多3次）...")
    executor_with_retry = FlakyExecutor(
        config=ExecutionConfig(
            mode=ExecutionMode.SEQUENTIAL,
            retry_on_failure=True,
            max_retries=3
        ),
        verbose=False
    )

    attempt_counter.clear()
    results_with_retry = executor_with_retry.run([1, 2, 3])
    success_with_retry = executor_with_retry._successful_tasks

    # 对比结果
    print("\n📊 重试机制对比:")
    print(f"  不启用重试: 成功 {success_no_retry}/3")
    print(f"  启用重试:   成功 {success_with_retry}/3")
    print(f"\n  💡 通过重试机制，成功率从 {success_no_retry/3*100:.0f}% 提升到 {success_with_retry/3*100:.0f}%")


def example_performance_comparison():
    """示例5: 性能对比"""
    print("\n" + "="*80)
    print("示例5: 不同执行模式性能对比")
    print("="*80)

    def cpu_intensive_task(n):
        """CPU密集型任务"""
        result = 0
        for i in range(10000):
            result += i ** 0.5
        return result * n

    tasks = list(range(20))

    modes = [
        (ExecutionMode.SEQUENTIAL, "串行"),
        (ExecutionMode.MULTITHREADED, "多线程"),
        (ExecutionMode.MULTIPROCESS, "多进程"),
    ]

    results_table = []

    for mode, name in modes:
        config = ExecutionConfig(mode=mode, max_workers=4)

        start = time.time()
        results = parallel_map(cpu_intensive_task, tasks, config=config, verbose=False)
        elapsed = time.time() - start

        results_table.append({
            'mode': name,
            'time': elapsed,
            'speedup': 1.0  # 将在后面计算
        })

    # 计算加速比
    baseline = results_table[0]['time']
    for row in results_table:
        row['speedup'] = baseline / row['time']

    # 显示结果
    print("\n📊 性能对比:")
    print(f"{'执行模式':12} {'时间(秒)':12} {'加速比':12}")
    print("-" * 40)
    for row in results_table:
        print(f"{row['mode']:12} {row['time']:>10.2f}s {row['speedup']:>10.2f}x")

    print("\n💡 最佳实践:")
    print("  - CPU密集型任务: 使用多进程")
    print("  - I/O密集型任务: 使用多线程")
    print("  - 调试/简单任务: 使用串行")


def main():
    """主函数"""
    print("="*80)
    print("并行框架使用示例")
    print("="*80)
    print("\n本示例展示如何使用 HydroSIS 并行框架加速计算")

    # 示例1: 简单并行映射
    example_simple_parallel_map()

    # 示例2: 自定义并行执行器
    example_custom_parallel_executor()

    # 示例3: 并行数据验证
    example_parallel_validation()

    # 示例4: 失败重试机制
    example_retry_mechanism()

    # 示例5: 性能对比
    example_performance_comparison()

    print("\n" + "="*80)
    print("✅ 所有示例运行完成！")
    print("="*80)

    print("\n💡 关键要点:")
    print("  1. 使用 parallel_map 进行简单并行映射")
    print("  2. 继承 ParallelExecutor 创建自定义执行器")
    print("  3. 根据任务类型选择合适的执行模式")
    print("  4. 启用重试机制提高可靠性")
    print("  5. 监控性能指标优化配置")


if __name__ == "__main__":
    main()
