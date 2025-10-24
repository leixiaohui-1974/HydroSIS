"""并行框架单元测试"""
import pytest
import time
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from hydrosis.parallel import (
    ExecutionMode,
    ExecutionConfig,
    TaskResult,
    ParallelExecutor,
    parallel_map,
    ParallelValidator,
    validate_datasets_parallel
)

from hydrosis.validation import (
    BaseValidator,
    ValidationResult,
    ValidationCriteria,
    PrecipitationCriteria,
    validate_precipitation_data
)


@pytest.mark.unit
@pytest.mark.parallel
class TestTaskResult:
    """测试 TaskResult 数据类"""

    def test_successful_result(self):
        """测试成功结果"""
        result = TaskResult(
            task_id="task1",
            success=True,
            result=42,
            execution_time=1.5
        )

        assert result.task_id == "task1"
        assert result.success is True
        assert result.result == 42
        assert result.execution_time == 1.5
        assert result.error is None

    def test_failed_result(self):
        """测试失败结果"""
        error = ValueError("test error")
        result = TaskResult(
            task_id="task2",
            success=False,
            error=error
        )

        assert result.task_id == "task2"
        assert result.success is False
        assert result.error == error

    def test_result_str(self):
        """测试字符串表示"""
        result = TaskResult(task_id=1, success=True, execution_time=2.5)
        str_repr = str(result)

        assert "task_id=1" in str_repr or "id=1" in str_repr
        assert "success=True" in str_repr
        assert "2.5" in str_repr or "2.50" in str_repr


@pytest.mark.unit
@pytest.mark.parallel
class TestExecutionConfig:
    """测试 ExecutionConfig"""

    def test_default_config(self):
        """测试默认配置"""
        config = ExecutionConfig()

        assert config.mode == ExecutionMode.MULTIPROCESS
        assert config.max_workers is not None
        assert config.chunk_size == 1
        assert config.show_progress is True

    def test_sequential_config(self):
        """测试串行配置"""
        config = ExecutionConfig(mode=ExecutionMode.SEQUENTIAL)

        assert config.mode == ExecutionMode.SEQUENTIAL
        assert config.max_workers == 1

    def test_custom_workers(self):
        """测试自定义worker数"""
        config = ExecutionConfig(max_workers=8)

        assert config.max_workers == 8

    def test_retry_config(self):
        """测试重试配置"""
        config = ExecutionConfig(
            retry_on_failure=True,
            max_retries=5
        )

        assert config.retry_on_failure is True
        assert config.max_retries == 5


@pytest.mark.unit
@pytest.mark.parallel
class TestParallelExecutor:
    """测试 ParallelExecutor 基类"""

    def test_cannot_instantiate_abstract_class(self):
        """测试不能实例化抽象类"""
        with pytest.raises(TypeError):
            ParallelExecutor()

    def test_simple_executor(self):
        """测试简单执行器"""
        class SquareExecutor(ParallelExecutor[int, int]):
            def execute_task(self, task: int) -> int:
                return task ** 2

        executor = SquareExecutor(
            config=ExecutionConfig(mode=ExecutionMode.SEQUENTIAL),
            verbose=False
        )

        tasks = [1, 2, 3, 4, 5]
        results = executor.run(tasks)

        assert len(results) == 5
        assert all(r.success for r in results)

        # 验证结果
        values = executor.get_successful_results(results)
        assert values == [1, 4, 9, 16, 25]

    def test_executor_with_failure(self):
        """测试带失败的执行器"""
        class FlakyExecutor(ParallelExecutor[int, int]):
            def execute_task(self, task: int) -> int:
                if task == 3:
                    raise ValueError("Task 3 always fails")
                return task * 10

        executor = FlakyExecutor(
            config=ExecutionConfig(
                mode=ExecutionMode.SEQUENTIAL,
                raise_on_error=False
            ),
            verbose=False
        )

        tasks = [1, 2, 3, 4, 5]
        results = executor.run(tasks)

        assert len(results) == 5

        # 1 个失败，4 个成功
        assert executor._successful_tasks == 4
        assert executor._failed_tasks == 1

        # 获取失败任务
        failed = executor.get_failed_tasks(results)
        assert len(failed) == 1
        assert failed[0].task_id == id(3)

    def test_executor_with_retry(self):
        """测试带重试的执行器"""
        attempt_counter = {}

        class RetryExecutor(ParallelExecutor[int, int]):
            def execute_task(self, task: int) -> int:
                # 计数尝试次数
                if task not in attempt_counter:
                    attempt_counter[task] = 0
                attempt_counter[task] += 1

                # 第一次失败，第二次成功
                if attempt_counter[task] == 1:
                    raise ValueError("First attempt fails")

                return task * 100

        executor = RetryExecutor(
            config=ExecutionConfig(
                mode=ExecutionMode.SEQUENTIAL,
                retry_on_failure=True,
                max_retries=2
            ),
            verbose=False
        )

        tasks = [1, 2, 3]
        results = executor.run(tasks)

        # 所有任务应该成功（通过重试）
        assert executor._successful_tasks == 3
        values = executor.get_successful_results(results)
        assert values == [100, 200, 300]

        # 验证重试次数
        for task in tasks:
            assert attempt_counter[task] == 2

    def test_sequential_execution(self):
        """测试串行执行"""
        class SlowExecutor(ParallelExecutor[int, int]):
            def execute_task(self, task: int) -> int:
                time.sleep(0.01)  # 模拟耗时操作
                return task + 1

        executor = SlowExecutor(
            config=ExecutionConfig(mode=ExecutionMode.SEQUENTIAL),
            verbose=False
        )

        start_time = time.time()
        results = executor.run([1, 2, 3, 4, 5])
        elapsed = time.time() - start_time

        assert len(results) == 5
        # 串行执行应该至少花费 5 * 0.01 = 0.05 秒
        assert elapsed >= 0.04

    def test_parallel_execution(self):
        """测试并行执行"""
        class SlowExecutor(ParallelExecutor[int, int]):
            def execute_task(self, task: int) -> int:
                time.sleep(0.05)  # 模拟耗时操作
                return task + 1

        executor = SlowExecutor(
            config=ExecutionConfig(
                mode=ExecutionMode.MULTIPROCESS,
                max_workers=4
            ),
            verbose=False
        )

        start_time = time.time()
        results = executor.run([1, 2, 3, 4, 5])
        elapsed = time.time() - start_time

        assert len(results) == 5

        # 并行执行应该快于串行（虽然有进程启动开销）
        # 这里不强制验证时间，因为CI环境可能不稳定

    def test_task_id_override(self):
        """测试自定义任务ID"""
        class NamedTaskExecutor(ParallelExecutor[dict, str]):
            def execute_task(self, task: dict) -> str:
                return task['name'].upper()

            def get_task_id(self, task: dict) -> str:
                return task['name']

        executor = NamedTaskExecutor(
            config=ExecutionConfig(mode=ExecutionMode.SEQUENTIAL),
            verbose=False
        )

        tasks = [
            {'name': 'alice'},
            {'name': 'bob'},
            {'name': 'charlie'}
        ]

        results = executor.run(tasks)

        assert len(results) == 3
        assert results[0].task_id == 'alice'
        assert results[1].task_id == 'bob'
        assert results[2].task_id == 'charlie'

    def test_success_rate(self):
        """测试成功率计算"""
        class FlakyExecutor(ParallelExecutor[int, int]):
            def execute_task(self, task: int) -> int:
                if task % 2 == 0:
                    raise ValueError("Even number")
                return task

        executor = FlakyExecutor(
            config=ExecutionConfig(mode=ExecutionMode.SEQUENTIAL, raise_on_error=False),
            verbose=False
        )

        results = executor.run([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # 5个奇数成功，5个偶数失败
        assert executor.success_rate == 0.5


@pytest.mark.unit
@pytest.mark.parallel
class TestParallelMap:
    """测试 parallel_map 便捷函数"""

    def test_simple_map(self):
        """测试简单映射"""
        def square(x):
            return x ** 2

        results = parallel_map(
            square,
            [1, 2, 3, 4, 5],
            config=ExecutionConfig(mode=ExecutionMode.SEQUENTIAL),
            verbose=False
        )

        assert results == [1, 4, 9, 16, 25]

    def test_map_with_lambda(self):
        """测试使用lambda"""
        results = parallel_map(
            lambda x: x * 10,
            range(5),
            config=ExecutionConfig(mode=ExecutionMode.SEQUENTIAL),
            verbose=False
        )

        assert results == [0, 10, 20, 30, 40]


@pytest.mark.unit
@pytest.mark.parallel
class TestParallelValidator:
    """测试 ParallelValidator"""

    @pytest.fixture
    def sample_precipitation_dfs(self):
        """生成多个示例降雨数据集"""
        np.random.seed(42)
        datasets = {}

        for year in [2020, 2021, 2022]:
            dates = pd.date_range(f'{year}-01-01', periods=100, freq='h')
            data = {
                'station_1': np.random.gamma(2, 2, size=100),
                'station_2': np.random.gamma(2, 2, size=100),
                'station_3': np.random.gamma(2, 2, size=100),
            }
            df = pd.DataFrame(data, index=dates)
            datasets[str(year)] = df

        return datasets

    @pytest.fixture
    def simple_validator(self):
        """创建简单的测试验证器"""
        class SimpleValidator(BaseValidator):
            def validate(self, data):
                # 简单验证：检查是否为DataFrame且有数据
                result = ValidationResult(
                    is_valid=isinstance(data, pd.DataFrame) and len(data) > 0,
                    errors=[],
                    warnings=[],
                    metrics={"row_count": len(data) if isinstance(data, pd.DataFrame) else 0}
                )
                if not isinstance(data, pd.DataFrame):
                    result.errors.append("Data is not a DataFrame")
                return result

        return SimpleValidator()

    def test_parallel_validator_creation(self, simple_validator):
        """测试并行验证器创建"""
        parallel_validator = ParallelValidator(simple_validator, verbose=False)

        assert parallel_validator.validator == simple_validator

    def test_validate_multiple_datasets(self, sample_precipitation_dfs, simple_validator):
        """测试验证多个数据集"""
        parallel_validator = ParallelValidator(
            simple_validator,
            config=ExecutionConfig(mode=ExecutionMode.SEQUENTIAL),
            verbose=False
        )

        results = parallel_validator.validate_multiple_datasets(sample_precipitation_dfs)

        assert len(results) == 3
        assert '2020' in results
        assert '2021' in results
        assert '2022' in results

        # 所有结果应该是 ValidationResult
        for year, result in results.items():
            assert hasattr(result, 'is_valid')
            assert hasattr(result, 'metrics')
            assert result.is_valid is True  # 所有数据集应该有效

    def test_validation_summary(self, sample_precipitation_dfs, simple_validator):
        """测试验证摘要"""
        parallel_validator = ParallelValidator(
            simple_validator,
            config=ExecutionConfig(mode=ExecutionMode.SEQUENTIAL),
            verbose=False
        )

        # 构建任务
        tasks = [
            {"data": df, "task_name": name}
            for name, df in sample_precipitation_dfs.items()
        ]

        # 执行验证
        task_results = parallel_validator.run(tasks)

        # 生成摘要
        summary = parallel_validator.get_validation_summary(task_results)

        assert summary['total_validations'] == 3
        assert summary['successful_validations'] >= 0
        assert 'validation_results' in summary
        assert len(summary['validation_results']) == 3

    def test_validate_datasets_parallel_function(self, sample_precipitation_dfs, simple_validator):
        """测试便捷函数 validate_datasets_parallel"""
        results = validate_datasets_parallel(
            simple_validator,
            sample_precipitation_dfs,
            config=ExecutionConfig(mode=ExecutionMode.SEQUENTIAL),
            verbose=False
        )

        assert len(results) == 3
        for year in ['2020', '2021', '2022']:
            assert year in results


# 定义在模块级别以支持pickle
class DoubleExecutor(ParallelExecutor[int, int]):
    def execute_task(self, task: int) -> int:
        return task * 2


@pytest.mark.unit
@pytest.mark.parallel
class TestExecutionModes:
    """测试不同执行模式"""

    def test_all_modes_produce_same_results(self):
        """测试所有执行模式产生相同结果"""
        tasks = list(range(10))
        expected = [x * 2 for x in tasks]

        for mode in [ExecutionMode.SEQUENTIAL, ExecutionMode.MULTIPROCESS, ExecutionMode.MULTITHREADED]:
            executor = DoubleExecutor(
                config=ExecutionConfig(mode=mode, max_workers=2),
                verbose=False
            )

            results = executor.run(tasks)
            values = executor.get_successful_results(results)

            # 所有模式应该产生相同结果（顺序可能不同）
            assert sorted(values) == sorted(expected)
