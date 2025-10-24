"""并行验证框架

提供并行数据验证能力，加速大规模数据集的验证过程。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..validation.base import BaseValidator, ValidationResult
from .executor import ParallelExecutor, ExecutionConfig, TaskResult


class ParallelValidator(ParallelExecutor[Dict[str, Any], ValidationResult]):
    """并行验证执行器

    支持并行验证多个数据集或验证规则。

    Parameters
    ----------
    validator : BaseValidator
        验证器实例
    config : ExecutionConfig, optional
        并行执行配置
    verbose : bool
        是否输出详细信息

    Example
    -------
    >>> from hydrosis.validation import PrecipitationValidator
    >>> validator = PrecipitationValidator()
    >>> parallel_validator = ParallelValidator(validator)
    >>>
    >>> tasks = [
    ...     {"data": df1, "task_name": "dataset1"},
    ...     {"data": df2, "task_name": "dataset2"},
    ... ]
    >>> results = parallel_validator.run(tasks)
    """

    def __init__(
        self,
        validator: BaseValidator,
        config: Optional[ExecutionConfig] = None,
        verbose: bool = True
    ):
        super().__init__(config=config, verbose=verbose)
        self.validator = validator

    def execute_task(self, task: Dict[str, Any]) -> ValidationResult:
        """执行单个验证任务

        Parameters
        ----------
        task : dict
            验证任务，应包含验证所需的数据和参数

        Returns
        -------
        ValidationResult
            验证结果
        """
        # 提取任务参数
        data = task.get("data")
        task_name = task.get("task_name", "unknown")

        # 执行验证
        result = self.validator.validate(data)

        # 添加任务名称到元数据
        if hasattr(result, 'metadata'):
            result.metadata["task_name"] = task_name

        return result

    def get_task_id(self, task: Dict[str, Any]) -> str:
        """获取任务ID"""
        return task.get("task_name", str(id(task)))

    def validate_multiple_datasets(
        self,
        datasets: Dict[str, Any]
    ) -> Dict[str, ValidationResult]:
        """验证多个数据集

        Parameters
        ----------
        datasets : dict
            数据集字典 {name: data}

        Returns
        -------
        dict
            验证结果字典 {name: ValidationResult}
        """
        # 构建任务列表
        tasks = [
            {"data": data, "task_name": name}
            for name, data in datasets.items()
        ]

        # 执行并行验证
        task_results = self.run(tasks)

        # 转换为字典格式
        results = {}
        for task_result in task_results:
            if task_result.success:
                name = task_result.task_id
                results[name] = task_result.result

        return results

    def get_validation_summary(
        self,
        results: List[TaskResult]
    ) -> Dict[str, Any]:
        """生成验证摘要

        Parameters
        ----------
        results : list of TaskResult
            任务结果列表

        Returns
        -------
        dict
            验证摘要
        """
        summary = {
            "total_validations": len(results),
            "successful_validations": sum(1 for r in results if r.success),
            "failed_validations": sum(1 for r in results if not r.success),
            "validation_results": {}
        }

        for task_result in results:
            if task_result.success and task_result.result:
                result: ValidationResult = task_result.result
                task_id = task_result.task_id

                summary["validation_results"][task_id] = {
                    "is_valid": result.is_valid,
                    "error_count": len(result.errors),
                    "warning_count": len(result.warnings),
                    "metrics": result.metrics
                }

        # 计算总体统计
        all_valid = all(
            info["is_valid"]
            for info in summary["validation_results"].values()
        )
        summary["all_valid"] = all_valid

        total_errors = sum(
            info["error_count"]
            for info in summary["validation_results"].values()
        )
        summary["total_errors"] = total_errors

        total_warnings = sum(
            info["warning_count"]
            for info in summary["validation_results"].values()
        )
        summary["total_warnings"] = total_warnings

        return summary


class BatchFileValidator(ParallelExecutor[Path, ValidationResult]):
    """批量文件验证器

    并行验证多个文件。

    Parameters
    ----------
    validator : BaseValidator
        验证器实例
    file_loader : callable
        文件加载函数 (Path) -> data
    config : ExecutionConfig, optional
        并行执行配置
    verbose : bool
        是否输出详细信息

    Example
    -------
    >>> def load_csv(path):
    ...     return pd.read_csv(path)
    >>>
    >>> validator = PrecipitationValidator()
    >>> batch_validator = BatchFileValidator(validator, load_csv)
    >>>
    >>> file_paths = [Path(f) for f in glob.glob("data/*.csv")]
    >>> results = batch_validator.run(file_paths)
    """

    def __init__(
        self,
        validator: BaseValidator,
        file_loader: callable,
        config: Optional[ExecutionConfig] = None,
        verbose: bool = True
    ):
        super().__init__(config=config, verbose=verbose)
        self.validator = validator
        self.file_loader = file_loader

    def execute_task(self, file_path: Path) -> ValidationResult:
        """执行单个文件验证"""
        # 加载文件
        data = self.file_loader(file_path)

        # 执行验证
        result = self.validator.validate(data)

        # 添加文件路径到元数据
        if hasattr(result, 'metadata'):
            result.metadata["file_path"] = str(file_path)

        return result

    def get_task_id(self, file_path: Path) -> str:
        """获取任务ID（使用文件名）"""
        return file_path.name


def validate_datasets_parallel(
    validator: BaseValidator,
    datasets: Dict[str, Any],
    config: Optional[ExecutionConfig] = None,
    verbose: bool = True
) -> Dict[str, ValidationResult]:
    """并行验证多个数据集（便捷接口）

    Parameters
    ----------
    validator : BaseValidator
        验证器实例
    datasets : dict
        数据集字典 {name: data}
    config : ExecutionConfig, optional
        并行执行配置
    verbose : bool
        是否输出详细信息

    Returns
    -------
    dict
        验证结果字典 {name: ValidationResult}

    Example
    -------
    >>> from hydrosis.validation import PrecipitationValidator
    >>>
    >>> datasets = {
    ...     "2020": df_2020,
    ...     "2021": df_2021,
    ...     "2022": df_2022,
    ... }
    >>>
    >>> validator = PrecipitationValidator()
    >>> results = validate_datasets_parallel(validator, datasets)
    """
    parallel_validator = ParallelValidator(
        validator=validator,
        config=config,
        verbose=verbose
    )

    return parallel_validator.validate_multiple_datasets(datasets)


def validate_files_parallel(
    validator: BaseValidator,
    file_paths: List[Union[str, Path]],
    file_loader: callable,
    config: Optional[ExecutionConfig] = None,
    verbose: bool = True
) -> List[ValidationResult]:
    """并行验证多个文件（便捷接口）

    Parameters
    ----------
    validator : BaseValidator
        验证器实例
    file_paths : list
        文件路径列表
    file_loader : callable
        文件加载函数
    config : ExecutionConfig, optional
        并行执行配置
    verbose : bool
        是否输出详细信息

    Returns
    -------
    list
        验证结果列表

    Example
    -------
    >>> from hydrosis.validation import PrecipitationValidator
    >>> import pandas as pd
    >>>
    >>> files = ["data1.csv", "data2.csv", "data3.csv"]
    >>> validator = PrecipitationValidator()
    >>>
    >>> results = validate_files_parallel(
    ...     validator,
    ...     files,
    ...     pd.read_csv
    ... )
    """
    paths = [Path(p) if isinstance(p, str) else p for p in file_paths]

    batch_validator = BatchFileValidator(
        validator=validator,
        file_loader=file_loader,
        config=config,
        verbose=verbose
    )

    task_results = batch_validator.run(paths)
    return batch_validator.get_successful_results(task_results)
