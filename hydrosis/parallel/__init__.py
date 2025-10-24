"""并行计算框架

提供统一的并行任务执行接口，支持多种并行模式和应用场景。
"""

from .executor import (
    ExecutionMode,
    ExecutionConfig,
    TaskResult,
    ParallelExecutor,
    parallel_map
)

from .validator import (
    ParallelValidator,
    BatchFileValidator,
    validate_datasets_parallel,
    validate_files_parallel
)

__all__ = [
    # 执行器核心
    "ExecutionMode",
    "ExecutionConfig",
    "TaskResult",
    "ParallelExecutor",
    "parallel_map",

    # 验证器
    "ParallelValidator",
    "BatchFileValidator",
    "validate_datasets_parallel",
    "validate_files_parallel",
]
