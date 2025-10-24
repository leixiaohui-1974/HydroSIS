"""通用并行执行框架

提供统一的并行任务执行接口，支持多进程、多线程和异步执行。
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union
)

import multiprocessing as mp


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"      # 串行执行
    MULTIPROCESS = "multiprocess"  # 多进程
    MULTITHREADED = "multithreaded"  # 多线程


@dataclass
class TaskResult:
    """任务执行结果

    Attributes
    ----------
    task_id : Any
        任务标识符
    success : bool
        是否成功
    result : Any
        执行结果
    error : Optional[Exception]
        错误信息（如果失败）
    execution_time : float
        执行时间（秒）
    metadata : dict
        额外元数据
    """
    task_id: Any
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.success:
            return f"TaskResult(id={self.task_id}, success=True, time={self.execution_time:.2f}s)"
        else:
            return f"TaskResult(id={self.task_id}, success=False, error={type(self.error).__name__})"


@dataclass
class ExecutionConfig:
    """并行执行配置

    Attributes
    ----------
    mode : ExecutionMode
        执行模式
    max_workers : int, optional
        最大worker数量，None表示自动检测
    chunk_size : int
        任务分块大小
    timeout : float, optional
        单个任务超时时间（秒）
    retry_on_failure : bool
        失败时是否重试
    max_retries : int
        最大重试次数
    show_progress : bool
        是否显示进度
    raise_on_error : bool
        遇到错误是否抛出异常
    """
    mode: ExecutionMode = ExecutionMode.MULTIPROCESS
    max_workers: Optional[int] = None
    chunk_size: int = 1
    timeout: Optional[float] = None
    retry_on_failure: bool = False
    max_retries: int = 3
    show_progress: bool = True
    raise_on_error: bool = False

    def __post_init__(self):
        """自动设置worker数量"""
        if self.max_workers is None:
            if self.mode == ExecutionMode.MULTIPROCESS:
                self.max_workers = mp.cpu_count()
            elif self.mode == ExecutionMode.MULTITHREADED:
                self.max_workers = mp.cpu_count() * 2
            else:  # SEQUENTIAL
                self.max_workers = 1


T = TypeVar('T')  # 任务参数类型
R = TypeVar('R')  # 任务结果类型


class ParallelExecutor(Generic[T, R], ABC):
    """并行执行器抽象基类

    提供统一的并行任务执行接口，子类需要实现 execute_task 方法。

    Parameters
    ----------
    config : ExecutionConfig, optional
        执行配置
    verbose : bool, default=True
        是否输出详细信息

    Example
    -------
    >>> class MyExecutor(ParallelExecutor):
    ...     def execute_task(self, task):
    ...         # 执行具体任务
    ...         return process(task)
    ...
    >>> executor = MyExecutor()
    >>> results = executor.run(tasks)
    """

    def __init__(
        self,
        config: Optional[ExecutionConfig] = None,
        verbose: bool = True
    ):
        self.config = config or ExecutionConfig()
        self.verbose = verbose

        # 执行统计
        self._total_tasks = 0
        self._successful_tasks = 0
        self._failed_tasks = 0
        self._total_time = 0.0

    @abstractmethod
    def execute_task(self, task: T) -> R:
        """执行单个任务（需要子类实现）

        Parameters
        ----------
        task : T
            任务参数

        Returns
        -------
        R
            任务结果
        """
        pass

    def prepare_task(self, task: T) -> T:
        """预处理任务（可选重写）

        Parameters
        ----------
        task : T
            原始任务

        Returns
        -------
        T
            处理后的任务
        """
        return task

    def get_task_id(self, task: T) -> Any:
        """获取任务ID（可选重写）

        Parameters
        ----------
        task : T
            任务

        Returns
        -------
        Any
            任务ID
        """
        return id(task)

    def run(self, tasks: Iterable[T]) -> List[TaskResult]:
        """运行所有任务

        Parameters
        ----------
        tasks : Iterable[T]
            任务列表

        Returns
        -------
        List[TaskResult]
            执行结果列表
        """
        tasks_list = list(tasks)
        self._total_tasks = len(tasks_list)

        if self.verbose:
            print(f"  ⚙ 开始执行 {self._total_tasks} 个任务 (mode={self.config.mode.value}, workers={self.config.max_workers})...")

        start_time = time.time()

        # 预处理任务
        prepared_tasks = [self.prepare_task(t) for t in tasks_list]

        # 选择执行模式
        if self.config.mode == ExecutionMode.SEQUENTIAL or self._total_tasks == 1:
            results = self._run_sequential(prepared_tasks)
        elif self.config.mode == ExecutionMode.MULTIPROCESS:
            results = self._run_parallel(prepared_tasks, use_processes=True)
        else:  # MULTITHREADED
            results = self._run_parallel(prepared_tasks, use_processes=False)

        self._total_time = time.time() - start_time

        # 统计结果
        self._successful_tasks = sum(1 for r in results if r.success)
        self._failed_tasks = sum(1 for r in results if not r.success)

        if self.verbose:
            self._print_summary(results)

        return results

    def _run_sequential(self, tasks: List[T]) -> List[TaskResult]:
        """串行执行任务"""
        results = []

        for i, task in enumerate(tasks, 1):
            task_id = self.get_task_id(task)
            result = self._execute_with_retry(task, task_id)
            results.append(result)

            if self.config.show_progress:
                status = "✓" if result.success else "❌"
                print(f"    {status} 任务 {task_id} ({i}/{self._total_tasks})")

        return results

    def _run_parallel(
        self,
        tasks: List[T],
        use_processes: bool = True
    ) -> List[TaskResult]:
        """并行执行任务"""
        executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

        results = []

        with executor_class(max_workers=self.config.max_workers) as executor:
            # 创建任务映射
            future_to_task = {}
            for task in tasks:
                task_id = self.get_task_id(task)
                future = executor.submit(
                    self._execute_task_wrapper,
                    task,
                    task_id
                )
                future_to_task[future] = (task, task_id)

            # 收集结果
            completed = 0
            for future in as_completed(
                future_to_task,
                timeout=self.config.timeout
            ):
                task, task_id = future_to_task[future]

                try:
                    result = future.result()
                    results.append(result)
                    completed += 1

                    if self.config.show_progress:
                        status = "✓" if result.success else "❌"
                        print(f"    {status} 任务 {task_id} ({completed}/{self._total_tasks})")

                except Exception as e:
                    result = TaskResult(
                        task_id=task_id,
                        success=False,
                        error=e
                    )
                    results.append(result)
                    completed += 1

                    if self.config.show_progress:
                        print(f"    ❌ 任务 {task_id} 失败: {e}")

                    if self.config.raise_on_error:
                        raise

        return results

    def _execute_task_wrapper(self, task: T, task_id: Any) -> TaskResult:
        """任务执行包装器（用于并行执行）"""
        return self._execute_with_retry(task, task_id)

    def _execute_with_retry(self, task: T, task_id: Any) -> TaskResult:
        """执行任务并支持重试"""
        retries = 0
        max_attempts = self.config.max_retries + 1 if self.config.retry_on_failure else 1

        last_error = None
        start_time = time.time()

        while retries < max_attempts:
            try:
                result = self.execute_task(task)
                execution_time = time.time() - start_time

                return TaskResult(
                    task_id=task_id,
                    success=True,
                    result=result,
                    execution_time=execution_time,
                    metadata={"retries": retries}
                )

            except Exception as e:
                last_error = e
                retries += 1

                if retries < max_attempts:
                    if self.verbose:
                        print(f"    ⚠ 任务 {task_id} 失败，重试 {retries}/{self.config.max_retries}...")
                    time.sleep(0.5 * retries)  # 指数退避

        # 所有重试失败
        execution_time = time.time() - start_time
        return TaskResult(
            task_id=task_id,
            success=False,
            error=last_error,
            execution_time=execution_time,
            metadata={"retries": retries - 1}
        )

    def _print_summary(self, results: List[TaskResult]) -> None:
        """打印执行摘要"""
        avg_time = self._total_time / self._total_tasks if self._total_tasks > 0 else 0

        print(f"\n  执行摘要:")
        print(f"    总任务数: {self._total_tasks}")
        print(f"    成功: {self._successful_tasks}")
        print(f"    失败: {self._failed_tasks}")
        print(f"    总时间: {self._total_time:.2f}s")
        print(f"    平均时间: {avg_time:.2f}s/task")

        if self._successful_tasks > 0:
            successful_results = [r for r in results if r.success]
            avg_success_time = sum(r.execution_time for r in successful_results) / len(successful_results)
            print(f"    成功任务平均时间: {avg_success_time:.2f}s")

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self._total_tasks == 0:
            return 0.0
        return self._successful_tasks / self._total_tasks

    def get_failed_tasks(self, results: List[TaskResult]) -> List[TaskResult]:
        """获取失败的任务"""
        return [r for r in results if not r.success]

    def get_successful_results(self, results: List[TaskResult]) -> List[R]:
        """获取成功的结果"""
        return [r.result for r in results if r.success and r.result is not None]


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    config: Optional[ExecutionConfig] = None,
    verbose: bool = True
) -> List[R]:
    """并行映射函数（便捷接口）

    Parameters
    ----------
    func : Callable
        要执行的函数
    items : Iterable
        输入项列表
    config : ExecutionConfig, optional
        执行配置
    verbose : bool
        是否输出详细信息

    Returns
    -------
    List[R]
        结果列表

    Example
    -------
    >>> def square(x):
    ...     return x ** 2
    >>> results = parallel_map(square, range(10))
    """
    class FunctionExecutor(ParallelExecutor[T, R]):
        def execute_task(self, task: T) -> R:
            return func(task)

    executor = FunctionExecutor(config=config, verbose=verbose)
    task_results = executor.run(items)
    return executor.get_successful_results(task_results)
