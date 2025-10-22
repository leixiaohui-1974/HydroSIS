"""Generic stage orchestrator for the HydroSIS three-phase product workflow.

The workflow被拆分为：
1. Preprocessing（流域划分、雨量、断面提取等）
2. Modeling（流域水文 / 河道水动力模拟）
3. Postprocessing（成果汇总、可视化、报告导出）

本模块提供通用的任务管理器，便于将现有模块化函数串联执行。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence


CallableLike = Callable[..., Any]


@dataclass(slots=True)
class StageTask:
    """Single task definition within a stage."""

    name: str
    func: CallableLike
    args: Sequence[Any] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    store_output_as: Optional[str] = None
    use_context: bool = False


@dataclass(slots=True)
class StageTaskResult:
    """Result of running one task."""

    name: str
    status: str
    output: Any = None
    error: Optional[BaseException] = None


@dataclass(slots=True)
class StageResult:
    """Aggregated results for an entire stage."""

    name: str
    tasks: List[StageTaskResult]
    context: Dict[str, Any]


@dataclass(slots=True)
class PipelineConfig:
    """Convenience container bundling tasks for all three stages."""

    preprocessing: Sequence[StageTask] = ()
    modeling: Sequence[StageTask] = ()
    postprocessing: Sequence[StageTask] = ()


class HydroProductPipeline:
    """Coordinator executing the three product stages sequentially."""

    def __init__(
        self,
        *,
        preprocessing: Sequence[StageTask] = (),
        modeling: Sequence[StageTask] = (),
        postprocessing: Sequence[StageTask] = (),
        context: Optional[MutableMapping[str, Any]] = None,
    ) -> None:
        self.preprocessing_tasks = list(preprocessing)
        self.modeling_tasks = list(modeling)
        self.postprocessing_tasks = list(postprocessing)
        self._context: Dict[str, Any] = dict(context or {})

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "HydroProductPipeline":
        return cls(
            preprocessing=config.preprocessing,
            modeling=config.modeling,
            postprocessing=config.postprocessing,
        )

    @property
    def context(self) -> Dict[str, Any]:
        return self._context

    def run_stage(self, stage_name: str, tasks: Sequence[StageTask]) -> StageResult:
        results: List[StageTaskResult] = []
        for task in tasks:
            call_args = task.args
            if task.use_context:
                call_args = (self._context,) + tuple(call_args)
            try:
                output = task.func(*call_args, **task.kwargs)
                status = "success"
                err: Optional[BaseException] = None
                if task.store_output_as:
                    self._context[task.store_output_as] = output
            except BaseException as exc:  # noqa: W0703
                output = None
                status = "failed"
                err = exc
            results.append(StageTaskResult(name=task.name, status=status, output=output, error=err))
        return StageResult(name=stage_name, tasks=results, context=dict(self._context))

    def run_preprocessing(self) -> StageResult:
        return self.run_stage("preprocessing", self.preprocessing_tasks)

    def run_modeling(self) -> StageResult:
        return self.run_stage("modeling", self.modeling_tasks)

    def run_postprocessing(self) -> StageResult:
        return self.run_stage("postprocessing", self.postprocessing_tasks)

    def run(self) -> Dict[str, StageResult]:
        """Execute the three stages sequentially."""
        outputs: Dict[str, StageResult] = {}
        outputs["preprocessing"] = self.run_preprocessing()
        outputs["modeling"] = self.run_modeling()
        outputs["postprocessing"] = self.run_postprocessing()
        return outputs
