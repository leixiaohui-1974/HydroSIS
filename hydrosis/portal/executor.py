"""Background execution utilities for workflow runs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from queue import Queue
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from hydrosis.config import ModelConfig
from hydrosis.workflow import WorkflowResult

from .analytics import summarise_workflow_result
from .schemas import RunRequest
from .state import PortalState

ProgressCallback = Callable[[str, Mapping[str, object]], None]
WorkflowRunner = Callable[
    [
        str,
        ModelConfig,
        RunRequest,
        Iterable[str],
        PortalState,
        Optional[ProgressCallback],
    ],
    WorkflowResult,
]


@dataclass
class ProgressTracker:
    """Track high-level progress milestones for a run."""

    scenario_total: int
    baseline_completed: bool = False
    scenarios_completed: int = 0
    evaluation_enabled: bool = False
    evaluation_completed: bool = False

    @property
    def total_segments(self) -> int:
        evaluation_segments = 1 if self.evaluation_enabled else 0
        return max(1, 1 + self.scenario_total + evaluation_segments)

    @property
    def completed_segments(self) -> int:
        completed = 1 if self.baseline_completed else 0
        completed += self.scenarios_completed
        if self.evaluation_enabled and self.evaluation_completed:
            completed += 1
        return min(completed, self.total_segments)

    def percent_complete(self) -> int:
        if self.total_segments == 0:
            return 100
        return int((self.completed_segments / self.total_segments) * 100)

    def mark_baseline_complete(self) -> None:
        self.baseline_completed = True

    def mark_scenario_complete(self) -> None:
        self.scenarios_completed = min(self.scenarios_completed + 1, self.scenario_total)

    def enable_evaluation(self) -> None:
        self.evaluation_enabled = True

    def mark_evaluation_complete(self) -> None:
        if self.evaluation_enabled:
            self.evaluation_completed = True


@dataclass
class RunProgressEvent:
    """Structured payload emitted to subscribers tracking a workflow run."""

    run_id: str
    stage: str
    status: str
    timestamp: str
    percent: int
    message: Optional[str] = None
    phase: Optional[str] = None
    progress: Mapping[str, int] = field(default_factory=dict)
    details: Mapping[str, object] = field(default_factory=dict)
    summary: Optional[Mapping[str, object]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "run_id": self.run_id,
            "stage": self.stage,
            "status": self.status,
            "timestamp": self.timestamp,
            "percent": self.percent,
        }
        if self.message is not None:
            payload["message"] = self.message
        if self.phase is not None:
            payload["phase"] = self.phase
        if self.progress:
            payload["progress"] = dict(self.progress)
        if self.details:
            payload["details"] = dict(self.details)
        if self.summary is not None:
            payload["summary"] = dict(self.summary)
        if self.error is not None:
            payload["error"] = self.error
        return payload


class RunEventBroker:
    """Multiplexes progress events to multiple subscribers using thread-safe queues."""

    def __init__(self) -> None:
        self._subscribers: Dict[str, List[Queue]] = {}
        self._latest: Dict[str, Dict[str, object]] = {}
        self._lock = threading.Lock()

    def subscribe(self, run_id: str) -> Queue:
        queue: Queue = Queue()
        with self._lock:
            self._subscribers.setdefault(run_id, []).append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: Queue) -> None:
        with self._lock:
            subscribers = self._subscribers.get(run_id)
            if not subscribers:
                return
            try:
                subscribers.remove(queue)
            except ValueError:
                return
            if not subscribers:
                self._subscribers.pop(run_id, None)

    def latest(self, run_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            payload = self._latest.get(run_id)
            return dict(payload) if payload is not None else None

    def publish(self, run_id: str, event: RunProgressEvent) -> None:
        payload = event.to_dict()
        with self._lock:
            self._latest[run_id] = dict(payload)
            subscribers = list(self._subscribers.get(run_id, []))
        for queue in subscribers:
            queue.put(dict(payload))


class RunExecutor:
    """Submit workflow runs to a background worker and emit progress events."""

    def __init__(
        self,
        state: PortalState,
        broker: RunEventBroker,
        workflow_runner: WorkflowRunner,
        max_workers: int = 2,
    ) -> None:
        self._state = state
        self._broker = broker
        self._workflow_runner = workflow_runner
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _build_message(self, stage: str, payload: Mapping[str, object]) -> str:
        phase = payload.get("phase")
        if stage == "baseline":
            return "基准情景模拟完成" if phase == "complete" else "开始基准情景模拟"
        if stage == "scenario":
            scenario_id = payload.get("scenario_id")
            index = payload.get("index")
            total = payload.get("total")
            label = f"情景 {scenario_id}" if scenario_id else "情景"
            if phase == "complete":
                return f"{label} 完成 ({index}/{total})"
            return f"开始 {label} ({index}/{total})"
        if stage == "evaluation":
            return "评估阶段完成" if phase == "complete" else "开始评估阶段"
        if stage == "evaluation_plan":
            plan_id = payload.get("plan_id")
            label = f"评估方案 {plan_id}" if plan_id else "评估方案"
            return f"{label} {payload.get('phase')} ({payload.get('index')}/{payload.get('total')})"
        if stage == "persistence":
            return "输出已写入结果目录" if phase == "complete" else "正在写入输出文件"
        if stage == "report":
            return "报告生成完成" if phase == "complete" else "正在生成评估报告"
        if stage == "workflow":
            return "工作流执行完成" if phase == "complete" else "准备执行工作流"
        return f"{stage} 阶段更新"

    def _emit_progress(
        self,
        run_id: str,
        stage: str,
        tracker: ProgressTracker,
        payload: Mapping[str, object],
    ) -> None:
        phase = payload.get("phase")
        if stage == "scenario" and "total" in payload:
            tracker.scenario_total = max(int(payload.get("total", tracker.scenario_total)), 0)
        if stage == "evaluation" and phase == "start":
            tracker.enable_evaluation()
        if stage == "baseline" and phase == "complete":
            tracker.mark_baseline_complete()
        if stage == "scenario" and phase == "complete":
            tracker.mark_scenario_complete()
        if stage == "evaluation" and phase == "complete":
            tracker.mark_evaluation_complete()

        message = str(payload.get("message")) if "message" in payload else self._build_message(stage, payload)
        event = RunProgressEvent(
            run_id=run_id,
            stage=stage,
            status="running",
            phase=phase if isinstance(phase, str) else None,
            message=message,
            timestamp=datetime.now(timezone.utc).isoformat(),
            percent=tracker.percent_complete(),
            progress={
                "completed": tracker.completed_segments,
                "total": tracker.total_segments,
                "scenarios_completed": tracker.scenarios_completed,
                "scenario_total": tracker.scenario_total,
            },
            details={key: value for key, value in payload.items() if key not in {"message"}},
        )
        self._broker.publish(run_id, event)

    def submit(
        self,
        run_id: str,
        project_id: str,
        config: ModelConfig,
        payload: RunRequest,
        scenario_ids: Sequence[str],
    ) -> None:
        tracker = ProgressTracker(scenario_total=len(list(scenario_ids)))
        queued_event = RunProgressEvent(
            run_id=run_id,
            stage="queue",
            status="queued",
            timestamp=datetime.now(timezone.utc).isoformat(),
            percent=0,
            message="运行已加入队列等待执行",
            progress={
                "completed": 0,
                "total": tracker.total_segments,
                "scenarios_completed": 0,
                "scenario_total": tracker.scenario_total,
            },
        )
        self._broker.publish(run_id, queued_event)

        def progress_callback(stage: str, progress_payload: Mapping[str, object]) -> None:
            self._emit_progress(run_id, stage, tracker, progress_payload)

        def task() -> None:
            self._state.start_run(run_id)
            start_event = RunProgressEvent(
                run_id=run_id,
                stage="workflow",
                status="running",
                timestamp=datetime.now(timezone.utc).isoformat(),
                percent=tracker.percent_complete(),
                message="工作流执行已开始",
                progress={
                    "completed": tracker.completed_segments,
                    "total": tracker.total_segments,
                    "scenarios_completed": tracker.scenarios_completed,
                    "scenario_total": tracker.scenario_total,
                },
            )
            self._broker.publish(run_id, start_event)
            try:
                result = self._workflow_runner(
                    project_id,
                    config,
                    payload,
                    scenario_ids,
                    self._state,
                    progress_callback,
                )
                tracker.mark_baseline_complete()
                tracker.scenarios_completed = tracker.scenario_total
                tracker.mark_evaluation_complete()
                summary = summarise_workflow_result(result)
                self._state.complete_run(run_id, result)
                completion_event = RunProgressEvent(
                    run_id=run_id,
                    stage="workflow",
                    status="completed",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    percent=100,
                    message="运行完成，结果已保存",
                    progress={
                        "completed": tracker.total_segments,
                        "total": tracker.total_segments,
                        "scenarios_completed": tracker.scenario_total,
                        "scenario_total": tracker.scenario_total,
                    },
                    summary=summary if summary is not None else None,
                )
                self._broker.publish(run_id, completion_event)
            except Exception as exc:  # pragma: no cover - runtime failures are reported to clients
                self._state.fail_run(run_id, str(exc))
                failure_event = RunProgressEvent(
                    run_id=run_id,
                    stage="workflow",
                    status="failed",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    percent=tracker.percent_complete(),
                    message="运行失败，请查看错误信息",
                    progress={
                        "completed": tracker.completed_segments,
                        "total": tracker.total_segments,
                        "scenarios_completed": tracker.scenarios_completed,
                        "scenario_total": tracker.scenario_total,
                    },
                    error=str(exc),
                )
                self._broker.publish(run_id, failure_event)

        self._executor.submit(task)


__all__ = ["RunExecutor", "RunEventBroker", "RunProgressEvent", "ProgressTracker"]
