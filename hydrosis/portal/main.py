"""FastAPI application exposing the HydroSIS portal."""
from __future__ import annotations

import json
import os
from pathlib import Path
import json
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from hydrosis.config import ModelConfig
from hydrosis.workflow import run_workflow, WorkflowResult

from .analytics import summarise_workflow_result
from .llm import IntentParser
from .executor import RunEventBroker, RunExecutor, RunProgressEvent
from .schemas import (
    ConversationMessageSchema,
    ConversationResponse,
    InputSeriesSummary,
    MapLayerPayload,
    MapLayerResponse,
    InputsOverview,
    ProjectConfigPayload,
    ProjectInputsPayload,
    ProjectInputsResponse,
    ProjectList,
    ProjectOverview,
    ProjectSummary,
    RunList,
    RunRequest,
    RunResponse,
    ScenarioCreatePayload,
    ScenarioList,
    ScenarioUpdatePayload,
    ProjectPermissionList,
    ProjectPermissionPayload,
    QueueEntry,
    QueueSnapshot,
    UserList,
    UserPayload,
    UserResponse,
)
from .state import (
    ConversationMessage,
    InMemoryPortalState,
    PortalState,
    ProjectInputs,
    serialize_evaluation_outcome,
    serialize_model_score,
    serialize_scenario_run,
)



def _remove_none(obj):
    if isinstance(obj, dict):
        return {key: _remove_none(value) for key, value in obj.items() if value is not None}
    if isinstance(obj, list):
        return [_remove_none(item) for item in obj]
    return obj

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _series_summary(data: Mapping[str, Iterable[float]] | None) -> Optional[InputSeriesSummary]:
    if not data:
        return None

    lengths: List[int] = []
    all_values: List[float] = []
    for series in data.values():
        numeric_series = [float(value) for value in series]
        lengths.append(len(numeric_series))
        all_values.extend(numeric_series)

    if not lengths:
        return None

    min_value = min(all_values) if all_values else None
    max_value = max(all_values) if all_values else None
    mean_length = sum(lengths) / len(lengths)

    return InputSeriesSummary(
        series_count=len(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        mean_length=mean_length,
        min_value=min_value,
        max_value=max_value,
    )


def _build_inputs_overview(dataset: Optional[ProjectInputs]) -> Optional[InputsOverview]:
    if dataset is None:
        return None

    forcing_summary = _series_summary(dataset.forcing)
    observations_summary = (
        _series_summary(dataset.observations) if dataset.observations else None
    )

    return InputsOverview(
        forcing=forcing_summary,
        observations=observations_summary,
        updated_at=dataset.updated_at.isoformat(),
    )

def create_app(
    state: PortalState | None = None,
    *,
    database_url: str | None = None,
    config_path: str | Path | None = None,
) -> FastAPI:
    app = FastAPI(title="HydroSIS Portal", version="0.1.0")

    portal_state = _initialise_state(state, database_url=database_url, config_path=config_path)
    intent_parser = IntentParser()
    event_broker = RunEventBroker()
    run_executor = RunExecutor(
        portal_state,
        event_broker,
        workflow_runner=_execute_workflow,
    )

    role_requirements: Dict[str, set[str]] = {
        "manage_project": {"admin", "modeler"},
        "manage_inputs": {"admin", "modeler"},
        "manage_scenarios": {"admin", "modeler"},
        "trigger_run": {"admin", "operator", "modeler"},
        "manage_map": {"admin", "gis"},
        "assign_permission": {"admin"},
        "view_queue": {"admin", "operator", "viewer"},
    }

    def _extract_field(
        raw_value: object | None,
        payload: object,
        field: str,
    ) -> tuple[Optional[str], object]:
        extracted: Optional[str] = None
        candidate: object | None = None
        if isinstance(raw_value, Mapping):
            candidate = raw_value.get(field)
        elif isinstance(raw_value, str):
            candidate = raw_value
        if candidate is None and isinstance(payload, Mapping):
            candidate = payload.get(field)
            if candidate is not None and isinstance(payload, dict):
                payload = dict(payload)
                payload.pop(field, None)
        if candidate is not None and not isinstance(candidate, str):
            candidate = str(candidate)
        if isinstance(candidate, str):
            extracted = candidate
        return extracted, payload

    def _require_role(
        action: str,
        user_id: Optional[str],
        *,
        project_id: Optional[str] = None,
    ) -> None:
        required_roles = role_requirements.get(action)
        if not required_roles:
            return
        if not portal_state.list_users():
            return
        if not user_id:
            raise HTTPException(
                status_code=403,
                detail="user_id is required when role-based permissions are enabled",
            )
        try:
            account = portal_state.get_user(user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        roles = set(account.roles)
        if project_id:
            project_role = account.project_roles.get(project_id)
            if project_role:
                roles.add(project_role)
        if roles.intersection(required_roles):
            return
        raise HTTPException(
            status_code=403,
            detail=f"User '{user_id}' lacks required role for action '{action}'",
        )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Portal assets not found")
        return index_path.read_text(encoding="utf-8")

    @app.post("/conversations/{conversation_id}/messages", response_model=ConversationResponse)
    def post_message(
        conversation_id: str, payload: ConversationMessageSchema | dict
    ) -> ConversationResponse:
        if isinstance(payload, dict):
            payload = ConversationMessageSchema.from_dict(payload)
        conversation = portal_state.get_conversation(conversation_id)
        user_message = ConversationMessage(role=payload.role, content=payload.content)
        conversation.add(user_message)

        intent = intent_parser.parse(payload.content)

        reply = _render_assistant_reply(intent, conversation_id)
        assistant_message = ConversationMessage(role="assistant", content=reply, intent=intent)
        conversation.add(assistant_message)

        return ConversationResponse(
            reply=reply,
            intent=intent,
            messages=[
                ConversationMessageSchema(
                    role=msg.role, content=msg.content, intent=msg.intent
                )
                for msg in conversation.messages
            ],
        )

    @app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
    def get_conversation(conversation_id: str) -> ConversationResponse:
        conversation = portal_state.get_conversation(conversation_id)
        intent = conversation.messages[-1].intent if conversation.messages else {}
        return ConversationResponse(
            reply=conversation.messages[-1].content if conversation.messages else "",
            intent=intent or {},
            messages=[
                ConversationMessageSchema(
                    role=msg.role, content=msg.content, intent=msg.intent
                )
                for msg in conversation.messages
            ],
        )

    @app.get("/projects", response_model=ProjectList)
    def list_projects() -> ProjectList:
        projects = [
            ProjectSummary(**project.to_summary())
            for project in portal_state.list_projects()
        ]
        return ProjectList(projects=projects)

    @app.get("/users", response_model=UserList)
    def list_users_endpoint() -> UserList:
        users = [
            UserResponse.from_state(account.to_dict())
            for account in portal_state.list_users()
        ]
        return UserList(users=users)

    @app.get("/users/{user_id}", response_model=UserResponse)
    def get_user_endpoint(user_id: str) -> UserResponse:
        try:
            account = portal_state.get_user(user_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return UserResponse.from_state(account.to_dict())

    @app.post("/users", response_model=UserResponse)
    def create_user_endpoint(
        payload: UserPayload | dict, actor_id: str | None = None
    ) -> UserResponse:
        actor_id, payload = _extract_field(actor_id, payload, "actor_id")
        _require_role("assign_permission", actor_id)
        if isinstance(payload, dict):
            payload = UserPayload.from_dict(payload)
        account = portal_state.upsert_user(payload.id, payload.name, payload.roles)
        return UserResponse.from_state(account.to_dict())

    @app.put("/users/{user_id}", response_model=UserResponse)
    def update_user_endpoint(
        user_id: str, payload: UserPayload | dict, actor_id: str | None = None
    ) -> UserResponse:
        actor_id, payload = _extract_field(actor_id, payload, "actor_id")
        _require_role("assign_permission", actor_id)
        if isinstance(payload, dict):
            payload = UserPayload.from_dict({"id": user_id, **payload})
        account = portal_state.upsert_user(user_id, payload.name, payload.roles)
        return UserResponse.from_state(account.to_dict())

    @app.post(
        "/projects/{project_id}/permissions",
        response_model=UserResponse,
    )
    def assign_project_permission(
        project_id: str,
        payload: ProjectPermissionPayload | dict,
        actor_id: str | None = None,
    ) -> UserResponse:
        actor_id, payload = _extract_field(actor_id, payload, "actor_id")
        _require_role("assign_permission", actor_id, project_id=project_id)
        if isinstance(payload, dict):
            payload = ProjectPermissionPayload.from_dict(payload)
        try:
            portal_state.get_project(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        account = portal_state.set_project_role(
            payload.user_id, project_id, payload.role
        )
        return UserResponse.from_state(account.to_dict())

    @app.get(
        "/projects/{project_id}/permissions",
        response_model=ProjectPermissionList,
    )
    def list_project_permissions(project_id: str) -> ProjectPermissionList:
        try:
            portal_state.get_project(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        permissions = portal_state.list_project_roles(project_id)
        return ProjectPermissionList(project_id=project_id, permissions=permissions)

    @app.get("/projects/{project_id}/overview", response_model=ProjectOverview)
    def project_overview(project_id: str) -> ProjectOverview:
        try:
            project = portal_state.get_project(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        try:
            dataset = portal_state.get_inputs(project_id)
        except KeyError:
            dataset = None

        inputs_overview = _build_inputs_overview(dataset)

        runs = portal_state.list_runs(project_id)
        latest_run = RunResponse(**runs[0].to_dict()) if runs else None
        latest_summary = (
            summarise_workflow_result(runs[0].result)
            if runs and runs[0].result is not None
            else None
        )
        try:
            map_layers = portal_state.get_map_layers(project_id)
        except KeyError:
            map_layers = None
        map_layers_updated_at = (
            map_layers.updated_at.isoformat() if map_layers is not None else None
        )

        return ProjectOverview(
            id=project.id,
            name=project.name,
            scenarios=[scenario.id for scenario in project.model_config.scenarios],
            scenario_count=len(project.model_config.scenarios),
            total_runs=len(runs),
            inputs=inputs_overview,
            latest_run=latest_run,
            latest_summary=latest_summary,
            map_layers_updated_at=map_layers_updated_at,
        )

    @app.post(
        "/projects/{project_id}/inputs",
        response_model=ProjectInputsResponse,
    )
    def upsert_inputs(
        project_id: str, payload: ProjectInputsPayload | dict, user_id: str | None = None
    ) -> ProjectInputsResponse:
        user_id, payload = _extract_field(user_id, payload, "user_id")
        _require_role("manage_inputs", user_id, project_id=project_id)
        if isinstance(payload, dict):
            payload = ProjectInputsPayload.from_dict(payload)
        try:
            existing = portal_state.get_inputs(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        observations = payload.observations
        if observations is None and existing is not None:
            observations = existing.observations
        dataset = portal_state.set_inputs(
            project_id,
            payload.forcing,
            observations,
        )
        return ProjectInputsResponse(
            forcing=dataset.forcing,
            observations=dataset.observations,
            updated_at=dataset.updated_at.isoformat(),
        )

    @app.get("/projects/{project_id}/inputs", response_model=ProjectInputsResponse)
    def get_inputs(project_id: str) -> ProjectInputsResponse:
        try:
            dataset = portal_state.get_inputs(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if dataset is None:
            raise HTTPException(status_code=404, detail="Inputs not configured")
        return ProjectInputsResponse(
            forcing=dataset.forcing,
            observations=dataset.observations,
            updated_at=dataset.updated_at.isoformat(),
        )

    @app.post("/projects/{project_id}/config", response_model=ProjectSummary)
    def upsert_project(
        project_id: str,
        payload: ProjectConfigPayload | dict,
        user_id: str | None = None,
    ) -> ProjectSummary:
        user_id, payload = _extract_field(user_id, payload, "user_id")
        _require_role("manage_project", user_id, project_id=project_id)
        if isinstance(payload, dict):
            payload = ProjectConfigPayload.from_dict(payload)
        model_config = ModelConfig.from_dict(_remove_none(payload.model))
        project = portal_state.upsert_project(project_id, payload.name, model_config)
        return ProjectSummary(**project.to_summary())

    @app.get("/projects/{project_id}", response_model=ProjectSummary)
    def get_project(project_id: str) -> ProjectSummary:
        try:
            project = portal_state.get_project(project_id)
        except KeyError as exc:  # pragma: no cover - defensive
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ProjectSummary(**project.to_summary())

    @app.get("/projects/{project_id}/scenarios", response_model=ScenarioList)
    def list_scenarios(project_id: str) -> ScenarioList:
        scenarios = portal_state.list_scenarios(project_id)
        return ScenarioList(
            scenarios=[
                ScenarioCreatePayload(
                    id=scenario.id,
                    description=scenario.description,
                    modifications=scenario.modifications,
                )
                for scenario in scenarios
            ]
        )

    @app.post("/projects/{project_id}/scenarios", response_model=ScenarioCreatePayload)
    def create_scenario(
        project_id: str,
        payload: ScenarioCreatePayload | dict,
        user_id: str | None = None,
    ) -> ScenarioCreatePayload:
        user_id, payload = _extract_field(user_id, payload, "user_id")
        _require_role("manage_scenarios", user_id, project_id=project_id)
        if isinstance(payload, dict):
            payload = ScenarioCreatePayload.from_dict(payload)
        try:
            scenario = portal_state.add_scenario(
                project_id,
                scenario_id=payload.id,
                description=payload.description,
                modifications=payload.modifications,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return ScenarioCreatePayload(
            id=scenario.id,
            description=scenario.description,
            modifications=scenario.modifications,
        )

    @app.put(
        "/projects/{project_id}/scenarios/{scenario_id}",
        response_model=ScenarioCreatePayload,
    )
    def update_scenario(
        project_id: str,
        scenario_id: str,
        payload: ScenarioUpdatePayload | dict,
        user_id: str | None = None,
    ) -> ScenarioCreatePayload:
        user_id, payload = _extract_field(user_id, payload, "user_id")
        _require_role("manage_scenarios", user_id, project_id=project_id)
        if isinstance(payload, dict):
            payload = ScenarioUpdatePayload.from_dict(payload)
        try:
            scenario = portal_state.update_scenario(
                project_id,
                scenario_id,
                description=payload.description,
                modifications=payload.modifications,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return ScenarioCreatePayload(
            id=scenario.id,
            description=scenario.description,
            modifications=scenario.modifications,
        )

    @app.delete("/projects/{project_id}/scenarios/{scenario_id}")
    def delete_scenario(
        project_id: str, scenario_id: str, user_id: str | None = None
    ) -> Response:
        user_id, _ = _extract_field(user_id, {}, "user_id")
        _require_role("manage_scenarios", user_id, project_id=project_id)
        try:
            portal_state.remove_scenario(project_id, scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return Response(status_code=204, data=None)

    @app.post("/projects/{project_id}/map", response_model=MapLayerResponse)
    def update_map_layers(
        project_id: str, payload: MapLayerPayload | dict, user_id: str | None = None
    ) -> MapLayerResponse:
        user_id, payload = _extract_field(user_id, payload, "user_id")
        _require_role("manage_map", user_id, project_id=project_id)
        if isinstance(payload, dict):
            payload = MapLayerPayload.from_dict(payload)
        try:
            dataset = portal_state.set_map_layers(project_id, payload.layers)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return MapLayerResponse(
            project_id=dataset.project_id,
            layers=dataset.layers,
            updated_at=dataset.updated_at.isoformat(),
        )

    @app.get("/projects/{project_id}/map", response_model=MapLayerResponse)
    def get_map_layers_endpoint(project_id: str) -> MapLayerResponse:
        try:
            dataset = portal_state.get_map_layers(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if dataset is None:
            raise HTTPException(status_code=404, detail="Map layers not configured")
        return MapLayerResponse(
            project_id=dataset.project_id,
            layers=dataset.layers,
            updated_at=dataset.updated_at.isoformat(),
        )

    @app.post("/projects/{project_id}/runs", response_model=RunResponse)
    def create_run(
        project_id: str, payload: RunRequest | dict, user_id: str | None = None
    ) -> RunResponse:
        user_id, payload = _extract_field(user_id, payload, "user_id")
        _require_role("trigger_run", user_id, project_id=project_id)
        if isinstance(payload, dict):
            payload = RunRequest.from_dict(payload)
        try:
            project = portal_state.get_project(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        scenario_ids = list(
            payload.scenario_ids
            or [scenario.id for scenario in project.model_config.scenarios]
        )
        run_record = portal_state.create_run(project_id, scenario_ids)

        run_executor.submit(
            run_record.id,
            project_id,
            project.model_config,
            payload,
            scenario_ids,
        )

        return RunResponse(**portal_state.get_run(run_record.id).to_dict())

    @app.get("/runs/{run_id}/stream")
    def stream_run(run_id: str) -> StreamingResponse:
        try:
            run = portal_state.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        queue = event_broker.subscribe(run_id)

        latest_event = event_broker.latest(run_id)
        if latest_event is not None:
            queue.put(latest_event)
        else:
            queue.put(
                RunProgressEvent(
                    run_id=run_id,
                    stage="snapshot",
                    status=run.status,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    percent=0,
                    message="已连接到运行进度流",
                ).to_dict()
            )

        def event_generator():
            try:
                while True:
                    event = queue.get()
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in {"completed", "failed"}:
                        break
            finally:
                event_broker.unsubscribe(run_id, queue)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    @app.get("/projects/{project_id}/runs", response_model=RunList)
    def list_project_runs(project_id: str) -> RunList:
        try:
            portal_state.get_project(project_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        runs = [RunResponse(**run.to_dict()) for run in portal_state.list_runs(project_id)]
        return RunList(runs=runs)

    @app.get("/runs", response_model=RunList)
    def list_runs() -> RunList:
        runs = [RunResponse(**run.to_dict()) for run in portal_state.list_runs()]
        return RunList(runs=runs)

    @app.get("/runs/queue", response_model=QueueSnapshot)
    def queue_status(user_id: str | None = None) -> QueueSnapshot:
        _require_role("view_queue", user_id)
        entries: List[QueueEntry] = []
        for entry in run_executor.queue_snapshot():
            scenario_ids = entry.get("scenario_ids") or []
            if not isinstance(scenario_ids, Sequence) or isinstance(
                scenario_ids, (str, bytes)
            ):
                scenario_ids = [str(scenario_ids)] if scenario_ids else []
            entries.append(
                QueueEntry(
                    run_id=str(entry.get("run_id")),
                    project_id=str(entry.get("project_id")),
                    status=str(entry.get("status")),
                    scenario_ids=list(scenario_ids),
                    submitted_at=str(entry.get("submitted_at")),
                    started_at=entry.get("started_at"),
                    finished_at=entry.get("finished_at"),
                )
            )
        return QueueSnapshot(entries=entries)

    @app.get("/runs/{run_id}", response_model=RunResponse)
    def get_run(run_id: str) -> RunResponse:
        try:
            run = portal_state.get_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return RunResponse(**run.to_dict())

    @app.get("/runs/{run_id}/report")
    def fetch_report(run_id: str) -> Dict[str, object]:
        run = portal_state.get_run(run_id)
        result = run.result
        if result is None or result.report_path is None:
            raise HTTPException(status_code=404, detail="Report not available")
        report_path = Path(result.report_path)
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report file missing")
        return {"path": str(report_path), "content": report_path.read_text(encoding="utf-8")}

    @app.get("/runs/{run_id}/figures")
    def list_figures(run_id: str) -> Dict[str, Iterable[str]]:
        run = portal_state.get_run(run_id)
        result = run.result
        if result is None or result.report_path is None:
            return {"figures": []}
        report_path = Path(result.report_path)
        figures_dir = report_path.parent.parent / "figures"
        if not figures_dir.exists():
            return {"figures": []}
        files = [str(path) for path in figures_dir.glob("*.png")]
        return {"figures": files}

    @app.get("/runs/{run_id}/summary")
    def summarise_run(run_id: str) -> Dict[str, object]:
        run = portal_state.get_run(run_id)
        if run.result is None:
            raise HTTPException(status_code=404, detail="Run result not available")
        return summarise_workflow_result(run.result)

    @app.get("/runs/{run_id}/timeseries")
    def run_timeseries(run_id: str) -> Dict[str, object]:
        run = portal_state.get_run(run_id)
        if run.result is None:
            raise HTTPException(status_code=404, detail="Run result not available")
        result = run.result
        return {
            "baseline": serialize_scenario_run(result.baseline),
            "scenarios": {
                scenario_id: serialize_scenario_run(scenario_result)
                for scenario_id, scenario_result in result.scenarios.items()
            },
        }

    @app.get("/runs/{run_id}/evaluation")
    def run_evaluation(run_id: str) -> Dict[str, object]:
        run = portal_state.get_run(run_id)
        if run.result is None:
            raise HTTPException(status_code=404, detail="Run result not available")
        result = run.result
        return {
            "overall_scores": [
                serialize_model_score(score) for score in result.overall_scores or []
            ],
            "evaluation_outcomes": [
                serialize_evaluation_outcome(outcome)
                for outcome in result.evaluation_outcomes
            ],
        }
    return app


def _initialise_state(
    supplied_state: PortalState | None,
    *,
    database_url: str | None = None,
    config_path: str | Path | None = None,
) -> PortalState:
    if supplied_state is not None:
        return supplied_state

    resolved_url = _resolve_database_url(database_url=database_url, config_path=config_path)
    if resolved_url:
        try:
            from .storage import create_sqlalchemy_state
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "SQLAlchemy backend requested but SQLAlchemy is not installed"
            ) from exc

        return create_sqlalchemy_state(resolved_url)

    return InMemoryPortalState()


def _resolve_database_url(
    *,
    database_url: str | None = None,
    config_path: str | Path | None = None,
) -> Optional[str]:
    if database_url:
        return database_url

    env_url = os.environ.get("HYDROSIS_PORTAL_DB_URL")
    if env_url:
        return env_url

    config_location = config_path or os.environ.get("HYDROSIS_PORTAL_CONFIG")
    if not config_location:
        return None

    config_file = Path(config_location)
    if not config_file.exists():
        raise FileNotFoundError(f"Portal configuration file '{config_file}' not found")

    try:
        config_data = json.loads(config_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Portal configuration file '{config_file}' must be valid JSON"
        ) from exc

    url = config_data.get("database_url")
    if url and not isinstance(url, str):
        raise ValueError(
            f"database_url in '{config_file}' must be a string if provided"
        )
    return url


def _render_assistant_reply(intent: Dict[str, object], conversation_id: str) -> str:
    action = intent.get("action", "general_chat")
    if action == "run_scenarios":
        scenarios = intent.get("parameters", {}).get("scenario_ids") or ["all configured scenarios"]
        return f"Conversation {conversation_id}: 已识别到运行情景请求，目标情景：{', '.join(scenarios)}。"
    if action == "create_scenario":
        name = intent.get("parameters", {}).get("name", "新情景")
        return f"Conversation {conversation_id}: 可以为你创建情景 {name}，请补充参数调整信息。"
    if action == "list_scenarios":
        return "我可以列出当前项目中的情景，请调用场景列表接口。"
    if action == "summarise_results":
        return "可根据最新运行结果生成报告摘要。"
    return "已收到你的请求，如需运行模型请说明需要的情景或分析目标。"


def _execute_workflow(
    project_id: str,
    config: ModelConfig,
    payload: RunRequest,
    scenario_ids: Iterable[str],
    state: PortalState,
    progress_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> WorkflowResult:
    config_copy = ModelConfig.from_dict(_remove_none(config.to_dict()))

    dataset = None
    try:
        dataset = state.get_inputs(project_id)
    except KeyError:
        pass

    forcing_payload = payload.forcing
    observations_payload = payload.observations

    if forcing_payload is not None:
        existing_observations = dataset.observations if dataset else None
        merged_observations = (
            observations_payload
            if observations_payload is not None
            else existing_observations
        )
        dataset = state.set_inputs(project_id, forcing_payload, merged_observations)
    else:
        if dataset is None:
            raise HTTPException(
                status_code=400,
                detail=f"Forcing inputs are required. Upload them via /projects/{project_id}/inputs first.",
            )
        if observations_payload is not None:
            dataset = state.set_inputs(project_id, dataset.forcing, observations_payload)

    if dataset is None:
        raise HTTPException(
            status_code=400,
            detail="Project inputs are not configured",
        )

    forcing = _to_float_series(dataset.forcing)
    observations = _to_optional_float_series(dataset.observations)

    if payload.persist_outputs:
        config_copy.io.results_directory.mkdir(parents=True, exist_ok=True)
        if config_copy.io.figures_directory:
            config_copy.io.figures_directory.mkdir(parents=True, exist_ok=True)
        if config_copy.io.reports_directory:
            config_copy.io.reports_directory.mkdir(parents=True, exist_ok=True)

    return run_workflow(
        config_copy,
        forcing,
        observations=observations,
        scenario_ids=list(scenario_ids),
        persist_outputs=payload.persist_outputs,
        generate_report=payload.generate_report,
        progress_callback=progress_callback,
    )


def _to_float_series(data: Mapping[str, Iterable[float]]) -> Dict[str, List[float]]:
    return {key: [float(value) for value in values] for key, values in data.items()}


def _to_optional_float_series(
    data: Optional[Mapping[str, Iterable[float]]]
) -> Optional[Dict[str, List[float]]]:
    if data is None:
        return None
    return _to_float_series(data)


__all__ = ["create_app"]
