"""In-memory state container backing the HydroSIS portal API."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence
import uuid

from hydrosis.config import ModelConfig, ScenarioConfig
from hydrosis.workflow import WorkflowResult, ScenarioRun, EvaluationOutcome


@dataclass
class ConversationMessage:
    """Single utterance exchanged between the user and the assistant."""

    role: str
    content: str
    intent: Optional[MutableMapping[str, object]] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Conversation:
    """Minimal conversation log with linked intents."""

    id: str
    messages: List[ConversationMessage] = field(default_factory=list)

    def add(self, message: ConversationMessage) -> None:
        self.messages.append(message)


@dataclass
class Project:
    """Project configuration stored within the portal."""

    id: str
    name: Optional[str]
    model_config: ModelConfig

    def to_summary(self) -> Dict[str, object]:
        config_dict = self.model_config.to_dict()
        return {
            "id": self.id,
            "name": self.name,
            "scenarios": [scenario["id"] for scenario in config_dict.get("scenarios", [])],
            "evaluation": config_dict.get("evaluation"),
        }


@dataclass
class ProjectInputs:
    """Persisted hydrological inputs associated with a project."""

    project_id: str
    forcing: Dict[str, List[float]]
    observations: Optional[Dict[str, List[float]]]
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, object]:
        return {
            "project_id": self.project_id,
            "forcing": {key: list(values) for key, values in self.forcing.items()},
            "observations": {
                key: list(values) for key, values in (self.observations or {}).items()
            }
            or None,
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class RunRecord:
    """Execution record of a workflow invocation."""

    id: str
    project_id: str
    scenario_ids: Sequence[str]
    created_at: datetime
    status: str
    result: Optional[WorkflowResult] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "scenario_ids": list(self.scenario_ids),
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "error": self.error,
            "result": serialize_workflow_result(self.result) if self.result else None,
        }


class PortalState:
    """Centralised in-memory state for the API."""

    def __init__(self) -> None:
        self._conversations: Dict[str, Conversation] = {}
        self._projects: Dict[str, Project] = {}
        self._runs: Dict[str, RunRecord] = {}
        self._inputs: Dict[str, ProjectInputs] = {}

    # Conversation helpers -------------------------------------------------
    def get_conversation(self, conversation_id: str) -> Conversation:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            conversation = Conversation(id=conversation_id)
            self._conversations[conversation_id] = conversation
        return conversation

    # Project helpers ------------------------------------------------------
    def upsert_project(
        self, project_id: str, name: Optional[str], model_config: ModelConfig
    ) -> Project:
        project = Project(id=project_id, name=name, model_config=model_config)
        self._projects[project_id] = project
        return project

    def get_project(self, project_id: str) -> Project:
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' is not registered")
        return self._projects[project_id]

    def list_projects(self) -> Sequence[Project]:
        return list(self._projects.values())

    def set_inputs(
        self,
        project_id: str,
        forcing: Mapping[str, Sequence[float]],
        observations: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> ProjectInputs:
        project = self.get_project(project_id)
        normalized_forcing = _normalise_series(forcing)
        normalized_observations = (
            _normalise_series(observations) if observations is not None else None
        )
        dataset = ProjectInputs(
            project_id=project.id,
            forcing=normalized_forcing,
            observations=normalized_observations,
        )
        self._inputs[project_id] = dataset
        return dataset

    def get_inputs(self, project_id: str) -> Optional[ProjectInputs]:
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' is not registered")
        return self._inputs.get(project_id)

    def add_scenario(
        self,
        project_id: str,
        scenario_id: str,
        description: str,
        modifications: Mapping[str, Mapping[str, Any]],
    ) -> ScenarioConfig:
        project = self.get_project(project_id)
        if any(scenario.id == scenario_id for scenario in project.model_config.scenarios):
            raise ValueError(f"Scenario '{scenario_id}' already exists")
        scenario = ScenarioConfig(
            id=scenario_id,
            description=description,
            modifications={
                basin: dict(params)
                for basin, params in modifications.items()
            },
        )
        project.model_config.scenarios.append(scenario)
        return scenario

    def update_scenario(
        self,
        project_id: str,
        scenario_id: str,
        *,
        description: Optional[str] = None,
        modifications: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> ScenarioConfig:
        project = self.get_project(project_id)
        for index, scenario in enumerate(project.model_config.scenarios):
            if scenario.id != scenario_id:
                continue
            if description is not None:
                scenario.description = description
            if modifications is not None:
                scenario.modifications = {
                    basin: dict(values)
                    for basin, values in modifications.items()
                }
            project.model_config.scenarios[index] = scenario
            return scenario
        raise KeyError(f"Scenario '{scenario_id}' not found")

    def remove_scenario(self, project_id: str, scenario_id: str) -> None:
        project = self.get_project(project_id)
        scenarios = project.model_config.scenarios
        for index, scenario in enumerate(scenarios):
            if scenario.id == scenario_id:
                del scenarios[index]
                return
        raise KeyError(f"Scenario '{scenario_id}' not found")

    def list_scenarios(self, project_id: str) -> Sequence[ScenarioConfig]:
        project = self.get_project(project_id)
        return list(project.model_config.scenarios)

    # Run helpers ----------------------------------------------------------
    def create_run(
        self,
        project_id: str,
        scenario_ids: Sequence[str],
    ) -> RunRecord:
        run_id = uuid.uuid4().hex
        record = RunRecord(
            id=run_id,
            project_id=project_id,
            scenario_ids=list(scenario_ids),
            created_at=datetime.now(timezone.utc),
            status="pending",
        )
        self._runs[run_id] = record
        return record

    def complete_run(self, run_id: str, result: WorkflowResult) -> RunRecord:
        record = self._runs[run_id]
        record.status = "completed"
        record.result = result
        return record

    def fail_run(self, run_id: str, error: str) -> RunRecord:
        record = self._runs[run_id]
        record.status = "failed"
        record.error = error
        return record

    def get_run(self, run_id: str) -> RunRecord:
        if run_id not in self._runs:
            raise KeyError(f"Run '{run_id}' not found")
        return self._runs[run_id]

    def list_runs(self, project_id: Optional[str] = None) -> Sequence[RunRecord]:
        runs = list(self._runs.values())
        if project_id is None:
            return sorted(runs, key=lambda record: record.created_at, reverse=True)
        return sorted(
            [run for run in runs if run.project_id == project_id],
            key=lambda record: record.created_at,
            reverse=True,
        )


# Serialisation utilities --------------------------------------------------

def serialize_workflow_result(result: Optional[WorkflowResult]) -> Optional[Dict[str, object]]:
    if result is None:
        return None
    return {
        "baseline": serialize_scenario_run(result.baseline),
        "scenarios": {
            scenario_id: serialize_scenario_run(run)
            for scenario_id, run in result.scenarios.items()
        },
        "overall_scores": [serialize_model_score(score) for score in result.overall_scores or []],
        "evaluation_outcomes": [
            serialize_evaluation_outcome(outcome)
            for outcome in result.evaluation_outcomes
        ],
        "report_path": str(result.report_path) if result.report_path else None,
    }


def serialize_scenario_run(run: ScenarioRun) -> Dict[str, object]:
    return {
        "scenario_id": run.scenario_id,
        "local": run.local,
        "aggregated": run.aggregated,
        "zone_discharge": run.zone_discharge,
    }


def serialize_model_score(score) -> Dict[str, object]:
    return {
        "model_id": score.model_id,
        "per_subbasin": score.per_subbasin,
        "aggregated": score.aggregated,
    }


def serialize_evaluation_outcome(outcome: EvaluationOutcome) -> Dict[str, object]:
    return {
        "plan": outcome.plan.to_dict(),
        "scores": [serialize_model_score(score) for score in outcome.scores],
        "ranking": [serialize_model_score(score) for score in outcome.ranking],
        "ranking_metric": outcome.ranking_metric,
    }



def _normalise_series(data: Mapping[str, Sequence[float]]) -> Dict[str, List[float]]:
    return {str(key): [float(value) for value in values] for key, values in data.items()}


__all__ = [
    "Conversation",
    "ConversationMessage",
    "PortalState",
    "Project",
    "RunRecord",
    "ProjectInputs",
    "serialize_workflow_result",
]
