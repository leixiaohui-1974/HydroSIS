from __future__ import annotations

import copy
from datetime import datetime, timezone
import threading
import uuid
from typing import Dict, Mapping, MutableMapping, Optional, Sequence

from hydrosis.config import ModelConfig, ScenarioConfig
from hydrosis.workflow import WorkflowResult

from ..state import (
    Conversation,
    PortalState,
    Project,
    ProjectInputs,
    ProjectMapLayers,
    RunRecord,
    UserAccount,
    _normalise_series,
)


class InMemoryPortalState(PortalState):
    """Thread-safe in-memory implementation of :class:`PortalState`."""

    def __init__(self) -> None:
        self._conversations: MutableMapping[str, Conversation] = {}
        self._projects: MutableMapping[str, Project] = {}
        self._inputs: MutableMapping[str, ProjectInputs] = {}
        self._runs: MutableMapping[str, RunRecord] = {}
        self._users: MutableMapping[str, UserAccount] = {}
        self._map_layers: MutableMapping[str, ProjectMapLayers] = {}
        self._run_lock = threading.Lock()

    # Conversation helpers -------------------------------------------------
    def get_conversation(self, conversation_id: str) -> Conversation:
        conversation = self._conversations.get(conversation_id)
        if conversation is None:
            conversation = Conversation(id=conversation_id)
            self._conversations[conversation_id] = conversation
        return conversation

    # Project helpers ------------------------------------------------------
    def add_project(
        self, project_id: str, name: Optional[str], model_config: ModelConfig
    ) -> Project:
        config_copy = copy.deepcopy(model_config)
        project = Project(id=project_id, name=name, model_config=config_copy)
        self._projects[project_id] = project
        return project

    def get_project(self, project_id: str) -> Project:
        try:
            return self._projects[project_id]
        except KeyError as exc:
            raise KeyError(f"Project '{project_id}' is not registered") from exc

    def list_projects(self) -> Sequence[Project]:
        return [self._projects[key] for key in sorted(self._projects.keys())]

    def set_inputs(
        self,
        project_id: str,
        forcing: Mapping[str, Sequence[float]],
        observations: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> ProjectInputs:
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' is not registered")
        normalised_forcing = _normalise_series(forcing)
        normalised_observations = (
            _normalise_series(observations) if observations is not None else None
        )
        dataset = ProjectInputs(
            project_id=project_id,
            forcing=normalised_forcing,
            observations=normalised_observations,
            updated_at=datetime.now(timezone.utc),
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
        modifications: Mapping[str, Mapping[str, float]],
    ) -> ScenarioConfig:
        project = self.get_project(project_id)
        if any(scenario.id == scenario_id for scenario in project.model_config.scenarios):
            raise ValueError(f"Scenario '{scenario_id}' already exists")
        scenario = ScenarioConfig(
            id=scenario_id,
            description=description,
            modifications={
                basin: dict(values) for basin, values in (modifications or {}).items()
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
        modifications: Optional[Mapping[str, Mapping[str, float]]] = None,
    ) -> ScenarioConfig:
        project = self.get_project(project_id)
        for scenario in project.model_config.scenarios:
            if scenario.id == scenario_id:
                if description is not None:
                    scenario.description = description
                if modifications is not None:
                    scenario.modifications = {
                        basin: dict(values) for basin, values in modifications.items()
                    }
                return scenario
        raise KeyError(f"Scenario '{scenario_id}' not found")

    def remove_scenario(self, project_id: str, scenario_id: str) -> None:
        project = self.get_project(project_id)
        before = len(project.model_config.scenarios)
        project.model_config.scenarios = [
            scenario for scenario in project.model_config.scenarios if scenario.id != scenario_id
        ]
        if len(project.model_config.scenarios) == before:
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
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' is not registered")
        run_id = uuid.uuid4().hex
        record = RunRecord(
            id=run_id,
            project_id=project_id,
            scenario_ids=list(scenario_ids),
            created_at=datetime.now(timezone.utc),
            status="queued",
        )
        with self._run_lock:
            self._runs[run_id] = record
        return record

    def start_run(self, run_id: str) -> RunRecord:
        with self._run_lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run '{run_id}' not found")
            run.status = "running"
            run.error = None
            return run

    def complete_run(self, run_id: str, result: WorkflowResult) -> RunRecord:
        with self._run_lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run '{run_id}' not found")
            run.status = "completed"
            run.result = result
            run.error = None
            return run

    def fail_run(self, run_id: str, error: str) -> RunRecord:
        with self._run_lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run '{run_id}' not found")
            run.status = "failed"
            run.error = error
            return run

    def get_run(self, run_id: str) -> RunRecord:
        with self._run_lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError(f"Run '{run_id}' not found")
            return run

    def list_runs(self, project_id: Optional[str] = None) -> Sequence[RunRecord]:
        with self._run_lock:
            records = list(self._runs.values())
        if project_id is not None:
            records = [run for run in records if run.project_id == project_id]
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records

    # User helpers ----------------------------------------------------------
    def upsert_user(
        self, user_id: str, name: Optional[str], roles: Sequence[str]
    ) -> UserAccount:
        account = self._users.get(user_id)
        if account is None:
            account = UserAccount(id=user_id, name=name)
            self._users[user_id] = account
        if name is not None:
            account.name = name
            account.updated_at = datetime.now(timezone.utc)
        account.set_roles(roles)
        return account

    def get_user(self, user_id: str) -> UserAccount:
        account = self._users.get(user_id)
        if account is None:
            raise KeyError(f"User '{user_id}' is not registered")
        return account

    def list_users(self) -> Sequence[UserAccount]:
        return sorted(self._users.values(), key=lambda account: account.created_at)

    def set_project_role(
        self, user_id: str, project_id: str, role: Optional[str]
    ) -> UserAccount:
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' is not registered")
        account = self.get_user(user_id)
        account.assign_project_role(project_id, role)
        return account

    def list_project_roles(self, project_id: str) -> Mapping[str, str]:
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' is not registered")
        assignments: Dict[str, str] = {}
        for user_id, account in self._users.items():
            role = account.project_roles.get(project_id)
            if role:
                assignments[user_id] = role
        return assignments

    # GIS helpers -----------------------------------------------------------
    def set_map_layers(
        self, project_id: str, layers: Mapping[str, Mapping[str, object]]
    ) -> ProjectMapLayers:
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' is not registered")
        dataset = ProjectMapLayers(
            project_id=project_id,
            layers={name: dict(payload) for name, payload in layers.items()},
            updated_at=datetime.now(timezone.utc),
        )
        self._map_layers[project_id] = dataset
        return dataset

    def get_map_layers(self, project_id: str) -> Optional[ProjectMapLayers]:
        if project_id not in self._projects:
            raise KeyError(f"Project '{project_id}' is not registered")
        return self._map_layers.get(project_id)


__all__ = ["InMemoryPortalState"]
