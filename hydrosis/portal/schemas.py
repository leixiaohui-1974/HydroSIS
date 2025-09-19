"""Dataclass-based models exposed through the portal API."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


def _normalise_modifications(mods: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    normalised: Dict[str, Dict[str, Any]] = {}
    for basin, values in mods.items():
        if not isinstance(values, Mapping):
            raise ValueError("each modification entry must be a mapping")
        basin_key = str(basin)
        basin_values: Dict[str, Any] = {}
        for key, value in values.items():
            if isinstance(value, (int, float)):
                basin_values[str(key)] = float(value)
            else:
                basin_values[str(key)] = value
        normalised[basin_key] = basin_values
    return normalised


@dataclass
class ConversationMessageSchema:
    role: str
    content: str
    intent: Optional[Dict[str, object]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ConversationMessageSchema":
        return cls(role=str(data["role"]), content=str(data["content"]), intent=data.get("intent"))

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ConversationResponse:
    reply: str
    intent: Dict[str, object]
    messages: List[ConversationMessageSchema] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "reply": self.reply,
            "intent": self.intent,
            "messages": [message.to_dict() for message in self.messages],
        }


@dataclass
class ProjectConfigPayload:
    model: Dict[str, object]
    name: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProjectConfigPayload":
        model = data.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("model configuration must be a mapping")
        name = data.get("name")
        return cls(model=dict(model), name=str(name) if name is not None else None)


@dataclass
class ProjectSummary:
    id: str
    name: Optional[str]
    scenarios: List[str]
    evaluation: Optional[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class UserPayload:
    id: str
    name: Optional[str]
    roles: List[str]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "UserPayload":
        roles = data.get("roles", [])
        if isinstance(roles, (str, bytes)):
            roles = [part.strip() for part in str(roles).split(",") if part.strip()]
        if not isinstance(roles, Sequence) or isinstance(roles, (str, bytes)):
            raise ValueError("roles must be provided as a list or comma separated string")
        parsed_roles = [str(role).strip() for role in roles if str(role).strip()]
        name = data.get("name")
        return cls(
            id=str(data.get("id")),
            name=str(name) if name is not None else None,
            roles=parsed_roles,
        )

    def to_dict(self) -> Dict[str, object]:
        return {"id": self.id, "name": self.name, "roles": list(self.roles)}


@dataclass
class UserResponse:
    id: str
    name: Optional[str]
    roles: List[str]
    project_roles: Mapping[str, str]
    created_at: str
    updated_at: str

    @classmethod
    def from_state(cls, data: Mapping[str, object]) -> "UserResponse":
        return cls(
            id=str(data.get("id")),
            name=data.get("name"),
            roles=list(data.get("roles", [])),
            project_roles=dict(data.get("project_roles", {})),
            created_at=str(data.get("created_at")),
            updated_at=str(data.get("updated_at")),
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "roles": list(self.roles),
            "project_roles": dict(self.project_roles),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class UserList:
    users: List[UserResponse]

    def to_dict(self) -> Dict[str, object]:
        return {"users": [user.to_dict() for user in self.users]}


@dataclass
class InputSeriesSummary:
    series_count: int
    min_length: int
    max_length: int
    mean_length: float
    min_value: Optional[float]
    max_value: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class InputsOverview:
    forcing: Optional[InputSeriesSummary]
    observations: Optional[InputSeriesSummary]
    updated_at: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "forcing": self.forcing.to_dict() if self.forcing else None,
            "observations": self.observations.to_dict() if self.observations else None,
            "updated_at": self.updated_at,
        }


@dataclass
class ScenarioCreatePayload:
    id: str
    description: str
    modifications: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ScenarioCreatePayload":
        mods = data.get("modifications")
        if not isinstance(mods, Mapping):
            raise ValueError("modifications must be a mapping")
        normalised = _normalise_modifications(mods)  # type: ignore[arg-type]
        return cls(
            id=str(data.get("id")),
            description=str(data.get("description", "")),
            modifications=normalised,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "description": self.description,
            "modifications": {basin: dict(values) for basin, values in self.modifications.items()},
        }


@dataclass
class RunRequest:
    forcing: Optional[Mapping[str, Sequence[float]]]
    observations: Optional[Mapping[str, Sequence[float]]]
    scenario_ids: Optional[Sequence[str]]
    persist_outputs: bool
    generate_report: bool

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "RunRequest":
        forcing = data.get("forcing")
        forcing_payload: Optional[Dict[str, List[float]]] = None
        if forcing is not None:
            if not isinstance(forcing, Mapping):
                raise ValueError("forcing must be provided as a mapping of sequences")
            forcing_payload = {str(key): list(value) for key, value in forcing.items()}
        observations = data.get("observations")
        observations_payload: Optional[Dict[str, List[float]]] = None
        if observations is not None:
            if not isinstance(observations, Mapping):
                raise ValueError("observations must be provided as a mapping of sequences")
            observations_payload = {
                str(key): list(value) for key, value in observations.items()
            }
        scenario_ids = data.get("scenario_ids")
        return cls(
            forcing=forcing_payload,
            observations=observations_payload,
            scenario_ids=list(scenario_ids)
            if isinstance(scenario_ids, Sequence)
            and not isinstance(scenario_ids, (str, bytes))
            else None,
            persist_outputs=bool(data.get("persist_outputs", False)),
            generate_report=bool(data.get("generate_report", False)),
        )


@dataclass
class RunResponse:
    id: str
    project_id: str
    scenario_ids: List[str]
    status: str
    error: Optional[str]
    result: Optional[Dict[str, object]]
    created_at: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ProjectInputsPayload:
    forcing: Mapping[str, Sequence[float]]
    observations: Optional[Mapping[str, Sequence[float]]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProjectInputsPayload":
        forcing = data.get("forcing")
        if not isinstance(forcing, Mapping):
            raise ValueError("forcing must be provided as a mapping of sequences")
        observations = data.get("observations")
        if observations is not None and not isinstance(observations, Mapping):
            raise ValueError("observations must be a mapping when provided")
        return cls(
            forcing={str(key): list(value) for key, value in forcing.items()},
            observations={
                str(key): list(value) for key, value in observations.items()
            }
            if isinstance(observations, Mapping)
            else None,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "forcing": {key: list(values) for key, values in self.forcing.items()},
            "observations": {
                key: list(values) for key, values in (self.observations or {}).items()
            }
            or None,
        }


@dataclass
class ProjectInputsResponse:
    forcing: Mapping[str, Sequence[float]]
    observations: Optional[Mapping[str, Sequence[float]]]
    updated_at: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class ProjectPermissionPayload:
    user_id: str
    role: Optional[str]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ProjectPermissionPayload":
        role = data.get("role")
        return cls(
            user_id=str(data.get("user_id")),
            role=str(role) if role is not None else None,
        )

    def to_dict(self) -> Dict[str, object]:
        return {"user_id": self.user_id, "role": self.role}


@dataclass
class ProjectPermissionList:
    project_id: str
    permissions: Mapping[str, str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "project_id": self.project_id,
            "permissions": dict(self.permissions),
        }


@dataclass
class ScenarioList:
    scenarios: List[ScenarioCreatePayload]

    def to_dict(self) -> Dict[str, object]:
        return {"scenarios": [scenario.to_dict() for scenario in self.scenarios]}


@dataclass
class ScenarioUpdatePayload:
    description: Optional[str] = None
    modifications: Optional[Mapping[str, Mapping[str, Any]]] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ScenarioUpdatePayload":
        description = data.get("description")
        mods = data.get("modifications")
        if mods is not None and not isinstance(mods, Mapping):
            raise ValueError("modifications must be a mapping when provided")
        return cls(
            description=str(description) if description is not None else None,
            modifications=_normalise_modifications(mods) if isinstance(mods, Mapping) else None,  # type: ignore[arg-type]
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "description": self.description,
            "modifications": {basin: dict(values) for basin, values in (self.modifications or {}).items()},
        }


@dataclass
class ProjectList:
    projects: List[ProjectSummary]

    def to_dict(self) -> Dict[str, object]:
        return {"projects": [project.to_dict() for project in self.projects]}


@dataclass
class ProjectOverview:
    id: str
    name: Optional[str]
    scenarios: List[str]
    scenario_count: int
    total_runs: int
    inputs: Optional[InputsOverview]
    latest_run: Optional["RunResponse"]
    latest_summary: Optional[Dict[str, object]]
    map_layers_updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "scenarios": list(self.scenarios),
            "scenario_count": self.scenario_count,
            "total_runs": self.total_runs,
            "inputs": self.inputs.to_dict() if self.inputs else None,
            "latest_run": self.latest_run.to_dict() if self.latest_run else None,
            "latest_summary": self.latest_summary,
            "map_layers_updated_at": self.map_layers_updated_at,
        }


@dataclass
class RunList:
    runs: List[RunResponse]

    def to_dict(self) -> Dict[str, object]:
        return {"runs": [run.to_dict() for run in self.runs]}


@dataclass
class MapLayerPayload:
    layers: Mapping[str, Mapping[str, Any]]

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "MapLayerPayload":
        layers = data.get("layers")
        if not isinstance(layers, Mapping):
            raise ValueError("layers must be provided as a mapping of GeoJSON objects")
        payload: Dict[str, Dict[str, Any]] = {}
        for name, layer in layers.items():
            if not isinstance(layer, Mapping):
                raise ValueError("each layer must be a mapping")
            payload[str(name)] = dict(layer)
        return cls(layers=payload)

    def to_dict(self) -> Dict[str, object]:
        return {"layers": {key: dict(value) for key, value in self.layers.items()}}


@dataclass
class MapLayerResponse:
    project_id: str
    layers: Mapping[str, Mapping[str, Any]]
    updated_at: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "project_id": self.project_id,
            "layers": {key: dict(value) for key, value in self.layers.items()},
            "updated_at": self.updated_at,
        }


@dataclass
class QueueEntry:
    run_id: str
    project_id: str
    status: str
    scenario_ids: Sequence[str]
    submitted_at: str
    started_at: Optional[str]
    finished_at: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "run_id": self.run_id,
            "project_id": self.project_id,
            "status": self.status,
            "scenario_ids": list(self.scenario_ids),
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class QueueSnapshot:
    entries: Sequence[QueueEntry]

    def to_dict(self) -> Dict[str, object]:
        return {"entries": [entry.to_dict() for entry in self.entries]}


__all__ = [
    "ConversationMessageSchema",
    "ConversationResponse",
    "ProjectConfigPayload",
    "ProjectSummary",
    "ProjectList",
    "RunRequest",
    "RunResponse",
    "RunList",
    "ScenarioCreatePayload",
    "ScenarioList",
    "ScenarioUpdatePayload",
    "ProjectInputsPayload",
    "ProjectInputsResponse",
    "InputSeriesSummary",
    "InputsOverview",
    "ProjectOverview",
    "UserPayload",
    "UserResponse",
    "UserList",
    "ProjectPermissionPayload",
    "ProjectPermissionList",
    "MapLayerPayload",
    "MapLayerResponse",
    "QueueEntry",
    "QueueSnapshot",
]
