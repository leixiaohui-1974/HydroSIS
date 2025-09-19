"""Dataclass-based models exposed through the portal API."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


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
        }


@dataclass

class RunList:
    runs: List[RunResponse]

    def to_dict(self) -> Dict[str, object]:
        return {"runs": [run.to_dict() for run in self.runs]}


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

]
