"""SQLAlchemy-backed persistence for the HydroSIS portal."""
from __future__ import annotations

from datetime import datetime, timezone
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    String,
    Text,
    create_engine,
    delete,
    select,
    func,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from ..state import (
    Conversation,
    Project,
    ProjectInputs,
    RunRecord,
    ScenarioConfig,
    deserialize_workflow_result,
    serialize_workflow_result,
)
from hydrosis.config import ModelConfig

Base = declarative_base()


class ProjectModel(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=True)
    config = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    scenarios = relationship(
        "ScenarioModel",
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    inputs = relationship(
        "ProjectInputModel",
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
        uselist=False,
    )
    runs = relationship(
        "RunModel",
        back_populates="project",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ScenarioModel(Base):
    __tablename__ = "project_scenarios"
    __table_args__ = (Index("ix_project_scenarios_project_id", "project_id"),)

    project_id = Column(
        String,
        ForeignKey("projects.id", ondelete="CASCADE"),
        primary_key=True,
    )
    scenario_id = Column(String, primary_key=True)
    description = Column(Text, nullable=True)
    modifications = Column(JSON, nullable=False, default=dict)

    project = relationship("ProjectModel", back_populates="scenarios")


class ProjectInputModel(Base):
    __tablename__ = "project_inputs"

    project_id = Column(
        String,
        ForeignKey("projects.id", ondelete="CASCADE"),
        primary_key=True,
    )
    forcing = Column(JSON, nullable=False)
    observations = Column(JSON, nullable=True)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    project = relationship("ProjectModel", back_populates="inputs")


class RunModel(Base):
    __tablename__ = "runs"
    __table_args__ = (
        Index("ix_runs_project_created", "project_id", "created_at"),
    )

    id = Column(String, primary_key=True)
    project_id = Column(
        String,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
    )
    scenario_ids = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    status = Column(String, nullable=False)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)

    project = relationship("ProjectModel", back_populates="runs")


def create_engine_from_url(database_url: str, **kwargs) -> Engine:
    """Create a SQLAlchemy engine configured for the given URL."""

    return create_engine(database_url, future=True, **kwargs)


def create_session_factory(engine: Engine) -> sessionmaker:
    """Construct a :class:`sessionmaker` bound to ``engine``."""

    return sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


class SQLAlchemyPortalState:
    """Persistence backend implementing :class:`PortalState` using SQLAlchemy."""

    def __init__(self, session_factory: sessionmaker) -> None:
        self._session_factory = session_factory
        self._conversations: Dict[str, Conversation] = {}

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
        config_dict = model_config.to_dict()
        scenarios = config_dict.get("scenarios", [])
        base_config = dict(config_dict)
        base_config["scenarios"] = []

        with self._session_factory() as session:
            project = session.get(ProjectModel, project_id)
            if project is None:
                project = ProjectModel(id=project_id)
                session.add(project)

            project.name = name
            project.config = base_config

            session.execute(
                delete(ScenarioModel).where(ScenarioModel.project_id == project_id)
            )
            for scenario in scenarios:
                session.add(
                    ScenarioModel(
                        project_id=project_id,
                        scenario_id=scenario["id"],
                        description=scenario.get("description"),
                        modifications=scenario.get("modifications", {}),
                    )
                )

            session.commit()
            session.refresh(project)

            return _build_project(session, project)

    def get_project(self, project_id: str) -> Project:
        with self._session_factory() as session:
            project = session.get(ProjectModel, project_id)
            if project is None:
                raise KeyError(f"Project '{project_id}' is not registered")
            return _build_project(session, project)

    def list_projects(self) -> Sequence[Project]:
        with self._session_factory() as session:
            projects = session.execute(select(ProjectModel)).scalars().all()
            return [_build_project(session, project) for project in projects]

    def set_inputs(
        self,
        project_id: str,
        forcing: Mapping[str, Sequence[float]],
        observations: Optional[Mapping[str, Sequence[float]]] = None,
    ) -> ProjectInputs:
        _ = self.get_project(project_id)

        normalized_forcing = _normalise_series(forcing)
        normalized_observations = (
            _normalise_series(observations) if observations is not None else None
        )

        with self._session_factory() as session:
            dataset = session.get(ProjectInputModel, project_id)
            if dataset is None:
                dataset = ProjectInputModel(project_id=project_id)
                session.add(dataset)

            dataset.forcing = normalized_forcing
            dataset.observations = normalized_observations
            dataset.updated_at = datetime.now(timezone.utc)

            session.commit()
            session.refresh(dataset)

            return _build_inputs(dataset)

    def get_inputs(self, project_id: str) -> Optional[ProjectInputs]:
        with self._session_factory() as session:
            project = session.get(ProjectModel, project_id)
            if project is None:
                raise KeyError(f"Project '{project_id}' is not registered")
            dataset = session.get(ProjectInputModel, project_id)
            if dataset is None:
                return None
            return _build_inputs(dataset)

    def add_scenario(
        self,
        project_id: str,
        scenario_id: str,
        description: str,
        modifications: Mapping[str, Mapping[str, Any]],
    ) -> ScenarioConfig:
        with self._session_factory() as session:
            project = session.get(ProjectModel, project_id)
            if project is None:
                raise KeyError(f"Project '{project_id}' is not registered")

            existing = session.get(
                ScenarioModel, {"project_id": project_id, "scenario_id": scenario_id}
            )
            if existing is not None:
                raise ValueError(f"Scenario '{scenario_id}' already exists")

            scenario = ScenarioModel(
                project_id=project_id,
                scenario_id=scenario_id,
                description=description,
                modifications={
                    basin: dict(values) for basin, values in modifications.items()
                },
            )
            session.add(scenario)
            session.commit()
            session.refresh(scenario)

            return _build_scenario(scenario)

    def update_scenario(
        self,
        project_id: str,
        scenario_id: str,
        *,
        description: Optional[str] = None,
        modifications: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> ScenarioConfig:
        with self._session_factory() as session:
            scenario = session.get(
                ScenarioModel, {"project_id": project_id, "scenario_id": scenario_id}
            )
            if scenario is None:
                raise KeyError(f"Scenario '{scenario_id}' not found")

            if description is not None:
                scenario.description = description
            if modifications is not None:
                scenario.modifications = {
                    basin: dict(values) for basin, values in modifications.items()
                }

            session.commit()
            session.refresh(scenario)

            return _build_scenario(scenario)

    def remove_scenario(self, project_id: str, scenario_id: str) -> None:
        with self._session_factory() as session:
            result = session.execute(
                delete(ScenarioModel).where(
                    ScenarioModel.project_id == project_id,
                    ScenarioModel.scenario_id == scenario_id,
                )
            )
            if result.rowcount == 0:
                raise KeyError(f"Scenario '{scenario_id}' not found")
            session.commit()

    def list_scenarios(self, project_id: str) -> Sequence[ScenarioConfig]:
        with self._session_factory() as session:
            project = session.get(ProjectModel, project_id)
            if project is None:
                raise KeyError(f"Project '{project_id}' is not registered")
            rows = (
                session.execute(
                    select(ScenarioModel).where(ScenarioModel.project_id == project_id)
                )
                .scalars()
                .all()
            )
            return [_build_scenario(row) for row in rows]

    # Run helpers ----------------------------------------------------------
    def create_run(
        self,
        project_id: str,
        scenario_ids: Sequence[str],
    ) -> RunRecord:
        self.get_project(project_id)
        run_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc)

        with self._session_factory() as session:
            run = RunModel(
                id=run_id,
                project_id=project_id,
                scenario_ids=list(scenario_ids),
                created_at=created_at,
                status="pending",
            )
            session.add(run)
            session.commit()
            session.refresh(run)
            return _build_run(run)

    def complete_run(self, run_id: str, result: WorkflowResult) -> RunRecord:
        serialized = serialize_workflow_result(result)
        with self._session_factory() as session:
            run = session.get(RunModel, run_id)
            if run is None:
                raise KeyError(f"Run '{run_id}' not found")
            run.status = "completed"
            run.result = serialized
            run.error = None
            session.commit()
            session.refresh(run)
            return _build_run(run)

    def fail_run(self, run_id: str, error: str) -> RunRecord:
        with self._session_factory() as session:
            run = session.get(RunModel, run_id)
            if run is None:
                raise KeyError(f"Run '{run_id}' not found")
            run.status = "failed"
            run.error = error
            session.commit()
            session.refresh(run)
            return _build_run(run)

    def get_run(self, run_id: str) -> RunRecord:
        with self._session_factory() as session:
            run = session.get(RunModel, run_id)
            if run is None:
                raise KeyError(f"Run '{run_id}' not found")
            return _build_run(run)

    def list_runs(self, project_id: Optional[str] = None) -> Sequence[RunRecord]:
        with self._session_factory() as session:
            statement = select(RunModel)
            if project_id is not None:
                statement = statement.where(RunModel.project_id == project_id)
            runs = session.execute(statement).scalars().all()
            runs.sort(key=lambda record: record.created_at, reverse=True)
            return [_build_run(run) for run in runs]


def create_sqlalchemy_state(database_url: str, **engine_kwargs) -> SQLAlchemyPortalState:
    engine = create_engine_from_url(database_url, **engine_kwargs)
    Base.metadata.create_all(engine)
    return SQLAlchemyPortalState(create_session_factory(engine))


def _build_project(session: Session, row: ProjectModel) -> Project:
    config_dict = dict(row.config or {})
    scenarios = (
        session.execute(
            select(ScenarioModel).where(ScenarioModel.project_id == row.id)
        )
        .scalars()
        .all()
    )
    config_dict["scenarios"] = [
        {
            "id": scenario.scenario_id,
            "description": scenario.description or "",
            "modifications": scenario.modifications or {},
        }
        for scenario in scenarios
    ]
    model_config = ModelConfig.from_dict(config_dict)
    return Project(id=row.id, name=row.name, model_config=model_config)


def _build_inputs(row: ProjectInputModel) -> ProjectInputs:
    return ProjectInputs(
        project_id=row.project_id,
        forcing={key: list(values) for key, values in (row.forcing or {}).items()},
        observations={
            key: list(values) for key, values in (row.observations or {}).items()
        }
        if row.observations
        else None,
        updated_at=row.updated_at
        if row.updated_at.tzinfo
        else row.updated_at.replace(tzinfo=timezone.utc),
    )


def _build_scenario(row: ScenarioModel) -> ScenarioConfig:
    return ScenarioConfig(
        id=row.scenario_id,
        description=row.description or "",
        modifications={
            basin: dict(values)
            for basin, values in (row.modifications or {}).items()
        },
    )


def _build_run(row: RunModel) -> RunRecord:
    created_at = row.created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    result = deserialize_workflow_result(row.result)
    return RunRecord(
        id=row.id,
        project_id=row.project_id,
        scenario_ids=list(row.scenario_ids or []),
        created_at=created_at,
        status=row.status,
        result=result,
        error=row.error,
    )


def _normalise_series(data: Mapping[str, Sequence[float]]) -> Dict[str, List[float]]:
    return {
        str(key): [float(value) for value in values]
        for key, values in (data or {}).items()
    }
