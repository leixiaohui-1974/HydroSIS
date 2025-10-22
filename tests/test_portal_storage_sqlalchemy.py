"""Tests verifying the SQLAlchemy-backed portal state implementation."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from hydrosis.config import ComparisonPlanConfig
from hydrosis.evaluation.comparison import ModelScore
from hydrosis.portal.storage import SqlAlchemyStorage
from hydrosis.workflow import EvaluationOutcome, ScenarioRun, WorkflowResult


def _dummy_workflow_result() -> WorkflowResult:
    """Instantiate a dummy workflow result for testing."""

    return WorkflowResult(
        baseline=ScenarioRun(
            scenario_id="baseline",
            local={"test_basin": [1.0, 2.0, 3.0]},
            aggregated={"test_basin": [1.5, 2.5]},
            zone_discharge={},
        ),
        scenarios={},
        overall_scores=[
            ModelScore(
                model_id="test_model",
                per_subbasin={"test_basin": {"kge": 0.5}},
                aggregated={"kge": 0.5},
            )
        ],
        evaluation_outcomes=[
            EvaluationOutcome(
                plan=ComparisonPlanConfig(
                    id="test_plan",
                    models=["test_model"],
                    subbasins=["test_basin"],
                ),
                scores=[
                    ModelScore(
                        model_id="test_model",
                        per_subbasin={"test_basin": {"kge": 0.5}},
                        aggregated={"kge": 0.5},
                    )
                ],
                ranking=[
                    ModelScore(
                        model_id="test_model",
                        per_subbasin={"test_basin": {"kge": 0.5}},
                        aggregated={"kge": 0.5},
                    )
                ],
                ranking_metric="kge",
            )
        ],
    )


def test_sqlalchemy_storage_round_trip(storage: SqlAlchemyStorage) -> None:
    """Verify that a workflow result survives a SQLAlchemy round trip."""

    result = _dummy_workflow_result()
    storage.write("dummy", result)
    read_result = storage.read("dummy")

    assert read_result == result
