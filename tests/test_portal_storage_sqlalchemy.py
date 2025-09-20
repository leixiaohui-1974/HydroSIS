"""Tests verifying the SQLAlchemy-backed portal state implementation."""
from __future__ import annotations

import pytest

pytest.importorskip("sqlalchemy")

from hydrosis.config import ComparisonPlanConfig
from hydrosis.evaluation.comparison import ModelScore
from hydrosis.portal.storage import create_sqlalchemy_state
from hydrosis.workflow import EvaluationOutcome, ScenarioRun, WorkflowResult

from tests.portal_test_utils import build_demo_model_config


def _dummy_workflow_result() -> WorkflowResult:
    baseline = ScenarioRun(
        scenario_id="baseline",
        local={"S1": [1.0, 2.0], "S3": [0.6, 1.2]},
        aggregated={"S3": [1.4, 2.6]},
        zone_discharge={"Z1": {"S1": [1.0, 2.0]}},
    )
    alternate = ScenarioRun(
        scenario_id="alternate_routing",
        local={"S1": [1.1, 2.1], "S3": [0.7, 1.3]},
        aggregated={"S3": [1.6, 2.8]},
        zone_discharge={"Z1": {"S1": [1.1, 2.1]}},
    )

    baseline_score = ModelScore(
        model_id="baseline",
        per_subbasin={"S3": {"rmse": 0.0, "nse": 1.0}},
        aggregated={"rmse": 0.0, "nse": 1.0},
    )
    alternate_score = ModelScore(
        model_id="alternate_routing",
        per_subbasin={"S3": {"rmse": 0.1, "nse": 0.9}},
        aggregated={"rmse": 0.1, "nse": 0.9},
    )

    plan = ComparisonPlanConfig(
        id="baseline_vs_scenario",
        description="",
        models=["baseline", "alternate_routing"],
        reference="observed",
        subbasins=["S3"],
        ranking_metric="rmse",
    )
    outcome = EvaluationOutcome(
        plan=plan,
        scores=[baseline_score, alternate_score],
        ranking=[baseline_score, alternate_score],
        ranking_metric="rmse",
    )

    return WorkflowResult(
        baseline=baseline,
        scenarios={"alternate_routing": alternate},
        overall_scores=[baseline_score, alternate_score],
        evaluation_outcomes=[outcome],
    )


def test_sqlalchemy_state_round_trip(tmp_path) -> None:
    db_path = tmp_path / "portal.db"
    db_url = f"sqlite:///{db_path}"

    results_dir = tmp_path / "results"
    results_dir.mkdir()

    state = create_sqlalchemy_state(db_url)
    config = build_demo_model_config(results_dir)

    project = state.upsert_project("demo", "Demo project", config)
    assert project.id == "demo"
    assert project.model_config.scenarios[0].id == "alternate_routing"

    inputs = state.set_inputs(
        "demo",
        forcing={"S1": [0.5, 1.5], "S3": [0.0, 0.0]},
        observations={"S3": [0.2, 0.4]},
    )
    assert inputs.forcing["S1"] == [0.5, 1.5]

    scenario = state.add_scenario(
        "demo",
        "expansion",
        "Extended routing adjustments",
        {"S1": {"routing_model": "lag_long"}},
    )
    assert scenario.id == "expansion"

    updated = state.update_scenario(
        "demo",
        "expansion",
        description="Updated expansion",
        modifications={"S1": {"precipitation_multiplier": 1.1}},
    )
    assert updated.description == "Updated expansion"
    assert updated.modifications["S1"]["precipitation_multiplier"] == 1.1

    scenario_ids = {item.id for item in state.list_scenarios("demo")}
    assert scenario_ids == {"alternate_routing", "expansion"}

    run = state.create_run("demo", ["alternate_routing", "expansion"])
    assert run.status == "pending"

    result = _dummy_workflow_result()
    state.complete_run(run.id, result)

    completed = state.get_run(run.id)
    assert completed.status == "completed"
    assert completed.result is not None
    assert completed.result.scenarios["alternate_routing"].aggregated["S3"][0] == 1.6

    failed_run = state.create_run("demo", ["alternate_routing"])
    state.fail_run(failed_run.id, "integration failure")

    failed = state.get_run(failed_run.id)
    assert failed.status == "failed"
    assert failed.error == "integration failure"

    ordered_runs = state.list_runs("demo")
    assert ordered_runs[0].id == failed_run.id
    assert ordered_runs[1].id == run.id

    state.remove_scenario("demo", "expansion")
    remaining = {item.id for item in state.list_scenarios("demo")}
    assert remaining == {"alternate_routing"}

    state_reloaded = create_sqlalchemy_state(db_url)
    project_reloaded = state_reloaded.get_project("demo")
    assert project_reloaded.name == "Demo project"

    inputs_reloaded = state_reloaded.get_inputs("demo")
    assert inputs_reloaded is not None
    assert inputs_reloaded.forcing["S1"] == [0.5, 1.5]

    persisted_run = state_reloaded.get_run(run.id)
    assert persisted_run.status == "completed"
    assert persisted_run.result is not None
    assert (
        persisted_run.result.overall_scores[0].aggregated["rmse"]
        == 0.0
    )

    runs_after_reload = state_reloaded.list_runs("demo")
    assert runs_after_reload[0].id == failed_run.id
    assert runs_after_reload[1].id == run.id
