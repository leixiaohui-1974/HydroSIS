"""Web and API behaviour tests exercising the FastAPI portal with persistence."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hydrosis import HydroSISModel
from hydrosis.portal import create_app

from tests.portal_test_utils import build_demo_forcing, build_demo_model_config


def test_portal_serves_index_page() -> None:
    app = create_app()
    client = TestClient(app)

    response = client.get("/")
    assert response.status_code == 200
    assert "HydroSIS Portal" in response.text


def test_portal_api_with_sqlalchemy_backend(tmp_path) -> None:
    pytest.importorskip("sqlalchemy")
    db_url = f"sqlite:///{tmp_path / 'portal.db'}"
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    app = create_app(database_url=db_url)
    client = TestClient(app)

    config = build_demo_model_config(results_dir)
    config_payload = config.to_dict()

    project_response = client.post(
        "/projects/demo/config",
        json={"model": config_payload, "name": "Demo"},
    )
    assert project_response.status_code == 200

    forcing = build_demo_forcing()
    model = HydroSISModel.from_config(config)
    baseline_local = model.run(forcing)
    observations = {
        basin: list(values)
        for basin, values in model.accumulate_discharge(baseline_local).items()
    }

    inputs_response = client.post(
        "/projects/demo/inputs",
        json={"forcing": forcing, "observations": observations},
    )
    assert inputs_response.status_code == 200

    run_response = client.post(
        "/projects/demo/runs",
        json={
            "scenario_ids": ["alternate_routing"],
            "persist_outputs": False,
            "generate_report": False,
        },
    )
    assert run_response.status_code == 200, run_response.text
    run_payload = run_response.json()
    assert run_payload["status"] == "completed"
    run_id = run_payload["id"]

    reloaded_app = create_app(database_url=db_url)
    reloaded_client = TestClient(reloaded_app)

    overview_response = reloaded_client.get("/projects/demo/overview")
    assert overview_response.status_code == 200
    overview = overview_response.json()
    assert overview["total_runs"] >= 1
    assert overview["latest_run"]["id"] == run_id

    summary_response = reloaded_client.get(f"/runs/{run_id}/summary")
    assert summary_response.status_code == 200
    assert summary_response.json()["baseline"]["scenario_id"] == "baseline"

    inputs_get = reloaded_client.get("/projects/demo/inputs")
    assert inputs_get.status_code == 200
    assert inputs_get.json()["forcing"]["S1"] == forcing["S1"]
