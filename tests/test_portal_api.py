"""Integration tests for the HydroSIS portal FastAPI application."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Dict, List

from fastapi.testclient import TestClient

from hydrosis import HydroSISModel
from hydrosis.config import (
    ComparisonPlanConfig,
    EvaluationConfig,
    IOConfig,
    ModelConfig,
    ScenarioConfig,
)
from hydrosis.delineation.dem_delineator import DelineationConfig
from hydrosis.parameters.zone import ParameterZoneConfig
from hydrosis.portal import create_app
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig


def _build_portal_config(results_directory: Path) -> ModelConfig:
    delineation = DelineationConfig(
        dem_path=Path("dem.tif"),
        pour_points_path=Path("pour_points.geojson"),
        precomputed_subbasins=[
            {"id": "S1", "area_km2": 1.0, "downstream": "S3", "parameters": {}},
            {"id": "S2", "area_km2": 1.0, "downstream": "S3", "parameters": {}},
            {"id": "S3", "area_km2": 1.0, "downstream": None, "parameters": {}},
        ],
    )

    runoff_models = [
        RunoffModelConfig(
            id="curve",
            model_type="scs_curve_number",
            parameters={"curve_number": 75, "initial_abstraction_ratio": 0.2},
        ),
        RunoffModelConfig(
            id="reservoir",
            model_type="linear_reservoir",
            parameters={"recession": 0.85, "conversion": 1.0},
        ),
    ]

    routing_models = [
        RoutingModelConfig(
            id="lag_short",
            model_type="lag",
            parameters={"lag_steps": 1},
        ),
        RoutingModelConfig(
            id="lag_long",
            model_type="lag",
            parameters={"lag_steps": 2},
        ),
    ]

    parameter_zones = [
        ParameterZoneConfig(
            id="Z1",
            description="Headwater gauge",
            control_points=["S1"],
            parameters={"runoff_model": "curve", "routing_model": "lag_short"},
        ),
        ParameterZoneConfig(
            id="Z2",
            description="Outlet station",
            control_points=["S3"],
            parameters={"runoff_model": "reservoir", "routing_model": "lag_short"},
        ),
    ]

    io_config = IOConfig(
        precipitation=Path("data/forcing/precipitation.csv"),
        results_directory=results_directory,
        figures_directory=results_directory / "figures",
        reports_directory=results_directory / "reports",
    )

    scenarios = [
        ScenarioConfig(
            id="alternate_routing",
            description="Slow down routing through the middle catchment",
            modifications={"S2": {"routing_model": "lag_long"}},
        )
    ]

    evaluation = EvaluationConfig(
        metrics=["rmse", "nse"],
        comparisons=[
            ComparisonPlanConfig(
                id="baseline_vs_scenario",
                description="Compare baseline and scenario accuracy at the outlet",
                models=["baseline", "alternate_routing"],
                reference="observed",
                subbasins=["S3"],
                ranking_metric="rmse",
            )
        ],
    )

    return ModelConfig(
        delineation=delineation,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=io_config,
        scenarios=scenarios,
        evaluation=evaluation,
    )


def test_portal_end_to_end_workflow() -> None:
    app = create_app()
    client = TestClient(app)

    forcing: Dict[str, List[float]] = {
        "S1": [0.0, 10.0, 30.0, 0.0],
        "S2": [5.0, 5.0, 5.0, 5.0],
        "S3": [0.0, 0.0, 0.0, 0.0],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _build_portal_config(Path(tmpdir) / "results")
        config_payload = config.to_dict()

        response = client.post(
            "/projects/demo/config",
            json={"model": config_payload, "name": "Demo"},
        )
        assert response.status_code == 200
        project_summary = response.json()
        assert project_summary["scenarios"] == ["alternate_routing"]

        overview_response = client.get("/projects/demo/overview")
        assert overview_response.status_code == 200
        overview = overview_response.json()
        assert overview["scenario_count"] == 1
        assert overview["total_runs"] == 0
        assert overview["inputs"] is None

        projects_response = client.get("/projects")
        assert projects_response.status_code == 200
        projects = projects_response.json()["projects"]
        assert any(project["id"] == "demo" for project in projects)

        model = HydroSISModel.from_config(config)
        baseline_local = model.run(forcing)
        observations = {
            basin: list(values) for basin, values in model.accumulate_discharge(baseline_local).items()
        }

        inputs_response = client.post(
            "/projects/demo/inputs",
            json={"forcing": forcing, "observations": observations},
        )
        assert inputs_response.status_code == 200
        inputs_payload = inputs_response.json()
        assert inputs_payload["forcing"]["S1"] == forcing["S1"]
        assert "updated_at" in inputs_payload

        overview_after_inputs = client.get("/projects/demo/overview").json()
        assert overview_after_inputs["inputs"]["forcing"]["series_count"] == len(forcing)
        assert overview_after_inputs["inputs"]["forcing"]["max_length"] == 4
        assert overview_after_inputs["inputs"]["observations"]["min_value"] is not None

        inputs_get = client.get("/projects/demo/inputs")
        assert inputs_get.status_code == 200
        assert inputs_get.json()["observations"]["S1"] == observations["S1"]

        scenario_create = client.post(
            "/projects/demo/scenarios",
            json={
                "id": "higher_inflow",
                "description": "Increase precipitation by 10%",
                "modifications": {"S1": {"precipitation_multiplier": 1.1}},
            },
        )
        assert scenario_create.status_code == 200
        assert scenario_create.json()["id"] == "higher_inflow"

        scenario_update = client.put(
            "/projects/demo/scenarios/alternate_routing",
            json={"description": "Updated description"},
        )
        assert scenario_update.status_code == 200
        assert scenario_update.json()["description"] == "Updated description"

        scenarios_response = client.get("/projects/demo/scenarios")
        assert scenarios_response.status_code == 200
        scenario_ids = [item["id"] for item in scenarios_response.json()["scenarios"]]
        assert set(scenario_ids) == {"alternate_routing", "higher_inflow"}

        delete_response = client.delete("/projects/demo/scenarios/higher_inflow")
        assert delete_response.status_code == 204
        scenarios_after_delete = client.get("/projects/demo/scenarios").json()["scenarios"]
        assert all(item["id"] != "higher_inflow" for item in scenarios_after_delete)

        chat_response = client.post(
            "/conversations/default/messages",
            json={"role": "user", "content": "请运行 scenario alternate_routing"},
        )
        assert chat_response.status_code == 200
        assert chat_response.json()["intent"]["action"] == "run_scenarios"

        run_response = client.post(
            "/projects/demo/runs",
            json={
                "scenario_ids": ["alternate_routing"],
                "persist_outputs": False,
                "generate_report": False,
            },
        )
        assert run_response.status_code == 200, run_response.text
        payload = run_response.json()
        assert payload["status"] == "completed"
        assert payload["result"]
        assert "baseline" in payload["result"]
        assert payload["result"]["overall_scores"]

        run_id = payload["id"]

        overview_after_run = client.get("/projects/demo/overview").json()
        assert overview_after_run["total_runs"] == 1
        assert overview_after_run["latest_run"]["id"] == run_id
        assert overview_after_run["latest_summary"]["baseline"]["scenario_id"] == "baseline"

        summary_response = client.get(f"/runs/{run_id}/summary")
        assert summary_response.status_code == 200
        summary = summary_response.json()
        assert summary["baseline"]["scenario_id"] == "baseline"
        assert "narrative" in summary and summary["narrative"]
        assert "alternate_routing" in summary["scenarios"]
        alt_summary = summary["scenarios"]["alternate_routing"]
        assert "aggregated" in alt_summary and "S3" in alt_summary["aggregated"]
        assert "delta_vs_baseline" in alt_summary

        timeseries_response = client.get(f"/runs/{run_id}/timeseries")
        assert timeseries_response.status_code == 200
        timeseries = timeseries_response.json()
        assert timeseries["baseline"]["scenario_id"] == "baseline"
        assert len(timeseries["baseline"]["aggregated"]["S3"]) == len(forcing["S3"])
        assert "alternate_routing" in timeseries["scenarios"]
        assert timeseries["scenarios"]["alternate_routing"]["local"]["S1"]

        evaluation_response = client.get(f"/runs/{run_id}/evaluation")
        assert evaluation_response.status_code == 200
        evaluation = evaluation_response.json()
        assert evaluation["overall_scores"], evaluation
        assert evaluation["evaluation_outcomes"], evaluation
        first_plan = evaluation["evaluation_outcomes"][0]
        assert first_plan["plan"]["id"] == "baseline_vs_scenario"
        assert first_plan["ranking"], first_plan

        project_runs = client.get("/projects/demo/runs")
        assert project_runs.status_code == 200
        assert any(run["id"] == run_id for run in project_runs.json()["runs"])

        all_runs = client.get("/runs")
        assert all_runs.status_code == 200
        assert any(run["id"] == run_id for run in all_runs.json()["runs"])

        get_response = client.get(f"/runs/{run_id}")
        assert get_response.status_code == 200
        assert get_response.json()["id"] == run_id


def test_portal_role_permissions_and_map_layers() -> None:
    app = create_app()
    client = TestClient(app)

    with tempfile.TemporaryDirectory() as tmpdir:
        config = _build_portal_config(Path(tmpdir) / "results")
        config_payload = config.to_dict()

        admin_create = client.post("/users", json={"id": "admin", "roles": ["admin"]})
        assert admin_create.status_code == 200

        unauthorized_user = client.post("/users", json={"id": "guest", "roles": []})
        assert unauthorized_user.status_code == 403

        modeler_create = client.post(
            "/users",
            json={"id": "modeler", "roles": ["modeler"], "actor_id": "admin"},
        )
        assert modeler_create.status_code == 200

        config_response = client.post(
            "/projects/demo/config",
            json={"model": config_payload, "name": "Permitted", "user_id": "admin"},
        )
        assert config_response.status_code == 200

        permission_assign = client.post(
            "/projects/demo/permissions",
            json={"user_id": "modeler", "role": "modeler", "actor_id": "admin"},
        )
        assert permission_assign.status_code == 200
        permissions_view = client.get("/projects/demo/permissions")
        assert permissions_view.status_code == 200
        assert permissions_view.json()["permissions"].get("modeler") == "modeler"

        map_payload = {
            "layers": {
                "subbasins": {"type": "FeatureCollection", "features": []},
                "stations": {"type": "FeatureCollection", "features": []},
            }
        }
        map_update = client.post(
            "/projects/demo/map",
            json={**map_payload, "user_id": "admin"},
        )
        assert map_update.status_code == 200

        map_get = client.get("/projects/demo/map")
        assert map_get.status_code == 200
        assert set(map_get.json()["layers"].keys()) == {"subbasins", "stations"}

        forcing: Dict[str, List[float]] = {
            "S1": [0.0, 5.0, 15.0, 5.0],
            "S2": [1.0, 2.0, 3.0, 4.0],
            "S3": [0.0, 0.0, 0.0, 0.0],
        }

        inputs_response = client.post(
            "/projects/demo/inputs",
            json={"forcing": forcing, "user_id": "admin"},
        )
        assert inputs_response.status_code == 200

        scenario_create = client.post(
            "/projects/demo/scenarios",
            json={
                "id": "perm_test",
                "description": "Permission controlled scenario",
                "modifications": {"S1": {"routing_model": "lag_long"}},
                "user_id": "modeler",
            },
        )
        assert scenario_create.status_code == 200

        queue_forbidden = client.get("/runs/queue")
        assert queue_forbidden.status_code == 403

        run_response = client.post(
            "/projects/demo/runs",
            json={
                "scenario_ids": ["perm_test"],
                "persist_outputs": False,
                "generate_report": False,
                "user_id": "admin",
            },
        )
        assert run_response.status_code == 200
        run_id = run_response.json()["id"]

        queue_response = client.get("/runs/queue", params={"user_id": "admin"})
        assert queue_response.status_code == 200
        entries = queue_response.json()["entries"]
        assert any(entry["run_id"] == run_id for entry in entries)

        users_list = client.get("/users")
        assert users_list.status_code == 200
        assert any(user["id"] == "admin" for user in users_list.json()["users"])
