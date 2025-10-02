"""Tests covering the high-level HydroSIS workflow orchestration helpers."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Dict, List
from unittest import mock

from hydrosis import HydroSISModel, run_workflow
from hydrosis.config import (
    ComparisonPlanConfig,
    EvaluationConfig,
    IOConfig,
    ModelConfig,
    ScenarioConfig,
)
from hydrosis.delineation.dem_delineator import DelineationConfig
from hydrosis.parameters.zone import ParameterZoneConfig
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig


def _build_workflow_config(results_directory: Path) -> ModelConfig:
    """Create a configuration exercising zoning, scenarios, and evaluation."""

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


class WorkflowIntegrationTests(unittest.TestCase):
    """Validate the high-level workflow runner across baseline and scenario runs."""

    def test_run_workflow_generates_outputs_and_evaluation(self) -> None:
        forcing: Dict[str, List[float]] = {
            "S1": [0.0, 10.0, 30.0, 0.0],
            "S2": [5.0, 5.0, 5.0, 5.0],
            "S3": [0.0, 0.0, 0.0, 0.0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_workflow_config(Path(tmpdir) / "results")

            baseline_model = HydroSISModel.from_config(config)
            baseline_local = baseline_model.run(forcing)
            observed = baseline_model.accumulate_discharge(baseline_local)

            result = run_workflow(
                config,
                forcing,
                observations=observed,
                scenario_ids=["alternate_routing"],
                persist_outputs=True,
                generate_report=True,
            )

            self.assertIn("alternate_routing", result.scenarios)
            self.assertIsNotNone(result.overall_scores)
            self.assertTrue(result.overall_scores)

            score_index = {score.model_id: score for score in result.overall_scores or []}
            self.assertIn("baseline", score_index)
            self.assertIn("alternate_routing", score_index)
            self.assertAlmostEqual(score_index["baseline"].aggregated["rmse"], 0.0)
            self.assertGreater(score_index["alternate_routing"].aggregated["rmse"], 0.0)

            self.assertTrue(result.evaluation_outcomes)
            outcome = result.evaluation_outcomes[0]
            self.assertEqual(outcome.plan.id, "baseline_vs_scenario")
            self.assertEqual(outcome.ranking[0].model_id, "baseline")

            baseline_output = (
                config.io.results_directory / "baseline" / "S3.csv"
            )
            scenario_output = (
                config.io.results_directory / "alternate_routing" / "S3.csv"
            )
            self.assertTrue(baseline_output.exists())
            self.assertTrue(scenario_output.exists())

            if result.report_path is not None:
                self.assertTrue(result.report_path.exists())

    def test_run_workflow_injects_qwen_via_environment(self) -> None:
        forcing: Dict[str, List[float]] = {
            "S1": [0.0, 1.0, 2.0],
            "S2": [0.0, 1.0, 2.0],
            "S3": [0.0, 1.0, 2.0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = _build_workflow_config(Path(tmpdir) / "results")
            baseline_model = HydroSISModel.from_config(config)
            observations = baseline_model.accumulate_discharge(baseline_model.run(forcing))

            with mock.patch(
                "hydrosis.reporting.markdown.qwen_narrative",
                return_value="LLM narrative",
            ) as mocked_qwen, mock.patch.dict(
                "os.environ",
                {"HYDROSIS_LLM_PROVIDER": "qwen", "DASHSCOPE_API_KEY": "token"},
                clear=False,
            ):
                result = run_workflow(
                    config,
                    forcing,
                    observations=observations,
                    scenario_ids=["alternate_routing"],
                    generate_report=True,
                )

            self.assertIsNotNone(result.report_path)
            self.assertTrue(result.report_path and result.report_path.exists())
            mocked_qwen.assert_called()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
