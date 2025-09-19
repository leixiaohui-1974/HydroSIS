"""Example-driven tests validating core HydroSIS functionality."""
from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import Dict, List

from hydrosis import HydroSISModel, ModelComparator, ModelConfig, SimulationEvaluator
from hydrosis.config import IOConfig, ScenarioConfig
from hydrosis.delineation.dem_delineator import DelineationConfig
from hydrosis.model import Subbasin
from hydrosis.parameters.zone import ParameterZoneBuilder, ParameterZoneConfig
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig


def _build_sample_config() -> ModelConfig:
    """Construct a minimal yet comprehensive configuration for examples."""

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
            description="Headwater zone controlled by gauge G1",
            control_points=["S1"],
            parameters={"runoff_model": "curve", "routing_model": "lag_short"},
        ),
        ParameterZoneConfig(
            id="Z2",
            description="Outlet control at station G2",
            control_points=["S3"],
            parameters={"runoff_model": "reservoir", "routing_model": "lag_short"},
        ),
    ]

    io_config = IOConfig(
        precipitation=Path("data/forcing/precipitation.csv"),
        results_directory=Path("results"),
    )

    scenarios = [
        ScenarioConfig(
            id="alternate_routing",
            description="Increase lag time for middle catchment",
            modifications={"S2": {"routing_model": "lag_long"}},
        )
    ]

    return ModelConfig(
        delineation=delineation,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=io_config,
        scenarios=scenarios,
    )


def _build_comparison_config() -> ModelConfig:
    """Configuration with multiple parameter zones for comparison tests."""

    delineation = DelineationConfig(
        dem_path=Path("dem.tif"),
        pour_points_path=Path("pour_points.geojson"),
        precomputed_subbasins=[
            {"id": "S1", "area_km2": 1.5, "downstream": "S3", "parameters": {}},
            {"id": "S2", "area_km2": 2.0, "downstream": "S3", "parameters": {}},
            {"id": "S3", "area_km2": 3.0, "downstream": "S4", "parameters": {}},
            {"id": "S4", "area_km2": 4.0, "downstream": None, "parameters": {}},
        ],
    )

    runoff_models = [
        RunoffModelConfig(
            id="headwater",
            model_type="scs_curve_number",
            parameters={"curve_number": 70, "initial_abstraction_ratio": 0.1},
        ),
        RunoffModelConfig(
            id="headwater_wet",
            model_type="scs_curve_number",
            parameters={"curve_number": 82, "initial_abstraction_ratio": 0.05},
        ),
        RunoffModelConfig(
            id="mid_storage",
            model_type="linear_reservoir",
            parameters={"recession": 0.82, "conversion": 0.9},
        ),
        RunoffModelConfig(
            id="lowland_storage",
            model_type="linear_reservoir",
            parameters={"recession": 0.9, "conversion": 0.75},
        ),
    ]

    routing_models = [
        RoutingModelConfig(id="lag_fast", model_type="lag", parameters={"lag_steps": 1}),
        RoutingModelConfig(id="lag_medium", model_type="lag", parameters={"lag_steps": 2}),
        RoutingModelConfig(id="lag_slow", model_type="lag", parameters={"lag_steps": 3}),
    ]

    parameter_zones = [
        ParameterZoneConfig(
            id="ZU",
            description="Upstream gauge",
            control_points=["S1"],
            parameters={"runoff_model": "headwater", "routing_model": "lag_fast"},
        ),
        ParameterZoneConfig(
            id="ZM",
            description="Mid catchment reservoir",
            control_points=["S2"],
            parameters={"runoff_model": "mid_storage", "routing_model": "lag_medium"},
        ),
        ParameterZoneConfig(
            id="ZD",
            description="Downstream station",
            control_points=["S4"],
            parameters={"runoff_model": "lowland_storage", "routing_model": "lag_medium"},
        ),
    ]

    io_config = IOConfig(
        precipitation=Path("data/forcing/precipitation.csv"),
        results_directory=Path("results"),
    )

    return ModelConfig(
        delineation=delineation,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=io_config,
    )


def _lag_route(values: List[float], lag_steps: int) -> List[float]:
    if lag_steps >= len(values):
        return [0.0] * lag_steps
    return [0.0] * lag_steps + values[:-lag_steps]


def _scs_runoff(precip: List[float], curve_number: float, ia_ratio: float) -> List[float]:
    s = max(0.0, (1000.0 / curve_number - 10.0) * 25.4)
    ia = ia_ratio * s
    runoff: List[float] = []
    for p in precip:
        if p <= ia:
            runoff.append(0.0)
        else:
            q = (p - ia) ** 2 / (p - ia + s)
            runoff.append(q)
    return runoff


def _linear_reservoir_runoff(
    precip: List[float], recession: float, conversion: float, initial_storage: float
) -> List[float]:
    state = initial_storage
    flows: List[float] = []
    for p in precip:
        state = state * recession + p * conversion
        direct = (1.0 - recession) * state
        flows.append(direct)
    return flows


class HydroSISExampleTests(unittest.TestCase):
    """High-level examples showcasing and validating HydroSIS behaviour."""

    def test_parameter_zone_assignment_resolves_upstream_overlap(self) -> None:
        """Zones assign exclusive downstream coverage for nested controllers."""

        subbasins = [
            Subbasin(id="S1", area_km2=10.0, downstream="S3"),
            Subbasin(id="S2", area_km2=12.0, downstream="S3"),
            Subbasin(id="S3", area_km2=20.0, downstream=None),
        ]

        configs = [
            ParameterZoneConfig(
                id="Z1",
                description="Upstream gauge",
                control_points=["S1"],
                parameters={"runoff_model": "curve", "routing_model": "lag_short"},
            ),
            ParameterZoneConfig(
                id="Z2",
                description="Downstream control",
                control_points=["S3"],
                parameters={"runoff_model": "reservoir", "routing_model": "lag_short"},
            ),
        ]

        zones = ParameterZoneBuilder.from_config(configs, subbasins)
        zone_map = {zone.id: list(zone.controlled_subbasins) for zone in zones}

        self.assertEqual(zone_map["Z1"], ["S1"])
        self.assertEqual(zone_map["Z2"], ["S2", "S3"])

    def test_model_run_matches_hand_calculated_results(self) -> None:
        """Full model run reproduces analytical expectations for simple inputs."""

        config = _build_sample_config()
        model = HydroSISModel.from_config(config)

        forcing: Dict[str, List[float]] = {
            "S1": [0.0, 20.0, 50.0],
            "S2": [5.0, 5.0, 5.0],
            "S3": [0.0, 0.0, 0.0],
        }

        routed = model.run(forcing)
        aggregated = model.accumulate_discharge(routed)

        expected_s1 = _lag_route(_scs_runoff(forcing["S1"], 75, 0.2), lag_steps=1)
        expected_s2 = _lag_route(_linear_reservoir_runoff(forcing["S2"], 0.85, 1.0, 0.0), 1)
        expected_s3 = _lag_route(_linear_reservoir_runoff(forcing["S3"], 0.85, 1.0, 0.0), 1)

        self.assertEqual(set(routed), {"S1", "S2", "S3"})

        for actual, expected in zip(routed["S1"], expected_s1):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))

        for actual, expected in zip(routed["S2"], expected_s2):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))

        for actual, expected in zip(routed["S3"], expected_s3):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))

        expected_total_s3 = [a + b + c for a, b, c in zip(expected_s1, expected_s2, expected_s3)]
        for actual, expected in zip(aggregated["S3"], expected_total_s3):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))

    def test_scenario_modification_updates_routing_choice(self) -> None:
        """Applying a scenario switches routing models and changes outputs."""

        forcing = {
            "S1": [0.0, 20.0, 50.0],
            "S2": [5.0, 5.0, 5.0],
            "S3": [0.0, 0.0, 0.0],
        }

        baseline_config = _build_sample_config()
        baseline_model = HydroSISModel.from_config(baseline_config)
        baseline_local = baseline_model.run(forcing)
        baseline_total = baseline_model.accumulate_discharge(baseline_local)

        scenario_config = _build_sample_config()
        scenario_model = HydroSISModel.from_config(scenario_config)
        scenario_config.apply_scenario("alternate_routing", scenario_model.subbasins.values())
        scenario_local = scenario_model.run(forcing)
        scenario_total = scenario_model.accumulate_discharge(scenario_local)

        expected_baseline_s2 = _lag_route(
            _linear_reservoir_runoff(forcing["S2"], 0.85, 1.0, 0.0), 1
        )
        expected_scenario_s2 = _lag_route(
            _linear_reservoir_runoff(forcing["S2"], 0.85, 1.0, 0.0), 2
        )

        for actual, expected in zip(baseline_local["S2"], expected_baseline_s2):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))

        for actual, expected in zip(scenario_local["S2"], expected_scenario_s2):
            self.assertTrue(math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9))

        self.assertNotEqual(baseline_local["S2"], scenario_local["S2"])

        zone_baseline = baseline_model.parameter_zone_discharge(baseline_local)
        zone_scenario = scenario_model.parameter_zone_discharge(scenario_local)
        self.assertIn("Z1", zone_baseline)
        self.assertIn("Z1", zone_scenario)
        self.assertIn("S1", zone_baseline["Z1"])
        self.assertIn("S1", zone_scenario["Z1"])
        self.assertEqual(zone_baseline["Z1"]["S1"], baseline_total["S1"])
        self.assertEqual(zone_scenario["Z1"]["S1"], scenario_total["S1"])


    def test_extended_runoff_models_are_buildable(self) -> None:
        """Ensure newly supported runoff models can be instantiated uniformly."""

        subbasin = Subbasin(id="TEST", area_km2=5.0, downstream=None)
        precipitation = [10.0, 0.0, 5.0]

        configs = [
            (
                "xin_an_jiang",
                {
                    "wm": 120.0,
                    "b": 0.4,
                    "imp": 0.02,
                    "recession": 0.7,
                },
            ),
            (
                "wetspa",
                {
                    "soil_storage_max": 250.0,
                    "infiltration_coefficient": 0.5,
                    "surface_runoff_coefficient": 0.35,
                },
            ),
            (
                "hymod",
                {
                    "max_storage": 90.0,
                    "beta": 1.2,
                    "quickflow_ratio": 0.6,
                    "num_quick_reservoirs": 3,
                },
            ),
        ]

        for idx, (model_type, parameters) in enumerate(configs):
            config = RunoffModelConfig(
                id=f"model_{idx}", model_type=model_type, parameters=parameters
            )
            model = config.build()
            flows = model.simulate(subbasin, precipitation)
            self.assertEqual(len(flows), len(precipitation))
            self.assertTrue(all(math.isfinite(flow) for flow in flows))

    def test_multi_model_comparison_identifies_best_performer(self) -> None:
        """Model comparator ranks multi-zone simulations by accuracy metrics."""

        config = _build_comparison_config()
        truth_model = HydroSISModel.from_config(config)

        forcing: Dict[str, List[float]] = {
            "S1": [5.0, 10.0, 20.0, 0.0],
            "S2": [0.0, 15.0, 10.0, 5.0],
            "S3": [0.0, 0.0, 5.0, 0.0],
            "S4": [1.0, 0.0, 2.0, 0.0],
        }

        observations = truth_model.accumulate_discharge(truth_model.run(forcing))

        calibrated_config = _build_comparison_config()
        calibrated_model = HydroSISModel.from_config(calibrated_config)
        calibrated_results = calibrated_model.accumulate_discharge(
            calibrated_model.run(forcing)
        )

        biased_config = _build_comparison_config()
        for runoff_cfg in biased_config.runoff_models:
            if runoff_cfg.id == "headwater":
                runoff_cfg.parameters["curve_number"] = 88
        biased_model = HydroSISModel.from_config(biased_config)
        biased_results = biased_model.accumulate_discharge(biased_model.run(forcing))

        sluggish_config = _build_comparison_config()
        for routing_cfg in sluggish_config.routing_models:
            if routing_cfg.id == "lag_medium":
                routing_cfg.parameters["lag_steps"] = 4
        sluggish_model = HydroSISModel.from_config(sluggish_config)
        sluggish_results = sluggish_model.accumulate_discharge(
            sluggish_model.run(forcing)
        )


        simulations = {
            "calibrated": calibrated_results,
            "biased": biased_results,
            "sluggish": sluggish_results,
        }

        comparator = ModelComparator(SimulationEvaluator())
        scores = comparator.compare(simulations, observations)
        ranking = comparator.rank(scores, metric="rmse")

        self.assertEqual(ranking[0].model_id, "calibrated")
        self.assertEqual(ranking[-1].model_id, "biased")

        aggregated = {score.model_id: score.aggregated for score in scores}
        self.assertLess(aggregated["calibrated"]["rmse"], aggregated["sluggish"]["rmse"])
        self.assertLess(abs(aggregated["calibrated"]["pbias"]), abs(aggregated["biased"]["pbias"]))



if __name__ == "__main__":  # pragma: no cover - allow direct execution
    unittest.main()
