import tempfile
import unittest
from pathlib import Path

from hydrosis.model import Subbasin
from hydrosis.parameters import (
    ObjectiveDefinition,
    ParameterZone,
    ParameterZoneOptimizer,
    UncertaintyAnalyzer,
)
from hydrosis.reporting import generate_evaluation_report
from hydrosis.evaluation import ModelScore, SimulationEvaluator
from hydrosis.runoff.hbv import HBVRunoff
from hydrosis.runoff.vic import VICRunoff
from hydrosis.routing.dynamic_wave import DynamicWaveRouting


class AdvancedCapabilityTests(unittest.TestCase):
    def test_vic_and_hbv_runoff_generation(self) -> None:
        subbasin = Subbasin(id="S1", area_km2=1.5, downstream=None, parameters={})
        vic = VICRunoff({
            "infiltration_shape": 0.4,
            "max_soil_moisture": 200,
            "baseflow_coefficient": 0.02,
            "recession": 0.9,
        })
        hbv = HBVRunoff({
            "degree_day_factor": 3.0,
            "field_capacity": 110,
            "beta": 1.1,
            "k0": 0.25,
            "k1": 0.1,
            "k2": 0.02,
            "percolation": 2.0,
        })

        precipitation = [10.0, 0.0, 25.0, 5.0, 0.0]
        vic_flow = vic.simulate(subbasin, precipitation)
        hbv_flow = hbv.simulate(subbasin, precipitation)

        self.assertEqual(len(vic_flow), len(precipitation))
        self.assertEqual(len(hbv_flow), len(precipitation))
        self.assertTrue(all(value >= 0.0 for value in vic_flow))
        self.assertTrue(all(value >= 0.0 for value in hbv_flow))

    def test_dynamic_wave_routing_produces_smoother_series(self) -> None:
        routing = DynamicWaveRouting({
            "time_step": 1.0,
            "reach_length": 10.0,
            "wave_celerity": 2.0,
            "diffusivity": 0.1,
        })
        subbasin = Subbasin(id="S1", area_km2=1.0, downstream=None, parameters={})
        inflow = [0.0, 10.0, 20.0, 5.0, 0.0]
        outflow = routing.route(subbasin, inflow)

        self.assertEqual(len(outflow), len(inflow))
        self.assertGreater(outflow[2], outflow[1])
        self.assertLess(outflow[3], outflow[2])

    def test_parameter_zone_optimizer_and_uncertainty(self) -> None:
        zone = ParameterZone(
            id="Z1",
            description="Test zone",
            controllers=["S1"],
            controlled_subbasins=["S1"],
            parameters={"scale": 0.2},
        )

        target = 0.05

        def evaluation(candidate):
            scale = candidate["Z1"]["scale"]
            error = abs(scale - target)
            return {"rmse": error, "nse": 1.0 - error}

        def cyclic_sampler(values):
            index = {"value": 0}

            def _sampler(_zone):
                value = values[index["value"] % len(values)]
                index["value"] += 1
                yield {"scale": value}

            return _sampler

        optimizer = ParameterZoneOptimizer(
            [zone],
            evaluation,
            [ObjectiveDefinition(id="rmse", weight=1.0, sense="min")],
        )
        samplers = {"Z1": cyclic_sampler([0.2, 0.05, 0.1])}
        result = optimizer.optimise(samplers, max_iterations=3)

        self.assertAlmostEqual(result.best_parameters["Z1"]["scale"], 0.05, places=6)
        self.assertLessEqual(result.objective_scores["rmse"], 0.05)

        analyzer = UncertaintyAnalyzer([zone], evaluation)
        summary = analyzer.analyse({"Z1": cyclic_sampler([0.04, 0.06])}, draws=2)
        self.assertIn("nse", summary)
        self.assertAlmostEqual(summary["nse"]["mean"], 0.99, places=2)

    def test_markdown_report_template_integration(self) -> None:
        evaluator = SimulationEvaluator(metrics={"rmse": lambda a, b: 0.1}, orientations={"rmse": "min"})
        scores = [ModelScore(model_id="baseline", per_subbasin={}, aggregated={"rmse": 0.1})]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.md"
            path = generate_evaluation_report(
                output_path,
                scores,
                evaluator,
                description="测试报告",
                figures_directory=Path(tmpdir) / "figures",
                template_context={"模型运行概述": "概述文本"},
                narrative_callback=lambda prompt: f"LLM:{prompt}",
            )
            content = path.read_text(encoding="utf-8")
        self.assertIn("模型运行概述", content)
        self.assertIn("LLM", content)


if __name__ == "__main__":
    unittest.main()
