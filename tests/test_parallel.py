"""Tests for the parallel computing capabilities of HydroSIS."""
from __future__ import annotations

import unittest
from typing import Dict, List

from hydrosis import HydroSISModel, ParallelConfig, ParallelHydroSISModel
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig
from hydrosis.parameters.zone import ParameterZoneBuilder, ParameterZoneConfig
from hydrosis.config import ModelConfig, DelineationConfig, IOConfig, ScenarioConfig
from pathlib import Path

class ParallelComputingTests(unittest.TestCase):
    """Test parallel computing functionality."""

    def setUp(self) -> None:
        """Set up a reusable model configuration for tests."""
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

        self.config = ModelConfig(
            delineation=delineation,
            runoff_models=runoff_models,
            routing_models=routing_models,
            parameter_zones=parameter_zones,
            io=io_config,
            scenarios=scenarios,
        )

    def test_parallel_config_initialization(self) -> None:
        """Test that ParallelConfig can be initialized with different parameters."""
        # Default configuration
        config1 = ParallelConfig()
        self.assertIsNone(config1.max_workers)
        self.assertTrue(config1.use_processes)
        self.assertEqual(config1.chunk_size, 1)
        self.assertFalse(config1.enable_progress)

        # Custom configuration
        config2 = ParallelConfig(
            max_workers=4,
            use_processes=False,
            chunk_size=2,
            enable_progress=True,
        )
        self.assertEqual(config2.max_workers, 4)
        self.assertFalse(config2.use_processes)
        self.assertEqual(config2.chunk_size, 2)
        self.assertTrue(config2.enable_progress)

    def test_parallel_model_creation(self) -> None:
        """Test that ParallelHydroSISModel can be created from configuration."""
        # Create with default parallel config
        parallel_model = ParallelHydroSISModel.from_config(self.config)
        self.assertIsInstance(parallel_model, ParallelHydroSISModel)
        self.assertIsNotNone(parallel_model.parallel_config)
        
        # Create with custom parallel config
        parallel_config = ParallelConfig(max_workers=2, use_processes=True)
        parallel_model = ParallelHydroSISModel.from_config(self.config, parallel_config)
        self.assertEqual(parallel_model.parallel_config.max_workers, 2)
        self.assertTrue(parallel_model.parallel_config.use_processes)

    def test_parallel_vs_sequential_results(self) -> None:
        """Test that parallel and sequential execution produce identical results."""
        # Create models
        sequential_model = HydroSISModel.from_config(self.config)
        parallel_config = ParallelConfig(max_workers=2, use_processes=False)  # Use threads for testing
        parallel_model = ParallelHydroSISModel.from_config(self.config, parallel_config)
        
        # Test data
        forcing: Dict[str, List[float]] = {
            "S1": [0.0, 10.0, 20.0, 5.0],
            "S2": [5.0, 5.0, 5.0, 5.0],
            "S3": [0.0, 0.0, 0.0, 0.0],
        }
        
        # Run simulations
        sequential_result, _ = sequential_model.run(forcing)
        parallel_result, _ = parallel_model.run(forcing)
        
        # Compare results
        self.assertEqual(set(sequential_result.keys()), set(parallel_result.keys()))
        
        for sub_id in sequential_result:
            self.assertEqual(len(sequential_result[sub_id]), len(parallel_result[sub_id]))
            for i in range(len(sequential_result[sub_id])):
                self.assertAlmostEqual(
                    sequential_result[sub_id][i],
                    parallel_result[sub_id][i],
                    places=6,
                    msg=f"Mismatch at {sub_id}[{i}]"
                )

    def test_parallel_accumulation(self) -> None:
        """Test that accumulation works correctly with parallel execution."""
        parallel_config = ParallelConfig(max_workers=2, use_processes=False)
        parallel_model = ParallelHydroSISModel.from_config(self.config, parallel_config)
        
        # Test data
        forcing: Dict[str, List[float]] = {
            "S1": [0.0, 10.0, 20.0, 5.0],
            "S2": [5.0, 5.0, 5.0, 5.0],
            "S3": [0.0, 0.0, 0.0, 0.0],
        }
        
        # Run simulation
        local_result, _ = parallel_model.run(forcing)
        accumulated_result = parallel_model.accumulate_discharge(local_result)
        
        # Verify accumulation
        self.assertIn("S3", accumulated_result)
        # S3 should have contributions from S1 and S2
        for i in range(4):
            expected = local_result["S1"][i] + local_result["S2"][i] + local_result["S3"][i]
            self.assertAlmostEqual(accumulated_result["S3"][i], expected, places=6)

    def test_parallel_parameter_zones(self) -> None:
        """Test that parameter zones work correctly with parallel execution."""
        parallel_config = ParallelConfig(max_workers=2, use_processes=False)
        parallel_model = ParallelHydroSISModel.from_config(self.config, parallel_config)
        
        # Test data
        forcing: Dict[str, List[float]] = {
            "S1": [0.0, 10.0, 20.0, 5.0],
            "S2": [5.0, 5.0, 5.0, 5.0],
            "S3": [0.0, 0.0, 0.0, 0.0],
        }
        
        # Run simulation
        local_result, _ = parallel_model.run(forcing)
        zone_discharge = parallel_model.parameter_zone_discharge(local_result)
        
        # Verify zone discharge
        self.assertIn("Z1", zone_discharge)
        self.assertIn("Z2", zone_discharge)
        self.assertIn("S1", zone_discharge["Z1"])
        self.assertIn("S3", zone_discharge["Z2"])


if __name__ == "__main__":
    unittest.main()
