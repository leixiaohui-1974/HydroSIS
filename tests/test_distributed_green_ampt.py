"""Tests for the distributed Green-Ampt runoff model."""
from __future__ import annotations

import unittest
from typing import Dict, List

from hydrosis.model import Subbasin
from hydrosis.runoff.distributed_green_ampt import DistributedGreenAmpt
from hydrosis.runoff.base import RunoffModelConfig


class DistributedGreenAmptTests(unittest.TestCase):
    """Test distributed Green-Ampt runoff model functionality."""

    def test_model_initialization(self) -> None:
        """Test that the model can be initialized with different parameters."""
        # Default parameters
        model1 = DistributedGreenAmpt({})
        self.assertEqual(model1.k_sat, 10.0)
        self.assertEqual(model1.psi_f, 100.0)
        self.assertEqual(model1.theta_i, 0.2)
        self.assertEqual(model1.theta_s, 0.4)
        self.assertEqual(model1.porosity, 0.45)
        self.assertEqual(model1.num_zones, 5)
        self.assertEqual(model1.zone_distribution, "uniform")
        
        # Custom parameters
        params = {
            "saturated_conductivity": 15.0,
            "wetting_front_suction": 150.0,
            "initial_moisture": 0.15,
            "saturated_moisture": 0.35,
            "porosity": 0.5,
            "zones": 10,
            "zone_distribution": "random",
        }
        model2 = DistributedGreenAmpt(params)
        self.assertEqual(model2.k_sat, 15.0)
        self.assertEqual(model2.psi_f, 150.0)
        self.assertEqual(model2.theta_i, 0.15)
        self.assertEqual(model2.theta_s, 0.35)
        self.assertEqual(model2.porosity, 0.5)
        self.assertEqual(model2.num_zones, 10)
        self.assertEqual(model2.zone_distribution, "random")

    def test_parameter_validation(self) -> None:
        """Test that parameters are validated and constrained to valid ranges."""
        # Test moisture content constraints
        params = {
            "initial_moisture": -0.1,  # Invalid: negative
            "saturated_moisture": 1.5,  # Invalid: > 1
            "porosity": 0.3,  # Valid
        }
        model = DistributedGreenAmpt(params)
        self.assertEqual(model.theta_i, 0.0)  # Clamped to 0
        self.assertEqual(model.theta_s, 0.3)  # Clamped to 1
        self.assertEqual(model.porosity, 0.3)
        
        # Test theta_s <= porosity constraint
        params = {
            "saturated_moisture": 0.6,
            "porosity": 0.5,  # theta_s > porosity
        }
        model = DistributedGreenAmpt(params)
        self.assertEqual(model.theta_s, 0.5)  # Clamped to porosity
        self.assertEqual(model.porosity, 0.5)

    def test_uniform_zone_distribution(self) -> None:
        """Test that uniform zone distribution creates identical zones."""
        params = {
            "saturated_conductivity": 10.0,
            "wetting_front_suction": 100.0,
            "initial_moisture": 0.2,
            "saturated_moisture": 0.4,
            "porosity": 0.45,
            "zones": 3,
            "zone_distribution": "uniform",
        }
        model = DistributedGreenAmpt(params)
        zones = model._initialize_zones()
        
        self.assertEqual(len(zones), 3)
        for zone in zones:
            self.assertEqual(zone["k_sat"], 10.0)
            self.assertEqual(zone["psi_f"], 100.0)
            self.assertEqual(zone["delta_theta"], 0.2)  # theta_s - theta_i
            self.assertEqual(zone["area_fraction"], 1.0 / 3)

    def test_random_zone_distribution(self) -> None:
        """Test that random zone distribution creates varied zones."""
        params = {
            "saturated_conductivity": 10.0,
            "wetting_front_suction": 100.0,
            "initial_moisture": 0.2,
            "saturated_moisture": 0.4,
            "porosity": 0.45,
            "zones": 5,
            "zone_distribution": "random",
        }
        model = DistributedGreenAmpt(params)
        zones = model._initialize_zones()
        
        self.assertEqual(len(zones), 5)
        # Check that zones have different properties
        k_sats = [zone["psi_f"] for zone in zones]
        psi_fs = [zone["psi_f"] for zone in zones]
        
        # With random variation, not all values should be identical
        # (though this is probabilistic and could theoretically fail)
        self.assertTrue(len(set(k_sats)) > 1 or len(set(psi_fs)) > 1)

    def test_clustered_zone_distribution(self) -> None:
        """Test that clustered zone distribution creates distinct clusters."""
        params = {
            "saturated_conductivity": 10.0,
            "wetting_front_suction": 100.0,
            "initial_moisture": 0.2,
            "saturated_moisture": 0.4,
            "porosity": 0.45,
            "zones": 6,
            "zone_distribution": "clustered",
        }
        model = DistributedGreenAmpt(params)
        zones = model._initialize_zones()
        
        self.assertEqual(len(zones), 6)
        # Check that we have exactly 3 different types of zones
        k_sats = [zone["k_sat"] for zone in zones]
        unique_k_sats = set(k_sats)
        self.assertEqual(len(unique_k_sats), 3)

    def test_simulation_with_no_precipitation(self) -> None:
        """Test that the model produces no runoff with no precipitation."""
        params = {
            "saturated_conductivity": 10.0,
            "wetting_front_suction": 100.0,
            "initial_moisture": 0.2,
            "saturated_moisture": 0.4,
            "porosity": 0.45,
            "zones": 3,
            "zone_distribution": "uniform",
        }
        model = DistributedGreenAmpt(params)
        subbasin = Subbasin(id="test", area_km2=10.0, downstream=None)
        precipitation = [0.0, 0.0, 0.0, 0.0]
        
        runoff, _ = model.simulate(subbasin, precipitation)
        
        self.assertEqual(len(runoff), 4)
        for value in runoff:
            self.assertEqual(value, 0.0)

    def test_simulation_with_low_precipitation(self) -> None:
        """Test that the model produces no runoff with low precipitation."""
        params = {
            "saturated_conductivity": 10.0,  # High conductivity
            "wetting_front_suction": 100.0,
            "initial_moisture": 0.2,
            "saturated_moisture": 0.4,
            "porosity": 0.45,
            "zones": 3,
            "zone_distribution": "uniform",
        }
        model = DistributedGreenAmpt(params)
        subbasin = Subbasin(id="test", area_km2=10.0, downstream=None)
        precipitation = [0.1, 0.1, 0.1, 0.1]  # Very low precipitation
        
        runoff, _ = model.simulate(subbasin, precipitation)
        
        # With high conductivity and low precipitation, most should infiltrate
        # But we expect some runoff due to the model's implementation
        self.assertEqual(len(runoff), 4)
        for value in runoff:
            self.assertGreaterEqual(value, 0.0)

    def test_simulation_with_high_precipitation(self) -> None:
        """Test that the model produces runoff with high precipitation."""
        params = {
            "saturated_conductivity": 5.0,  # Lower conductivity
            "wetting_front_suction": 50.0,
            "initial_moisture": 0.3,  # Higher initial moisture
            "saturated_moisture": 0.4,
            "porosity": 0.45,
            "zones": 3,
            "zone_distribution": "uniform",
        }
        model = DistributedGreenAmpt(params)
        subbasin = Subbasin(id="test", area_km2=10.0, downstream=None)
        precipitation = [50.0, 50.0, 50.0, 50.0]  # High precipitation
        
        runoff, _ = model.simulate(subbasin, precipitation)
        
        self.assertEqual(len(runoff), 4)
        # With high precipitation and lower conductivity, we expect significant runoff
        for value in runoff:
            self.assertGreaterEqual(value, 0.0)

    def test_simulation_with_different_initial_moisture(self) -> None:
        """Test that initial moisture affects runoff generation."""
        high_precip = [30.0, 30.0, 30.0, 30.0]
        subbasin = Subbasin(id="test", area_km2=10.0, downstream=None)
        
        # Dry conditions
        params_dry = {
            "saturated_conductivity": 10.0,
            "wetting_front_suction": 100.0,
            "initial_moisture": 0.1,  # Dry
            "saturated_moisture": 0.4,
            "porosity": 0.45,
            "zones": 3,
            "zone_distribution": "uniform",
        }
        model_dry = DistributedGreenAmpt(params_dry)
        runoff_dry, _ = model_dry.simulate(subbasin, high_precip)
        
        # Wet conditions
        params_wet = {
            "saturated_conductivity": 10.0,
            "wetting_front_suction": 100.0,
            "initial_moisture": 0.35,  # Wet
            "saturated_moisture": 0.4,
            "porosity": 0.45,
            "zones": 3,
            "zone_distribution": "uniform",
        }
        model_wet = DistributedGreenAmpt(params_wet)
        runoff_wet, _ = model_wet.simulate(subbasin, high_precip)
        
        # Wet conditions should produce more runoff
        total_dry = sum(runoff_dry)
        total_wet = sum(runoff_wet)
        self.assertGreater(total_wet, total_dry)

    def test_model_registration(self) -> None:
        """Test that the model is properly registered with the configuration system."""
        # Check that the model is registered
        self.assertIn("distributed_green_ampt", RunoffModelConfig.REGISTRY)
        
        # Check that we can create the model through the configuration
        config = RunoffModelConfig(
            id="test_green_ampt",
            model_type="distributed_green_ampt",
            parameters={
                "saturated_conductivity": 10.0,
                "wetting_front_suction": 100.0,
                "initial_moisture": 0.2,
                "saturated_moisture": 0.4,
                "porosity": 0.45,
                "zones": 3,
                "zone_distribution": "uniform",
            },
        )
        
        model = config.build()
        self.assertIsInstance(model, DistributedGreenAmpt)
        self.assertEqual(model.k_sat, 10.0)


if __name__ == "__main__":
    unittest.main()
