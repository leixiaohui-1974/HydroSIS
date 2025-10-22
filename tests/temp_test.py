import math
from pathlib import Path
from typing import Dict, List

from hydrosis.config import (
    DelineationConfig,
    ModelConfig,
    RunoffModelConfig,
    RoutingModelConfig,
    ParameterZoneConfig,
    IOConfig,
    ScenarioConfig,
)
from hydrosis.model import HydroSISModel, Subbasin
from hydrosis.testing.example_documenter import (
    build_sample_config,
    lag_route,
    scs_runoff,
    linear_reservoir_runoff,
)

def _hand_calculated_run_example():
    config = build_sample_config()
    model = HydroSISModel.from_config(config)

    forcing: Dict[str, List[float]] = {
        "S1": [0.0, 20.0, 50.0],
        "S2": [5.0, 5.0, 5.0],
        "S3": [0.0, 0.0, 0.0],
    }

    routed = model.run(forcing)
    aggregated = model.accumulate_discharge(routed)

    expected_s1 = lag_route(scs_runoff(forcing["S1"], 75, 0.2), lag_steps=1)
    expected_s2 = lag_route(linear_reservoir_runoff(forcing["S2"], 0.85, 1.0, 0.0), 1)
    expected_s3 = lag_route(linear_reservoir_runoff(forcing["S3"], 0.85, 1.0, 0.0), 1)

    print(f"Routed S1: {routed['S1']}")
    print(f"Expected S1: {expected_s1}")
    print(f"Routed S2: {routed['S2']}")
    print(f"Expected S2: {expected_s2}")
    print(f"Routed S3: {routed['S3']}")
    print(f"Expected S3: {expected_s3}")
    print(f"Aggregated S3: {aggregated['S3']}")
    expected_total_s3 = [a + b + c for a, b, c in zip(expected_s1, expected_s2, expected_s3)]
    print(f"Expected Total S3: {expected_total_s3}")

    for actual, expected in zip(routed["S1"], expected_s1):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("S1 routing does not match analytical expectation")

    for actual, expected in zip(routed["S2"], expected_s2):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("S2 routing does not match analytical expectation")

    for actual, expected in zip(aggregated["S3"], expected_total_s3):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("Aggregated discharge does not match combined expectation")

if __name__ == "__main__":
    _hand_calculated_run_example()