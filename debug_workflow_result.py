#!/usr/bin/env python3
"""Debug what's actually in workflow_result.baseline"""
import numpy as np
import pandas as pd
from pathlib import Path

# Re-run a minimal simulation to check the outputs
from hydrosis.config import (
    DelineationConfig, IOConfig, ModelConfig,
    RunoffModelConfig, RoutingModelConfig,
    ParameterZoneConfig, EvaluationConfig
)
from hydrosis.workflow import run_workflow

print("="*80)
print("Debugging workflow_result structure")
print("="*80)

# Minimal test: 2 subbasins, 10 timesteps
print("\n1. Setting up minimal test configuration...")

# Load real parameter zones
results_dir = Path("results/upper_truckee_complete_11steps")
parameter_dir = results_dir / "parameters"

# Load subbasins
subbasin_csv = parameter_dir / "parameter_subbasins.csv"
subbasin_df = pd.read_csv(subbasin_csv)

# Just use first 2 subbasins
test_subbasins = subbasin_df.head(2).copy()
print(f"  Test subbasins: {test_subbasins['subzone_id'].tolist()}")

# Create test configuration
runoff_model = RunoffModelConfig(
    id="hbv",
    model_type="hbv",
    parameters={
        "FC": 100.0,
        "BETA": 1.0,
        "K0": 0.15,
        "K1": 0.05,
        "K2": 0.01,
        "PERC": 1.0,
    }
)

routing_model = RoutingModelConfig(
    id="muskingum",
    model_type="muskingum",
    parameters={"K": 1.0, "x": 0.2}
)

# Build subbasin list
subbasin_list = []
for _, row in test_subbasins.iterrows():
    downstream_val = row.get('downstream_subzone_id', '')
    subbasin_list.append({
        "id": str(row['subzone_id']),
        "area_km2": float(row['area_km2']),
        "downstream": str(downstream_val) if pd.notna(downstream_val) and downstream_val else None,
        "parameters": {"runoff_model": "hbv", "routing_model": "muskingum"},
    })

delineation_cfg = DelineationConfig(subbasins=subbasin_list)

parameter_zones = [
    ParameterZoneConfig(
        zone_id="1",
        subbasins=[str(sb['id']) for sb in subbasin_list],
        control_points=[],
        parameters={"runoff_model": "hbv", "routing_model": "muskingum"}
    )
]

# Dummy precipitation
forcing = {
    str(sb['id']): [10.0, 5.0, 2.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
    for sb in subbasin_list
}

io_config = IOConfig(results_directory=results_dir)
evaluation_config = EvaluationConfig(metrics=["rmse"])

model_config = ModelConfig(
    delineation=delineation_cfg,
    runoff_models=[runoff_model],
    routing_models=[routing_model],
    parameter_zones=parameter_zones,
    io=io_config,
    evaluation=evaluation_config,
)

print("\n2. Running workflow...")
workflow_result = run_workflow(
    model_config,
    forcing,
    observations={},
    persist_outputs=False,
)

print("\n3. Checking workflow_result.baseline...")
baseline = workflow_result.baseline

print(f"\n  baseline.local type: {type(baseline.local)}")
print(f"  baseline.aggregated type: {type(baseline.aggregated)}")

print(f"\n  baseline.local keys: {list(baseline.local.keys())}")
print(f"  baseline.aggregated keys: {list(baseline.aggregated.keys())}")

# Check first subbasin
first_sb = list(baseline.local.keys())[0]
print(f"\n  Subbasin {first_sb}:")
print(f"    local (runoff):     {baseline.local[first_sb][:5]}")
print(f"    aggregated (discharge): {baseline.aggregated[first_sb][:5]}")

# Check if they're identical
local_arr = np.array(baseline.local[first_sb])
aggr_arr = np.array(baseline.aggregated[first_sb])
print(f"    Are identical? {np.allclose(local_arr, aggr_arr)}")

# If not identical, show difference
if not np.allclose(local_arr, aggr_arr):
    print(f"    Difference: {aggr_arr[:5] - local_arr[:5]}")
else:
    print(f"    ⚠ WARNING: local and aggregated are IDENTICAL!")
    print(f"    This means routing is not working or data is not being separated correctly.")
