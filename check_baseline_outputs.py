#!/usr/bin/env python3
"""Check what's in baseline.local vs baseline.aggregated"""
import sys
import pandas as pd
from pathlib import Path

# Load the saved files
runoff_file = Path("results/upper_truckee_complete_11steps/step_09_runoff/9.1_runoff_timeseries.csv")
discharge_file = Path("results/upper_truckee_complete_11steps/step_10_routing/10.1_discharge_timeseries.csv")

runoff_df = pd.read_csv(runoff_file, index_col=0)
discharge_df = pd.read_csv(discharge_file, index_col=0)

print("="*80)
print("Comparing runoff (local) vs discharge (aggregated)")
print("="*80)

print("\nRunoff file columns (first 10):", runoff_df.columns[:10].tolist())
print("Discharge file columns (first 10):", discharge_df.columns[:10].tolist())

print("\nRunoff file shape:", runoff_df.shape)
print("Discharge file shape:", discharge_df.shape)

# Check if they're identical
print("\n" + "="*80)
print("Checking if files are identical...")
print("="*80)

# Compare a few subbasins
for col in ['222', '231', '101']:
    if col in runoff_df.columns and int(col) in discharge_df.columns:
        runoff_vals = runoff_df[col].values[:5]
        discharge_vals = discharge_df[int(col)].values[:5]

        print(f"\nSubbasin {col}:")
        print(f"  Runoff (local):    {runoff_vals}")
        print(f"  Discharge (aggr):  {discharge_vals}")
        print(f"  Are identical? {(runoff_vals == discharge_vals).all()}")
