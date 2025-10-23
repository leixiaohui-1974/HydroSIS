#!/usr/bin/env python3
"""Debug script to trace runoff coefficient calculation"""
import pandas as pd
import numpy as np
from pathlib import Path

# Load data
print("="*80)
print("DEBUG: Runoff Coefficient Calculation")
print("="*80)

# Load precipitation
precip_file = Path("results/upper_truckee_complete_11steps/step_08_areal_rainfall/8.2_subbasin_areal_precipitation.csv")
precip_df = pd.read_csv(precip_file, index_col=0)
precip_df.columns = precip_df.columns.astype(str)

# Load discharge
discharge_file = Path("results/upper_truckee_complete_11steps/step_10_routing/10.1_discharge_timeseries.csv")
discharge_df = pd.read_csv(discharge_file, index_col=0)
discharge_df.columns = discharge_df.columns.astype(int)

# Load subbasin info
subbasin_df = pd.read_csv('results/upper_truckee_complete_11steps/parameters/parameter_subbasins.csv')
subbasin_df['zone_id'] = subbasin_df['subzone_id'].astype(str).str[0].astype(int)

# Focus on Zone 1
zone_id = 1
print(f"\n{'='*80}")
print(f"Analyzing Zone {zone_id}")
print(f"{'='*80}")

# Get subbasins for Zone 1
zone_subbasins = subbasin_df[subbasin_df['zone_id'] == zone_id]
print(f"\nSubbasins in Zone {zone_id}:")
print(zone_subbasins[['subzone_id', 'area_km2']])

subbasin_ids_str = zone_subbasins['subzone_id'].astype(str).tolist()
subbasin_ids_int = zone_subbasins['subzone_id'].astype(int).tolist()
zone_area = zone_subbasins['area_km2'].sum()

print(f"\nZone {zone_id} total area: {zone_area:.2f} km²")
print(f"Subbasin IDs (str): {subbasin_ids_str}")
print(f"Subbasin IDs (int): {subbasin_ids_int}")

# Calculate precipitation (area-weighted average)
print(f"\n{'='*80}")
print("PRECIPITATION CALCULATION")
print(f"{'='*80}")

total_area = 0
weighted_precip = pd.Series(0, index=precip_df.index)

for sid_str in subbasin_ids_str:
    if sid_str in precip_df.columns:
        area = zone_subbasins[zone_subbasins['subzone_id'].astype(str) == sid_str]['area_km2'].values[0]
        print(f"  Subbasin {sid_str}: area={area:.2f} km², precip[0]={precip_df[sid_str].iloc[0]:.3f} mm/hr")
        weighted_precip += precip_df[sid_str] * area
        total_area += area

precip_avg = weighted_precip / total_area

print(f"\nArea-weighted average precipitation:")
print(f"  First 5 timesteps: {precip_avg.iloc[:5].values}")
print(f"  Total precip depth: {precip_avg.sum() * 1.0:.2f} mm")  # 1 hour timestep

# Calculate discharge (sum of all subbasins)
print(f"\n{'='*80}")
print("DISCHARGE CALCULATION")
print(f"{'='*80}")

available_discharge_ids = [sid for sid in subbasin_ids_int if sid in discharge_df.columns]
print(f"Available discharge subbasins: {available_discharge_ids}")

for sid_int in available_discharge_ids[:3]:  # Show first 3
    print(f"  Subbasin {sid_int}: discharge[0]={discharge_df[sid_int].iloc[0]:.3f} m³/s")

discharge_sum = discharge_df[available_discharge_ids].sum(axis=1)

print(f"\nTotal discharge (sum of subbasins):")
print(f"  First 5 timesteps: {discharge_sum.iloc[:5].values}")
print(f"  Mean discharge: {discharge_sum.mean():.2f} m³/s")

# Calculate runoff coefficient
print(f"\n{'='*80}")
print("RUNOFF COEFFICIENT CALCULATION")
print(f"{'='*80}")

dt_hours = 1

# Total precipitation depth (mm)
total_precip_depth = precip_avg.sum() * dt_hours
print(f"\nTotal precipitation depth: {total_precip_depth:.2f} mm")

# Total discharge volume (m³)
total_discharge_volume_m3 = discharge_sum.sum() * dt_hours * 3600
print(f"Total discharge volume: {total_discharge_volume_m3:.2f} m³")

# Total discharge depth (mm)
total_discharge_depth_mm = (total_discharge_volume_m3 / (zone_area * 1e6)) * 1000
print(f"Total discharge depth: {total_discharge_depth_mm:.2f} mm")
print(f"  (volume {total_discharge_volume_m3:.2f} m³ / area {zone_area * 1e6:.0f} m²) * 1000")

# Runoff coefficient
runoff_coeff = total_discharge_depth_mm / total_precip_depth
print(f"\nRunoff coefficient: {runoff_coeff:.3f}")
print(f"  = {total_discharge_depth_mm:.2f} mm / {total_precip_depth:.2f} mm")

if 0.1 <= runoff_coeff <= 0.8:
    print("✓ Runoff coefficient is REASONABLE")
else:
    print("⚠ Runoff coefficient is UNREASONABLE")

# Now check what routing might have done
print(f"\n{'='*80}")
print("CHECKING ROUTING OUTPUT")
print(f"{'='*80}")

# Check discharge statistics
stats_file = Path("results/upper_truckee_complete_11steps/step_10_routing/10.2_discharge_statistics.csv")
stats_df = pd.read_csv(stats_file)
print("\nDischarge statistics for Zone 1 subbasins:")
zone1_stats = stats_df[stats_df['Subbasin_ID'].isin(subbasin_ids_int)]
print(zone1_stats[['Subbasin_ID', 'Peak_Discharge_m3s', 'Total_Volume_m3', 'Mean_Discharge_m3s']])

total_volume_from_stats = zone1_stats['Total_Volume_m3'].sum()
print(f"\nTotal volume from statistics: {total_volume_from_stats:.2f} m³")
print(f"Total volume from timeseries: {total_discharge_volume_m3:.2f} m³")
print(f"Difference: {abs(total_volume_from_stats - total_discharge_volume_m3):.2f} m³")
