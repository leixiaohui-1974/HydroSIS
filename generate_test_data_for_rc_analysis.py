#!/usr/bin/env python3
"""Generate test data for runoff coefficient analysis"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Create output directory
output_dir = Path("results/test_data_for_rc_analysis")
output_dir.mkdir(parents=True, exist_ok=True)

# Generate synthetic precipitation data (6 zones, 168 hours)
n_hours = 168
n_zones = 6

# Different precipitation patterns for each zone
precipitation = {}
precipitation['time'] = list(range(n_hours))

for i in range(n_zones):
    zone_id = f'Zone{i+1}'
    # Base intensity with random variation
    base_intensity = 5.0 + i * 0.5  # mm/h
    # Add some storm events
    precip = np.random.uniform(0, 2, n_hours) * base_intensity
    # Add 2-3 storm events
    for storm in range(2):
        start = np.random.randint(20, n_hours-20)
        duration = np.random.randint(6, 12)
        precip[start:start+duration] += np.random.uniform(15, 25, duration)
    
    precipitation[zone_id] = precip

precip_df = pd.DataFrame(precipitation)
precip_file = output_dir / 'precipitation_timeseries.csv'
precip_df.to_csv(precip_file, index=False)
print(f"Generated precipitation data: {precip_file}")

# Generate discharge data with realistic runoff coefficients
discharge = {}
discharge['time'] = list(range(n_hours))

# Different runoff coefficients for each zone
target_rcs = [0.35, 0.45, 0.52, 0.38, 0.48, 0.42]

# Area is 100 km² = 100,000,000 m²
area_km2 = 100.0
area_m2 = area_km2 * 1e6

for i, zone_id in enumerate([f'Zone{i+1}' for i in range(n_zones)]):
    rc = target_rcs[i]
    precip_series = precip_df[zone_id].values
    
    # Simple runoff generation with lag
    # precip is in mm/h, we want runoff in mm/h
    runoff_mm_h = precip_series * rc
    
    # Add baseflow
    baseflow_mm_h = 0.5
    runoff_mm_h += baseflow_mm_h
    
    # Apply simple routing (lag)
    lag = np.random.randint(2, 5)
    runoff_mm_h = np.roll(runoff_mm_h, lag)
    runoff_mm_h[:lag] = baseflow_mm_h
    
    # Convert mm/h to m³/s
    # mm/h * area_m2 / 1000 / 3600 = m³/s
    runoff_m3s = (runoff_mm_h * area_m2) / (1000 * 3600)
    
    discharge[zone_id] = runoff_m3s

print("\nVerification:")
print(f"Area: {area_km2} km² = {area_m2} m²")
for i, zone_id in enumerate([f'Zone{i+1}' for i in range(n_zones)]):
    total_precip_mm = precip_df[zone_id].sum()  # mm (hourly sum)
    total_discharge_m3s_sum = discharge[zone_id].sum()  # sum of m³/s values
    # Each m³/s value represents flow rate for 1 hour
    # Total volume = sum(m³/s) * 3600 s = m³
    total_discharge_m3 = total_discharge_m3s_sum * 3600  # m³
    # Convert to mm: m³ / area_m2 * 1000 = mm
    total_runoff_mm = (total_discharge_m3 / area_m2) * 1000
    actual_rc = total_runoff_mm / total_precip_mm if total_precip_mm > 0 else 0
    print(f"{zone_id}: Target RC={target_rcs[i]:.2f}, Actual RC={actual_rc:.2f}")

discharge_df = pd.DataFrame(discharge)
discharge_file = output_dir / 'discharge_timeseries.csv'
discharge_df.to_csv(discharge_file, index=False)
print(f"Generated discharge data: {discharge_file}")

# Generate simple watershed GeoJSON with proper area (100 km²)
import json

# Use UTM-like coordinates (meters) to get proper area
# 100 km² = 100,000,000 m² = 10km × 10km square

features = []
for i in range(n_zones):
    # Create 10km x 10km squares (= 100 km²)
    x_offset = (i % 3) * 10000  # 10 km spacing
    y_offset = (i // 3) * 10000
    
    coordinates = [[
        [x_offset, y_offset],
        [x_offset + 10000, y_offset],
        [x_offset + 10000, y_offset + 10000],
        [x_offset, y_offset + 10000],
        [x_offset, y_offset]
    ]]
    
    feature = {
        "type": "Feature",
        "properties": {
            "id": f"Zone{i+1}",
            "name": f"Zone {i+1}",
            "area_km2": 100.0
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": coordinates
        }
    }
    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

watershed_file = output_dir / 'watersheds.geojson'
with open(watershed_file, 'w') as f:
    json.dump(geojson, f, indent=2)

print(f"Generated watershed data: {watershed_file}")

# Print summary
print("\nData Summary:")
print(f"- Time steps: {n_hours}")
print(f"- Zones: {n_zones}")
print(f"- Target runoff coefficients: {target_rcs}")
print(f"\nTotal precipitation by zone:")
for zone in [f'Zone{i+1}' for i in range(n_zones)]:
    total_precip = precip_df[zone].sum()
    print(f"  {zone}: {total_precip:.2f} mm")
