# P1 Hydrodynamic Integration Plan

## Geometry Preparation
- Use `hydrosis.hydrodynamics.zone_geometry.build_zone_geometry` with Step04 outputs (`channel_centerlines.csv`, `channel_cross_sections_corrected.csv`) to generate the solver geometry bundle (`ZoneGeometry`).
- Persist the sampled cross-section table as `results/<project>/04_channel_profile/p1_mainstem_geometry.csv` (or similar) so later stages can reuse it without re-reading raw Step04 files.
- Keep `depth_step_m`, `centerline_bed_slope`, and the cross-section ↔ centerline alignment that the helper already provides; these feed directly into rating-curve generation or per-reach hydraulics.

## Solver Hook-Up
- Wrap the existing 1D solver (`hydrosis.hydrodynamics.core.SaintVenantSolver`) with an adapter that accepts irregular cross-sections:
  1. Map each `station_global_m` to a solver reach node, interpolating centerline slope and bed elevation along the chainage grid expected by the solver.
  2. For each node, provide a callable that returns wetted area/perimeter for the instantaneous depth (re-using the sampled table or evaluating on-demand via `CrossSection.sample_geometry`).
  3. Ingest Step09 hydro outputs (`channel_flow_timeseries.csv` or per-zone inflow series) as lateral inflows aligned to the node spacing; keep the API open so external hydrographs can be injected later.
- If the implicit Saint-Venant implementation proves heavy for early comparisons, add a thin explicit solver (`explicit_dynamic_wave.py`) that consumes the same geometry bundle and lateral inflows; make sure both solvers honor the same input schema.
- As an immediate baseline, the lightweight rating-curve solver `hydrosis.hydrodynamics.cross_section_solver.CrossSectionSolver` can translate segment discharge to stage/velocity using the sampled Manning geometry; treat this as the fallback path while the full Saint-Venant adapter matures.

### Configuration switches

```yaml
channel_dynamics:
  enable_cross_section_solver: true          # Step10 will execute the rating-curve branch
  flow_source: subbasin                      # Use Step09 subbasin hydrographs as lateral inflow
  flow_scenario: hydraulic_p3p4              # Name of the Step09 scenario (maps to hydro/<scenario>_subbasin)
  zones:                                     # Zones to evaluate (matches Step04 outputs)
    - P1
  report_segments:
    P1:
      - P1_sub1                              # Optional: figures for selected segment IDs
  cross_sections_path: ../results/upper_truckee_project/04_channel_profile/channel_cross_sections_corrected.csv
  centerline_path: ../results/upper_truckee_project/04_channel_profile/channel_centerlines.csv
  flow_timeseries_path: ../results/upper_truckee_project/09_hydrologic_run/channel_flow_timeseries.csv
  time_column: Timestamp
  mannings_n: 0.04
  depth_step_m: 0.25
  slope_floor_m_per_m: 1.0e-5
  slope_cap_m_per_m: 0.01
  active_half_width_m: 75.0
  depth_cap_m: 20.0
```

启用后，Step10 会在 `results/upper_truckee_project/10_hydrodynamic_run/cross_section_solver/`
下生成每个指定分区的时序 CSV、极值统计及关键断面对比图。

## Step10 Integration
- Create a new optional branch in Step10 (e.g., `hydrodynamic_channel_run`) guarded by a configuration flag (`project_config["channel_dynamics"]["enable_cross_section_solver"]`).
- When enabled:
  1. Load (or regenerate) the `ZoneGeometry` bundle for all zones flagged in the config (start with `P1`).
  2. Route Step09 hydrographs through the 1D solver to obtain stage/flow time series per reporting cross-section.
  3. Emit comparison artefacts under `results/<project>/10_hydrodynamic_run/cross_section_solver/`:
     - CSV time series for discharge, stage, and velocity.
     - Matched plots versus the hydrologic routing (e.g., `p1_outlet_flow_stage_comparison_cross_section.png`).
     - Optional animations if the solver output cadence is fine enough.
- Document the new flag and outputs in `docs/examples/upper_truckee_project.md` and the Step10 README once the implementation is in place.

## Validation & QA
- Before enabling the branch by default, run the solver against a synthetic steady-flow case to verify the geometry conversion has no artefacts (monotonic bed, reasonable depths).
- Add unit tests that exercise `build_zone_geometry` (e.g., confirm slope/bed alignment and depth capping) to guard against regressions when Step04 formats evolve.
