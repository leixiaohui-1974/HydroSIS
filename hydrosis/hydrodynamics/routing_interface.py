# ============ HydroSIS Integration Interface ============

import math
from typing import List, Mapping

from .core import SaintVenantSolver, BoundaryCondition, RiverReach


class HydrodynamicRoutingModel:
    """Hydrodynamic routing implementation as HydroSIS RoutingModel"""

    def __init__(self, parameters: Mapping[str, float]):
        self.parameters = dict(parameters)

        # Extract channel configuration from parameters
        self.reach = RiverReach(
            id=str(parameters.get('reach_id', 'main')),
            length=float(parameters.get('length', 10000)),  # Default 10km
            bed_slope=float(parameters.get('bed_slope', 0.001)),
            manning_n=float(parameters.get('manning_n', 0.03)),
            width=float(parameters.get('width', 30)),
            num_sections=int(parameters.get('num_sections', 20))
        )

        self.dt = float(parameters.get('time_step', 300))  # Default 5 minutes
        self.solver = SaintVenantSolver(self.reach, dt=self.dt)

    def route(self, subbasin, inflow: List[float]) -> List[float]:
        """Implement HydroSIS RoutingModel interface

        Args:
            subbasin: Subbasin object
            inflow: Runoff time series (m³/s)

        Returns:
            Outlet discharge time series (m³/s)
        """
        num_steps = len(inflow)

        # Distribute runoff uniformly across reach
        lateral_per_section = [q / self.reach.num_sections for q in inflow]

        # Configure boundary conditions
        bc = BoundaryCondition(
            upstream_type="discharge",
            upstream_values=[0.0] * num_steps,  # No upstream inflow
            downstream_type="stage",
            downstream_values=[2.0] * num_steps  # Constant downstream stage
        )
        
        outflow = []
        for t in range(num_steps):
            self.solver.set_lateral_inflow(
                [lateral_per_section[t]] * self.reach.num_sections
            )
            self.solver.solve_timestep(bc, t)
            outflow.append(float(self.solver.state.discharge[-1]))
        
        return outflow


def create_coupled_model_config():
    """Generate example configuration for coupled simulation"""
    config = {
        "routing_models": [
            {
                "id": "hydrodynamic",
                "model_type": "saint_venant_1d",
                "parameters": {
                    "reach_id": "main_channel",
                    "length": 15000,  # 15 km
                    "bed_slope": 0.0005,
                    "manning_n": 0.035,
                    "width": 40,
                    "num_sections": 30,
                    "time_step": 300
                }
            }
        ]
    }
    return config


# Register with HydroSIS framework
try:
    from hydrosis.routing.base import RoutingModelConfig
    RoutingModelConfig.register("saint_venant_1d", HydrodynamicRoutingModel)
    print("✓ 1D hydrodynamic model registered to HydroSIS routing model library")
except ImportError:
    print("⚠ HydroSIS framework not detected, model can run independently")


if __name__ == "__main__":
    # Standalone execution example
    print("=" * 60)
    print("1D Hydrodynamic Model Standalone Test")
    print("=" * 60)

    reach = RiverReach(
        id="test_reach",
        length=5000,
        bed_slope=0.001,
        manning_n=0.03,
        width=25,
        num_sections=15
    )

    solver = SaintVenantSolver(reach, dt=60)

    # Set flood hydrograph boundary
    num_steps = 100
    peak_time = 30
    upstream_q = [10 + 50 * math.exp(-((t-peak_time)/10)**2) for t in range(num_steps)]

    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q,
        downstream_type="stage",
        downstream_values=[2.5] * num_steps
    )

    # Set uniform lateral inflow
    solver.set_lateral_inflow([0.01] * reach.num_sections)

    print(f"Simulated reach: {reach.length}m, {reach.num_sections} cross-sections")
    print(f"Upstream peak discharge: {max(upstream_q):.1f} m³/s")
    print("Starting simulation...\n")

    results = solver.run_simulation(bc, num_steps)

    # Output results summary
    peak_discharge_outlet = max(d[-1] for d in results['discharge'])
    peak_depth = max(max(d) for d in results['depth'])

    print(f"✓ Simulation completed {num_steps} time steps")
    print(f"  Outlet peak discharge: {peak_discharge_outlet:.2f} m³/s")
    print(f"  Maximum depth: {peak_depth:.2f} m")
    print(f"  Routing attenuation: {(max(upstream_q) - peak_discharge_outlet)/max(upstream_q)*100:.1f}%")