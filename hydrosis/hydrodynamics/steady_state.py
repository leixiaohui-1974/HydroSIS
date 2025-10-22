"""Steady-state flow computation module - provides initial conditions for unsteady flow simulation"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import fsolve

from .geometry import CrossSection, RectangleSection
from .core import RiverReach


class SteadyStateCalculator:
    """Steady-state flow calculator

    Used to calculate steady-state water surface profile under given discharge and boundary conditions,
    providing reasonable initial conditions for unsteady flow simulation.
    """

    def __init__(self, reach: RiverReach, cross_section: Optional[CrossSection] = None):
        """
        Parameters:
            reach: River reach geometry information
            cross_section: Cross-section geometry, defaults to rectangular cross-section
        """
        self.reach = reach
        self.cross_section = cross_section or RectangleSection(width=reach.width)
        self.g = 9.81

    def compute_normal_depth(self, discharge: float, tolerance: float = 1e-6) -> float:
        """Calculate normal depth

        Using Manning formula: Q = (1/n) * A * R^(2/3) * S^(1/2)
        """
        def manning_residual(h):
            if h <= 0.01:
                return 1e6  # Penalty for negative depth

            props = self.cross_section.compute_properties(h)
            if props.area < 1e-6 or props.hydraulic_radius < 1e-6:
                return 1e6

            Q_calc = (1/self.reach.manning_n) * props.area * \
                    props.hydraulic_radius**(2/3) * self.reach.bed_slope**0.5

            return Q_calc - discharge

        # Initial guess: based on rectangular cross-section approximation
        if hasattr(self.cross_section, 'width'):
            h_guess = (discharge * self.reach.manning_n /
                      (self.cross_section.width * self.reach.bed_slope**0.5))**(3/5)
        else:
            h_guess = 1.0

        try:
            h_normal = fsolve(manning_residual, h_guess, xtol=tolerance)[0]
            return max(h_normal, 0.1)
        except (ValueError, RuntimeError, ZeroDivisionError):
            # If solution fails, use iterative method
            return self._iterative_normal_depth(discharge, tolerance)

    def _iterative_normal_depth(self, discharge: float, tolerance: float = 1e-6) -> float:
        """Iteratively solve for normal depth"""
        h = 1.0
        for _ in range(50):
            props = self.cross_section.compute_properties(h)
            if props.area < 1e-6:
                h += 0.1
                continue

            Q_calc = (1/self.reach.manning_n) * props.area * \
                    props.hydraulic_radius**(2/3) * self.reach.bed_slope**0.5

            if abs(Q_calc - discharge) < tolerance:
                break

            # Newton method update
            dQ_dh = (1/self.reach.manning_n) * props.top_width * \
                   props.hydraulic_radius**(2/3) * self.reach.bed_slope**0.5

            if dQ_dh > 1e-6:
                h += (discharge - Q_calc) / dQ_dh
                h = max(h, 0.1)
            else:
                h += 0.1 if Q_calc < discharge else -0.05

        return max(h, 0.1)

    def compute_critical_depth(self, discharge: float, tolerance: float = 1e-6) -> float:
        """Calculate critical depth

        Critical condition: Fr = V/sqrt(g*D) = 1
        where D = A/T is hydraulic depth
        """
        def froude_residual(h):
            if h <= 0.01:
                return 1e6

            props = self.cross_section.compute_properties(h)
            if props.area < 1e-6 or props.top_width < 1e-6:
                return 1e6

            velocity = discharge / props.area
            hydraulic_depth = props.area / props.top_width
            froude = velocity / np.sqrt(self.g * hydraulic_depth)

            return froude - 1.0

        try:
            h_critical = fsolve(froude_residual, 0.5, xtol=tolerance)[0]
            return max(h_critical, 0.1)
        except (ValueError, RuntimeError, ZeroDivisionError):
            return 0.5  # Default value

    def compute_steady_profile(self,
                              upstream_discharge: float,
                              downstream_condition: Tuple[str, float],
                              method: str = "standard_step") -> Tuple[np.ndarray, np.ndarray]:
        """Calculate steady-state water surface profile

        Parameters:
            upstream_discharge: Upstream discharge (m³/s)
            downstream_condition: Downstream boundary condition ("depth", value) or ("normal", 0)
            method: Computation method, currently supports "standard_step"

        Returns:
            (depths, elevations): Depth and water surface elevation at each cross-section
        """
        n_sections = self.reach.num_sections
        x_coords = np.array(self.reach.x_coords)
        bed_elevations = np.zeros(n_sections)

        # Calculate bed elevation (increasing from downstream to upstream)
        for i in range(n_sections):
            bed_elevations[i] = (n_sections - 1 - i) * self.reach.dx * self.reach.bed_slope

        # Determine downstream boundary condition
        if downstream_condition[0] == "depth":
            h_downstream = downstream_condition[1]
        elif downstream_condition[0] == "normal":
            h_downstream = self.compute_normal_depth(upstream_discharge)
        else:
            raise ValueError(f"Unsupported downstream boundary condition: {downstream_condition[0]}")

        # Use standard step method to calculate water surface profile
        depths = np.zeros(n_sections)
        depths[-1] = h_downstream  # Downstream boundary

        # Calculate step by step from downstream to upstream
        for i in range(n_sections - 2, -1, -1):
            depths[i] = self._compute_upstream_depth(
                upstream_discharge,
                depths[i + 1],
                self.reach.dx
            )

        # Calculate water surface elevation
        elevations = bed_elevations + depths

        return depths, elevations

    def _compute_upstream_depth(self, discharge: float, h_downstream: float, dx: float) -> float:
        """Calculate upstream depth using energy equation"""

        # Downstream cross-section parameters
        props_down = self.cross_section.compute_properties(h_downstream)
        v_down = discharge / props_down.area if props_down.area > 1e-6 else 0
        
        # Friction slope
        Sf_down = (self.reach.manning_n * v_down * abs(v_down)) / \
                 (props_down.hydraulic_radius**(4/3)) if props_down.hydraulic_radius > 1e-6 else 0

        def energy_residual(h_up):
            if h_up <= 0.01:
                return 1e6

            props_up = self.cross_section.compute_properties(h_up)
            if props_up.area < 1e-6:
                return 1e6

            v_up = discharge / props_up.area

            # Friction slope
            Sf_up = (self.reach.manning_n * v_up * abs(v_up)) / \
                   (props_up.hydraulic_radius**(4/3)) if props_up.hydraulic_radius > 1e-6 else 0

            # Average friction slope
            Sf_avg = (Sf_up + Sf_down) / 2

            # Energy equation: E_up = E_down + (S0 - Sf_avg) * dx
            E_up = h_up + v_up**2 / (2 * self.g)
            E_down = h_downstream + v_down**2 / (2 * self.g)

            return E_up - E_down - (self.reach.bed_slope - Sf_avg) * dx

        # Initial guess
        h_guess = h_downstream + self.reach.bed_slope * dx

        try:
            h_upstream = fsolve(energy_residual, h_guess, xtol=1e-6)[0]
            return max(h_upstream, 0.1)
        except (ValueError, RuntimeError, ZeroDivisionError):
            # If solution fails, use simple approximation
            return h_downstream + self.reach.bed_slope * dx

    def create_initial_conditions(self,
                                 discharge: float,
                                 boundary_type: str = "normal") -> Tuple[np.ndarray, np.ndarray]:
        """Create initial conditions for unsteady flow simulation

        Parameters:
            discharge: Initial discharge (m³/s)
            boundary_type: Boundary type ("normal" or "critical")

        Returns:
            (initial_depths, initial_discharges): Initial depth and discharge distribution
        """
        if boundary_type == "normal":
            # Use normal depth as uniform initial condition
            h_normal = self.compute_normal_depth(discharge)
            depths = np.full(self.reach.num_sections, h_normal)
        elif boundary_type == "critical":
            # Use critical depth
            h_critical = self.compute_critical_depth(discharge)
            depths = np.full(self.reach.num_sections, h_critical)
        else:
            # Calculate complete steady-state water surface profile
            depths, _ = self.compute_steady_profile(
                discharge,
                ("normal", 0)
            )

        # Uniform discharge distribution
        discharges = np.full(self.reach.num_sections, discharge)

        return depths, discharges

    def validate_initial_conditions(self, depths: np.ndarray, discharges: np.ndarray) -> dict:
        """Validate reasonableness of initial conditions

        Returns validation report
        """
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {}
        }

        # Check depths
        if np.any(depths <= 0):
            report["errors"].append("Negative or zero depths exist")
            report["valid"] = False

        if np.any(depths > 10):
            report["warnings"].append("Abnormally large depths exist (>10m)")

        # Check discharges
        if np.any(discharges <= 0):
            report["errors"].append("Negative or zero discharges exist")
            report["valid"] = False

        # Mass conservation check
        q_variation = np.std(discharges) / np.mean(discharges) * 100
        if q_variation > 5:
            report["warnings"].append(f"Large discharge variation ({q_variation:.1f}%)")

        # Statistics
        report["statistics"] = {
            "depth_range": (np.min(depths), np.max(depths)),
            "discharge_range": (np.min(discharges), np.max(discharges)),
            "avg_depth": np.mean(depths),
            "avg_discharge": np.mean(discharges),
            "flow_variation": q_variation
        }
        
        return report


def compute_normal_depth(cross_section: CrossSection,
                        discharge: float,
                        bed_slope: float,
                        manning_n: float,
                        tolerance: float = 1e-6) -> float:
    """Standalone function: calculate normal depth

    This is a convenience function that doesn't require creating a complete SteadyStateCalculator instance
    """
    # Create temporary reach
    temp_reach = RiverReach(
        id="temp",
        length=1000,
        bed_slope=bed_slope,
        manning_n=manning_n,
        width=getattr(cross_section, 'width', 20),
        num_sections=2
    )

    calculator = SteadyStateCalculator(temp_reach, cross_section)
    return calculator.compute_normal_depth(discharge, tolerance)


def compute_critical_depth(cross_section: CrossSection,
                          discharge: float,
                          tolerance: float = 1e-6) -> float:
    """Standalone function: calculate critical depth"""
    temp_reach = RiverReach(
        id="temp",
        length=1000,
        bed_slope=0.001,
        manning_n=0.03,
        width=getattr(cross_section, 'width', 20),
        num_sections=2
    )
    
    calculator = SteadyStateCalculator(temp_reach, cross_section)
    return calculator.compute_critical_depth(discharge, tolerance)