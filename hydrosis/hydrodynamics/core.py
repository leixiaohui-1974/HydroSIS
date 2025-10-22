"""1D Channel Hydrodynamics Model - Implicit Solver Based on Saint-Venant Equations

This module provides 1D hydrodynamic simulation capabilities compatible with HydroSIS, supporting:
- Complete Saint-Venant equations (continuity + momentum equations)
- Preissmann implicit four-point finite difference scheme
- Variable cross-section channel geometry
- Lateral inflow (from hydrological model runoff)
- Downstream boundary conditions (stage/discharge control)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# Cross-section geometry support
from .geometry import CrossSection, RectangleSection


@dataclass
class RiverReach:
    """River reach geometry definition"""

    id: str
    length: float  # Reach length (m)
    bed_slope: float  # Bed slope (dimensionless)
    manning_n: float  # Manning roughness coefficient
    width: float  # Channel width (m), simplified as rectangular cross-section
    num_sections: int = 10  # Number of computational cross-sections
    
    def __post_init__(self):
        self.dx = self.length / (self.num_sections - 1)
        self.x_coords = [i * self.dx for i in range(self.num_sections)]


@dataclass
class BoundaryCondition:
    """Boundary condition definition"""

    upstream_type: str = "discharge"  # "discharge" or "stage"
    upstream_values: List[float] = field(default_factory=list)
    downstream_type: str = "stage"  # "stage" or "rating_curve"
    downstream_values: List[float] = field(default_factory=list)
    downstream_rating: Optional[Tuple[float, float]] = None  # (a, b) for Q = a * H^b


@dataclass
class HydraulicState:
    """Hydraulic state variables"""

    depth: np.ndarray  # Depth (m)
    discharge: np.ndarray  # Discharge (m³/s)
    velocity: np.ndarray  # Velocity (m/s)
    area: np.ndarray  # Flow area (m²)

    @classmethod
    def initialize(cls, num_sections: int, initial_depth: float = 1.0,
                   initial_q: float = 10.0, width: float = 20.0):
        """Initialize static water state"""
        depth = np.full(num_sections, initial_depth)
        area = depth * width
        discharge = np.full(num_sections, initial_q)
        velocity = discharge / area
        return cls(depth=depth, discharge=discharge, velocity=velocity, area=area)


class SaintVenantSolver:
    """Saint-Venant Equations Implicit Solver

    Uses Preissmann four-point implicit scheme to solve:
    - Continuity equation: ∂A/∂t + ∂Q/∂x = q_lateral
    - Momentum equation: ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀ - Sf)

    Where:
    - A: Flow area
    - Q: Discharge
    - h: Water stage
    - S₀: Bed slope
    - Sf: Friction slope = n²Q|Q|/(A²R^(4/3)), R is hydraulic radius
    - q_lateral: Lateral inflow
    """

    def __init__(self, reach: RiverReach, dt: float = 60.0,
                 theta: float = 0.6, epsilon: float = 1e-4,
                 cross_section: Optional[CrossSection] = None):
        """
        Parameters:
            reach: River reach geometry information
            dt: Time step (s)
            theta: Time weighting factor (0.5=Crank-Nicolson, 1.0=fully implicit)
            epsilon: Newton iteration convergence tolerance
        """
        self.reach = reach
        self.dt = dt
        self.theta = theta
        self.epsilon = epsilon
        self.g = 9.81  # Gravitational acceleration
        # Cross-section geometry: defaults to rectangle for backward compatibility
        self.cross_section: CrossSection = cross_section or RectangleSection(width=reach.width)

        # Use more reasonable initialization
        self._initialize_steady_state()
        self.lateral_inflow = np.zeros(reach.num_sections)

    def _initialize_steady_state(self):
        """Initialize as steady flow state"""
        n = self.reach.num_sections

        # Estimate reasonable initial discharge and depth
        initial_q = 20.0  # Base discharge

        # Estimate corresponding normal depth using Manning formula
        S = max(self.reach.bed_slope, 1e-6)
        n_manning = self.reach.manning_n

        # Normal depth approximation for rectangular cross-section
        if hasattr(self.cross_section, 'width'):
            width = self.cross_section.width
            # Simplified normal depth calculation: h = (Q*n/(width*S^0.5))^(3/5)
            h_normal = (initial_q * n_manning / (width * S**0.5))**(3/5)
        else:
            # For other cross-section types, use iterative solution for normal depth
            h_normal = self._compute_normal_depth_iterative(initial_q)

        h_normal = max(h_normal, 0.5)  # Ensure minimum depth

        # Initialize state
        self.state = HydraulicState.initialize(
            n,
            initial_depth=h_normal,
            initial_q=initial_q,
            width=getattr(self.cross_section, 'width', self.reach.width)
        )

        # Update geometric parameters
        self.update_hydraulic_properties(self.state)

    def _compute_normal_depth_iterative(self, discharge: float, max_iter: int = 20) -> float:
        """Iteratively calculate normal depth"""
        h = 1.0  # Initial guess
        S = max(self.reach.bed_slope, 1e-6)
        n = self.reach.manning_n
        
        for _ in range(max_iter):
            props = self.cross_section.compute_properties(h)
            if props.area < 1e-6 or props.hydraulic_radius < 1e-6:
                h += 0.1
                continue
                
            Q_calc = (1/n) * props.area * props.hydraulic_radius**(2/3) * S**0.5

            if abs(Q_calc - discharge) < 0.1:
                break

            # Simple Newton method update
            dQ_dh = (1/n) * props.top_width * props.hydraulic_radius**(2/3) * S**0.5
            if dQ_dh > 1e-6:
                h += (discharge - Q_calc) / dQ_dh
                h = max(h, 0.1)  # Ensure positive value
            else:
                h += 0.1

        return max(h, 0.5)

    def _compute_section_properties(self, depths: np.ndarray):
        """Calculate geometric parameters of each cross-section based on current depth array.

        Returns: (area, top_width, hydraulic_radius)
        """
        n = len(depths)
        area = np.zeros(n)
        top_width = np.zeros(n)
        hydraulic_radius = np.zeros(n)
        # Calculate section by section (cross-section type may be compound/irregular, point-wise is safer)
        for i in range(n):
            props = self.cross_section.compute_properties(float(depths[i]))
            area[i] = props.area
            top_width[i] = props.top_width
            hydraulic_radius[i] = props.hydraulic_radius
        return area, top_width, hydraulic_radius

    def update_hydraulic_properties(self, state: HydraulicState):
        """Update hydraulic geometric parameters, supporting variable cross-sections."""
        area, _, _ = self._compute_section_properties(state.depth)
        state.area = area
        state.velocity = np.where(state.area > 1e-6,
                                   state.discharge / state.area, 0.0)

    def friction_slope(self, Q: np.ndarray, A: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Calculate friction slope Sf, using hydraulic radius provided by geometric cross-section."""
        R = np.maximum(R, 0.01)  # Prevent division by zero
        n = self.reach.manning_n
        Sf = n**2 * Q * np.abs(Q) / (A**2 * R**(4/3))
        return Sf
    
    def build_jacobian_and_residual(self, Q_new: np.ndarray, h_new: np.ndarray,
                                     Q_old: np.ndarray, h_old: np.ndarray,
                                     bc: BoundaryCondition, time_idx: int):
        """Build Jacobian matrix and residual vector for Newton method

        Unknowns: [Q₁, h₁, Q₂, h₂, ..., Qₙ, hₙ]
        Equations:
        - Continuity: A^{n+1} - A^n + θΔt/Δx(Q_{i+1} - Q_i) + (1-θ)Δt/Δx(Q_{i+1}^n - Q_i^n) = Δt·q
        - Momentum: Q^{n+1} - Q^n + θΔt[∂(Q²/A)/∂x + gA∂h/∂x - gA(S₀-Sf)] = 0
        """
        n = self.reach.num_sections
        N = 2 * n  # Total number of unknowns
        dx = self.reach.dx

        # Calculate cross-section geometric parameters at current time
        A_new, T_new, R_new = self._compute_section_properties(h_new)
        A_old, _, _ = self._compute_section_properties(h_old)

        J = np.zeros((N, N))
        R = np.zeros(N)

        # Use last value when boundary sequence is out of bounds (adapt to adaptive time step scenarios)
        up_idx = min(time_idx, len(bc.upstream_values) - 1) if bc.upstream_values else 0
        dn_idx = min(time_idx, len(bc.downstream_values) - 1) if bc.downstream_values else 0

        # === Upstream boundary condition (i=0) ===
        if bc.upstream_type == "discharge":
            # Upstream discharge boundary: directly set discharge
            J[0, 0] = 1.0  # ∂R/∂Q_0 = 1
            R[0] = Q_new[0] - bc.upstream_values[up_idx]

            # Upstream continuity equation (1st equation)
            J[1, 0] = -self.theta * self.dt / dx
            J[1, 1] = T_new[0]  # ∂A/∂h
            J[1, 2] = self.theta * self.dt / dx
            R[1] = (A_new[0] - A_old[0] +
                   self.theta * self.dt / dx * (Q_new[1] - Q_new[0]) +
                   (1 - self.theta) * self.dt / dx * (Q_old[1] - Q_old[0]) -
                   self.dt * self.lateral_inflow[0])
        else:  # stage boundary
            # Upstream stage boundary: directly set stage
            J[0, 1] = 1.0  # ∂R/∂h_0 = 1
            R[0] = h_new[0] - bc.upstream_values[up_idx]

            # Upstream momentum equation
            v_0 = Q_new[0] / A_new[0] if A_new[0] > 1e-6 else 0
            v_1 = Q_new[1] / A_new[1] if A_new[1] > 1e-6 else 0
            Sf_0 = self.friction_slope(Q_new[0:1], A_new[0:1], R_new[0:1])[0]
            dh = (h_new[1] - h_new[0]) / dx

            J[1, 0] = 1.0
            J[1, 1] = self.theta * self.dt * self.g * T_new[0] * (dh - (self.reach.bed_slope - Sf_0))
            R[1] = (Q_new[0] - Q_old[0] +
                   self.theta * self.dt * (
                       (v_1 * Q_new[1] - v_0 * Q_new[0]) / dx +
                       self.g * A_new[0] * dh -
                       self.g * A_new[0] * (self.reach.bed_slope - Sf_0)
                   ))

        # === Internal node equations (i=1 to n-2) ===
        for i in range(1, n - 1):
            # Continuity equation (even rows)
            row = 2 * i
            R[row] = (A_new[i] - A_old[i] +
                     self.theta * self.dt / dx * (Q_new[i+1] - Q_new[i]) +
                     (1 - self.theta) * self.dt / dx * (Q_old[i+1] - Q_old[i]) -
                     self.dt * self.lateral_inflow[i])

            # Jacobian matrix - continuity
            J[row, 2*i] = -self.theta * self.dt / dx  # ∂R/∂Q_i
            J[row, 2*i+1] = T_new[i]  # ∂R/∂h_i = ∂A/∂h
            J[row, 2*(i+1)] = self.theta * self.dt / dx  # ∂R/∂Q_{i+1}

            # Momentum equation (odd rows) - improved version
            row = 2 * i + 1

            # Calculate more accurate derivative of convection term ∂(Q²/A)/∂x
            v_i = Q_new[i] / A_new[i] if A_new[i] > 1e-6 else 0
            v_ip1 = Q_new[i+1] / A_new[i+1] if A_new[i+1] > 1e-6 else 0

            # Friction slope and its derivative
            Sf_i = self.friction_slope(Q_new[i:i+1], A_new[i:i+1], R_new[i:i+1])[0]

            # Improved derivative of friction slope with respect to discharge ∂Sf/∂Q
            if A_new[i] > 1e-6 and R_new[i] > 1e-6:
                dSf_dQ = 2 * self.reach.manning_n**2 * np.abs(Q_new[i]) / (A_new[i]**2 * R_new[i]**(4/3))
            else:
                dSf_dQ = 0

            # Water surface slope
            dh = (h_new[i+1] - h_new[i]) / dx

            # Residual
            R[row] = (Q_new[i] - Q_old[i] +
                     self.theta * self.dt * (
                         (v_ip1 * Q_new[i+1] - v_i * Q_new[i]) / dx +
                         self.g * A_new[i] * dh -
                         self.g * A_new[i] * (self.reach.bed_slope - Sf_i)
                     ))
            
            # Improved Jacobian matrix elements
            # ∂R/∂Q_i
            J[row, 2*i] = (1 + self.theta * self.dt * (
                -2 * v_i / dx +  # Convection term derivative
                self.g * A_new[i] * dSf_dQ  # Friction term derivative
            ))

            # ∂R/∂h_i - More accurate pressure and geometric terms
            J[row, 2*i+1] = self.theta * self.dt * self.g * (
                T_new[i] * (dh - (self.reach.bed_slope - Sf_i)) -  # Pressure term
                A_new[i] / dx  # Water surface slope term
            )

            # ∂R/∂Q_{i+1}
            if A_new[i+1] > 1e-6:
                J[row, 2*(i+1)] = self.theta * self.dt * 2 * v_ip1 / dx

            # ∂R/∂h_{i+1}
            J[row, 2*(i+1)+1] = self.theta * self.dt * self.g * A_new[i] / dx

        # === Downstream boundary condition (i=n-1) ===
        i = n - 1
        if bc.downstream_type == "stage":
            # Downstream stage boundary: directly set stage
            J[2*i+1, 2*i+1] = 1.0  # ∂R/∂h_{n-1} = 1
            R[2*i+1] = h_new[i] - bc.downstream_values[dn_idx]

            # Downstream continuity equation
            J[2*i, 2*i-2] = self.theta * self.dt / dx    # ∂R/∂Q_{i-1}
            J[2*i, 2*i] = -self.theta * self.dt / dx     # ∂R/∂Q_i
            J[2*i, 2*i+1] = T_new[i]                     # ∂R/∂h_i
            R[2*i] = (A_new[i] - A_old[i] +
                     self.theta * self.dt / dx * (-Q_new[i] + Q_new[i-1]) +
                     (1 - self.theta) * self.dt / dx * (-Q_old[i] + Q_old[i-1]) -
                     self.dt * self.lateral_inflow[i])
        else:  # discharge boundary or free outflow
            # Downstream discharge boundary or free outflow
            if hasattr(bc, 'downstream_rating') and bc.downstream_rating and len(bc.downstream_rating) == 2:
                # Stage-discharge relationship
                a, b = bc.downstream_rating
                if h_new[i] > 0.1:
                    J[2*i+1, 2*i] = 1.0
                    J[2*i+1, 2*i+1] = -a * b * h_new[i]**(b-1)
                    R[2*i+1] = Q_new[i] - a * h_new[i]**b
                else:
                    J[2*i+1, 2*i] = 1.0
                    J[2*i+1, 2*i+1] = -a * b
                    R[2*i+1] = Q_new[i] - a * b * h_new[i]
            else:
                # Free outflow boundary (normal depth approximation)
                if A_new[i] > 1e-6 and R_new[i] > 1e-6:
                    S = max(self.reach.bed_slope, 1e-6)
                    Q_normal = (1/self.reach.manning_n) * A_new[i] * R_new[i]**(2/3) * S**0.5
                    J[2*i+1, 2*i] = 1.0
                    J[2*i+1, 2*i+1] = -(1/self.reach.manning_n) * T_new[i] * R_new[i]**(2/3) * S**0.5
                    R[2*i+1] = Q_new[i] - Q_normal
                else:
                    J[2*i+1, 2*i] = 1.0
                    R[2*i+1] = Q_new[i] - max(Q_old[i], 1.0)

            # Downstream continuity equation
            J[2*i, 2*i-2] = self.theta * self.dt / dx
            J[2*i, 2*i] = -self.theta * self.dt / dx
            J[2*i, 2*i+1] = T_new[i]
            R[2*i] = (A_new[i] - A_old[i] +
                     self.theta * self.dt / dx * (-Q_new[i] + Q_new[i-1]) +
                     (1 - self.theta) * self.dt / dx * (-Q_old[i] + Q_old[i-1]) -
                     self.dt * self.lateral_inflow[i])
        
        return J, R
    
    def solve_timestep(self, bc: BoundaryCondition, time_idx: int,
                       max_iter: int = 15) -> bool:
        """Solve single time step - simplified version focused on stability

        Returns:
            Whether converged
        """
        Q_old = self.state.discharge.copy()
        h_old = self.state.depth.copy()

        Q_new = Q_old.copy()
        h_new = h_old.copy()

        # Boundary sequence out-of-bounds protection
        up_idx = min(time_idx, len(bc.upstream_values) - 1) if bc.upstream_values else 0
        dn_idx = min(time_idx, len(bc.downstream_values) - 1) if bc.downstream_values else 0

        # Directly apply boundary conditions (simplified method)
        if bc.upstream_type == "discharge":
            Q_new[0] = bc.upstream_values[up_idx]
        else:  # stage
            h_new[0] = bc.upstream_values[up_idx]

        if bc.downstream_type == "stage":
            h_new[-1] = bc.downstream_values[dn_idx]

        # Use explicit method to update internal nodes (more stable)
        success = self._explicit_update(Q_new, h_new, Q_old, h_old)
        
        if success:
            # Update state
            self.state.discharge = Q_new
            self.state.depth = h_new
            self.update_hydraulic_properties(self.state)
            return True

        return False

    def _explicit_update(self, Q_new: np.ndarray, h_new: np.ndarray,
                        Q_old: np.ndarray, h_old: np.ndarray) -> bool:
        """Use explicit method to update internal nodes"""
        try:
            n = self.reach.num_sections
            dx = self.reach.dx
            dt = self.dt

            # Calculate geometric parameters
            A_old, T_old, R_old = self._compute_section_properties(h_old)

            # Explicit update for internal nodes
            for i in range(1, n - 1):
                # Continuity equation: ∂A/∂t + ∂Q/∂x = 0
                dQ_dx = (Q_old[i+1] - Q_old[i-1]) / (2 * dx)
                A_new_i = A_old[i] - dt * dQ_dx + dt * self.lateral_inflow[i]

                # Calculate depth from area
                h_new[i] = self._depth_from_area(A_new_i, h_old[i])

                # Momentum equation: ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀ - Sf)
                # Simplified to: Q_new = Q_old + dt * [gA(S₀ - Sf) - ∂(Q²/A)/∂x - gA∂h/∂x]

                # Convection term
                v_i = Q_old[i] / A_old[i] if A_old[i] > 1e-6 else 0
                v_ip1 = Q_old[i+1] / A_old[i+1] if A_old[i+1] > 1e-6 else 0
                v_im1 = Q_old[i-1] / A_old[i-1] if A_old[i-1] > 1e-6 else 0

                dQQ_A_dx = ((v_ip1 * Q_old[i+1]) - (v_im1 * Q_old[i-1])) / (2 * dx)

                # Pressure term
                dh_dx = (h_old[i+1] - h_old[i-1]) / (2 * dx)

                # Friction term
                Sf = self.friction_slope(Q_old[i:i+1], A_old[i:i+1], R_old[i:i+1])[0]

                # Update discharge
                Q_new[i] = Q_old[i] + dt * (
                    self.g * A_old[i] * (self.reach.bed_slope - Sf) -
                    dQQ_A_dx -
                    self.g * A_old[i] * dh_dx
                )

                # Stability check
                if h_new[i] <= 0.01 or not np.isfinite(Q_new[i]) or not np.isfinite(h_new[i]):
                    return False

            return True

        except Exception:
            return False

    def _depth_from_area(self, area: float, h_guess: float) -> float:
        """Calculate depth from area"""
        try:
            # Use inverse function from cross-section geometry
            return self.cross_section.compute_depth_from_area(area)
        except (AttributeError, ValueError, ZeroDivisionError, RuntimeError):
            # If failed, use simple approximation
            if hasattr(self.cross_section, 'width'):
                return max(area / self.cross_section.width, 0.01)
            else:
                return max(h_guess, 0.01)

    def set_lateral_inflow(self, inflow: Sequence[float]):
        """Set lateral inflow (m³/s/m), usually from hydrological model runoff"""
        if len(inflow) != self.reach.num_sections:
            raise ValueError(f"Inflow length {len(inflow)} != sections {self.reach.num_sections}")
        self.lateral_inflow = np.array(inflow)

    def run_simulation(self, bc: BoundaryCondition, num_steps: int) -> Dict[str, List]:
        """Run complete simulation

        Returns:
            Time series dictionary {'discharge': [...], 'depth': [...], 'velocity': [...]}
        """
        results = {
            'time': [],
            'discharge': [],
            'depth': [],
            'velocity': []
        }

        for t in range(num_steps):
            success = self.solve_timestep(bc, t)
            if not success:
                print(f"Warning: time step {t} did not converge")

            results['time'].append(t * self.dt)
            results['discharge'].append(self.state.discharge.copy())
            results['depth'].append(self.state.depth.copy())
            results['velocity'].append(self.state.velocity.copy())

        return results


