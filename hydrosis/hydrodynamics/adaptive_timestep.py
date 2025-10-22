"""Adaptive timestep controller

Provides multiple strategies for dynamically adjusting timestep size to ensure numerical
stability while improving computational efficiency:
- CFL condition control
- Depth change rate control
- Newton iteration performance control
- Hybrid strategy
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class AdaptiveStrategy(Enum):
    """Adaptive strategy enumeration"""
    CFL_BASED = "cfl"                    # Based on CFL number
    DEPTH_CHANGE = "depth_change"        # Based on depth change rate
    CONVERGENCE = "convergence"          # Based on convergence performance
    HYBRID = "hybrid"                    # Hybrid strategy


@dataclass
class TimeStepMetrics:
    """Timestep metric indicators"""

    current_dt: float              # Current timestep size (s)
    cfl_number: float             # CFL number
    max_depth_change_rate: float  # Maximum depth change rate (m/s)
    newton_iterations: int        # Newton iteration count
    convergence_achieved: bool    # Whether convergence was achieved
    suggested_dt: float           # Suggested next timestep size (s)
    reason: str                   # Adjustment reason


class AdaptiveTimeStepController:
    """Adaptive timestep controller

    Dynamically adjusts timestep size based on simulation state to balance stability and efficiency
    """

    def __init__(self,
                 initial_dt: float = 60.0,
                 min_dt: float = 10.0,
                 max_dt: float = 600.0,
                 target_cfl: float = 0.5,
                 max_cfl: float = 0.8,
                 strategy: AdaptiveStrategy = AdaptiveStrategy.HYBRID,
                 safety_factor: float = 0.9):
        """
        Args:
            initial_dt: Initial timestep size (s)
            min_dt: Minimum allowed timestep (s)
            max_dt: Maximum allowed timestep (s)
            target_cfl: Target CFL number
            max_cfl: Maximum allowed CFL number
            strategy: Adaptive strategy
            safety_factor: Safety factor (damping factor for adjustments)
        """
        self.current_dt = initial_dt
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.target_cfl = target_cfl
        self.max_cfl = max_cfl
        self.strategy = strategy
        self.safety_factor = safety_factor

        # History for smooth adjustments
        self.dt_history: List[float] = [initial_dt]
        self.cfl_history: List[float] = []
        self.convergence_history: List[bool] = []

        # Statistics
        self.total_steps = 0
        self.adjustments = 0
        self.failed_steps = 0
    
    def compute_cfl_number(self, velocity: np.ndarray,
                          dx: float, dt: float) -> float:
        """Compute CFL number

        CFL = max(|v|) * dt / dx

        Args:
            velocity: Velocity array (m/s)
            dx: Spatial step size (m)
            dt: Timestep size (s)

        Returns:
            CFL number
        """
        max_velocity = np.max(np.abs(velocity))
        return max_velocity * dt / dx if dx > 0 else 0.0
    
    def compute_depth_change_rate(self, depth_current: np.ndarray,
                                  depth_previous: np.ndarray,
                                  dt: float) -> float:
        """Compute depth change rate

        Args:
            depth_current: Current depth (m)
            depth_previous: Previous timestep depth (m)
            dt: Timestep size (s)

        Returns:
            Maximum depth change rate (m/s)
        """
        depth_change = np.abs(depth_current - depth_previous)
        return np.max(depth_change) / dt if dt > 0 else 0.0
    
    def adjust_by_cfl(self, cfl_current: float) -> tuple[float, str]:
        """Adjust timestep based on CFL condition

        Returns:
            (suggested timestep, adjustment reason)
        """
        if cfl_current > self.max_cfl:
            # CFL too large, reduce timestep
            reduction_factor = self.target_cfl / cfl_current
            new_dt = self.current_dt * reduction_factor * self.safety_factor
            reason = f"CFL={cfl_current:.3f} exceeded limit, reducing timestep"
        elif cfl_current < self.target_cfl * 0.5:
            # CFL too small, increase timestep to improve efficiency
            increase_factor = self.target_cfl / cfl_current
            new_dt = self.current_dt * increase_factor * self.safety_factor
            reason = f"CFL={cfl_current:.3f} too small, increasing timestep"
        else:
            # Within reasonable range, maintain current value
            new_dt = self.current_dt
            reason = f"CFL={cfl_current:.3f} acceptable, maintaining"

        return new_dt, reason
    
    def adjust_by_depth_change(self, depth_change_rate: float,
                               max_allowed_rate: float = 0.5) -> tuple[float, str]:
        """Adjust timestep based on depth change rate

        Args:
            depth_change_rate: Current depth change rate (m/s)
            max_allowed_rate: Maximum allowed change rate (m/s)

        Returns:
            (suggested timestep, adjustment reason)
        """
        if depth_change_rate > max_allowed_rate:
            reduction_factor = max_allowed_rate / depth_change_rate
            new_dt = self.current_dt * reduction_factor * self.safety_factor
            reason = f"Depth change rate {depth_change_rate:.3f} m/s too fast"
        elif depth_change_rate < max_allowed_rate * 0.2 and depth_change_rate > 1e-6:
            increase_factor = min(2.0, max_allowed_rate / depth_change_rate)
            new_dt = self.current_dt * increase_factor * self.safety_factor
            reason = f"Depth change rate {depth_change_rate:.3f} m/s slow"
        else:
            new_dt = self.current_dt
            reason = "Depth change rate acceptable"

        return new_dt, reason
    
    def adjust_by_convergence(self, newton_iterations: int,
                              converged: bool,
                              max_iterations: int = 20) -> tuple[float, str]:
        """Adjust timestep based on Newton iteration performance

        Args:
            newton_iterations: Actual iteration count
            converged: Whether convergence was achieved
            max_iterations: Maximum allowed iterations

        Returns:
            (suggested timestep, adjustment reason)
        """
        if not converged:
            # Not converged, significantly reduce timestep
            new_dt = self.current_dt * 0.5
            reason = f"Newton iteration did not converge, halving timestep"
        elif newton_iterations > max_iterations * 0.8:
            # Iteration count approaching limit, preemptively reduce
            new_dt = self.current_dt * 0.8
            reason = f"Iterations {newton_iterations} approaching limit"
        elif newton_iterations < max_iterations * 0.3 and len(self.convergence_history) > 5:
            # Fast and stable convergence, can increase timestep
            if all(self.convergence_history[-5:]):
                new_dt = self.current_dt * 1.2
                reason = f"Iterations {newton_iterations} fast convergence"
            else:
                new_dt = self.current_dt
                reason = "Fast convergence but unstable history"
        else:
            new_dt = self.current_dt
            reason = f"Iterations {newton_iterations} normal"

        return new_dt, reason
    
    def update(self,
               velocity: np.ndarray,
               depth_current: np.ndarray,
               depth_previous: np.ndarray,
               dx: float,
               newton_iterations: int = 0,
               converged: bool = True) -> TimeStepMetrics:
        """Update timestep and return metric information

        Args:
            velocity: Current velocity field (m/s)
            depth_current: Current depth (m)
            depth_previous: Previous timestep depth (m)
            dx: Spatial step size (m)
            newton_iterations: Newton iteration count
            converged: Whether convergence was achieved

        Returns:
            TimeStepMetrics object
        """
        # Compute metrics
        cfl = self.compute_cfl_number(velocity, dx, self.current_dt)
        depth_change_rate = self.compute_depth_change_rate(
            depth_current, depth_previous, self.current_dt
        )

        # Select adjustment method based on strategy
        if self.strategy == AdaptiveStrategy.CFL_BASED:
            suggested_dt, reason = self.adjust_by_cfl(cfl)
        
        elif self.strategy == AdaptiveStrategy.DEPTH_CHANGE:
            suggested_dt, reason = self.adjust_by_depth_change(depth_change_rate)
        
        elif self.strategy == AdaptiveStrategy.CONVERGENCE:
            suggested_dt, reason = self.adjust_by_convergence(
                newton_iterations, converged
            )
        
        elif self.strategy == AdaptiveStrategy.HYBRID:
            # Hybrid strategy: take most conservative suggestion
            dt_cfl, reason_cfl = self.adjust_by_cfl(cfl)
            dt_depth, reason_depth = self.adjust_by_depth_change(depth_change_rate)
            dt_conv, reason_conv = self.adjust_by_convergence(
                newton_iterations, converged
            )

            suggested_dt = min(dt_cfl, dt_depth, dt_conv)

            # Identify primary limiting factor
            if suggested_dt == dt_cfl:
                reason = f"Hybrid strategy: CFL dominant ({reason_cfl})"
            elif suggested_dt == dt_depth:
                reason = f"Hybrid strategy: Depth change dominant ({reason_depth})"
            else:
                reason = f"Hybrid strategy: Convergence dominant ({reason_conv})"

        else:
            suggested_dt = self.current_dt
            reason = "Unknown strategy"

        # Limit to allowed range
        suggested_dt = np.clip(suggested_dt, self.min_dt, self.max_dt)

        # Smooth adjustment: avoid drastic changes
        if len(self.dt_history) > 0:
            max_change_ratio = 2.0
            suggested_dt = np.clip(
                suggested_dt,
                self.current_dt / max_change_ratio,
                self.current_dt * max_change_ratio
            )
        
        # Update history
        self.cfl_history.append(cfl)
        self.convergence_history.append(converged)
        self.total_steps += 1

        if abs(suggested_dt - self.current_dt) > 1.0:
            self.adjustments += 1

        if not converged:
            self.failed_steps += 1

        # Create metrics object
        metrics = TimeStepMetrics(
            current_dt=self.current_dt,
            cfl_number=cfl,
            max_depth_change_rate=depth_change_rate,
            newton_iterations=newton_iterations,
            convergence_achieved=converged,
            suggested_dt=suggested_dt,
            reason=reason
        )

        # Apply new timestep
        self.current_dt = suggested_dt
        self.dt_history.append(suggested_dt)

        # Limit history length
        if len(self.dt_history) > 100:
            self.dt_history = self.dt_history[-100:]
        if len(self.cfl_history) > 100:
            self.cfl_history = self.cfl_history[-100:]
        if len(self.convergence_history) > 100:
            self.convergence_history = self.convergence_history[-100:]
        
        return metrics
    
    def get_statistics(self) -> dict:
        """Get statistical information"""
        avg_dt = np.mean(self.dt_history) if self.dt_history else 0
        avg_cfl = np.mean(self.cfl_history) if self.cfl_history else 0
        success_rate = (self.total_steps - self.failed_steps) / max(self.total_steps, 1)

        return {
            'total_steps': self.total_steps,
            'adjustments': self.adjustments,
            'failed_steps': self.failed_steps,
            'current_dt': self.current_dt,
            'avg_dt': avg_dt,
            'min_dt_used': min(self.dt_history) if self.dt_history else 0,
            'max_dt_used': max(self.dt_history) if self.dt_history else 0,
            'avg_cfl': avg_cfl,
            'success_rate': success_rate * 100
        }

    def reset(self, new_initial_dt: Optional[float] = None):
        """Reset controller"""
        if new_initial_dt is not None:
            self.current_dt = new_initial_dt
        self.dt_history = [self.current_dt]
        self.cfl_history = []
        self.convergence_history = []
        self.total_steps = 0
        self.adjustments = 0
        self.failed_steps = 0


class VariableTimeStepSimulator:
    """Variable timestep simulator wrapper class

    Works with fixed timestep solvers to automatically manage timestep adjustments
    """

    def __init__(self, controller: AdaptiveTimeStepController):
        self.controller = controller
        self.metrics_history: List[TimeStepMetrics] = []

    def run_adaptive_simulation(self,
                                solver,  # Original solver object
                                total_time: float,
                                boundary_conditions,
                                verbose: bool = True) -> dict:
        """Run adaptive timestep simulation

        Args:
            solver: Original solver instance (must have state, solve_timestep methods)
            total_time: Total simulation time (s)
            boundary_conditions: Boundary condition object
            verbose: Whether to output detailed information

        Returns:
            Results dictionary {'time': [...], 'discharge': [...], 'depth': [...]}
        """
        results = {
            'time': [],
            'discharge': [],
            'depth': [],
            'velocity': [],
            'dt': []
        }
        
        current_time = 0.0
        time_step_index = 0
        
        depth_prev = solver.state.depth.copy()

        if verbose:
            print(f"Starting adaptive simulation (target time: {total_time}s)")
            print(f"Initial timestep: {self.controller.current_dt}s")
            print("-" * 70)

        while current_time < total_time:
            # Update solver timestep
            solver.dt = self.controller.current_dt

            # Execute one timestep computation
            converged = solver.solve_timestep(boundary_conditions, time_step_index)

            # Get current state
            velocity = solver.state.velocity
            depth_current = solver.state.depth
            dx = solver.reach.dx

            # Estimate Newton iteration count (use default if solver doesn't provide)
            newton_iters = getattr(solver, 'last_newton_iterations', 10)

            # Update timestep controller
            metrics = self.controller.update(
                velocity, depth_current, depth_prev,
                dx, newton_iters, converged
            )

            self.metrics_history.append(metrics)

            # Record results
            results['time'].append(current_time)
            results['discharge'].append(solver.state.discharge.copy())
            results['depth'].append(depth_current.copy())
            results['velocity'].append(velocity.copy())
            results['dt'].append(self.controller.current_dt)

            # Update time
            current_time += metrics.current_dt
            time_step_index += 1

            # Save current depth for next step
            depth_prev = depth_current.copy()

            # Periodically output progress
            if verbose and time_step_index % 10 == 0:
                progress = current_time / total_time * 100
                print(f"Step {time_step_index:4d} | "
                      f"Time {current_time:7.1f}s ({progress:5.1f}%) | "
                      f"dt={metrics.current_dt:5.1f}s | "
                      f"CFL={metrics.cfl_number:.3f} | "
                      f"{metrics.reason}")

        if verbose:
            print("-" * 70)
            stats = self.controller.get_statistics()
            print(f"Simulation complete!")
            print(f"  Total steps: {stats['total_steps']}")
            print(f"  Timestep adjustments: {stats['adjustments']}")
            print(f"  Average timestep: {stats['avg_dt']:.1f}s")
            print(f"  Timestep range: {stats['min_dt_used']:.1f} - {stats['max_dt_used']:.1f}s")
            print(f"  Average CFL: {stats['avg_cfl']:.3f}")
            print(f"  Success rate: {stats['success_rate']:.1f}%")
        
        return results
    
    def plot_metrics(self, save_path: str = 'adaptive_metrics.png'):
        """Plot adaptive control metrics charts"""
        if not self.dt_history:
            print("No history data to plot")
            return

        import matplotlib.pyplot as plt

        # Set font for Unicode minus sign
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        times = np.cumsum(self.dt_history)

        # Timestep history
        ax1.plot(times, self.dt_history, 'b-', linewidth=2)
        ax1.axhline(y=self.min_dt, color='r', linestyle='--', alpha=0.7, label=f'Min timestep {self.min_dt}s')
        ax1.axhline(y=self.max_dt, color='g', linestyle='--', alpha=0.7, label=f'Max timestep {self.max_dt}s')
        ax1.set_xlabel('Cumulative time (s)')
        ax1.set_ylabel('Timestep size (s)')
        ax1.set_title('Adaptive Timestep History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # CFL number history
        if self.cfl_history:
            ax2.plot(times, self.cfl_history, 'r-', linewidth=2)
            ax2.axhline(y=self.target_cfl, color='k', linestyle='--', alpha=0.7, label=f'Target CFL {self.target_cfl}')
            ax2.set_xlabel('Cumulative time (s)')
            ax2.set_ylabel('CFL number')
            ax2.set_title('CFL Number Monitoring')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Adjustment strategy distribution
        if hasattr(self, 'strategy_history') and self.strategy_history:
            strategy_counts = {}
            for strategy in self.strategy_history:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            strategies = list(strategy_counts.keys())
            counts = list(strategy_counts.values())

            ax3.pie(counts, labels=strategies, autopct='%1.1f%%')
            ax3.set_title('Adjustment Strategy Distribution')

        # Efficiency metrics
        total_time = sum(self.dt_history)
        fixed_dt_time = len(self.dt_history) * self.initial_dt
        efficiency = fixed_dt_time / total_time if total_time > 0 else 1

        ax4.bar(['Fixed timestep', 'Adaptive timestep'], [fixed_dt_time, total_time],
                color=['lightblue', 'lightgreen'])
        ax4.set_ylabel('Total compute time (s)')
        ax4.set_title(f'Efficiency comparison ({efficiency:.1f}x improvement)')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")


if __name__ == "__main__":
    # Test example
    print("="*70)
    print("Adaptive Timestep Controller Test")
    print("="*70)

    # Create controller
    controller = AdaptiveTimeStepController(
        initial_dt=60,
        min_dt=10,
        max_dt=300,
        target_cfl=0.5,
        strategy=AdaptiveStrategy.HYBRID
    )

    # Simulate series of state changes
    print("\nSimulation scenario: Flood event")
    print("-" * 70)

    # Simulate velocity and depth changes
    num_sections = 20
    dx = 500.0

    for step in range(20):
        # Simulate flood rise and fall
        t_normalized = step / 20.0
        peak_velocity = 2.0 + 3.0 * np.sin(t_normalized * np.pi)

        velocity = np.full(num_sections, peak_velocity) + \
                  np.random.normal(0, 0.2, num_sections)
        depth_current = np.full(num_sections, 2.0 + peak_velocity * 0.5)
        depth_previous = depth_current - 0.1 * np.random.random(num_sections)

        # Simulate convergence status
        converged = np.random.random() > 0.1
        newton_iters = np.random.randint(5, 15) if converged else 25

        # Update controller
        metrics = controller.update(
            velocity, depth_current, depth_previous,
            dx, newton_iters, converged
        )

        if step % 5 == 0:
            print(f"Step {step:2d}: dt={metrics.current_dt:5.1f}s | "
                  f"CFL={metrics.cfl_number:.3f} | "
                  f"Converged={metrics.convergence_achieved} | "
                  f"{metrics.reason}")

    # Display statistics
    print("\n" + "="*70)
    stats = controller.get_statistics()
    print("Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")