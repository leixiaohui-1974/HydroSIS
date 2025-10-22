"""HydroSIS 1D Hydrodynamics Module

Provides complete Saint-Venant equation solving capabilities, supporting:
- Multiple cross-section types (rectangle, trapezoid, compound, irregular)
- Adaptive time step control
- GPU-accelerated computation
- Steady-state flow calculation
- Seamless integration with HydroSIS
"""

from .geometry import (
    CrossSection,
    RectangleSection,
    TrapezoidSection,
    CompoundSection,
    IrregularSection,
    create_cross_section,
    compute_normal_depth,
    compute_critical_depth
)

from .adaptive_timestep import (
    AdaptiveTimeStepController,
    AdaptiveStrategy,
    TimeStepMetrics,
    VariableTimeStepSimulator
)

try:  # pragma: no cover - optional GPU support
    from .gpu_solver import (
        GPUSaintVenantSolver,
        BatchSimulator,
        DeviceManager,
        GPU_AVAILABLE,
    )
except Exception:  # pragma: no cover - fallback when cupy unavailable
    GPUSaintVenantSolver = None  # type: ignore[assignment]
    BatchSimulator = None  # type: ignore[assignment]
    DeviceManager = None  # type: ignore[assignment]
    GPU_AVAILABLE = False

from .steady_state import (
    SteadyStateCalculator,
    compute_normal_depth as compute_normal_depth_standalone,
    compute_critical_depth as compute_critical_depth_standalone
)

from .zone_geometry import (
    ZoneGeometry,
    build_zone_geometry,
    load_zone_centerline,
)

from .cross_section_solver import (
    CrossSectionSolver,
    RatingCurve,
)

# Maintain backward compatibility
from .core import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    HydraulicState
)

__all__ = [
    # Cross-section geometry
    'CrossSection',
    'RectangleSection',
    'TrapezoidSection',
    'CompoundSection',
    'IrregularSection',
    'create_cross_section',
    'compute_normal_depth',
    'compute_critical_depth',

    # Adaptive control
    'AdaptiveTimeStepController',
    'AdaptiveStrategy',
    'TimeStepMetrics',
    'VariableTimeStepSimulator',

    # GPU acceleration
    'GPUSaintVenantSolver',
    'BatchSimulator',
    'DeviceManager',
    'GPU_AVAILABLE',

    # Steady-state computation
    'SteadyStateCalculator',
    'compute_normal_depth_standalone',
    'compute_critical_depth_standalone',

    # Core solver
    'SaintVenantSolver',
    'RiverReach',
    'BoundaryCondition',
    'HydraulicState',

    # Zone geometry
    'ZoneGeometry',
    'build_zone_geometry',
    'load_zone_centerline',

    # Simplified cross-section solver
    'CrossSectionSolver',
    'RatingCurve',
]

if GPUSaintVenantSolver is None:  # pragma: no cover
    for name in ['GPUSaintVenantSolver', 'BatchSimulator', 'DeviceManager', 'GPU_AVAILABLE']:
        try:
            __all__.remove(name)
        except ValueError:
            pass

__version__ = '1.2.0'
