"""Simplified wrapper - maintain backward compatibility

New code should use the hydrosis.hydrodynamics module
"""

import warnings
from hydrosis.hydrodynamics import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    HydrodynamicRoutingModel
)

warnings.warn(
    "routing.hydrodynamic_1d is deprecated, please use hydrosis.hydrodynamics",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'SaintVenantSolver',
    'RiverReach',
    'BoundaryCondition',
    'HydrodynamicRoutingModel'
]