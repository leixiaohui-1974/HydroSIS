"""简化包装 - 保持向后兼容

建议新代码使用 hydrosis.hydrodynamics 模块
"""

import warnings
from hydrosis.hydrodynamics import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    HydrodynamicRoutingModel
)

warnings.warn(
    "routing.hydrodynamic_1d 已废弃，请使用 hydrosis.hydrodynamics",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'SaintVenantSolver',
    'RiverReach',
    'BoundaryCondition',
    'HydrodynamicRoutingModel'
]