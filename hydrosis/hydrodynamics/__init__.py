"""HydroSIS 一维水动力模块

提供完整的圣维南方程求解能力，支持：
- 多种断面形式（矩形、梯形、复合、不规则）
- 自适应时间步长控制
- GPU加速计算
- 稳态流计算
- 与HydroSIS无缝集成
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

from .gpu_solver import (
    GPUSaintVenantSolver,
    BatchSimulator,
    DeviceManager,
    GPU_AVAILABLE
)

from .steady_state import (
    SteadyStateCalculator,
    compute_normal_depth as compute_normal_depth_standalone,
    compute_critical_depth as compute_critical_depth_standalone
)

# 保持向后兼容
from .core import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    HydraulicState
)

__all__ = [
    # 断面几何
    'CrossSection',
    'RectangleSection',
    'TrapezoidSection',
    'CompoundSection',
    'IrregularSection',
    'create_cross_section',
    'compute_normal_depth',
    'compute_critical_depth',
    
    # 自适应控制
    'AdaptiveTimeStepController',
    'AdaptiveStrategy',
    'TimeStepMetrics',
    'VariableTimeStepSimulator',
    
    # GPU加速
    'GPUSaintVenantSolver',
    'BatchSimulator',
    'DeviceManager',
    'GPU_AVAILABLE',
    
    # 稳态计算
    'SteadyStateCalculator',
    'compute_normal_depth_standalone',
    'compute_critical_depth_standalone',
    
    # 核心求解器
    'SaintVenantSolver',
    'RiverReach',
    'BoundaryCondition',
    'HydraulicState'
]

__version__ = '1.2.0'