"""Parameter management utilities.

This package has been modularized for better maintainability:
- zone.py: Parameter zone definitions and builders
- optimization.py: Parameter optimization and uncertainty analysis
- partition_models.py: Data models for partitioning
- partition_grid.py: Grid operations and flow analysis
- partition_rebalance.py: Zone rebalancing logic
- partition_builder.py: Main partitioning algorithm
- partition.py: Original combined file (maintained for backward compatibility)
"""

from .zone import ParameterZone, ParameterZoneBuilder, ParameterZoneConfig
from .optimization import (
    ObjectiveDefinition,
    OptimizationResult,
    ParameterZoneOptimizer,
    UncertaintyAnalyzer,
)

# New modular imports
from .partition_models import (
    ZoneSummary,
    SubzoneSummary,
    ChannelSummary,
    PartitionOutputs,
)

# Main partition function from builder
try:
    from .partition_builder import partition_parameter_zones
except ImportError:
    # Fallback to original if new modules have issues
    try:
        from .partition import partition_parameter_zones
    except ImportError:
        partition_parameter_zones = None


__all__ = [
    # Core zone management
    "ParameterZone",
    "ParameterZoneBuilder",
    "ParameterZoneConfig",
    # Optimization
    "ObjectiveDefinition",
    "OptimizationResult",
    "ParameterZoneOptimizer",
    "UncertaintyAnalyzer",
    # Partition data models
    "ZoneSummary",
    "SubzoneSummary",
    "ChannelSummary",
    "PartitionOutputs",
    # Main partition function
    "partition_parameter_zones",
]
