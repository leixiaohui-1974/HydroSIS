"""Delineation utilities.

This package has been modularized for better maintainability:
- dem_delineator.py: DEM-based watershed delineation
- pour_points.py: Pour point data structures and I/O
- tree_generation.py: Tree-based pour point generation algorithms
- network.py: Flow network construction and analysis
- visualization.py: Plotting and output utilities
- utils.py: Original combined file (maintained for backward compatibility)
"""

from .dem_delineator import DelineationConfig

# Import from new modular structure
try:
    from .pour_points import PourPoint, read_pour_points_geojson, write_pour_points_geojson
    from .tree_generation import generate_tree_pour_points
    from .network import build_flow_network, delineate_watershed
except ImportError:
    # Fallback to original utils.py if new modules have issues
    try:
        from .utils import (
            PourPoint,
            read_pour_points_geojson,
            write_pour_points_geojson,
            generate_tree_pour_points,
            build_flow_network,
            delineate_watershed,
        )
    except ImportError:
        # Set to None if neither works
        PourPoint = None
        read_pour_points_geojson = None
        write_pour_points_geojson = None
        generate_tree_pour_points = None
        build_flow_network = None
        delineate_watershed = None


__all__ = [
    "DelineationConfig",
    "PourPoint",
    "read_pour_points_geojson",
    "write_pour_points_geojson",
    "generate_tree_pour_points",
    "build_flow_network",
    "delineate_watershed",
]
