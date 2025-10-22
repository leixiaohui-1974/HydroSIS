"""Channel cross-section geometry computation module

Supports hydraulic calculations for multiple cross-section types:
- Rectangle cross-section
- Trapezoid cross-section
- Compound cross-section - floodplain + main channel
- Irregular cross-section - based on surveyed point interpolation
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HydraulicProperties:
    """Hydraulic geometry parameter collection"""

    area: float          # Flow area (m²)
    wetted_perimeter: float  # Wetted perimeter (m)
    hydraulic_radius: float  # Hydraulic radius R=A/P (m)
    top_width: float     # Water surface width (m)
    hydraulic_depth: float   # Hydraulic depth D=A/T (m)
    conveyance: float    # Conveyance K=A*R^(2/3) (m^(8/3))


class CrossSection(ABC):
    """Cross-section base class - defines uniform interface"""

    @abstractmethod
    def compute_properties(self, depth: float) -> HydraulicProperties:
        """Calculate hydraulic parameters from depth"""
        pass

    @abstractmethod
    def compute_depth_from_area(self, area: float,
                                tolerance: float = 1e-4) -> float:
        """Calculate depth from flow area (for implicit solution)"""
        pass

    @abstractmethod
    def get_max_depth(self) -> float:
        """Return maximum allowable depth of cross-section"""
        pass


class RectangleSection(CrossSection):
    """Rectangle cross-section

    ┌────────────┐  ← Top width b
    │            │
    │            │  h (depth)
    │            │
    └────────────┘
    """

    def __init__(self, width: float, max_depth: float = 10.0):
        """
        Parameters:
            width: Channel width (m)
            max_depth: Maximum depth limit (m)
        """
        if width <= 0:
            raise ValueError("Channel width must be positive")
        self.width = width
        self._max_depth = max_depth
    
    def compute_properties(self, depth: float) -> HydraulicProperties:
        depth = max(0.0, min(depth, self._max_depth))
        
        area = self.width * depth
        wetted_perimeter = self.width + 2 * depth
        hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0 else 0
        
        return HydraulicProperties(
            area=area,
            wetted_perimeter=wetted_perimeter,
            hydraulic_radius=hydraulic_radius,
            top_width=self.width,
            hydraulic_depth=depth,
            conveyance=area * hydraulic_radius**(2/3) if hydraulic_radius > 0 else 0
        )
    
    def compute_depth_from_area(self, area: float,
                                tolerance: float = 1e-4) -> float:
        # Rectangle cross-section: h = A / b (analytical solution)
        return min(area / self.width, self._max_depth)

    def get_max_depth(self) -> float:
        return self._max_depth


class TrapezoidSection(CrossSection):
    """Trapezoid cross-section

          T (top width)
    ╱‾‾‾‾‾‾‾‾‾‾‾╲
    ╱            ╲
   ╱ main channel╲  h (depth)
  ╱   m:1  |  m:1 ╲
 ╱_________________╲
        b

    where m is side slope coefficient (horizontal:vertical)
    """

    def __init__(self, bottom_width: float, side_slope: float,
                 max_depth: float = 10.0):
        """
        Parameters:
            bottom_width: Bottom width b (m)
            side_slope: Side slope coefficient m (horizontal/vertical ratio)
                       e.g., m=2 means 2:1 slope (2m horizontal per 1m vertical)
            max_depth: Maximum depth limit (m)
        """
        if bottom_width <= 0 or side_slope < 0:
            raise ValueError("Bottom width must be positive, side slope must be non-negative")

        self.bottom_width = bottom_width
        self.side_slope = side_slope
        self._max_depth = max_depth

    def compute_properties(self, depth: float) -> HydraulicProperties:
        depth = max(0.0, min(depth, self._max_depth))

        # Trapezoid area: A = (b + m*h) * h
        area = (self.bottom_width + self.side_slope * depth) * depth

        # Top width: T = b + 2*m*h
        top_width = self.bottom_width + 2 * self.side_slope * depth

        # Wetted perimeter: P = b + 2*h*sqrt(1 + m²)
        side_length = depth * np.sqrt(1 + self.side_slope**2)
        wetted_perimeter = self.bottom_width + 2 * side_length
        
        hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0 else 0
        hydraulic_depth = area / top_width if top_width > 0 else 0
        
        return HydraulicProperties(
            area=area,
            wetted_perimeter=wetted_perimeter,
            hydraulic_radius=hydraulic_radius,
            top_width=top_width,
            hydraulic_depth=hydraulic_depth,
            conveyance=area * hydraulic_radius**(2/3) if hydraulic_radius > 0 else 0
        )
    
    def compute_depth_from_area(self, area: float,
                                tolerance: float = 1e-4) -> float:
        # Trapezoid cross-section: A = (b + m*h) * h
        # Rearrange to: m*h² + b*h - A = 0
        # Solve quadratic equation: h = (-b + sqrt(b² + 4*m*A)) / (2*m)

        if self.side_slope < 1e-6:  # Degenerates to rectangle
            return area / self.bottom_width

        b = self.bottom_width
        m = self.side_slope

        discriminant = b**2 + 4 * m * area
        if discriminant < 0:
            return 0.0

        depth = (-b + np.sqrt(discriminant)) / (2 * m)
        return min(depth, self._max_depth)

    def get_max_depth(self) -> float:
        return self._max_depth


class CompoundSection(CrossSection):
    """Compound cross-section - main channel + left/right floodplains

    Left FP  |  Main Ch  | Right FP
    ────┐   ┌────────┐   ┌────
        │   │        │   │
        │   │        │   │  Main channel depth h
        └───┘        └───┘  Floodplain height h_fp
    """

    def __init__(self,
                 main_bottom_width: float,
                 main_side_slope: float,
                 floodplain_height: float,
                 left_floodplain_width: float = 0.0,
                 right_floodplain_width: float = 0.0,
                 max_depth: float = 15.0):
        """
        Parameters:
            main_bottom_width: Main channel bottom width (m)
            main_side_slope: Main channel side slope coefficient
            floodplain_height: Floodplain height from main channel bottom (m)
            left_floodplain_width: Left floodplain width (m)
            right_floodplain_width: Right floodplain width (m)
            max_depth: Maximum depth limit (m)
        """
        self.main_channel = TrapezoidSection(
            main_bottom_width, 
            main_side_slope,
            max_depth=floodplain_height
        )
        self.floodplain_height = floodplain_height
        self.left_fp_width = left_floodplain_width
        self.right_fp_width = right_floodplain_width
        self._max_depth = max_depth
    
    def compute_properties(self, depth: float) -> HydraulicProperties:
        depth = max(0.0, min(depth, self._max_depth))

        if depth <= self.floodplain_height:
            # Water level below floodplain, only main channel carries flow
            return self.main_channel.compute_properties(depth)

        # Water level inundates floodplain, calculate main channel and floodplain separately
        main_props = self.main_channel.compute_properties(self.floodplain_height)

        # Floodplain depth
        fp_depth = depth - self.floodplain_height

        # Floodplain area (rectangular)
        fp_area = (self.left_fp_width + self.right_fp_width) * fp_depth

        # Total area
        total_area = main_props.area + fp_area

        # Floodplain wetted perimeter (only underwater portion)
        fp_perimeter = self.left_fp_width + self.right_fp_width

        # Total wetted perimeter
        total_perimeter = main_props.wetted_perimeter + fp_perimeter

        # Top width
        main_top_width = self.main_channel.bottom_width + \
                        2 * self.main_channel.side_slope * self.floodplain_height
        total_top_width = main_top_width + self.left_fp_width + self.right_fp_width

        hydraulic_radius = total_area / total_perimeter if total_perimeter > 0 else 0
        hydraulic_depth = total_area / total_top_width if total_top_width > 0 else 0

        return HydraulicProperties(
            area=total_area,
            wetted_perimeter=total_perimeter,
            hydraulic_radius=hydraulic_radius,
            top_width=total_top_width,
            hydraulic_depth=hydraulic_depth,
            conveyance=total_area * hydraulic_radius**(2/3) if hydraulic_radius > 0 else 0
        )

    def compute_depth_from_area(self, area: float,
                                tolerance: float = 1e-4) -> float:
        # Newton iteration method
        depth = 1.0  # Initial value

        for _ in range(20):
            props = self.compute_properties(depth)
            residual = props.area - area

            if abs(residual) < tolerance:
                return depth

            # Numerical differentiation for derivative dA/dh
            delta = 0.01
            props_plus = self.compute_properties(depth + delta)
            derivative = (props_plus.area - props.area) / delta

            if abs(derivative) < 1e-10:
                break

            # Newton update
            depth -= residual / derivative
            depth = max(0.01, min(depth, self._max_depth))

        return depth

    def get_max_depth(self) -> float:
        return self._max_depth


class IrregularSection(CrossSection):
    """Irregular cross-section - based on surveyed points

    Defined by a series of (y, z) coordinate points, where:
    - y: Transverse coordinate (m)
    - z: Elevation (m, relative to datum)
    """

    def __init__(self, y_coords: List[float], z_coords: List[float],
                 base_elevation: float = 0.0):
        """
        Parameters:
            y_coords: Transverse coordinate sequence, from left bank to right bank (m)
            z_coords: Corresponding elevation sequence (m)
            base_elevation: Channel bottom datum elevation (m)
        """
        if len(y_coords) != len(z_coords) or len(y_coords) < 3:
            raise ValueError("At least 3 cross-section points required, and x/y coordinates must have equal length")

        # Ensure sorted from left to right
        sorted_pairs = sorted(zip(y_coords, z_coords))
        self.y_coords = np.array([y for y, z in sorted_pairs])
        self.z_coords = np.array([z for y, z in sorted_pairs])

        self.base_elevation = base_elevation
        self._max_depth = np.max(self.z_coords) - np.min(self.z_coords) + 5.0
    
    def compute_properties(self, depth: float) -> HydraulicProperties:
        depth = max(0.0, min(depth, self._max_depth))

        # Current water surface elevation
        water_surface = self.base_elevation + depth

        # Find submerged cross-section segments
        area = 0.0
        wetted_perimeter = 0.0
        submerged_y = []

        for i in range(len(self.y_coords) - 1):
            y1, z1 = self.y_coords[i], self.z_coords[i]
            y2, z2 = self.y_coords[i + 1], self.z_coords[i + 1]

            # Check if segment is underwater
            if z1 > water_surface and z2 > water_surface:
                continue  # Completely above water surface

            # Handle partial submersion
            if z1 < water_surface and z2 < water_surface:
                # Completely submerged
                segment_width = y2 - y1
                segment_height = (water_surface - z1 + water_surface - z2) / 2
                area += segment_width * segment_height
                wetted_perimeter += np.sqrt((y2 - y1)**2 + (z2 - z1)**2)
                submerged_y.extend([y1, y2])
            else:
                # Partially submerged, need interpolation to find intersection
                if z1 > water_surface:
                    z1, z2 = z2, z1
                    y1, y2 = y2, y1

                # Linear interpolation to find water surface intersection
                t = (water_surface - z1) / (z2 - z1) if abs(z2 - z1) > 1e-10 else 0
                y_intersect = y1 + t * (y2 - y1)

                segment_width = abs(y_intersect - y1)
                segment_height = (water_surface - z1) / 2
                area += segment_width * segment_height
                wetted_perimeter += np.sqrt((y_intersect - y1)**2 +
                                           (water_surface - z1)**2)
                submerged_y.extend([y1, y_intersect])

        # Water surface width
        if len(submerged_y) >= 2:
            top_width = max(submerged_y) - min(submerged_y)
        else:
            top_width = 0.0

        hydraulic_radius = area / wetted_perimeter if wetted_perimeter > 0 else 0
        hydraulic_depth = area / top_width if top_width > 0 else 0

        return HydraulicProperties(
            area=area,
            wetted_perimeter=wetted_perimeter,
            hydraulic_radius=hydraulic_radius,
            top_width=top_width,
            hydraulic_depth=hydraulic_depth,
            conveyance=area * hydraulic_radius**(2/3) if hydraulic_radius > 0 else 0
        )

    def compute_depth_from_area(self, area: float,
                                tolerance: float = 1e-4) -> float:
        # Bisection method
        depth_min, depth_max = 0.0, self._max_depth

        for _ in range(50):
            depth_mid = (depth_min + depth_max) / 2
            props = self.compute_properties(depth_mid)

            if abs(props.area - area) < tolerance:
                return depth_mid

            if props.area < area:
                depth_min = depth_mid
            else:
                depth_max = depth_mid

        return (depth_min + depth_max) / 2

    def get_max_depth(self) -> float:
        return self._max_depth


# ============ Factory Functions ============

def create_cross_section(section_type: str, **parameters) -> CrossSection:
    """Factory function: create cross-section object based on type

    Parameters:
        section_type: 'rectangle', 'trapezoid', 'compound', 'irregular'
        **parameters: Cross-section specific parameters

    Example:
        >>> section = create_cross_section('trapezoid',
        ...                                bottom_width=10,
        ...                                side_slope=2.0)
    """
    section_type = section_type.lower()
    
    if section_type == 'rectangle':
        return RectangleSection(
            width=parameters['width'],
            max_depth=parameters.get('max_depth', 10.0)
        )
    
    elif section_type == 'trapezoid':
        return TrapezoidSection(
            bottom_width=parameters['bottom_width'],
            side_slope=parameters['side_slope'],
            max_depth=parameters.get('max_depth', 10.0)
        )
    
    elif section_type == 'compound':
        return CompoundSection(
            main_bottom_width=parameters['main_bottom_width'],
            main_side_slope=parameters['main_side_slope'],
            floodplain_height=parameters['floodplain_height'],
            left_floodplain_width=parameters.get('left_floodplain_width', 0),
            right_floodplain_width=parameters.get('right_floodplain_width', 0),
            max_depth=parameters.get('max_depth', 15.0)
        )
    
    elif section_type == 'irregular':
        return IrregularSection(
            y_coords=parameters['y_coords'],
            z_coords=parameters['z_coords'],
            base_elevation=parameters.get('base_elevation', 0.0)
        )
    
    else:
        raise ValueError(f"Unsupported cross-section type: {section_type}")


# ============ Utility Functions ============

def compute_normal_depth(section: CrossSection, discharge: float,
                        bed_slope: float, manning_n: float,
                        tolerance: float = 1e-4) -> float:
    """Calculate normal depth (Manning formula)

    Q = (1/n) * A * R^(2/3) * S^(1/2)

    Parameters:
        section: Cross-section object
        discharge: Discharge (m³/s)
        bed_slope: Bed slope
        manning_n: Manning coefficient
        tolerance: Convergence tolerance

    Returns:
        Normal depth (m)
    """
    depth = 1.0  # Initial value

    for _ in range(50):
        props = section.compute_properties(depth)

        # Manning discharge calculation
        Q_computed = (1 / manning_n) * props.area * \
                     props.hydraulic_radius**(2/3) * np.sqrt(bed_slope)

        residual = Q_computed - discharge

        if abs(residual) < tolerance:
            return depth

        # Numerical derivative
        delta = 0.01
        props_plus = section.compute_properties(depth + delta)
        Q_plus = (1 / manning_n) * props_plus.area * \
                 props_plus.hydraulic_radius**(2/3) * np.sqrt(bed_slope)

        derivative = (Q_plus - Q_computed) / delta

        if abs(derivative) < 1e-10:
            break

        # Newton update
        depth -= residual / derivative
        depth = max(0.1, min(depth, section.get_max_depth()))

    return depth


def compute_critical_depth(section: CrossSection, discharge: float,
                          tolerance: float = 1e-4) -> float:
    """Calculate critical depth

    Depth when Fr = 1, i.e., Q²T/(gA³) = 1

    Parameters:
        section: Cross-section object
        discharge: Discharge (m³/s)
        tolerance: Convergence tolerance

    Returns:
        Critical depth (m)
    """
    g = 9.81
    depth = 1.0
    
    for _ in range(50):
        props = section.compute_properties(depth)
        
        if props.area < 1e-6:
            depth += 0.1
            continue

        # Froude number
        Fr_squared = discharge**2 * props.top_width / (g * props.area**3)
        residual = Fr_squared - 1.0

        if abs(residual) < tolerance:
            return depth

        # Numerical derivative
        delta = 0.01
        props_plus = section.compute_properties(depth + delta)
        Fr_squared_plus = discharge**2 * props_plus.top_width / \
                         (g * props_plus.area**3)
        derivative = (Fr_squared_plus - Fr_squared) / delta

        if abs(derivative) < 1e-10:
            break

        depth -= residual / derivative
        depth = max(0.1, min(depth, section.get_max_depth()))

    return depth


if __name__ == "__main__":
    # Test different cross-section types
    print("="*60)
    print("Cross-Section Geometry Module Test")
    print("="*60)

    test_depth = 3.0

    # Test 1: Rectangle cross-section
    print("\n1. Rectangle cross-section (width 30m)")
    rect = RectangleSection(width=30)
    props = rect.compute_properties(test_depth)
    print(f"   Depth = {test_depth} m")
    print(f"   Area = {props.area:.2f} m²")
    print(f"   Hydraulic radius = {props.hydraulic_radius:.2f} m")

    # Test 2: Trapezoid cross-section
    print("\n2. Trapezoid cross-section (bottom width 10m, side slope 2:1)")
    trap = TrapezoidSection(bottom_width=10, side_slope=2.0)
    props = trap.compute_properties(test_depth)
    print(f"   Depth = {test_depth} m")
    print(f"   Area = {props.area:.2f} m²")
    print(f"   Top width = {props.top_width:.2f} m")
    print(f"   Hydraulic radius = {props.hydraulic_radius:.2f} m")

    # Test 3: Compound cross-section
    print("\n3. Compound cross-section (main channel 8m + floodplains 15m each)")
    compound = CompoundSection(
        main_bottom_width=8,
        main_side_slope=1.5,
        floodplain_height=2.5,
        left_floodplain_width=15,
        right_floodplain_width=15
    )

    print("   In main channel (h=2m):")
    props = compound.compute_properties(2.0)
    print(f"      Area = {props.area:.2f} m²")

    print("   After overbank (h=4m):")
    props = compound.compute_properties(4.0)
    print(f"      Area = {props.area:.2f} m²")
    print(f"      Top width = {props.top_width:.2f} m")

    # Test 4: Normal depth calculation
    print("\n4. Normal depth calculation (Q=50 m³/s, S=0.001, n=0.03)")
    normal_depth = compute_normal_depth(trap, 50, 0.001, 0.03)
    print(f"   Normal depth = {normal_depth:.2f} m")

    # Test 5: Critical depth
    print("\n5. Critical depth calculation (Q=50 m³/s)")
    critical_depth = compute_critical_depth(trap, 50)
    print(f"   Critical depth = {critical_depth:.2f} m")

    if normal_depth > critical_depth:
        print("   → Subcritical flow")
    else:
        print("   → Supercritical flow")