"""河道断面几何计算模块

支持多种断面形式的水力计算：
- 矩形断面 (Rectangle)
- 梯形断面 (Trapezoid)
- 复合断面 (Compound) - 滩地+主槽
- 自定义断面 (Irregular) - 基于测点插值
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HydraulicProperties:
    """水力几何参数集合"""
    
    area: float          # 过水面积 (m²)
    wetted_perimeter: float  # 湿周 (m)
    hydraulic_radius: float  # 水力半径 R=A/P (m)
    top_width: float     # 水面宽度 (m)
    hydraulic_depth: float   # 水力水深 D=A/T (m)
    conveyance: float    # 流量模数 K=A*R^(2/3) (m^(8/3))


class CrossSection(ABC):
    """断面基类 - 定义统一接口"""
    
    @abstractmethod
    def compute_properties(self, depth: float) -> HydraulicProperties:
        """根据水深计算水力参数"""
        pass
    
    @abstractmethod
    def compute_depth_from_area(self, area: float, 
                                tolerance: float = 1e-4) -> float:
        """根据过水面积反算水深 (用于隐式求解)"""
        pass
    
    @abstractmethod
    def get_max_depth(self) -> float:
        """返回断面最大允许水深"""
        pass


class RectangleSection(CrossSection):
    """矩形断面
    
    ┌────────────┐  ← 顶部宽度 b
    │            │
    │            │  h (水深)
    │            │
    └────────────┘
    """
    
    def __init__(self, width: float, max_depth: float = 10.0):
        """
        参数:
            width: 河宽 (m)
            max_depth: 最大水深限制 (m)
        """
        if width <= 0:
            raise ValueError("河宽必须为正数")
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
        # 矩形断面: h = A / b (解析解)
        return min(area / self.width, self._max_depth)
    
    def get_max_depth(self) -> float:
        return self._max_depth


class TrapezoidSection(CrossSection):
    """梯形断面
    
          T (顶宽)
    ╱‾‾‾‾‾‾‾‾‾‾‾╲
    ╱            ╲
   ╱   主槽宽b    ╲  h (水深)
  ╱   m:1  |  m:1 ╲
 ╱_________________╲
        b
    
    其中 m 为边坡系数 (水平:垂直)
    """
    
    def __init__(self, bottom_width: float, side_slope: float, 
                 max_depth: float = 10.0):
        """
        参数:
            bottom_width: 底宽 b (m)
            side_slope: 边坡系数 m (水平/垂直比)
                       例: m=2 表示2:1边坡 (2米水平对应1米垂直)
            max_depth: 最大水深限制 (m)
        """
        if bottom_width <= 0 or side_slope < 0:
            raise ValueError("底宽必须为正，边坡系数非负")
        
        self.bottom_width = bottom_width
        self.side_slope = side_slope
        self._max_depth = max_depth
    
    def compute_properties(self, depth: float) -> HydraulicProperties:
        depth = max(0.0, min(depth, self._max_depth))
        
        # 梯形面积: A = (b + m*h) * h
        area = (self.bottom_width + self.side_slope * depth) * depth
        
        # 顶宽: T = b + 2*m*h
        top_width = self.bottom_width + 2 * self.side_slope * depth
        
        # 湿周: P = b + 2*h*sqrt(1 + m²)
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
        # 梯形断面: A = (b + m*h) * h
        # 整理为: m*h² + b*h - A = 0
        # 求解二次方程: h = (-b + sqrt(b² + 4*m*A)) / (2*m)
        
        if self.side_slope < 1e-6:  # 退化为矩形
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
    """复合断面 - 主槽+左右滩地
    
    左滩地  |   主槽    |  右滩地
    ────┐   ┌────────┐   ┌────
        │   │        │   │
        │   │        │   │  主槽水深 h
        └───┘        └───┘  滩地高度 h_fp
    """
    
    def __init__(self, 
                 main_bottom_width: float,
                 main_side_slope: float,
                 floodplain_height: float,
                 left_floodplain_width: float = 0.0,
                 right_floodplain_width: float = 0.0,
                 max_depth: float = 15.0):
        """
        参数:
            main_bottom_width: 主槽底宽 (m)
            main_side_slope: 主槽边坡系数
            floodplain_height: 滩地高度，从主槽底部算起 (m)
            left_floodplain_width: 左滩地宽度 (m)
            right_floodplain_width: 右滩地宽度 (m)
            max_depth: 最大水深限制 (m)
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
            # 水位未漫滩，仅主槽过流
            return self.main_channel.compute_properties(depth)
        
        # 水位漫滩，分别计算主槽和滩地
        main_props = self.main_channel.compute_properties(self.floodplain_height)
        
        # 滩地水深
        fp_depth = depth - self.floodplain_height
        
        # 滩地面积 (矩形)
        fp_area = (self.left_fp_width + self.right_fp_width) * fp_depth
        
        # 总面积
        total_area = main_props.area + fp_area
        
        # 滩地湿周 (只计算水下部分)
        fp_perimeter = self.left_fp_width + self.right_fp_width
        
        # 总湿周
        total_perimeter = main_props.wetted_perimeter + fp_perimeter
        
        # 顶宽
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
        # 牛顿迭代法求解
        depth = 1.0  # 初值
        
        for _ in range(20):
            props = self.compute_properties(depth)
            residual = props.area - area
            
            if abs(residual) < tolerance:
                return depth
            
            # 数值微分求导数 dA/dh
            delta = 0.01
            props_plus = self.compute_properties(depth + delta)
            derivative = (props_plus.area - props.area) / delta
            
            if abs(derivative) < 1e-10:
                break
            
            # 牛顿更新
            depth -= residual / derivative
            depth = max(0.01, min(depth, self._max_depth))
        
        return depth
    
    def get_max_depth(self) -> float:
        return self._max_depth


class IrregularSection(CrossSection):
    """不规则断面 - 基于实测断面点
    
    通过一系列 (y, z) 坐标点定义，其中:
    - y: 横向坐标 (m)
    - z: 高程 (m，相对于基准面)
    """
    
    def __init__(self, y_coords: List[float], z_coords: List[float],
                 base_elevation: float = 0.0):
        """
        参数:
            y_coords: 横坐标序列，从左岸到右岸 (m)
            z_coords: 对应的高程序列 (m)
            base_elevation: 河底基准高程 (m)
        """
        if len(y_coords) != len(z_coords) or len(y_coords) < 3:
            raise ValueError("断面点数量至少3个，且横纵坐标数量相等")
        
        # 确保从左到右排序
        sorted_pairs = sorted(zip(y_coords, z_coords))
        self.y_coords = np.array([y for y, z in sorted_pairs])
        self.z_coords = np.array([z for y, z in sorted_pairs])
        
        self.base_elevation = base_elevation
        self._max_depth = np.max(self.z_coords) - np.min(self.z_coords) + 5.0
    
    def compute_properties(self, depth: float) -> HydraulicProperties:
        depth = max(0.0, min(depth, self._max_depth))
        
        # 当前水位高程
        water_surface = self.base_elevation + depth
        
        # 找到水下的断面段
        area = 0.0
        wetted_perimeter = 0.0
        submerged_y = []
        
        for i in range(len(self.y_coords) - 1):
            y1, z1 = self.y_coords[i], self.z_coords[i]
            y2, z2 = self.y_coords[i + 1], self.z_coords[i + 1]
            
            # 判断该段是否在水下
            if z1 > water_surface and z2 > water_surface:
                continue  # 完全露出水面
            
            # 处理部分淹没情况
            if z1 < water_surface and z2 < water_surface:
                # 完全淹没
                segment_width = y2 - y1
                segment_height = (water_surface - z1 + water_surface - z2) / 2
                area += segment_width * segment_height
                wetted_perimeter += np.sqrt((y2 - y1)**2 + (z2 - z1)**2)
                submerged_y.extend([y1, y2])
            else:
                # 部分淹没，需要插值找交点
                if z1 > water_surface:
                    z1, z2 = z2, z1
                    y1, y2 = y2, y1
                
                # 线性插值找水面交点
                t = (water_surface - z1) / (z2 - z1) if abs(z2 - z1) > 1e-10 else 0
                y_intersect = y1 + t * (y2 - y1)
                
                segment_width = abs(y_intersect - y1)
                segment_height = (water_surface - z1) / 2
                area += segment_width * segment_height
                wetted_perimeter += np.sqrt((y_intersect - y1)**2 + 
                                           (water_surface - z1)**2)
                submerged_y.extend([y1, y_intersect])
        
        # 水面宽度
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
        # 二分法求解
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


# ============ 工厂函数 ============

def create_cross_section(section_type: str, **parameters) -> CrossSection:
    """工厂函数：根据类型创建断面对象
    
    参数:
        section_type: 'rectangle', 'trapezoid', 'compound', 'irregular'
        **parameters: 断面特定参数
    
    示例:
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
        raise ValueError(f"不支持的断面类型: {section_type}")


# ============ 实用工具函数 ============

def compute_normal_depth(section: CrossSection, discharge: float, 
                        bed_slope: float, manning_n: float,
                        tolerance: float = 1e-4) -> float:
    """计算正常水深 (Manning公式)
    
    Q = (1/n) * A * R^(2/3) * S^(1/2)
    
    参数:
        section: 断面对象
        discharge: 流量 (m³/s)
        bed_slope: 河床坡度
        manning_n: 曼宁系数
        tolerance: 收敛容差
    
    返回:
        正常水深 (m)
    """
    depth = 1.0  # 初值
    
    for _ in range(50):
        props = section.compute_properties(depth)
        
        # Manning流量计算
        Q_computed = (1 / manning_n) * props.area * \
                     props.hydraulic_radius**(2/3) * np.sqrt(bed_slope)
        
        residual = Q_computed - discharge
        
        if abs(residual) < tolerance:
            return depth
        
        # 数值导数
        delta = 0.01
        props_plus = section.compute_properties(depth + delta)
        Q_plus = (1 / manning_n) * props_plus.area * \
                 props_plus.hydraulic_radius**(2/3) * np.sqrt(bed_slope)
        
        derivative = (Q_plus - Q_computed) / delta
        
        if abs(derivative) < 1e-10:
            break
        
        # 牛顿更新
        depth -= residual / derivative
        depth = max(0.1, min(depth, section.get_max_depth()))
    
    return depth


def compute_critical_depth(section: CrossSection, discharge: float,
                          tolerance: float = 1e-4) -> float:
    """计算临界水深
    
    Fr = 1 时的水深，即 Q²T/(gA³) = 1
    
    参数:
        section: 断面对象
        discharge: 流量 (m³/s)
        tolerance: 收敛容差
    
    返回:
        临界水深 (m)
    """
    g = 9.81
    depth = 1.0
    
    for _ in range(50):
        props = section.compute_properties(depth)
        
        if props.area < 1e-6:
            depth += 0.1
            continue
        
        # Froude数
        Fr_squared = discharge**2 * props.top_width / (g * props.area**3)
        residual = Fr_squared - 1.0
        
        if abs(residual) < tolerance:
            return depth
        
        # 数值导数
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
    # 测试不同断面类型
    print("="*60)
    print("断面几何模块测试")
    print("="*60)
    
    test_depth = 3.0
    
    # 测试1: 矩形断面
    print("\n1. 矩形断面 (宽30m)")
    rect = RectangleSection(width=30)
    props = rect.compute_properties(test_depth)
    print(f"   水深 = {test_depth} m")
    print(f"   面积 = {props.area:.2f} m²")
    print(f"   水力半径 = {props.hydraulic_radius:.2f} m")
    
    # 测试2: 梯形断面
    print("\n2. 梯形断面 (底宽10m, 边坡2:1)")
    trap = TrapezoidSection(bottom_width=10, side_slope=2.0)
    props = trap.compute_properties(test_depth)
    print(f"   水深 = {test_depth} m")
    print(f"   面积 = {props.area:.2f} m²")
    print(f"   顶宽 = {props.top_width:.2f} m")
    print(f"   水力半径 = {props.hydraulic_radius:.2f} m")
    
    # 测试3: 复合断面
    print("\n3. 复合断面 (主槽8m+滩地各15m)")
    compound = CompoundSection(
        main_bottom_width=8,
        main_side_slope=1.5,
        floodplain_height=2.5,
        left_floodplain_width=15,
        right_floodplain_width=15
    )
    
    print("   主槽内 (h=2m):")
    props = compound.compute_properties(2.0)
    print(f"      面积 = {props.area:.2f} m²")
    
    print("   漫滩后 (h=4m):")
    props = compound.compute_properties(4.0)
    print(f"      面积 = {props.area:.2f} m²")
    print(f"      顶宽 = {props.top_width:.2f} m")
    
    # 测试4: 正常水深计算
    print("\n4. 正常水深计算 (Q=50 m³/s, S=0.001, n=0.03)")
    normal_depth = compute_normal_depth(trap, 50, 0.001, 0.03)
    print(f"   正常水深 = {normal_depth:.2f} m")
    
    # 测试5: 临界水深
    print("\n5. 临界水深计算 (Q=50 m³/s)")
    critical_depth = compute_critical_depth(trap, 50)
    print(f"   临界水深 = {critical_depth:.2f} m")
    
    if normal_depth > critical_depth:
        print("   → 缓流状态")
    else:
        print("   → 急流状态")