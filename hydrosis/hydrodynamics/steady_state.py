"""稳态流计算模块 - 为非恒定流仿真提供初始条件"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import fsolve

from .geometry import CrossSection, RectangleSection
from .core import RiverReach


class SteadyStateCalculator:
    """稳态流计算器
    
    用于计算给定流量和边界条件下的稳态水面线，
    为非恒定流仿真提供合理的初始条件。
    """
    
    def __init__(self, reach: RiverReach, cross_section: Optional[CrossSection] = None):
        """
        参数:
            reach: 河段几何信息
            cross_section: 断面几何，默认为矩形断面
        """
        self.reach = reach
        self.cross_section = cross_section or RectangleSection(width=reach.width)
        self.g = 9.81
        
    def compute_normal_depth(self, discharge: float, tolerance: float = 1e-6) -> float:
        """计算正常水深
        
        使用曼宁公式: Q = (1/n) * A * R^(2/3) * S^(1/2)
        """
        def manning_residual(h):
            if h <= 0.01:
                return 1e6  # 惩罚负水深
            
            props = self.cross_section.compute_properties(h)
            if props.area < 1e-6 or props.hydraulic_radius < 1e-6:
                return 1e6
            
            Q_calc = (1/self.reach.manning_n) * props.area * \
                    props.hydraulic_radius**(2/3) * self.reach.bed_slope**0.5
            
            return Q_calc - discharge
        
        # 初始猜测：基于矩形断面的近似解
        if hasattr(self.cross_section, 'width'):
            h_guess = (discharge * self.reach.manning_n / 
                      (self.cross_section.width * self.reach.bed_slope**0.5))**(3/5)
        else:
            h_guess = 1.0
        
        try:
            h_normal = fsolve(manning_residual, h_guess, xtol=tolerance)[0]
            return max(h_normal, 0.1)
        except (ValueError, RuntimeError, ZeroDivisionError):
            # 如果求解失败，使用迭代方法
            return self._iterative_normal_depth(discharge, tolerance)
    
    def _iterative_normal_depth(self, discharge: float, tolerance: float = 1e-6) -> float:
        """迭代求解正常水深"""
        h = 1.0
        for _ in range(50):
            props = self.cross_section.compute_properties(h)
            if props.area < 1e-6:
                h += 0.1
                continue
                
            Q_calc = (1/self.reach.manning_n) * props.area * \
                    props.hydraulic_radius**(2/3) * self.reach.bed_slope**0.5
            
            if abs(Q_calc - discharge) < tolerance:
                break
                
            # 牛顿法更新
            dQ_dh = (1/self.reach.manning_n) * props.top_width * \
                   props.hydraulic_radius**(2/3) * self.reach.bed_slope**0.5
            
            if dQ_dh > 1e-6:
                h += (discharge - Q_calc) / dQ_dh
                h = max(h, 0.1)
            else:
                h += 0.1 if Q_calc < discharge else -0.05
                
        return max(h, 0.1)
    
    def compute_critical_depth(self, discharge: float, tolerance: float = 1e-6) -> float:
        """计算临界水深
        
        临界条件: Fr = V/sqrt(g*D) = 1
        其中 D = A/T 为水力水深
        """
        def froude_residual(h):
            if h <= 0.01:
                return 1e6
            
            props = self.cross_section.compute_properties(h)
            if props.area < 1e-6 or props.top_width < 1e-6:
                return 1e6
            
            velocity = discharge / props.area
            hydraulic_depth = props.area / props.top_width
            froude = velocity / np.sqrt(self.g * hydraulic_depth)
            
            return froude - 1.0
        
        try:
            h_critical = fsolve(froude_residual, 0.5, xtol=tolerance)[0]
            return max(h_critical, 0.1)
        except (ValueError, RuntimeError, ZeroDivisionError):
            return 0.5  # 默认值
    
    def compute_steady_profile(self, 
                              upstream_discharge: float,
                              downstream_condition: Tuple[str, float],
                              method: str = "standard_step") -> Tuple[np.ndarray, np.ndarray]:
        """计算稳态水面线
        
        参数:
            upstream_discharge: 上游流量 (m³/s)
            downstream_condition: 下游边界条件 ("depth", value) 或 ("normal", 0)
            method: 计算方法，目前支持 "standard_step"
            
        返回:
            (depths, elevations): 各断面的水深和水位
        """
        n_sections = self.reach.num_sections
        x_coords = np.array(self.reach.x_coords)
        bed_elevations = np.zeros(n_sections)
        
        # 计算河床高程（从下游向上游递增）
        for i in range(n_sections):
            bed_elevations[i] = (n_sections - 1 - i) * self.reach.dx * self.reach.bed_slope
        
        # 确定下游边界条件
        if downstream_condition[0] == "depth":
            h_downstream = downstream_condition[1]
        elif downstream_condition[0] == "normal":
            h_downstream = self.compute_normal_depth(upstream_discharge)
        else:
            raise ValueError(f"不支持的下游边界条件: {downstream_condition[0]}")
        
        # 使用标准步长法计算水面线
        depths = np.zeros(n_sections)
        depths[-1] = h_downstream  # 下游边界
        
        # 从下游向上游逐步计算
        for i in range(n_sections - 2, -1, -1):
            depths[i] = self._compute_upstream_depth(
                upstream_discharge, 
                depths[i + 1], 
                self.reach.dx
            )
        
        # 计算水位高程
        elevations = bed_elevations + depths
        
        return depths, elevations
    
    def _compute_upstream_depth(self, discharge: float, h_downstream: float, dx: float) -> float:
        """使用能量方程计算上游水深"""
        
        # 下游断面参数
        props_down = self.cross_section.compute_properties(h_downstream)
        v_down = discharge / props_down.area if props_down.area > 1e-6 else 0
        
        # 摩阻坡度
        Sf_down = (self.reach.manning_n * v_down * abs(v_down)) / \
                 (props_down.hydraulic_radius**(4/3)) if props_down.hydraulic_radius > 1e-6 else 0
        
        def energy_residual(h_up):
            if h_up <= 0.01:
                return 1e6
            
            props_up = self.cross_section.compute_properties(h_up)
            if props_up.area < 1e-6:
                return 1e6
            
            v_up = discharge / props_up.area
            
            # 摩阻坡度
            Sf_up = (self.reach.manning_n * v_up * abs(v_up)) / \
                   (props_up.hydraulic_radius**(4/3)) if props_up.hydraulic_radius > 1e-6 else 0
            
            # 平均摩阻坡度
            Sf_avg = (Sf_up + Sf_down) / 2
            
            # 能量方程: E_up = E_down + (S0 - Sf_avg) * dx
            E_up = h_up + v_up**2 / (2 * self.g)
            E_down = h_downstream + v_down**2 / (2 * self.g)
            
            return E_up - E_down - (self.reach.bed_slope - Sf_avg) * dx
        
        # 初始猜测
        h_guess = h_downstream + self.reach.bed_slope * dx
        
        try:
            h_upstream = fsolve(energy_residual, h_guess, xtol=1e-6)[0]
            return max(h_upstream, 0.1)
        except (ValueError, RuntimeError, ZeroDivisionError):
            # 如果求解失败，使用简单近似
            return h_downstream + self.reach.bed_slope * dx
    
    def create_initial_conditions(self, 
                                 discharge: float,
                                 boundary_type: str = "normal") -> Tuple[np.ndarray, np.ndarray]:
        """为非恒定流仿真创建初始条件
        
        参数:
            discharge: 初始流量 (m³/s)
            boundary_type: 边界类型 ("normal" 或 "critical")
            
        返回:
            (initial_depths, initial_discharges): 初始水深和流量分布
        """
        if boundary_type == "normal":
            # 使用正常水深作为均匀初始条件
            h_normal = self.compute_normal_depth(discharge)
            depths = np.full(self.reach.num_sections, h_normal)
        elif boundary_type == "critical":
            # 使用临界水深
            h_critical = self.compute_critical_depth(discharge)
            depths = np.full(self.reach.num_sections, h_critical)
        else:
            # 计算完整的稳态水面线
            depths, _ = self.compute_steady_profile(
                discharge, 
                ("normal", 0)
            )
        
        # 均匀流量分布
        discharges = np.full(self.reach.num_sections, discharge)
        
        return depths, discharges
    
    def validate_initial_conditions(self, depths: np.ndarray, discharges: np.ndarray) -> dict:
        """验证初始条件的合理性
        
        返回验证报告
        """
        report = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {}
        }
        
        # 检查水深
        if np.any(depths <= 0):
            report["errors"].append("存在负水深或零水深")
            report["valid"] = False
        
        if np.any(depths > 10):
            report["warnings"].append("存在异常大的水深 (>10m)")
        
        # 检查流量
        if np.any(discharges <= 0):
            report["errors"].append("存在负流量或零流量")
            report["valid"] = False
        
        # 质量守恒检查
        q_variation = np.std(discharges) / np.mean(discharges) * 100
        if q_variation > 5:
            report["warnings"].append(f"流量变化较大 ({q_variation:.1f}%)")
        
        # 统计信息
        report["statistics"] = {
            "depth_range": (np.min(depths), np.max(depths)),
            "discharge_range": (np.min(discharges), np.max(discharges)),
            "avg_depth": np.mean(depths),
            "avg_discharge": np.mean(discharges),
            "flow_variation": q_variation
        }
        
        return report


def compute_normal_depth(cross_section: CrossSection, 
                        discharge: float,
                        bed_slope: float,
                        manning_n: float,
                        tolerance: float = 1e-6) -> float:
    """独立函数：计算正常水深
    
    这是一个便利函数，不需要创建完整的 SteadyStateCalculator 实例
    """
    # 创建临时河段
    temp_reach = RiverReach(
        id="temp",
        length=1000,
        bed_slope=bed_slope,
        manning_n=manning_n,
        width=getattr(cross_section, 'width', 20),
        num_sections=2
    )
    
    calculator = SteadyStateCalculator(temp_reach, cross_section)
    return calculator.compute_normal_depth(discharge, tolerance)


def compute_critical_depth(cross_section: CrossSection,
                          discharge: float,
                          tolerance: float = 1e-6) -> float:
    """独立函数：计算临界水深"""
    temp_reach = RiverReach(
        id="temp",
        length=1000,
        bed_slope=0.001,
        manning_n=0.03,
        width=getattr(cross_section, 'width', 20),
        num_sections=2
    )
    
    calculator = SteadyStateCalculator(temp_reach, cross_section)
    return calculator.compute_critical_depth(discharge, tolerance)