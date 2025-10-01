"""一维河道水动力模型 - 基于圣维方形程组的隐式求解器

本模块提供与 HydroSIS 兼容的一维水动力模拟能力,支持:
- 完整圣维方形程组(连续性+动量方程)
- Preissmann 隐式四点差分格式
- 变断面河道几何
- 侧向入流(来自水文模型产流)
- 下游边界条件(水位/流量控制)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

# 断面几何支持
from .geometry import CrossSection, RectangleSection


@dataclass
class RiverReach:
    """河段几何定义"""
    
    id: str
    length: float  # 河段长度 (m)
    bed_slope: float  # 河床坡度 (无量纲)
    manning_n: float  # 曼宁糙率系数
    width: float  # 河宽 (m), 简化为矩形断面
    num_sections: int = 10  # 计算断面数量
    
    def __post_init__(self):
        self.dx = self.length / (self.num_sections - 1)
        self.x_coords = [i * self.dx for i in range(self.num_sections)]


@dataclass
class BoundaryCondition:
    """边界条件定义"""
    
    upstream_type: str = "discharge"  # "discharge" 或 "stage"
    upstream_values: List[float] = field(default_factory=list)
    downstream_type: str = "stage"  # "stage" 或 "rating_curve"
    downstream_values: List[float] = field(default_factory=list)
    downstream_rating: Optional[Tuple[float, float]] = None  # (a, b) for Q = a * H^b


@dataclass
class HydraulicState:
    """水力状态变量"""
    
    depth: np.ndarray  # 水深 (m)
    discharge: np.ndarray  # 流量 (m³/s)
    velocity: np.ndarray  # 流速 (m/s)
    area: np.ndarray  # 过水断面积 (m²)
    
    @classmethod
    def initialize(cls, num_sections: int, initial_depth: float = 1.0, 
                   initial_q: float = 10.0, width: float = 20.0):
        """初始化静水状态"""
        depth = np.full(num_sections, initial_depth)
        area = depth * width
        discharge = np.full(num_sections, initial_q)
        velocity = discharge / area
        return cls(depth=depth, discharge=discharge, velocity=velocity, area=area)


class SaintVenantSolver:
    """圣维方形程组隐式求解器
    
    采用 Preissmann 四点隐式格式求解:
    - 连续性方程: ∂A/∂t + ∂Q/∂x = q_lateral
    - 动量方程: ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀ - Sf)
    
    其中:
    - A: 过水断面积
    - Q: 流量
    - h: 水位
    - S₀: 河床坡度
    - Sf: 摩阻坡度 = n²Q|Q|/(A²R^(4/3)), R为水力半径
    - q_lateral: 侧向入流
    """
    
    def __init__(self, reach: RiverReach, dt: float = 60.0, 
                 theta: float = 0.6, epsilon: float = 1e-4,
                 cross_section: Optional[CrossSection] = None):
        """
        参数:
            reach: 河段几何信息
            dt: 时间步长 (s)
            theta: 时间权重因子 (0.5=Crank-Nicolson, 1.0=完全隐式)
            epsilon: 牛顿迭代收敛容差
        """
        self.reach = reach
        self.dt = dt
        self.theta = theta
        self.epsilon = epsilon
        self.g = 9.81  # 重力加速度
        # 断面几何：默认使用矩形，与旧版保持兼容
        self.cross_section: CrossSection = cross_section or RectangleSection(width=reach.width)
        
        # 使用更合理的初始化
        self._initialize_steady_state()
        self.lateral_inflow = np.zeros(reach.num_sections)
        
    def _initialize_steady_state(self):
        """初始化为稳定流状态"""
        n = self.reach.num_sections
        
        # 估算合理的初始流量和水深
        initial_q = 20.0  # 基础流量
        
        # 使用曼宁公式估算对应的正常水深
        S = max(self.reach.bed_slope, 1e-6)
        n_manning = self.reach.manning_n
        
        # 对于矩形断面的正常水深近似
        if hasattr(self.cross_section, 'width'):
            width = self.cross_section.width
            # 简化的正常水深计算: h = (Q*n/(width*S^0.5))^(3/5)
            h_normal = (initial_q * n_manning / (width * S**0.5))**(3/5)
        else:
            # 对于其他断面类型，使用迭代求解正常水深
            h_normal = self._compute_normal_depth_iterative(initial_q)
        
        h_normal = max(h_normal, 0.5)  # 确保最小水深
        
        # 初始化状态
        self.state = HydraulicState.initialize(
            n,
            initial_depth=h_normal,
            initial_q=initial_q,
            width=getattr(self.cross_section, 'width', self.reach.width)
        )
        
        # 更新几何参数
        self.update_hydraulic_properties(self.state)
        
    def _compute_normal_depth_iterative(self, discharge: float, max_iter: int = 20) -> float:
        """迭代计算正常水深"""
        h = 1.0  # 初始猜测
        S = max(self.reach.bed_slope, 1e-6)
        n = self.reach.manning_n
        
        for _ in range(max_iter):
            props = self.cross_section.compute_properties(h)
            if props.area < 1e-6 or props.hydraulic_radius < 1e-6:
                h += 0.1
                continue
                
            Q_calc = (1/n) * props.area * props.hydraulic_radius**(2/3) * S**0.5
            
            if abs(Q_calc - discharge) < 0.1:
                break
                
            # 简单的牛顿法更新
            dQ_dh = (1/n) * props.top_width * props.hydraulic_radius**(2/3) * S**0.5
            if dQ_dh > 1e-6:
                h += (discharge - Q_calc) / dQ_dh
                h = max(h, 0.1)  # 确保正值
            else:
                h += 0.1
                
        return max(h, 0.5)
    
    def _compute_section_properties(self, depths: np.ndarray):
        """根据当前水深数组计算各断面的几何参数。

        返回: (area, top_width, hydraulic_radius)
        """
        n = len(depths)
        area = np.zeros(n)
        top_width = np.zeros(n)
        hydraulic_radius = np.zeros(n)
        # 逐断面计算（断面类型可能为复合/不规则，逐点更安全）
        for i in range(n):
            props = self.cross_section.compute_properties(float(depths[i]))
            area[i] = props.area
            top_width[i] = props.top_width
            hydraulic_radius[i] = props.hydraulic_radius
        return area, top_width, hydraulic_radius

    def update_hydraulic_properties(self, state: HydraulicState):
        """更新水力几何参数，支持可变断面。"""
        area, _, _ = self._compute_section_properties(state.depth)
        state.area = area
        state.velocity = np.where(state.area > 1e-6,
                                   state.discharge / state.area, 0.0)
        
    def friction_slope(self, Q: np.ndarray, A: np.ndarray, R: np.ndarray) -> np.ndarray:
        """计算摩阻坡度 Sf，使用几何断面提供的水力半径。"""
        R = np.maximum(R, 0.01)  # 防止除零
        n = self.reach.manning_n
        Sf = n**2 * Q * np.abs(Q) / (A**2 * R**(4/3))
        return Sf
    
    def build_jacobian_and_residual(self, Q_new: np.ndarray, h_new: np.ndarray,
                                     Q_old: np.ndarray, h_old: np.ndarray,
                                     bc: BoundaryCondition, time_idx: int):
        """构建牛顿法雅可比矩阵和残差向量
        
        未知量: [Q₁, h₁, Q₂, h₂, ..., Qₙ, hₙ]
        方程:
        - 连续性: A^{n+1} - A^n + θΔt/Δx(Q_{i+1} - Q_i) + (1-θ)Δt/Δx(Q_{i+1}^n - Q_i^n) = Δt·q
        - 动量: Q^{n+1} - Q^n + θΔt[∂(Q²/A)/∂x + gA∂h/∂x - gA(S₀-Sf)] = 0
        """
        n = self.reach.num_sections
        N = 2 * n  # 总未知量数
        dx = self.reach.dx
        
        # 计算当前时刻的断面几何参数
        A_new, T_new, R_new = self._compute_section_properties(h_new)
        A_old, _, _ = self._compute_section_properties(h_old)
        
        J = np.zeros((N, N))
        R = np.zeros(N)
        
        # 边界序列越界时采用最后一个值（适配自适应时间步长场景）
        up_idx = min(time_idx, len(bc.upstream_values) - 1) if bc.upstream_values else 0
        dn_idx = min(time_idx, len(bc.downstream_values) - 1) if bc.downstream_values else 0
        
        # === 上游边界条件 (i=0) ===
        if bc.upstream_type == "discharge":
            # 上游流量边界：直接设定流量
            J[0, 0] = 1.0  # ∂R/∂Q_0 = 1
            R[0] = Q_new[0] - bc.upstream_values[up_idx]
            
            # 上游连续性方程（第1个方程）
            J[1, 0] = -self.theta * self.dt / dx
            J[1, 1] = T_new[0]  # ∂A/∂h
            J[1, 2] = self.theta * self.dt / dx
            R[1] = (A_new[0] - A_old[0] + 
                   self.theta * self.dt / dx * (Q_new[1] - Q_new[0]) +
                   (1 - self.theta) * self.dt / dx * (Q_old[1] - Q_old[0]) -
                   self.dt * self.lateral_inflow[0])
        else:  # stage boundary
            # 上游水位边界：直接设定水位
            J[0, 1] = 1.0  # ∂R/∂h_0 = 1
            R[0] = h_new[0] - bc.upstream_values[up_idx]
            
            # 上游动量方程
            v_0 = Q_new[0] / A_new[0] if A_new[0] > 1e-6 else 0
            v_1 = Q_new[1] / A_new[1] if A_new[1] > 1e-6 else 0
            Sf_0 = self.friction_slope(Q_new[0:1], A_new[0:1], R_new[0:1])[0]
            dh = (h_new[1] - h_new[0]) / dx
            
            J[1, 0] = 1.0
            J[1, 1] = self.theta * self.dt * self.g * T_new[0] * (dh - (self.reach.bed_slope - Sf_0))
            R[1] = (Q_new[0] - Q_old[0] +
                   self.theta * self.dt * (
                       (v_1 * Q_new[1] - v_0 * Q_new[0]) / dx +
                       self.g * A_new[0] * dh -
                       self.g * A_new[0] * (self.reach.bed_slope - Sf_0)
                   ))
        
        # === 内部节点方程 (i=1 to n-2) ===
        for i in range(1, n - 1):
            # 连续性方程 (偶数行)
            row = 2 * i
            R[row] = (A_new[i] - A_old[i] + 
                     self.theta * self.dt / dx * (Q_new[i+1] - Q_new[i]) +
                     (1 - self.theta) * self.dt / dx * (Q_old[i+1] - Q_old[i]) -
                     self.dt * self.lateral_inflow[i])
            
            # 雅可比矩阵 - 连续性
            J[row, 2*i] = -self.theta * self.dt / dx  # ∂R/∂Q_i
            J[row, 2*i+1] = T_new[i]  # ∂R/∂h_i = ∂A/∂h
            J[row, 2*(i+1)] = self.theta * self.dt / dx  # ∂R/∂Q_{i+1}
            
            # 动量方程 (奇数行) - 改进版本
            row = 2 * i + 1
            
            # 计算对流项 ∂(Q²/A)/∂x 的更精确导数
            v_i = Q_new[i] / A_new[i] if A_new[i] > 1e-6 else 0
            v_ip1 = Q_new[i+1] / A_new[i+1] if A_new[i+1] > 1e-6 else 0
            
            # 摩阻坡度及其导数
            Sf_i = self.friction_slope(Q_new[i:i+1], A_new[i:i+1], R_new[i:i+1])[0]
            
            # 改进的摩阻坡度对流量的导数 ∂Sf/∂Q
            if A_new[i] > 1e-6 and R_new[i] > 1e-6:
                dSf_dQ = 2 * self.reach.manning_n**2 * np.abs(Q_new[i]) / (A_new[i]**2 * R_new[i]**(4/3))
            else:
                dSf_dQ = 0
            
            # 水面坡度
            dh = (h_new[i+1] - h_new[i]) / dx
            
            # 残差
            R[row] = (Q_new[i] - Q_old[i] +
                     self.theta * self.dt * (
                         (v_ip1 * Q_new[i+1] - v_i * Q_new[i]) / dx +
                         self.g * A_new[i] * dh -
                         self.g * A_new[i] * (self.reach.bed_slope - Sf_i)
                     ))
            
            # 改进的雅可比矩阵元素
            # ∂R/∂Q_i
            J[row, 2*i] = (1 + self.theta * self.dt * (
                -2 * v_i / dx +  # 对流项导数
                self.g * A_new[i] * dSf_dQ  # 摩阻项导数
            ))
            
            # ∂R/∂h_i - 更精确的压力项和几何项
            J[row, 2*i+1] = self.theta * self.dt * self.g * (
                T_new[i] * (dh - (self.reach.bed_slope - Sf_i)) -  # 压力项
                A_new[i] / dx  # 水面坡度项
            )
            
            # ∂R/∂Q_{i+1}
            if A_new[i+1] > 1e-6:
                J[row, 2*(i+1)] = self.theta * self.dt * 2 * v_ip1 / dx
            
            # ∂R/∂h_{i+1}
            J[row, 2*(i+1)+1] = self.theta * self.dt * self.g * A_new[i] / dx
        
        # === 下游边界条件 (i=n-1) ===
        i = n - 1
        if bc.downstream_type == "stage":
            # 下游水位边界：直接设定水位
            J[2*i+1, 2*i+1] = 1.0  # ∂R/∂h_{n-1} = 1
            R[2*i+1] = h_new[i] - bc.downstream_values[dn_idx]
            
            # 下游连续性方程
            J[2*i, 2*i-2] = self.theta * self.dt / dx    # ∂R/∂Q_{i-1}
            J[2*i, 2*i] = -self.theta * self.dt / dx     # ∂R/∂Q_i
            J[2*i, 2*i+1] = T_new[i]                     # ∂R/∂h_i
            R[2*i] = (A_new[i] - A_old[i] + 
                     self.theta * self.dt / dx * (-Q_new[i] + Q_new[i-1]) +
                     (1 - self.theta) * self.dt / dx * (-Q_old[i] + Q_old[i-1]) -
                     self.dt * self.lateral_inflow[i])
        else:  # discharge boundary or free outflow
            # 下游流量边界或自由出流
            if hasattr(bc, 'downstream_rating') and bc.downstream_rating and len(bc.downstream_rating) == 2:
                # 水位-流量关系
                a, b = bc.downstream_rating
                if h_new[i] > 0.1:
                    J[2*i+1, 2*i] = 1.0
                    J[2*i+1, 2*i+1] = -a * b * h_new[i]**(b-1)
                    R[2*i+1] = Q_new[i] - a * h_new[i]**b
                else:
                    J[2*i+1, 2*i] = 1.0
                    J[2*i+1, 2*i+1] = -a * b
                    R[2*i+1] = Q_new[i] - a * b * h_new[i]
            else:
                # 自由出流边界（正常水深近似）
                if A_new[i] > 1e-6 and R_new[i] > 1e-6:
                    S = max(self.reach.bed_slope, 1e-6)
                    Q_normal = (1/self.reach.manning_n) * A_new[i] * R_new[i]**(2/3) * S**0.5
                    J[2*i+1, 2*i] = 1.0
                    J[2*i+1, 2*i+1] = -(1/self.reach.manning_n) * T_new[i] * R_new[i]**(2/3) * S**0.5
                    R[2*i+1] = Q_new[i] - Q_normal
                else:
                    J[2*i+1, 2*i] = 1.0
                    R[2*i+1] = Q_new[i] - max(Q_old[i], 1.0)
            
            # 下游连续性方程
            J[2*i, 2*i-2] = self.theta * self.dt / dx
            J[2*i, 2*i] = -self.theta * self.dt / dx
            J[2*i, 2*i+1] = T_new[i]
            R[2*i] = (A_new[i] - A_old[i] + 
                     self.theta * self.dt / dx * (-Q_new[i] + Q_new[i-1]) +
                     (1 - self.theta) * self.dt / dx * (-Q_old[i] + Q_old[i-1]) -
                     self.dt * self.lateral_inflow[i])
        
        return J, R
    
    def solve_timestep(self, bc: BoundaryCondition, time_idx: int, 
                       max_iter: int = 15) -> bool:
        """求解单个时间步 - 简化版本专注于稳定性
        
        返回:
            是否收敛
        """
        Q_old = self.state.discharge.copy()
        h_old = self.state.depth.copy()
        
        Q_new = Q_old.copy()
        h_new = h_old.copy()
        
        # 边界序列越界保护
        up_idx = min(time_idx, len(bc.upstream_values) - 1) if bc.upstream_values else 0
        dn_idx = min(time_idx, len(bc.downstream_values) - 1) if bc.downstream_values else 0
        
        # 直接应用边界条件（简化方法）
        if bc.upstream_type == "discharge":
            Q_new[0] = bc.upstream_values[up_idx]
        else:  # stage
            h_new[0] = bc.upstream_values[up_idx]
        
        if bc.downstream_type == "stage":
            h_new[-1] = bc.downstream_values[dn_idx]
        
        # 使用显式方法更新内部节点（更稳定）
        success = self._explicit_update(Q_new, h_new, Q_old, h_old)
        
        if success:
            # 更新状态
            self.state.discharge = Q_new
            self.state.depth = h_new
            self.update_hydraulic_properties(self.state)
            return True
        
        return False
    
    def _explicit_update(self, Q_new: np.ndarray, h_new: np.ndarray,
                        Q_old: np.ndarray, h_old: np.ndarray) -> bool:
        """使用显式方法更新内部节点"""
        try:
            n = self.reach.num_sections
            dx = self.reach.dx
            dt = self.dt
            
            # 计算几何参数
            A_old, T_old, R_old = self._compute_section_properties(h_old)
            
            # 显式更新内部节点
            for i in range(1, n - 1):
                # 连续性方程：∂A/∂t + ∂Q/∂x = 0
                dQ_dx = (Q_old[i+1] - Q_old[i-1]) / (2 * dx)
                A_new_i = A_old[i] - dt * dQ_dx + dt * self.lateral_inflow[i]
                
                # 从面积反算水深
                h_new[i] = self._depth_from_area(A_new_i, h_old[i])
                
                # 动量方程：∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀ - Sf)
                # 简化为：Q_new = Q_old + dt * [gA(S₀ - Sf) - ∂(Q²/A)/∂x - gA∂h/∂x]
                
                # 对流项
                v_i = Q_old[i] / A_old[i] if A_old[i] > 1e-6 else 0
                v_ip1 = Q_old[i+1] / A_old[i+1] if A_old[i+1] > 1e-6 else 0
                v_im1 = Q_old[i-1] / A_old[i-1] if A_old[i-1] > 1e-6 else 0
                
                dQQ_A_dx = ((v_ip1 * Q_old[i+1]) - (v_im1 * Q_old[i-1])) / (2 * dx)
                
                # 压力项
                dh_dx = (h_old[i+1] - h_old[i-1]) / (2 * dx)
                
                # 摩阻项
                Sf = self.friction_slope(Q_old[i:i+1], A_old[i:i+1], R_old[i:i+1])[0]
                
                # 更新流量
                Q_new[i] = Q_old[i] + dt * (
                    self.g * A_old[i] * (self.reach.bed_slope - Sf) -
                    dQQ_A_dx -
                    self.g * A_old[i] * dh_dx
                )
                
                # 稳定性检查
                if h_new[i] <= 0.01 or not np.isfinite(Q_new[i]) or not np.isfinite(h_new[i]):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _depth_from_area(self, area: float, h_guess: float) -> float:
        """从面积反算水深"""
        try:
            # 使用断面几何的反函数
            return self.cross_section.compute_depth_from_area(area)
        except:
            # 如果失败，使用简单近似
            if hasattr(self.cross_section, 'width'):
                return max(area / self.cross_section.width, 0.01)
            else:
                return max(h_guess, 0.01)
    
    def set_lateral_inflow(self, inflow: Sequence[float]):
        """设置侧向入流 (m³/s/m), 通常来自水文模型产流"""
        if len(inflow) != self.reach.num_sections:
            raise ValueError(f"Inflow length {len(inflow)} != sections {self.reach.num_sections}")
        self.lateral_inflow = np.array(inflow)
    
    def run_simulation(self, bc: BoundaryCondition, num_steps: int) -> Dict[str, List]:
        """运行完整模拟
        
        返回:
            时间序列字典 {'discharge': [...], 'depth': [...], 'velocity': [...]}
        """
        results = {
            'time': [],
            'discharge': [],
            'depth': [],
            'velocity': []
        }
        
        for t in range(num_steps):
            success = self.solve_timestep(bc, t)
            if not success:
                print(f"Warning: 时间步 {t} 未收敛")
            
            results['time'].append(t * self.dt)
            results['discharge'].append(self.state.discharge.copy())
            results['depth'].append(self.state.depth.copy())
            results['velocity'].append(self.state.velocity.copy())
        
        return results


