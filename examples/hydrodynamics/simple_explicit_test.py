"""简单显式方法测试 - 验证边界条件传播"""

from hydrosis.hydrodynamics import (
    RiverReach,
    BoundaryCondition,
    RectangleSection
)
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class SimpleExplicitSolver:
    """简化的显式求解器 - 专注于边界条件传播"""
    
    def __init__(self, reach, cross_section, dt=60):
        self.reach = reach
        self.cross_section = cross_section
        self.dt = dt
        self.g = 9.81
        
        # 初始化状态
        n = reach.num_sections
        self.Q = np.full(n, 20.0)  # 初始流量
        self.h = np.full(n, 1.0)   # 初始水深
        self.A = np.full(n, cross_section.width * 1.0)  # 初始面积
        
    def solve_step(self, bc, time_idx):
        """求解一个时间步 - 修复版本"""
        n = self.reach.num_sections
        dx = self.reach.dx
        dt = self.dt
        
        Q_old = self.Q.copy()
        h_old = self.h.copy()
        A_old = self.A.copy()
        
        # 应用边界条件
        up_idx = min(time_idx, len(bc.upstream_values) - 1) if bc.upstream_values else 0
        dn_idx = min(time_idx, len(bc.downstream_values) - 1) if bc.downstream_values else 0
        
        # 上游边界条件
        if bc.upstream_type == "discharge":
            self.Q[0] = bc.upstream_values[up_idx]
        
        # 下游边界条件
        if bc.downstream_type == "stage":
            self.h[-1] = bc.downstream_values[dn_idx]
            self.A[-1] = self.cross_section.width * self.h[-1]
        
        # 显式更新内部节点 - 修复版本
        for i in range(1, n-1):
            # 连续性方程: ∂A/∂t + ∂Q/∂x = 0
            # 使用中心差分: ∂Q/∂x ≈ (Q[i+1] - Q[i-1]) / (2*dx)
            dQ_dx = (Q_old[i+1] - Q_old[i-1]) / (2 * dx)
            self.A[i] = A_old[i] - dt * dQ_dx
            
            # 确保面积为正
            self.A[i] = max(self.A[i], 0.1 * self.cross_section.width)
            
            # 从面积计算水深
            self.h[i] = self.A[i] / self.cross_section.width
            
            # 动量方程: ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gAS₀ - gASf
            # 简化版本，忽略摩阻和对流项的非线性部分
            
            # 压力项: gA∂h/∂x
            dh_dx = (h_old[i+1] - h_old[i-1]) / (2 * dx)
            pressure_term = self.g * A_old[i] * dh_dx
            
            # 重力项: gAS₀
            gravity_term = self.g * A_old[i] * self.reach.bed_slope
            
            # 更新流量
            self.Q[i] = Q_old[i] + dt * (gravity_term - pressure_term)
        
        # 下游边界的流量通过连续性方程计算 - 修复版本
        if bc.downstream_type == "stage":
            # 使用连续性方程: ∂A/∂t + ∂Q/∂x = 0
            # 在下游边界: ∂Q/∂x = -∂A/∂t
            i = n - 1
            dA_dt = (self.A[i] - A_old[i]) / dt
            
            # 使用向后差分估算流量梯度
            # ∂Q/∂x ≈ (Q[i] - Q[i-1]) / dx = -dA_dt
            # 所以: Q[i] = Q[i-1] - dx * dA_dt
            self.Q[i] = self.Q[i-1] - dx * dA_dt
            
            # 确保流量为正（物理约束）
            self.Q[i] = max(self.Q[i], 0.1)
        
        return True
    
    def run_simulation(self, bc, num_steps):
        """运行完整仿真"""
        results = {
            'time': [],
            'Q': [],
            'h': [],
            'A': []
        }
        
        for t in range(num_steps):
            success = self.solve_step(bc, t)
            
            results['time'].append(t * self.dt)
            results['Q'].append(self.Q.copy())
            results['h'].append(self.h.copy())
            results['A'].append(self.A.copy())
            
            if not success:
                print(f"时间步 {t} 求解失败")
                break
        
        return results

def test_explicit_boundary():
    """测试显式方法的边界条件传播"""
    
    print("=== 简单显式方法边界条件测试 ===\n")
    
    # 创建河段
    section = RectangleSection(width=20)
    reach = RiverReach(
        id="explicit_test",
        length=2000,
        bed_slope=0.002,
        manning_n=0.03,
        width=20,
        num_sections=6
    )
    
    print(f"河段参数:")
    print(f"  长度: {reach.length}m")
    print(f"  网格数: {reach.num_sections}")
    print(f"  网格间距: {reach.dx:.0f}m")
    
    # 创建求解器
    solver = SimpleExplicitSolver(reach, section, dt=120)
    
    print(f"\n初始状态:")
    print(f"  流量: {solver.Q}")
    print(f"  水深: {solver.h}")
    
    # 设计边界条件
    num_steps = 15
    t = np.arange(num_steps)
    
    # 阶跃流量变化
    upstream_q = np.where(t < 5, 20.0, 60.0)
    downstream_h = np.full(num_steps, 1.5)  # 固定下游水位
    
    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q.tolist(),
        downstream_type="stage",
        downstream_values=downstream_h.tolist()
    )
    
    print(f"\n边界条件:")
    print(f"  上游流量: {upstream_q[0]:.0f} → {upstream_q[-1]:.0f} m³/s (第5步变化)")
    print(f"  下游水位: {downstream_h[0]:.1f} m (固定)")
    
    # 运行仿真
    print(f"\n开始仿真...")
    results = solver.run_simulation(bc, num_steps)
    
    # 分析结果
    print(f"\n=== 结果分析 ===")
    
    time_array = np.array(results['time']) / 3600  # 转换为小时
    
    # 提取上下游数据
    upstream_q_results = [Q[0] for Q in results['Q']]
    downstream_q_results = [Q[-1] for Q in results['Q']]
    upstream_h_results = [h[0] for h in results['h']]
    downstream_h_results = [h[-1] for h in results['h']]
    
    print(f"流量变化:")
    print(f"  上游: {min(upstream_q_results):.1f} - {max(upstream_q_results):.1f} m³/s")
    print(f"  下游: {min(downstream_q_results):.1f} - {max(downstream_q_results):.1f} m³/s")
    
    print(f"水深变化:")
    print(f"  上游: {min(upstream_h_results):.2f} - {max(upstream_h_results):.2f} m")
    print(f"  下游: {min(downstream_h_results):.2f} - {max(downstream_h_results):.2f} m")
    
    # 检查传播效应
    q_change_up = max(upstream_q_results) - min(upstream_q_results)
    q_change_down = max(downstream_q_results) - min(downstream_q_results)
    
    print(f"传播效应:")
    print(f"  上游流量变化: {q_change_up:.1f} m³/s")
    print(f"  下游流量变化: {q_change_down:.1f} m³/s")
    print(f"  传播效率: {q_change_down/q_change_up*100:.1f}%")
    
    # 绘制结果
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    # 流量对比
    ax1.plot(time_array, upstream_q_results, 'b-o', label='上游流量', linewidth=2)
    ax1.plot(time_array, downstream_q_results, 'r-s', label='下游流量', linewidth=2)
    ax1.plot(time_array, upstream_q[:len(time_array)], 'b--', label='设定上游流量', alpha=0.7)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('流量 (m³/s)')
    ax1.set_title('显式方法：流量传播')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 水深对比
    ax2.plot(time_array, upstream_h_results, 'b-o', label='上游水深', linewidth=2)
    ax2.plot(time_array, downstream_h_results, 'r-s', label='下游水深', linewidth=2)
    ax2.axhline(y=downstream_h[0], color='orange', linestyle=':', label='设定下游水位')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('水深 (m)')
    ax2.set_title('显式方法：水深变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 沿程分布（最后时刻）
    x_coords = np.array(reach.x_coords) / 1000
    final_Q = results['Q'][-1]
    final_h = results['h'][-1]
    
    ax3.plot(x_coords, final_Q, 'g-o', label='最终流量分布', linewidth=2)
    ax3.set_xlabel('距离 (km)')
    ax3.set_ylabel('流量 (m³/s)')
    ax3.set_title('最终沿程流量分布')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4.plot(x_coords, final_h, 'm-o', label='最终水深分布', linewidth=2)
    ax4.set_xlabel('距离 (km)')
    ax4.set_ylabel('水深 (m)')
    ax4.set_title('最终沿程水深分布')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_explicit_test.png', dpi=150, bbox_inches='tight')
    print(f"\n结果图表已保存: simple_explicit_test.png")
    
    return results

if __name__ == "__main__":
    results = test_explicit_boundary()