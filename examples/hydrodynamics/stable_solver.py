"""数值稳定的水动力求解器"""

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

class StableSaintVenantSolver:
    """数值稳定的圣维南求解器"""
    
    def __init__(self, reach, cross_section, dt=60):
        self.reach = reach
        self.cross_section = cross_section
        self.dt = dt
        self.g = 9.81
        
        # 计算CFL稳定性条件
        self.max_velocity = 5.0  # 估算最大流速
        self.cfl_limit = 0.5
        self.stable_dt = min(dt, self.cfl_limit * reach.dx / self.max_velocity)
        
        print(f"稳定性分析:")
        print(f"  设定时间步: {dt}s")
        print(f"  稳定时间步: {self.stable_dt:.1f}s")
        print(f"  CFL数: {self.max_velocity * dt / reach.dx:.2f}")
        
        # 初始化为稳态
        n = reach.num_sections
        initial_q = 30.0
        initial_h = self._compute_normal_depth(initial_q)
        
        self.Q = np.full(n, initial_q)
        self.h = np.full(n, initial_h)
        
        print(f"初始稳态: Q={initial_q:.1f} m³/s, h={initial_h:.2f} m")
        
    def _compute_normal_depth(self, discharge):
        """计算正常水深"""
        width = self.cross_section.width
        S = max(self.reach.bed_slope, 1e-6)
        n = self.reach.manning_n
        
        # 矩形断面正常水深
        h_normal = (discharge * n / (width * S**0.5))**(3/5)
        return max(h_normal, 0.1)
    
    def solve_step(self, bc, time_idx):
        """求解一个时间步 - 稳定版本"""
        n = self.reach.num_sections
        dx = self.reach.dx
        dt = self.stable_dt  # 使用稳定的时间步
        
        Q_old = self.Q.copy()
        h_old = self.h.copy()
        
        # 边界条件
        up_idx = min(time_idx, len(bc.upstream_values) - 1) if bc.upstream_values else 0
        dn_idx = min(time_idx, len(bc.downstream_values) - 1) if bc.downstream_values else 0
        
        # 使用分步法：先更新连续性方程，再更新动量方程
        
        # 第一步：更新面积（连续性方程）
        A_new = np.zeros(n)
        for i in range(n):
            A_new[i] = h_old[i] * self.cross_section.width
        
        # 应用边界条件
        if bc.upstream_type == "discharge":
            self.Q[0] = bc.upstream_values[up_idx]
        
        if bc.downstream_type == "stage":
            self.h[-1] = bc.downstream_values[dn_idx]
            A_new[-1] = self.h[-1] * self.cross_section.width
        
        # 更新内部节点的面积
        for i in range(1, n-1):
            # 连续性方程: ∂A/∂t + ∂Q/∂x = 0
            dQ_dx = (Q_old[i+1] - Q_old[i-1]) / (2 * dx)
            A_new[i] = A_new[i] - dt * dQ_dx
            
            # 限制面积变化幅度（稳定性控制）
            max_change = 0.1 * A_new[i]  # 最大10%变化
            A_change = A_new[i] - h_old[i] * self.cross_section.width
            A_change = np.clip(A_change, -max_change, max_change)
            A_new[i] = h_old[i] * self.cross_section.width + A_change
            
            # 确保面积为正
            A_new[i] = max(A_new[i], 0.1 * self.cross_section.width)
            
            # 更新水深
            self.h[i] = A_new[i] / self.cross_section.width
        
        # 第二步：更新流量（动量方程）
        for i in range(1, n-1):
            # 简化的动量方程（忽略对流项，专注于压力和重力平衡）
            dh_dx = (self.h[i+1] - self.h[i-1]) / (2 * dx)
            
            # 重力项
            gravity_term = self.g * A_new[i] * self.reach.bed_slope
            
            # 压力项
            pressure_term = self.g * A_new[i] * dh_dx
            
            # 简化的摩阻项
            v_i = Q_old[i] / A_new[i] if A_new[i] > 1e-6 else 0
            R_i = A_new[i] / (self.cross_section.width + 2 * self.h[i])
            Sf = (self.reach.manning_n * abs(v_i) * v_i) / (R_i**(4/3)) if R_i > 0 else 0
            friction_term = self.g * A_new[i] * Sf
            
            # 更新流量（带稳定性限制）
            dQ_dt = gravity_term - pressure_term - friction_term
            Q_change = dt * dQ_dt
            
            # 限制流量变化幅度
            max_q_change = 0.2 * abs(Q_old[i])  # 最大20%变化
            Q_change = np.clip(Q_change, -max_q_change, max_q_change)
            
            self.Q[i] = Q_old[i] + Q_change
            
            # 确保流量为正
            self.Q[i] = max(self.Q[i], 0.1)
        
        # 下游流量通过质量守恒计算
        if bc.downstream_type == "stage":
            # 使用简单的质量守恒：Q_out ≈ Q_in（忽略蓄水项）
            # 加上小的调整项来反映水位变化的影响
            i = n - 1
            
            # 基于上游流量和水位差异计算下游流量
            upstream_avg_q = np.mean(self.Q[:-1])  # 上游平均流量
            
            # 水位差异对流量的影响（简化）
            h_diff = self.h[-1] - np.mean(self.h[:-1])
            flow_adjustment = 0.1 * upstream_avg_q * h_diff  # 简单的线性调整
            
            self.Q[-1] = upstream_avg_q + flow_adjustment
            self.Q[-1] = max(self.Q[-1], 0.1)
        
        return True
    
    def run_simulation(self, bc, num_steps):
        """运行稳定仿真"""
        results = {
            'time': [],
            'Q': [],
            'h': []
        }
        
        print("步骤 | 时间(h) | 上游Q | 下游Q | 上游h | 下游h")
        print("-" * 50)
        
        for t in range(num_steps):
            success = self.solve_step(bc, t)
            
            time_h = t * self.dt / 3600
            
            results['time'].append(t * self.dt)
            results['Q'].append(self.Q.copy())
            results['h'].append(self.h.copy())
            
            # 打印进度
            print(f"{t:4d} | {time_h:6.2f} | {self.Q[0]:5.1f} | {self.Q[-1]:5.1f} | "
                  f"{self.h[0]:5.2f} | {self.h[-1]:5.2f}")
            
            # 检查数值稳定性
            if not success or np.any(~np.isfinite(self.Q)) or np.any(~np.isfinite(self.h)):
                print(f"数值不稳定，停止于第{t}步")
                break
                
            if np.any(self.h > 100) or np.any(self.Q > 1000):
                print(f"数值发散，停止于第{t}步")
                break
        
        return results

def test_stable_solver():
    """测试稳定求解器"""
    
    print("=== 数值稳定的水动力求解器测试 ===\n")
    
    # 创建河段
    section = RectangleSection(width=20)
    reach = RiverReach(
        id="stable_test",
        length=1200,
        bed_slope=0.001,
        manning_n=0.03,
        width=20,
        num_sections=4  # 很少的网格点
    )
    
    # 创建求解器
    solver = StableSaintVenantSolver(reach, section, dt=240)
    
    # 设计温和的边界条件
    num_steps = 10
    t = np.arange(num_steps)
    
    # 温和的流量变化
    base_q = 30.0
    peak_q = 45.0
    upstream_q = base_q + (peak_q - base_q) * np.sin(np.pi * t / (num_steps - 1))
    
    # 合理的下游水位
    downstream_h = np.full(num_steps, 1.3)
    
    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q.tolist(),
        downstream_type="stage",
        downstream_values=downstream_h.tolist()
    )
    
    print(f"\n边界条件:")
    print(f"  上游流量: {base_q:.0f} - {peak_q:.0f} m³/s")
    print(f"  下游水位: {downstream_h[0]:.1f} m (固定)")
    
    # 运行仿真
    results = solver.run_simulation(bc, num_steps)
    
    # 分析和绘图
    time_array = np.array(results['time']) / 3600
    
    upstream_q_results = [Q[0] for Q in results['Q']]
    downstream_q_results = [Q[-1] for Q in results['Q']]
    upstream_h_results = [h[0] for h in results['h']]
    downstream_h_results = [h[-1] for h in results['h']]
    
    print(f"\n=== 最终结果 ===")
    print(f"流量变化:")
    print(f"  上游: {min(upstream_q_results):.1f} - {max(upstream_q_results):.1f} m³/s")
    print(f"  下游: {min(downstream_q_results):.1f} - {max(downstream_q_results):.1f} m³/s")
    
    print(f"水深变化:")
    print(f"  上游: {min(upstream_h_results):.2f} - {max(upstream_h_results):.2f} m")
    print(f"  下游: {min(downstream_h_results):.2f} - {max(downstream_h_results):.2f} m")
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 流量对比
    ax1.plot(time_array, upstream_q_results, 'b-o', label='上游流量', linewidth=2)
    ax1.plot(time_array, downstream_q_results, 'r-s', label='下游流量', linewidth=2)
    ax1.plot(time_array, upstream_q[:len(time_array)], 'b--', label='设定上游流量', alpha=0.7)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('流量 (m³/s)')
    ax1.set_title('稳定求解器：流量传播')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 水深对比
    ax2.plot(time_array, upstream_h_results, 'b-o', label='上游水深', linewidth=2)
    ax2.plot(time_array, downstream_h_results, 'r-s', label='下游水深', linewidth=2)
    ax2.axhline(y=downstream_h[0], color='orange', linestyle=':', label='设定下游水位')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('水深 (m)')
    ax2.set_title('稳定求解器：水深变化')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stable_solver_test.png', dpi=150, bbox_inches='tight')
    print(f"\n结果图表已保存: stable_solver_test.png")
    
    return results

if __name__ == "__main__":
    results = test_stable_solver()