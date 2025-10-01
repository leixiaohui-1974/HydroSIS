"""物理正确的水动力求解器 - 修复版本"""

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

class CorrectedSaintVenantSolver:
    """修正的圣维南求解器 - 物理正确版本"""
    
    def __init__(self, reach, cross_section, dt=60):
        self.reach = reach
        self.cross_section = cross_section
        self.dt = dt
        self.g = 9.81
        
        # 初始化为稳态
        n = reach.num_sections
        initial_q = 25.0
        initial_h = self._compute_normal_depth(initial_q)
        
        self.Q = np.full(n, initial_q)
        self.h = np.full(n, initial_h)
        self.A = np.full(n, cross_section.width * initial_h)
        
        print(f"初始化稳态: Q={initial_q:.1f} m³/s, h={initial_h:.2f} m")
        
    def _compute_normal_depth(self, discharge):
        """计算正常水深"""
        width = self.cross_section.width
        S = self.reach.bed_slope
        n = self.reach.manning_n
        
        # 矩形断面正常水深: h = (Q*n/(width*S^0.5))^(3/5)
        h_normal = (discharge * n / (width * S**0.5))**(3/5)
        return max(h_normal, 0.1)
    
    def solve_step(self, bc, time_idx):
        """求解一个时间步 - 使用MacCormack格式"""
        n = self.reach.num_sections
        dx = self.reach.dx
        dt = self.dt
        
        # 保存旧值
        Q_old = self.Q.copy()
        h_old = self.h.copy()
        A_old = self.A.copy()
        
        # 边界条件索引
        up_idx = min(time_idx, len(bc.upstream_values) - 1) if bc.upstream_values else 0
        dn_idx = min(time_idx, len(bc.downstream_values) - 1) if bc.downstream_values else 0
        
        # === 预测步 (Predictor) ===
        Q_pred = Q_old.copy()
        h_pred = h_old.copy()
        A_pred = A_old.copy()
        
        # 应用上游边界条件
        if bc.upstream_type == "discharge":
            Q_pred[0] = bc.upstream_values[up_idx]
        
        # 预测步：使用向前差分更新内部节点
        for i in range(0, n-1):  # 包括边界节点
            if i == 0 and bc.upstream_type == "discharge":
                continue  # 上游流量已设定
                
            # 连续性方程: ∂A/∂t + ∂Q/∂x = 0
            if i < n-1:
                dQ_dx = (Q_old[i+1] - Q_old[i]) / dx
                A_pred[i] = A_old[i] - dt * dQ_dx
                h_pred[i] = A_pred[i] / self.cross_section.width
            
            # 动量方程: ∂Q/∂t + ∂(Q²/A)/∂x + gA∂h/∂x = gA(S₀ - Sf)
            if i < n-1:
                # 对流项
                v_i = Q_old[i] / A_old[i] if A_old[i] > 1e-6 else 0
                v_ip1 = Q_old[i+1] / A_old[i+1] if A_old[i+1] > 1e-6 else 0
                convection = (v_ip1 * Q_old[i+1] - v_i * Q_old[i]) / dx
                
                # 压力项
                pressure = self.g * A_old[i] * (h_old[i+1] - h_old[i]) / dx
                
                # 重力项
                gravity = self.g * A_old[i] * self.reach.bed_slope
                
                # 摩阻项（简化）
                R = A_old[i] / (self.cross_section.width + 2 * h_old[i]) if h_old[i] > 0 else 0.1
                Sf = (self.reach.manning_n * abs(v_i) * v_i) / (R**(4/3)) if R > 0 else 0
                friction = self.g * A_old[i] * Sf
                
                Q_pred[i] = Q_old[i] + dt * (gravity - pressure - convection - friction)
        
        # 应用下游边界条件到预测值
        if bc.downstream_type == "stage":
            h_pred[-1] = bc.downstream_values[dn_idx]
            A_pred[-1] = self.cross_section.width * h_pred[-1]
            
            # 下游流量通过连续性方程计算
            i = n - 1
            if i > 0:
                dA_dt = (A_pred[i] - A_old[i]) / dt
                Q_pred[i] = Q_pred[i-1] + dx * dA_dt
        
        # === 校正步 (Corrector) ===
        # 使用预测值计算校正
        for i in range(1, n):
            # 连续性方程校正（使用向后差分）
            dQ_dx = (Q_pred[i] - Q_pred[i-1]) / dx
            self.A[i] = A_old[i] - dt * dQ_dx
            self.h[i] = self.A[i] / self.cross_section.width
            
            # 动量方程校正
            if i < n:
                # 使用预测值计算项
                v_i = Q_pred[i] / A_pred[i] if A_pred[i] > 1e-6 else 0
                v_im1 = Q_pred[i-1] / A_pred[i-1] if A_pred[i-1] > 1e-6 else 0
                
                convection = (v_i * Q_pred[i] - v_im1 * Q_pred[i-1]) / dx
                pressure = self.g * A_pred[i] * (h_pred[i] - h_pred[i-1]) / dx
                gravity = self.g * A_pred[i] * self.reach.bed_slope
                
                # 摩阻项
                R = A_pred[i] / (self.cross_section.width + 2 * h_pred[i]) if h_pred[i] > 0 else 0.1
                Sf = (self.reach.manning_n * abs(v_i) * v_i) / (R**(4/3)) if R > 0 else 0
                friction = self.g * A_pred[i] * Sf
                
                self.Q[i] = Q_old[i] + dt * (gravity - pressure - convection - friction)
        
        # 重新应用边界条件
        if bc.upstream_type == "discharge":
            self.Q[0] = bc.upstream_values[up_idx]
            # 上游水深通过连续性方程计算
            if n > 1:
                dQ_dx = (self.Q[1] - self.Q[0]) / dx
                self.A[0] = A_old[0] - dt * dQ_dx
                self.h[0] = self.A[0] / self.cross_section.width
        
        if bc.downstream_type == "stage":
            self.h[-1] = bc.downstream_values[dn_idx]
            self.A[-1] = self.cross_section.width * self.h[-1]
            # 下游流量通过连续性方程计算
            i = n - 1
            if i > 0:
                dA_dt = (self.A[i] - A_old[i]) / dt
                self.Q[i] = self.Q[i-1] + dx * dA_dt
        
        # 物理约束
        self.h = np.maximum(self.h, 0.01)
        self.A = np.maximum(self.A, 0.01 * self.cross_section.width)
        
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
            
            # 打印详细进度
            print(f"步骤 {t:2d}: 上游Q={self.Q[0]:5.1f}, 下游Q={self.Q[-1]:5.1f}, "
                  f"上游h={self.h[0]:.2f}, 下游h={self.h[-1]:.2f}")
            
            if not success:
                print(f"时间步 {t} 求解失败")
                break
        
        return results

def test_corrected_solver():
    """测试修正的求解器"""
    
    print("=== 物理正确的水动力求解器测试 ===\n")
    
    # 创建河段
    section = RectangleSection(width=20)
    reach = RiverReach(
        id="corrected_test",
        length=1600,  # 更短的河段
        bed_slope=0.001,  # 更小的坡度
        manning_n=0.03,
        width=20,
        num_sections=5  # 更少的网格
    )
    
    print(f"河段参数:")
    print(f"  长度: {reach.length}m")
    print(f"  坡度: {reach.bed_slope*1000:.1f}‰")
    print(f"  网格数: {reach.num_sections}")
    print(f"  网格间距: {reach.dx:.0f}m")
    
    # 创建求解器
    solver = CorrectedSaintVenantSolver(reach, section, dt=300)  # 更大的时间步
    
    # 设计边界条件
    num_steps = 12
    t = np.arange(num_steps)
    
    # 平缓的流量变化
    base_q = 25.0
    peak_q = 50.0
    upstream_q = base_q + (peak_q - base_q) * np.sin(np.pi * t / (num_steps - 1))**2
    
    # 固定下游水位
    downstream_h = np.full(num_steps, 1.2)
    
    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q.tolist(),
        downstream_type="stage",
        downstream_values=downstream_h.tolist()
    )
    
    print(f"\n边界条件:")
    print(f"  上游流量: {base_q:.0f} → {peak_q:.0f} m³/s (正弦变化)")
    print(f"  下游水位: {downstream_h[0]:.1f} m (固定)")
    
    # 运行仿真
    print(f"\n开始仿真...")
    results = solver.run_simulation(bc, num_steps)
    
    # 分析结果
    print(f"\n=== 结果分析 ===")
    
    time_array = np.array(results['time']) / 3600
    
    # 提取数据
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
    
    # 绘制结果
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 流量对比
    ax1.plot(time_array, upstream_q_results, 'b-o', label='上游流量', linewidth=2, markersize=4)
    ax1.plot(time_array, downstream_q_results, 'r-s', label='下游流量', linewidth=2, markersize=4)
    ax1.plot(time_array, upstream_q[:len(time_array)], 'b--', label='设定上游流量', alpha=0.7)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('流量 (m³/s)')
    ax1.set_title('修正求解器：上下游流量对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 水深对比
    ax2.plot(time_array, upstream_h_results, 'b-o', label='上游水深', linewidth=2, markersize=4)
    ax2.plot(time_array, downstream_h_results, 'r-s', label='下游水深', linewidth=2, markersize=4)
    ax2.axhline(y=downstream_h[0], color='orange', linestyle=':', label='设定下游水位')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('水深 (m)')
    ax2.set_title('修正求解器：上下游水深对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 沿程分布对比（初始 vs 最终）
    x_coords = np.array(reach.x_coords) / 1000
    initial_Q = results['Q'][0]
    final_Q = results['Q'][-1]
    
    ax3.plot(x_coords, initial_Q, 'g-o', label='初始流量分布', linewidth=2, markersize=4)
    ax3.plot(x_coords, final_Q, 'r-s', label='最终流量分布', linewidth=2, markersize=4)
    ax3.set_xlabel('距离 (km)')
    ax3.set_ylabel('流量 (m³/s)')
    ax3.set_title('沿程流量分布对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 沿程水深分布
    initial_h = results['h'][0]
    final_h = results['h'][-1]
    
    ax4.plot(x_coords, initial_h, 'g-o', label='初始水深分布', linewidth=2, markersize=4)
    ax4.plot(x_coords, final_h, 'r-s', label='最终水深分布', linewidth=2, markersize=4)
    ax4.set_xlabel('距离 (km)')
    ax4.set_ylabel('水深 (m)')
    ax4.set_title('沿程水深分布对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('corrected_solver_test.png', dpi=150, bbox_inches='tight')
    print(f"\n结果图表已保存: corrected_solver_test.png")
    
    # 物理合理性检查
    print(f"\n=== 物理合理性检查 ===")
    
    # 检查质量守恒
    mass_errors = []
    for i, (Q_array, A_array) in enumerate(zip(results['Q'], results['A'])):
        if i == 0:
            continue
        Q_in = Q_array[0]
        Q_out = Q_array[-1]
        mass_error = abs(Q_out - Q_in) / Q_in * 100
        mass_errors.append(mass_error)
    
    avg_mass_error = np.mean(mass_errors) if mass_errors else 0
    print(f"平均质量守恒误差: {avg_mass_error:.1f}%")
    
    # 检查流量传播
    q_up_change = max(upstream_q_results) - min(upstream_q_results)
    q_down_change = max(downstream_q_results) - min(downstream_q_results)
    
    if q_up_change > 0:
        propagation_ratio = q_down_change / q_up_change
        print(f"流量传播比: {propagation_ratio:.2f}")
        print(f"传播效率: {propagation_ratio*100:.1f}%")
    
    # 检查水位响应
    h_up_change = max(upstream_h_results) - min(upstream_h_results)
    h_down_change = max(downstream_h_results) - min(downstream_h_results)
    
    print(f"水位变化:")
    print(f"  上游变化: {h_up_change:.3f} m")
    print(f"  下游变化: {h_down_change:.3f} m (应该为0)")
    
    return results

if __name__ == "__main__":
    results = test_corrected_solver()