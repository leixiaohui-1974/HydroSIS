"""边界条件分析 - 验证上下游流量关系"""

from hydrosis.hydrodynamics import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    RectangleSection
)
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_boundary_response():
    """分析固定下游水位边界条件下的流量响应"""
    
    print("=== 边界条件响应分析 ===\n")
    
    # 创建河段
    section = RectangleSection(width=20)
    reach = RiverReach(
        id="boundary_test",
        length=3000,
        bed_slope=0.003,  # 较大坡度
        manning_n=0.03,
        width=20,
        num_sections=8
    )
    
    solver = SaintVenantSolver(reach, dt=240, theta=0.9, epsilon=1e-3, cross_section=section)
    
    print(f"河段参数:")
    print(f"  长度: {reach.length}m, 坡度: {reach.bed_slope}")
    print(f"  网格数: {reach.num_sections}")
    print(f"  初始流量: {solver.state.discharge[0]:.1f} m³/s")
    print(f"  初始水深: {solver.state.depth[0]:.2f} m")
    
    # 设计变化的上游流量
    num_steps = 30
    t = np.arange(num_steps)
    
    # 三角波洪水过程
    base_q = 20
    peak_q = 80
    peak_time = 12
    
    upstream_q = np.zeros(num_steps)
    for i in range(num_steps):
        if i <= peak_time:
            upstream_q[i] = base_q + (peak_q - base_q) * i / peak_time
        else:
            upstream_q[i] = peak_q - (peak_q - base_q) * (i - peak_time) / (num_steps - peak_time - 1)
    
    print(f"\n上游流量变化:")
    print(f"  基流: {base_q} m³/s")
    print(f"  峰值: {peak_q} m³/s")
    print(f"  峰现时间: 第{peak_time}步")
    
    # 固定下游水位边界
    # 使用峰值流量对应的正常水深作为固定水位
    peak_depth = solver._compute_normal_depth_iterative(peak_q)
    fixed_stage = peak_depth + 0.1  # 稍高于正常水深
    
    print(f"\n边界条件:")
    print(f"  上游: 变化流量 ({base_q}-{peak_q} m³/s)")
    print(f"  下游: 固定水位 {fixed_stage:.2f} m")
    
    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q.tolist(),
        downstream_type="stage",
        downstream_values=[fixed_stage] * num_steps
    )
    
    # 存储每个时间步的详细结果
    time_history = []
    upstream_q_history = []
    downstream_q_history = []
    upstream_h_history = []
    downstream_h_history = []
    mass_balance_error = []
    
    print(f"\n开始模拟...")
    print("时间步 | 上游Q | 下游Q | 上游h | 下游h | 质量误差")
    print("-" * 55)
    
    for step in range(num_steps):
        # 记录模拟前状态
        q_up_before = solver.state.discharge[0]
        q_down_before = solver.state.discharge[-1]
        h_up_before = solver.state.depth[0]
        h_down_before = solver.state.depth[-1]
        
        # 执行一个时间步
        converged = solver.solve_timestep(bc, step)
        
        # 记录模拟后状态
        q_up_after = solver.state.discharge[0]
        q_down_after = solver.state.discharge[-1]
        h_up_after = solver.state.depth[0]
        h_down_after = solver.state.depth[-1]
        
        # 计算质量平衡误差
        mass_error = abs(q_down_after - upstream_q[step]) / upstream_q[step] * 100
        
        # 存储结果
        time_history.append(step * solver.dt / 3600)  # 转换为小时
        upstream_q_history.append(q_up_after)
        downstream_q_history.append(q_down_after)
        upstream_h_history.append(h_up_after)
        downstream_h_history.append(h_down_after)
        mass_balance_error.append(mass_error)
        
        # 打印详细信息
        status = "✓" if converged else "✗"
        print(f"{step:6d} | {q_up_after:5.1f} | {q_down_after:5.1f} | {h_up_after:5.2f} | {h_down_after:5.2f} | {mass_error:6.1f}% {status}")
        
        if not converged and step > 5:  # 允许前几步不收敛
            print("⚠️ 连续不收敛，停止模拟")
            break
    
    # 绘制分析图表
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    time_array = np.array(time_history)
    
    # 1. 上下游流量对比
    ax1.plot(time_array, upstream_q_history, 'b-o', label='上游流量', linewidth=2, markersize=4)
    ax1.plot(time_array, downstream_q_history, 'r-s', label='下游流量', linewidth=2, markersize=4)
    ax1.plot(time_array, upstream_q[:len(time_array)], 'b--', label='设定上游流量', alpha=0.7)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('流量 (m³/s)')
    ax1.set_title('上下游流量对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 上下游水深对比
    ax2.plot(time_array, upstream_h_history, 'b-o', label='上游水深', linewidth=2, markersize=4)
    ax2.plot(time_array, downstream_h_history, 'r-s', label='下游水深', linewidth=2, markersize=4)
    ax2.axhline(y=fixed_stage, color='orange', linestyle=':', label=f'固定下游水位 {fixed_stage:.2f}m')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('水深 (m)')
    ax2.set_title('上下游水深对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 流量差异分析
    q_diff = np.array(downstream_q_history) - np.array(upstream_q_history)
    ax3.plot(time_array, q_diff, 'g-o', label='下游-上游流量差', linewidth=2, markersize=4)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('流量差 (m³/s)')
    ax3.set_title('上下游流量差异')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 质量平衡误差
    ax4.plot(time_array, mass_balance_error, 'm-o', label='质量平衡误差', linewidth=2, markersize=4)
    ax4.axhline(y=5, color='r', linestyle='--', alpha=0.7, label='5%误差线')
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('误差 (%)')
    ax4.set_title('质量平衡误差')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('boundary_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n分析图表已保存: boundary_analysis.png")
    
    # 统计分析
    print(f"\n=== 统计分析 ===")
    print(f"流量变化范围:")
    print(f"  上游: {min(upstream_q_history):.1f} - {max(upstream_q_history):.1f} m³/s")
    print(f"  下游: {min(downstream_q_history):.1f} - {max(downstream_q_history):.1f} m³/s")
    
    print(f"水深变化范围:")
    print(f"  上游: {min(upstream_h_history):.2f} - {max(upstream_h_history):.2f} m")
    print(f"  下游: {min(downstream_h_history):.2f} - {max(downstream_h_history):.2f} m")
    
    avg_q_diff = np.mean(np.abs(q_diff))
    max_q_diff = np.max(np.abs(q_diff))
    avg_mass_error = np.mean(mass_balance_error)
    
    print(f"流量传播特性:")
    print(f"  平均流量差: {avg_q_diff:.2f} m³/s")
    print(f"  最大流量差: {max_q_diff:.2f} m³/s")
    print(f"  平均质量误差: {avg_mass_error:.1f}%")
    
    # 分析滞后效应
    if len(upstream_q_history) > 15:
        up_peak_idx = np.argmax(upstream_q_history)
        down_peak_idx = np.argmax(downstream_q_history)
        lag_steps = down_peak_idx - up_peak_idx
        lag_time = lag_steps * solver.dt / 3600
        
        print(f"洪峰传播:")
        print(f"  上游峰现: 第{up_peak_idx}步")
        print(f"  下游峰现: 第{down_peak_idx}步")
        print(f"  传播滞后: {lag_steps}步 ({lag_time:.2f}小时)")
    
    return {
        'time': time_array,
        'upstream_q': upstream_q_history,
        'downstream_q': downstream_q_history,
        'upstream_h': upstream_h_history,
        'downstream_h': downstream_h_history,
        'mass_error': mass_balance_error
    }

if __name__ == "__main__":
    results = analyze_boundary_response()