"""真正工作的非恒定流演示 - 使用稳态初始条件"""

from hydrosis.hydrodynamics import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    RectangleSection,
    TrapezoidSection,
    SteadyStateCalculator
)
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def working_unsteady_demo():
    """演示真正工作的非恒定流仿真"""
    
    print("=== 非恒定流仿真演示 ===\n")
    
    # 1. 创建河段和断面
    section = RectangleSection(width=25)
    reach = RiverReach(
        id="unsteady_river",
        length=4000,      # 4km河段
        bed_slope=0.002,  # 2‰坡度
        manning_n=0.03,
        width=25,
        num_sections=10   # 10个断面
    )
    
    print(f"河段参数:")
    print(f"  长度: {reach.length/1000:.1f} km")
    print(f"  坡度: {reach.bed_slope*1000:.1f} ‰")
    print(f"  河宽: {section.width} m")
    print(f"  网格数: {reach.num_sections}")
    print(f"  网格间距: {reach.dx:.0f} m")
    
    # 2. 计算稳态初始条件
    steady_calc = SteadyStateCalculator(reach, section)
    
    base_discharge = 30.0  # 基础流量
    h_normal = steady_calc.compute_normal_depth(base_discharge)
    h_critical = steady_calc.compute_critical_depth(base_discharge)
    
    print(f"\n水力特性:")
    print(f"  基础流量: {base_discharge} m³/s")
    print(f"  正常水深: {h_normal:.2f} m")
    print(f"  临界水深: {h_critical:.2f} m")
    print(f"  流态: {'缓流' if h_normal > h_critical else '急流'}")
    
    # 3. 创建求解器并设置初始条件
    solver = SaintVenantSolver(reach, dt=180, theta=0.8, epsilon=1e-3, cross_section=section)
    
    # 使用稳态计算的初始条件
    initial_depths, initial_discharges = steady_calc.create_initial_conditions(
        base_discharge, boundary_type="normal"
    )
    
    # 手动设置初始状态
    solver.state.depth = initial_depths.copy()
    solver.state.discharge = initial_discharges.copy()
    solver.update_hydraulic_properties(solver.state)
    
    print(f"\n初始条件:")
    print(f"  初始水深: {solver.state.depth[0]:.2f} m (均匀)")
    print(f"  初始流量: {solver.state.discharge[0]:.1f} m³/s (均匀)")
    
    # 验证初始条件
    validation = steady_calc.validate_initial_conditions(
        solver.state.depth, solver.state.discharge
    )
    print(f"  初始条件验证: {'✓ 通过' if validation['valid'] else '✗ 失败'}")
    if validation['warnings']:
        for warning in validation['warnings']:
            print(f"    警告: {warning}")
    
    # 4. 设计洪水过程
    num_steps = 25
    t = np.arange(num_steps)
    
    # 设计一个明显的洪水过程
    peak_discharge = 120.0  # 峰值流量
    peak_time = 10
    
    # 三角形洪水过程
    upstream_q = np.zeros(num_steps)
    for i in range(num_steps):
        if i <= peak_time:
            upstream_q[i] = base_discharge + (peak_discharge - base_discharge) * i / peak_time
        else:
            upstream_q[i] = peak_discharge - (peak_discharge - base_discharge) * (i - peak_time) / (num_steps - peak_time - 1)
    
    print(f"\n洪水过程:")
    print(f"  基流: {base_discharge} m³/s")
    print(f"  峰值: {peak_discharge} m³/s")
    print(f"  峰现时间: 第{peak_time}步 ({peak_time * solver.dt / 3600:.1f}小时)")
    print(f"  总时长: {num_steps * solver.dt / 3600:.1f}小时")
    
    # 5. 设置边界条件
    # 下游使用自由出流（正常水深边界）
    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q.tolist(),
        downstream_type="discharge",  # 使用自由出流
        downstream_values=[0] * num_steps  # 这个值会被忽略
    )
    
    print(f"\n边界条件:")
    print(f"  上游: 变化流量边界")
    print(f"  下游: 自由出流边界（正常水深）")
    
    # 6. 运行非恒定流仿真
    print(f"\n开始非恒定流仿真...")
    print("步数 | 时间(h) | 上游Q | 下游Q | 上游h | 下游h | 状态")
    print("-" * 60)
    
    # 存储结果
    time_history = []
    upstream_q_history = []
    downstream_q_history = []
    upstream_h_history = []
    downstream_h_history = []
    convergence_history = []
    
    converged_steps = 0
    
    for step in range(num_steps):
        time_hours = step * solver.dt / 3600
        
        # 记录仿真前状态
        q_up_before = solver.state.discharge[0]
        q_down_before = solver.state.discharge[-1]
        h_up_before = solver.state.depth[0]
        h_down_before = solver.state.depth[-1]
        
        # 执行一个时间步
        converged = solver.solve_timestep(bc, step)
        
        # 记录仿真后状态
        q_up_after = solver.state.discharge[0]
        q_down_after = solver.state.discharge[-1]
        h_up_after = solver.state.depth[0]
        h_down_after = solver.state.depth[-1]
        
        # 存储结果
        time_history.append(time_hours)
        upstream_q_history.append(q_up_after)
        downstream_q_history.append(q_down_after)
        upstream_h_history.append(h_up_after)
        downstream_h_history.append(h_down_after)
        convergence_history.append(converged)
        
        if converged:
            converged_steps += 1
        
        # 打印进度
        status = "✓" if converged else "✗"
        print(f"{step:4d} | {time_hours:6.2f} | {q_up_after:5.1f} | {q_down_after:5.1f} | "
              f"{h_up_after:5.2f} | {h_down_after:5.2f} | {status}")
    
    print(f"\n仿真完成!")
    print(f"收敛率: {converged_steps}/{num_steps} ({converged_steps/num_steps*100:.1f}%)")
    
    # 7. 分析结果
    time_array = np.array(time_history)
    
    print(f"\n=== 结果分析 ===")
    print(f"流量变化:")
    print(f"  上游: {min(upstream_q_history):.1f} - {max(upstream_q_history):.1f} m³/s")
    print(f"  下游: {min(downstream_q_history):.1f} - {max(downstream_q_history):.1f} m³/s")
    
    print(f"水深变化:")
    print(f"  上游: {min(upstream_h_history):.2f} - {max(upstream_h_history):.2f} m")
    print(f"  下游: {min(downstream_h_history):.2f} - {max(downstream_h_history):.2f} m")
    
    # 分析洪峰传播
    up_peak_idx = np.argmax(upstream_q_history)
    down_peak_idx = np.argmax(downstream_q_history)
    
    if up_peak_idx < len(upstream_q_history) - 2 and down_peak_idx < len(downstream_q_history) - 2:
        lag_time = (down_peak_idx - up_peak_idx) * solver.dt / 3600
        peak_attenuation = (max(upstream_q_history) - max(downstream_q_history)) / max(upstream_q_history) * 100
        
        print(f"洪峰传播:")
        print(f"  传播时间: {lag_time:.2f} 小时")
        print(f"  峰值衰减: {peak_attenuation:.1f}%")
        print(f"  上游峰值: {max(upstream_q_history):.1f} m³/s")
        print(f"  下游峰值: {max(downstream_q_history):.1f} m³/s")
    
    # 8. 绘制结果
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 流量过程对比
    ax1.plot(time_array, upstream_q_history, 'b-o', label='上游流量', linewidth=2, markersize=3)
    ax1.plot(time_array, downstream_q_history, 'r-s', label='下游流量', linewidth=2, markersize=3)
    ax1.plot(time_array, upstream_q[:len(time_array)], 'b--', label='设定上游流量', alpha=0.7)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('流量 (m³/s)')
    ax1.set_title('非恒定流流量过程')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 水深过程对比
    ax2.plot(time_array, upstream_h_history, 'b-o', label='上游水深', linewidth=2, markersize=3)
    ax2.plot(time_array, downstream_h_history, 'r-s', label='下游水深', linewidth=2, markersize=3)
    ax2.axhline(y=h_normal, color='g', linestyle=':', label=f'正常水深 {h_normal:.2f}m')
    ax2.axhline(y=h_critical, color='orange', linestyle=':', label=f'临界水深 {h_critical:.2f}m')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('水深 (m)')
    ax2.set_title('非恒定流水深过程')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 流量传播分析
    q_diff = np.array(downstream_q_history) - np.array(upstream_q_history)
    ax3.plot(time_array, q_diff, 'g-o', label='下游-上游流量差', linewidth=2, markersize=3)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('流量差 (m³/s)')
    ax3.set_title('流量传播效应')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 收敛性分析
    convergence_rate = np.cumsum(convergence_history) / np.arange(1, len(convergence_history) + 1) * 100
    ax4.plot(time_array, convergence_rate, 'm-o', label='累积收敛率', linewidth=2, markersize=3)
    ax4.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80%基准线')
    ax4.set_xlabel('时间 (小时)')
    ax4.set_ylabel('收敛率 (%)')
    ax4.set_title('数值收敛性')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('working_unsteady_flow.png', dpi=150, bbox_inches='tight')
    print(f"\n结果图表已保存: working_unsteady_flow.png")
    
    return {
        'time': time_array,
        'upstream_q': upstream_q_history,
        'downstream_q': downstream_q_history,
        'upstream_h': upstream_h_history,
        'downstream_h': downstream_h_history,
        'convergence': convergence_history
    }

if __name__ == "__main__":
    results = working_unsteady_demo()