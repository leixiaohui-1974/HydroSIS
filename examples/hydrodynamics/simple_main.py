"""简化版水动力主示例 - 直接使用求解器API"""

from hydrosis.hydrodynamics import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    TrapezoidSection,
    CompoundSection,
    RectangleSection
)
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def demo_multiple_sections():
    """演示多种断面类型的水动力模拟"""
    
    print("=== HydroSIS 水动力模块演示 ===\n")
    
    # 1. 矩形断面演示
    print("1. 矩形断面河道")
    rect_section = RectangleSection(width=25)
    rect_reach = RiverReach(
        id="rect_river",
        length=5000,  # 减少长度
        bed_slope=0.002,  # 增大坡度提升稳定性
        manning_n=0.03,
        width=25,
        num_sections=15  # 减少网格数
    )
    
    rect_solver = SaintVenantSolver(rect_reach, dt=180, theta=0.8, cross_section=rect_section)
    
    # 更温和的洪水过程
    num_steps = 40
    t = np.linspace(0, num_steps-1, num_steps)
    base_q = 20  # 与初始化一致
    peak_q = 40  # 减小峰值
    upstream_q = base_q + peak_q * np.exp(-((t - 15)/6)**2)
    
    # 使用更合理的下游边界
    downstream_h = []
    for q in upstream_q:
        # 根据流量估算下游水位（简单的水位-流量关系）
        h_est = rect_solver._compute_normal_depth_iterative(q)
        downstream_h.append(h_est + 0.2)  # 稍高于正常水深
    
    bc_rect = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q.tolist(),
        downstream_type="stage",
        downstream_values=downstream_h
    )
    
    print(f"   初始状态: Q={rect_solver.state.discharge[0]:.1f} m³/s, h={rect_solver.state.depth[0]:.2f} m")
    results_rect = rect_solver.run_simulation(bc_rect, num_steps)
    print(f"   ✓ 模拟完成，出口峰值流量: {max(q[-1] for q in results_rect['discharge']):.1f} m³/s")
    
    # 2. 梯形断面演示
    print("\n2. 梯形断面河道")
    trap_section = TrapezoidSection(bottom_width=12, side_slope=1.5)  # 减小边坡
    trap_reach = RiverReach(
        id="trap_river", 
        length=6000,
        bed_slope=0.0015,
        manning_n=0.035,
        width=12,
        num_sections=18
    )
    
    trap_solver = SaintVenantSolver(trap_reach, dt=200, theta=0.8, cross_section=trap_section)
    
    # 为梯形断面计算合适的下游边界
    downstream_h_trap = []
    for q in upstream_q:
        h_est = trap_solver._compute_normal_depth_iterative(q)
        downstream_h_trap.append(h_est + 0.15)
    
    bc_trap = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q.tolist(),
        downstream_type="stage", 
        downstream_values=downstream_h_trap
    )
    
    print(f"   初始状态: Q={trap_solver.state.discharge[0]:.1f} m³/s, h={trap_solver.state.depth[0]:.2f} m")
    results_trap = trap_solver.run_simulation(bc_trap, num_steps)
    print(f"   ✓ 模拟完成，出口峰值流量: {max(q[-1] for q in results_trap['discharge']):.1f} m³/s")
    
    # 3. 复合断面演示（简化参数）
    print("\n3. 复合断面河道（主槽+滩地）")
    comp_section = CompoundSection(
        main_bottom_width=8,
        main_side_slope=1.0,  # 简化边坡
        floodplain_height=1.8,  # 降低滩地高度
        left_floodplain_width=12,
        right_floodplain_width=12
    )
    
    comp_reach = RiverReach(
        id="comp_river",
        length=4000,
        bed_slope=0.002,  # 增大坡度
        manning_n=0.04,
        width=8,
        num_sections=12
    )
    
    comp_solver = SaintVenantSolver(comp_reach, dt=240, theta=0.85, cross_section=comp_section)
    
    # 为复合断面计算下游边界
    downstream_h_comp = []
    for q in upstream_q:
        h_est = comp_solver._compute_normal_depth_iterative(q)
        downstream_h_comp.append(h_est + 0.1)
    
    bc_comp = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q.tolist(),
        downstream_type="stage",
        downstream_values=downstream_h_comp
    )
    
    print(f"   初始状态: Q={comp_solver.state.discharge[0]:.1f} m³/s, h={comp_solver.state.depth[0]:.2f} m")
    results_comp = comp_solver.run_simulation(bc_comp, num_steps)
    
    # 检查是否发生漫滩
    max_depth = max(max(h) for h in results_comp['depth'])
    if max_depth > comp_section.floodplain_height:
        print(f"   ⚠️ 发生漫滩！最大水深: {max_depth:.2f}m (滩地高度: {comp_section.floodplain_height:.1f}m)")
    else:
        print(f"   ✓ 未发生漫滩，最大水深: {max_depth:.2f}m")
    
    print(f"   ✓ 模拟完成，出口峰值流量: {max(q[-1] for q in results_comp['discharge']):.1f} m³/s")
    
    # 4. 绘制对比图
    print("\n4. 生成对比图表")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    
    time_hours = np.array(results_rect['time']) / 3600
    
    # 流量对比
    ax1.plot(time_hours, [q[-1] for q in results_rect['discharge']], 'b-', label='矩形断面', linewidth=2)
    ax1.plot(time_hours, [q[-1] for q in results_trap['discharge']], 'r-', label='梯形断面', linewidth=2)
    ax1.plot(time_hours, [q[-1] for q in results_comp['discharge']], 'g-', label='复合断面', linewidth=2)
    ax1.plot(time_hours, upstream_q[:len(time_hours)], 'k--', label='入口流量', alpha=0.7)
    ax1.set_xlabel('时间 (小时)')
    ax1.set_ylabel('流量 (m³/s)')
    ax1.set_title('出口流量对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 水深对比
    ax2.plot(time_hours, [h[-1] for h in results_rect['depth']], 'b-', label='矩形断面')
    ax2.plot(time_hours, [h[-1] for h in results_trap['depth']], 'r-', label='梯形断面')
    ax2.plot(time_hours, [h[-1] for h in results_comp['depth']], 'g-', label='复合断面')
    ax2.axhline(y=comp_section.floodplain_height, color='orange', linestyle=':', label='滩地高度')
    ax2.set_xlabel('时间 (小时)')
    ax2.set_ylabel('水深 (m)')
    ax2.set_title('出口水深对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 流速对比
    rect_v = [q[-1]/((h[-1]*rect_section.width) if h[-1] > 0.01 else 1) 
              for q, h in zip(results_rect['discharge'], results_rect['depth'])]
    trap_props = [trap_section.compute_properties(h[-1]) for h in results_trap['depth']]
    trap_v = [q[-1]/props.area if props.area > 0.01 else 0 
              for q, props in zip(results_trap['discharge'], trap_props)]
    comp_props = [comp_section.compute_properties(h[-1]) for h in results_comp['depth']]
    comp_v = [q[-1]/props.area if props.area > 0.01 else 0 
              for q, props in zip(results_comp['discharge'], comp_props)]
    
    ax3.plot(time_hours, rect_v, 'b-', label='矩形断面')
    ax3.plot(time_hours, trap_v, 'r-', label='梯形断面')
    ax3.plot(time_hours, comp_v, 'g-', label='复合断面')
    ax3.set_xlabel('时间 (小时)')
    ax3.set_ylabel('流速 (m/s)')
    ax3.set_title('出口流速对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 断面几何对比
    depths = np.linspace(0.5, 3.0, 50)
    rect_areas = [rect_section.compute_properties(d).area for d in depths]
    trap_areas = [trap_section.compute_properties(d).area for d in depths]
    comp_areas = [comp_section.compute_properties(d).area for d in depths]
    
    ax4.plot(depths, rect_areas, 'b-', label='矩形断面')
    ax4.plot(depths, trap_areas, 'r-', label='梯形断面')
    ax4.plot(depths, comp_areas, 'g-', label='复合断面')
    ax4.axvline(x=comp_section.floodplain_height, color='orange', linestyle=':', alpha=0.7)
    ax4.set_xlabel('水深 (m)')
    ax4.set_ylabel('过水面积 (m²)')
    ax4.set_title('断面几何特性')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hydrodynamics_comparison_fixed.png', dpi=150, bbox_inches='tight')
    print("   ✓ 对比图表已保存: hydrodynamics_comparison_fixed.png")
    
    # 5. 性能统计
    print("\n=== 模拟性能统计 ===")
    print(f"矩形断面: 网格数={rect_reach.num_sections}, 时间步长={rect_solver.dt}s")
    print(f"梯形断面: 网格数={trap_reach.num_sections}, 时间步长={trap_solver.dt}s") 
    print(f"复合断面: 网格数={comp_reach.num_sections}, 时间步长={comp_solver.dt}s")
    print("\n✓ 水动力模块演示完成！")

if __name__ == "__main__":
    demo_multiple_sections()