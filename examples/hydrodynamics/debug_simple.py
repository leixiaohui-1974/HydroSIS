"""极简调试版本 - 验证数值方法基本正确性"""

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

def debug_basic_flow():
    """最基本的稳定流验证"""
    
    print("=== 基础数值方法调试 ===\n")
    
    # 使用最简单的矩形断面
    section = RectangleSection(width=20)
    reach = RiverReach(
        id="debug_river",
        length=2000,      # 很短的河段
        bed_slope=0.005,  # 较大坡度确保稳定
        manning_n=0.03,
        width=20,
        num_sections=6    # 很少的网格点
    )
    
    solver = SaintVenantSolver(reach, dt=300, theta=0.9, epsilon=1e-2, cross_section=section)
    
    print(f"河段参数:")
    print(f"  长度: {reach.length}m, 坡度: {reach.bed_slope}")
    print(f"  网格数: {reach.num_sections}, 网格间距: {reach.dx:.1f}m")
    print(f"  时间步长: {solver.dt}s")
    
    print(f"\n初始状态:")
    print(f"  流量: {solver.state.discharge}")
    print(f"  水深: {solver.state.depth}")
    
    # 验证初始状态是否为稳定流
    initial_q = solver.state.discharge[0]
    initial_h = solver.state.depth[0]
    
    # 用曼宁公式验证
    A = initial_h * section.width
    P = section.width + 2 * initial_h
    R = A / P
    Q_manning = (1/reach.manning_n) * A * R**(2/3) * reach.bed_slope**0.5
    
    print(f"\n稳定流验证:")
    print(f"  初始流量: {initial_q:.2f} m³/s")
    print(f"  曼宁公式流量: {Q_manning:.2f} m³/s")
    print(f"  误差: {abs(initial_q - Q_manning)/initial_q*100:.1f}%")
    
    # 极简边界条件：恒定流量
    num_steps = 10
    constant_q = initial_q
    
    # 下游边界：自由出流（正常水深）
    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=[constant_q] * num_steps,
        downstream_type="stage",  # 这里实际会被忽略，使用自由出流
        downstream_values=[initial_h] * num_steps
    )
    
    print(f"\n开始模拟 (恒定流量 {constant_q:.1f} m³/s)...")
    
    # 逐步模拟，观察每一步
    for step in range(num_steps):
        print(f"\n--- 时间步 {step} ---")
        print(f"模拟前: Q={solver.state.discharge[-1]:.2f}, h={solver.state.depth[-1]:.3f}")
        
        converged = solver.solve_timestep(bc, step)
        
        print(f"模拟后: Q={solver.state.discharge[-1]:.2f}, h={solver.state.depth[-1]:.3f}")
        print(f"收敛状态: {'✓' if converged else '✗'}")
        
        # 检查质量守恒
        Q_in = solver.state.discharge[0]
        Q_out = solver.state.discharge[-1]
        mass_error = abs(Q_out - Q_in) / Q_in * 100
        print(f"质量守恒误差: {mass_error:.2f}%")
        
        if not converged:
            print("⚠️ 未收敛，停止模拟")
            break
    
    print(f"\n=== 调试完成 ===")
    
    # 绘制结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    x = np.array(reach.x_coords) / 1000  # 转换为km
    
    ax1.plot(x, solver.state.discharge, 'b-o', label='流量')
    ax1.set_xlabel('距离 (km)')
    ax1.set_ylabel('流量 (m³/s)')
    ax1.set_title('沿程流量分布')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(x, solver.state.depth, 'r-o', label='水深')
    ax2.set_xlabel('距离 (km)')
    ax2.set_ylabel('水深 (m)')
    ax2.set_title('沿程水深分布')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('debug_basic_flow.png', dpi=150, bbox_inches='tight')
    print("调试图表已保存: debug_basic_flow.png")

if __name__ == "__main__":
    debug_basic_flow()