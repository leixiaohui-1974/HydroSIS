"""梯形断面河道洪水演进"""

from hydrosis.hydrodynamics import (
    TrapezoidSection,
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    compute_normal_depth
)
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建梯形断面
section = TrapezoidSection(
    bottom_width=10,    # 底宽10m
    side_slope=2.0      # 边坡2:1
)

# 创建河段
reach = RiverReach(
    id="trapezoid_reach",
    length=8000,
    bed_slope=0.001,
    manning_n=0.03,
    width=10,  # 这里的width在使用自定义断面时会被覆盖
    num_sections=30
)

# 创建求解器（使用自定义断面）
solver = SaintVenantSolver(reach, dt=60, cross_section=section)

# 洪水过程边界条件
num_steps = 120
t = np.arange(num_steps)
upstream_q = 20 + 80 * np.exp(-((t - 40)/15)**2)  # 高斯型洪峰

bc = BoundaryCondition(
    upstream_type="discharge",
    upstream_values=upstream_q.tolist(),
    downstream_type="stage",
    downstream_values=[3.0] * num_steps
)

# 运行模拟
solver.set_lateral_inflow([0.01] * reach.num_sections)
results = solver.run_simulation(bc, num_steps)

# 可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 流量过程
outlet_q = [q[-1] for q in results['discharge']]
ax1.plot(results['time'], upstream_q, 'b-', label='入口流量')
ax1.plot(results['time'], outlet_q, 'r-', label='出口流量')
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('流量 (m³/s)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_title('梯形断面河道洪水演进')

# 水深过程
outlet_h = [h[-1] for h in results['depth']]
ax2.plot(results['time'], outlet_h, 'g-')
ax2.set_xlabel('时间 (s)')
ax2.set_ylabel('水深 (m)')
ax2.grid(True, alpha=0.3)
ax2.set_title('出口断面水深变化')

plt.tight_layout()
plt.savefig('trapezoid_flood.png', dpi=150)
print("✓ 图表已保存至 trapezoid_flood.png")

# 计算正常水深
normal_depth = compute_normal_depth(
    section, 
    discharge=max(upstream_q),
    bed_slope=reach.bed_slope,
    manning_n=reach.manning_n
)
print(f"峰值流量对应的正常水深: {normal_depth:.2f} m")