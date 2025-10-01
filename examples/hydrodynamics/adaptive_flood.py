"""自适应时间步长洪水模拟"""

from hydrosis.hydrodynamics import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    TrapezoidSection,
    AdaptiveTimeStepController,
    AdaptiveStrategy,
    VariableTimeStepSimulator
)

# 创建河段和断面
section = TrapezoidSection(bottom_width=12, side_slope=2.0)
reach = RiverReach(
    id="adaptive_reach",
    length=10000,
    bed_slope=0.001,
    manning_n=0.03,
    width=12,
    num_sections=40
)

# 创建自适应控制器
controller = AdaptiveTimeStepController(
    initial_dt=60,
    min_dt=15,
    max_dt=300,
    target_cfl=0.5,
    strategy=AdaptiveStrategy.HYBRID
)

# 创建求解器
solver = SaintVenantSolver(reach, dt=60, cross_section=section)

# 边界条件（急涨急落的洪水）
num_steps = 200
import numpy as np
t = np.arange(num_steps)
# 急涨缓落型洪水
upstream_q = 15 + 120 * (np.exp(-((t-50)/20)**2) + 0.3*np.exp(-((t-80)/30)**2))

bc = BoundaryCondition(
    upstream_type="discharge",
    upstream_values=upstream_q.tolist(),
    downstream_type="stage",
    downstream_values=[2.8] * num_steps
)

# 使用自适应模拟器
simulator = VariableTimeStepSimulator(controller)
results = simulator.run_adaptive_simulation(
    solver,
    total_time=num_steps * 60,  # 总时间（秒）
    boundary_conditions=bc,
    verbose=True
)

# 绘制时间步长历史
simulator.plot_metrics(save_path='adaptive_timestep_history.png')

# 统计报告
stats = controller.get_statistics()
print("\n自适应控制统计:")
print(f"  步长范围: {stats['min_dt_used']:.0f} - {stats['max_dt_used']:.0f} 秒")
print(f"  平均步长: {stats['avg_dt']:.1f} 秒")
print(f"  调整次数: {stats['adjustments']}")
print(f"  成功率: {stats['success_rate']:.1f}%")

# 与固定步长对比
fixed_dt = 60
efficiency_gain = (num_steps * fixed_dt) / sum(results['dt'])
print(f"\n效率提升: {efficiency_gain:.1f}x")
print(f"  (相比固定步长 {fixed_dt}s)")