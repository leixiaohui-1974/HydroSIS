from hydrodynamic_1d import RiverReach, BoundaryCondition, SaintVenantSolver

# 1. 定义河段
reach = RiverReach(
    id="test_river",
    length=5000,        # 长度 5km
    bed_slope=0.001,    # 坡度 1‰
    manning_n=0.03,     # 糙率
    width=30,           # 河宽 30m
    num_sections=20     # 20个计算断面
)

# 2. 创建求解器
solver = SaintVenantSolver(reach, dt=60)  # 时间步长60秒

# 3. 设置边界条件
num_steps = 100
bc = BoundaryCondition(
    upstream_type="discharge",
    upstream_values=[20.0] * num_steps,  # 上游流量 20 m³/s
    downstream_type="stage",
    downstream_values=[2.5] * num_steps  # 下游水位 2.5m
)

# 4. 运行模拟
results = solver.run_simulation(bc, num_steps)

# 5. 查看结果
print(f"出口流量: {results['discharge'][-1][-1]:.2f} m³/s")