"""复合断面漫滩过程模拟"""

from hydrosis.hydrodynamics import (
    CompoundSection,
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition
)
import numpy as np

# 创建复合断面（主槽+滩地）
section = CompoundSection(
    main_bottom_width=8,
    main_side_slope=1.5,
    floodplain_height=2.5,      # 滩地高出主槽底2.5m
    left_floodplain_width=20,
    right_floodplain_width=20
)

# 测试不同水深的水力参数
print("复合断面水力参数:")
print("-" * 50)
for depth in [1.0, 2.0, 2.5, 3.0, 4.0]:
    props = section.compute_properties(depth)
    print(f"水深 {depth:.1f}m: 面积={props.area:.1f}m², "
          f"顶宽={props.top_width:.1f}m, "
          f"水力半径={props.hydraulic_radius:.2f}m")
    if depth <= 2.5:
        print("  (主槽内)")
    else:
        print("  (已漫滩)")

# 运行洪水模拟 - 使用更温和的参数
reach = RiverReach(
    id="compound_reach",
    length=5000,
    bed_slope=0.0008,  # 稍大的坡度提升稳定性
    manning_n=0.035,
    width=8,
    num_sections=15    # 减少网格数提升稳定性
)

# 使用更大的时间步长和更保守的theta值
solver = SaintVenantSolver(reach, dt=180, theta=0.7, cross_section=section)

# 设置合理的初始条件
initial_depth = 1.8  # 主槽内的初始水深
initial_q = 25.0     # 初始流量
solver.state.depth[:] = initial_depth
solver.state.discharge[:] = initial_q
solver.update_hydraulic_properties(solver.state)

# 更平缓的洪水过程（避免急剧变化）
num_steps = 80
t = np.linspace(0, 1, num_steps)
# 使用平滑的三角波而非正弦波
peak_factor = np.where(t <= 0.3, t / 0.3, 
                      np.where(t <= 0.7, 1.0, (1.0 - t) / 0.3))
upstream_q = 25 + 80 * peak_factor  # 从25到105 m³/s的平缓变化

bc = BoundaryCondition(
    upstream_type="discharge",
    upstream_values=upstream_q.tolist(),
    downstream_type="stage",
    downstream_values=[2.8] * num_steps  # 稍低的下游水位
)

print(f"\n开始模拟 (时间步长: {solver.dt}s, 网格数: {reach.num_sections})")
print(f"初始条件: 水深={initial_depth:.1f}m, 流量={initial_q:.1f}m³/s")

results = solver.run_simulation(bc, num_steps)

# 分析漫滩时刻
print("\n漫滩分析:")
floodplain_threshold = section.floodplain_height
overbank_detected = False

for t, depth_array in enumerate(results['depth']):
    max_depth = max(depth_array)
    if max_depth > floodplain_threshold and not overbank_detected:
        time_hours = t * solver.dt / 3600
        print(f"⚠️ 时刻 {t*solver.dt:.0f}s ({time_hours:.1f}h): 开始漫滩")
        print(f"   最大水深: {max_depth:.2f}m (超出滩地高度 {max_depth-floodplain_threshold:.2f}m)")
        overbank_detected = True
        
        # 分析漫滩时的流量分配
        props_main = section.main_channel.compute_properties(floodplain_threshold)
        props_total = section.compute_properties(max_depth)
        fp_area = props_total.area - props_main.area
        
        print(f"   主槽面积: {props_main.area:.1f}m², 滩地面积: {fp_area:.1f}m²")
        print(f"   面积比例: 主槽{props_main.area/props_total.area*100:.0f}%, "
              f"滩地{fp_area/props_total.area*100:.0f}%")
        break

if not overbank_detected:
    max_depth_overall = max(max(depth_array) for depth_array in results['depth'])
    print(f"未发生漫滩，最大水深: {max_depth_overall:.2f}m")

# 输出最终状态
final_q = results['discharge'][-1][-1]
final_h = results['depth'][-1][-1]
print(f"\n最终状态:")
print(f"出口流量: {final_q:.1f} m³/s")
print(f"出口水深: {final_h:.2f} m")

print("\n✓ 复合断面模拟完成")