# ============ HydroSIS 集成接口 ============

class HydrodynamicRoutingModel:
    """作为 HydroSIS RoutingModel 的水动力路由实现"""
    
    def __init__(self, parameters: Mapping[str, float]):
        self.parameters = dict(parameters)
        
        # 从参数提取河道配置
        self.reach = RiverReach(
            id=str(parameters.get('reach_id', 'main')),
            length=float(parameters.get('length', 10000)),  # 默认10km
            bed_slope=float(parameters.get('bed_slope', 0.001)),
            manning_n=float(parameters.get('manning_n', 0.03)),
            width=float(parameters.get('width', 30)),
            num_sections=int(parameters.get('num_sections', 20))
        )
        
        self.dt = float(parameters.get('time_step', 300))  # 默认5分钟
        self.solver = SaintVenantSolver(self.reach, dt=self.dt)
        
    def route(self, subbasin, inflow: List[float]) -> List[float]:
        """实现 HydroSIS RoutingModel 接口
        
        参数:
            subbasin: 子流域对象
            inflow: 产流时间序列 (m³/s)
        
        返回:
            出口流量时间序列 (m³/s)
        """
        num_steps = len(inflow)
        
        # 将产流均匀分布到河段
        lateral_per_section = [q / self.reach.num_sections for q in inflow]
        
        # 配置边界条件
        bc = BoundaryCondition(
            upstream_type="discharge",
            upstream_values=[0.0] * num_steps,  # 无上游来水
            downstream_type="stage",
            downstream_values=[2.0] * num_steps  # 下游恒定水位
        )
        
        outflow = []
        for t in range(num_steps):
            self.solver.set_lateral_inflow(
                [lateral_per_section[t]] * self.reach.num_sections
            )
            self.solver.solve_timestep(bc, t)
            outflow.append(float(self.solver.state.discharge[-1]))
        
        return outflow


def create_coupled_model_config():
    """生成耦合模拟的示例配置"""
    config = {
        "routing_models": [
            {
                "id": "hydrodynamic",
                "model_type": "saint_venant_1d",
                "parameters": {
                    "reach_id": "main_channel",
                    "length": 15000,  # 15 km
                    "bed_slope": 0.0005,
                    "manning_n": 0.035,
                    "width": 40,
                    "num_sections": 30,
                    "time_step": 300
                }
            }
        ]
    }
    return config


# 注册到 HydroSIS 框架
try:
    from hydrosis.routing.base import RoutingModelConfig
    RoutingModelConfig.register("saint_venant_1d", HydrodynamicRoutingModel)
    print("✓ 一维水动力模型已注册到 HydroSIS 路由模型库")
except ImportError:
    print("⚠ 未检测到 HydroSIS 框架,模型可独立运行")


if __name__ == "__main__":
    # 独立运行示例
    print("=" * 60)
    print("一维水动力模型独立测试")
    print("=" * 60)
    
    reach = RiverReach(
        id="test_reach",
        length=5000,
        bed_slope=0.001,
        manning_n=0.03,
        width=25,
        num_sections=15
    )
    
    solver = SaintVenantSolver(reach, dt=60)
    
    # 设置洪水过程边界
    num_steps = 100
    peak_time = 30
    upstream_q = [10 + 50 * math.exp(-((t-peak_time)/10)**2) for t in range(num_steps)]
    
    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=upstream_q,
        downstream_type="stage",
        downstream_values=[2.5] * num_steps
    )
    
    # 设置均匀侧向入流
    solver.set_lateral_inflow([0.01] * reach.num_sections)
    
    print(f"模拟河段: {reach.length}m, {reach.num_sections}个断面")
    print(f"上游峰值流量: {max(upstream_q):.1f} m³/s")
    print("开始模拟...\n")
    
    results = solver.run_simulation(bc, num_steps)
    
    # 输出结果摘要
    peak_discharge_outlet = max(d[-1] for d in results['discharge'])
    peak_depth = max(max(d) for d in results['depth'])
    
    print(f"✓ 模拟完成 {num_steps} 个时间步")
    print(f"  出口峰值流量: {peak_discharge_outlet:.2f} m³/s")
    print(f"  最大水深: {peak_depth:.2f} m")
    print(f"  演算衰减: {(max(upstream_q) - peak_discharge_outlet)/max(upstream_q)*100:.1f}%")