"""GPU加速性能测试与参数敏感性分析"""

from hydrosis.hydrodynamics import (
    RiverReach,
    BoundaryCondition,
    GPUSaintVenantSolver,
    BatchSimulator,
    DeviceManager,
    GPU_AVAILABLE
)
import time

# 检查GPU状态
gpu_info = DeviceManager.get_gpu_info()
print(f"GPU可用: {gpu_info.available}")
if gpu_info.available:
    print(f"设备: {gpu_info.device_name}")
    print(f"内存: {gpu_info.memory_free:.1f} GB")

# 创建大规模网格测试
print("\n" + "="*70)
print("大规模网格性能测试")
print("="*70)

for num_sections in [20, 50, 100, 200]:
    reach = RiverReach(
        id=f"test_{num_sections}",
        length=20000,
        bed_slope=0.0008,
        manning_n=0.03,
        width=35,
        num_sections=num_sections
    )
    
    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=[60.0] * 100,
        downstream_type="stage",
        downstream_values=[3.0] * 100
    )
    
    print(f"\n网格规模: {num_sections} 个断面")
    
    # CPU测试
    solver_cpu = GPUSaintVenantSolver(reach, dt=60, use_gpu=False)
    solver_cpu.set_lateral_inflow([0.02] * num_sections)
    
    start = time.time()
    results_cpu = solver_cpu.run_simulation_gpu(bc, 100, verbose=False)
    cpu_time = time.time() - start
    print(f"  CPU: {cpu_time:.2f}秒 ({100/cpu_time:.1f} 步/秒)")
    
    # GPU测试
    if GPU_AVAILABLE:
        solver_gpu = GPUSaintVenantSolver(reach, dt=60, use_gpu=True)
        solver_gpu.set_lateral_inflow([0.02] * num_sections)
        
        start = time.time()
        results_gpu = solver_gpu.run_simulation_gpu(bc, 100, verbose=False)
        gpu_time = time.time() - start
        print(f"  GPU: {gpu_time:.2f}秒 ({100/gpu_time:.1f} 步/秒)")
        print(f"  加速比: {cpu_time/gpu_time:.2f}x")

# 参数敏感性分析
if GPU_AVAILABLE:
    print("\n" + "="*70)
    print("GPU批量参数敏感性分析")
    print("="*70)
    
    base_reach = RiverReach(
        id="sensitivity",
        length=8000,
        bed_slope=0.001,
        manning_n=0.03,
        width=25,
        num_sections=30
    )
    
    batch_sim = BatchSimulator(base_reach, use_gpu=True)
    
    bc = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=[80.0] * 80,
        downstream_type="stage",
        downstream_values=[2.5] * 80
    )
    
    # 分析曼宁系数影响
    sensitivity = batch_sim.sensitivity_analysis(
        'manning_n',
        [0.020, 0.025, 0.030, 0.035, 0.040, 0.045],
        bc,
        80
    )
    
    print(f"\n参数: {sensitivity['parameter_name']}")
    print("取值 | 峰值流量 | 峰值水深")
    print("-" * 40)
    for i, val in enumerate(sensitivity['parameter_values']):
        print(f"{val:.3f} | {sensitivity['peak_discharges'][i]:7.1f} | "
              f"{sensitivity['peak_depths'][i]:7.2f}")
    
    print(f"\n敏感度:")
    print(f"  流量: {sensitivity['sensitivity_discharge']:.1f}%")
    print(f"  水深: {sensitivity['sensitivity_depth']:.1f}%")