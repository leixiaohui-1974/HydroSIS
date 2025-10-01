# HydroSIS 水动力模块组织与集成指南

## 📁 完整目录结构

```
hydrosis/
├── hydrodynamics/                    # 🆕 水动力模块（新增）
│   ├── __init__.py                   # 模块导出
│   ├── core.py                       # 核心求解器（重构原 hydrodynamic_1d.py）
│   ├── geometry.py                   # 🌟 断面几何模块
│   ├── adaptive_timestep.py          # 🌟 自适应时间步长
│   ├── gpu_solver.py                 # 🌟 GPU加速求解器
│   └── routing_interface.py          # HydroSIS 集成接口
│
├── routing/                          # 已有的路由模块
│   ├── __init__.py
│   ├── base.py
│   ├── lag.py
│   ├── muskingum.py
│   ├── dynamic_wave.py
│   └── hydrodynamic_1d.py           # 简化包装（兼容旧代码）
│
├── runoff/                           # 产流模块（已有）
├── parameters/                       # 参数管理（已有）
├── evaluation/                       # 评估模块（已有）
├── reporting/                        # 报告生成（已有）
│
├── examples/                         # 示例代码
│   └── hydrodynamics/               # 🆕 水动力示例
│       ├── basic_trapezoid.py       # 梯形断面示例
│       ├── compound_floodplain.py   # 复合断面示例
│       ├── adaptive_flood.py        # 自适应步长洪水模拟
│       ├── gpu_benchmark.py         # GPU性能测试
│       └── irregular_section.py     # 不规则断面示例
│
├── tests/                            # 单元测试
│   └── hydrodynamics/               # 🆕 水动力测试
│       ├── test_geometry.py
│       ├── test_adaptive.py
│       └── test_gpu_solver.py
│
└── docs/                             # 文档
    └── hydrodynamics/               # 🆕 水动力文档
        ├── user_guide.md
        ├── api_reference.md
        └── theory.md
```

---

## 📝 文件说明与建议命名

### 核心模块文件

| 文件路径 | 建议名称 | 功能 | 依赖 |
|---------|---------|------|------|
| `hydrosis/hydrodynamics/geometry.py` | ✅ 当前名称合适 | 断面几何计算 | numpy |
| `hydrosis/hydrodynamics/adaptive_timestep.py` | ✅ 当前名称合适 | 自适应时间步长控制 | numpy |
| `hydrosis/hydrodynamics/gpu_solver.py` | ✅ 当前名称合适 | GPU加速求解器 | numpy, cupy(可选) |
| `hydrosis/hydrodynamics/core.py` | 建议名称 | 核心圣维南求解器 | geometry, scipy |
| `hydrosis/hydrodynamics/routing_interface.py` | 建议名称 | HydroSIS路由接口 | core |

### `__init__.py` 内容

**`hydrosis/hydrodynamics/__init__.py`**:
```python
"""HydroSIS 一维水动力模块

提供完整的圣维南方程求解能力，支持：
- 多种断面形式（矩形、梯形、复合、不规则）
- 自适应时间步长控制
- GPU加速计算
- 与HydroSIS无缝集成
"""

from .geometry import (
    CrossSection,
    RectangleSection,
    TrapezoidSection,
    CompoundSection,
    IrregularSection,
    create_cross_section,
    compute_normal_depth,
    compute_critical_depth
)

from .adaptive_timestep import (
    AdaptiveTimeStepController,
    AdaptiveStrategy,
    TimeStepMetrics,
    VariableTimeStepSimulator
)

from .gpu_solver import (
    GPUSaintVenantSolver,
    BatchSimulator,
    DeviceManager,
    GPU_AVAILABLE
)

# 保持向后兼容
from .core import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    HydraulicState
)

__all__ = [
    # 断面几何
    'CrossSection',
    'RectangleSection',
    'TrapezoidSection',
    'CompoundSection',
    'IrregularSection',
    'create_cross_section',
    'compute_normal_depth',
    'compute_critical_depth',
    
    # 自适应控制
    'AdaptiveTimeStepController',
    'AdaptiveStrategy',
    'TimeStepMetrics',
    'VariableTimeStepSimulator',
    
    # GPU加速
    'GPUSaintVenantSolver',
    'BatchSimulator',
    'DeviceManager',
    'GPU_AVAILABLE',
    
    # 核心求解器
    'SaintVenantSolver',
    'RiverReach',
    'BoundaryCondition',
    'HydraulicState'
]

__version__ = '1.1.0'
```

---

## 🔄 代码迁移步骤

### 步骤1: 重构现有代码

**将原 `hydrodynamic_1d.py` 拆分为 `core.py`**:

```python
# hydrosis/hydrodynamics/core.py
"""核心圣维南方程求解器（从 hydrodynamic_1d.py 重构）"""

from .geometry import CrossSection, RectangleSection
import numpy as np
# ... 原有代码，增强支持自定义断面

class SaintVenantSolver:
    def __init__(self, reach, dt=60.0, 
                 cross_section: CrossSection = None,  # 🆕 支持自定义断面
                 theta=0.6, epsilon=1e-4):
        self.reach = reach
        self.dt = dt
        
        # 🆕 断面对象
        if cross_section is None:
            self.cross_section = RectangleSection(reach.width)
        else:
            self.cross_section = cross_section
        
        # ... 其余代码
```

### 步骤2: 创建集成接口

**`hydrosis/hydrodynamics/routing_interface.py`**:
```python
"""HydroSIS 路由模块集成接口"""

from typing import List, Mapping
from .core import SaintVenantSolver, RiverReach, BoundaryCondition
from .geometry import create_cross_section
from .adaptive_timestep import AdaptiveTimeStepController, AdaptiveStrategy

class HydrodynamicRoutingModel:
    """水动力路由模型 - HydroSIS RoutingModel 接口实现"""
    
    def __init__(self, parameters: Mapping[str, float]):
        self.parameters = dict(parameters)
        
        # 解析断面配置
        section_type = parameters.get('section_type', 'rectangle')
        section_params = self._extract_section_params(section_type)
        cross_section = create_cross_section(section_type, **section_params)
        
        # 创建河段
        self.reach = RiverReach(
            id=str(parameters.get('reach_id', 'main')),
            length=float(parameters.get('length', 10000)),
            bed_slope=float(parameters.get('bed_slope', 0.001)),
            manning_n=float(parameters.get('manning_n', 0.03)),
            width=float(parameters.get('width', 30)),
            num_sections=int(parameters.get('num_sections', 20))
        )
        
        # 是否使用自适应步长
        use_adaptive = parameters.get('adaptive_timestep', False)
        
        if use_adaptive:
            self.adaptive_controller = AdaptiveTimeStepController(
                initial_dt=float(parameters.get('time_step', 300)),
                strategy=AdaptiveStrategy.HYBRID
            )
            self.solver = None  # 在运行时创建
        else:
            self.solver = SaintVenantSolver(
                self.reach,
                dt=float(parameters.get('time_step', 300)),
                cross_section=cross_section
            )
            self.adaptive_controller = None
    
    def route(self, subbasin, inflow: List[float]) -> List[float]:
        """HydroSIS RoutingModel 接口"""
        # ... 实现路由逻辑
        pass
    
    def _extract_section_params(self, section_type: str) -> dict:
        """从parameters提取断面参数"""
        if section_type == 'trapezoid':
            return {
                'bottom_width': self.parameters.get('bottom_width', 10),
                'side_slope': self.parameters.get('side_slope', 2.0)
            }
        elif section_type == 'compound':
            return {
                'main_bottom_width': self.parameters.get('main_bottom_width', 8),
                'main_side_slope': self.parameters.get('main_side_slope', 1.5),
                'floodplain_height': self.parameters.get('floodplain_height', 2.5),
                'left_floodplain_width': self.parameters.get('left_fp_width', 10),
                'right_floodplain_width': self.parameters.get('right_fp_width', 10)
            }
        else:  # rectangle
            return {'width': self.parameters.get('width', 30)}

# 注册到 HydroSIS
from hydrosis.routing.base import RoutingModelConfig
RoutingModelConfig.register("hydrodynamic_1d", HydrodynamicRoutingModel)
```

### 步骤3: 保持向后兼容

**保留 `routing/hydrodynamic_1d.py` 作为简化包装**:
```python
"""简化包装 - 保持向后兼容

建议新代码使用 hydrosis.hydrodynamics 模块
"""

import warnings
from hydrosis.hydrodynamics import (
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition,
    HydrodynamicRoutingModel
)

warnings.warn(
    "routing.hydrodynamic_1d 已废弃，请使用 hydrosis.hydrodynamics",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    'SaintVenantSolver',
    'RiverReach',
    'BoundaryCondition',
    'HydrodynamicRoutingModel'
]
```

---

## 🚀 使用示例

### 示例1: 使用梯形断面

**`examples/hydrodynamics/basic_trapezoid.py`**:
```python
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
```

### 示例2: 复合断面漫滩模拟

**`examples/hydrodynamics/compound_floodplain.py`**:
```python
"""复合断面漫滩过程模拟"""

from hydrosis.hydrodynamics import (
    CompoundSection,
    SaintVenantSolver,
    RiverReach,
    BoundaryCondition
)

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

# 运行洪水模拟
reach = RiverReach(
    id="compound_reach",
    length=5000,
    bed_slope=0.0005,
    manning_n=0.035,
    width=8,
    num_sections=20
)

solver = SaintVenantSolver(reach, dt=120, cross_section=section)

# 大洪水过程（会发生漫滩）
num_steps = 100
import numpy as np
upstream_q = 50 + 150 * np.sin(np.linspace(0, np.pi, num_steps))

bc = BoundaryCondition(
    upstream_type="discharge",
    upstream_values=upstream_q.tolist(),
    downstream_type="stage",
    downstream_values=[3.5] * num_steps  # 下游高水位
)

results = solver.run_simulation(bc, num_steps)

# 分析漫滩时刻
print("\n漫滩分析:")
floodplain_threshold = section.floodplain_height
for t, depth_array in enumerate(results['depth']):
    max_depth = max(depth_array)
    if max_depth > floodplain_threshold:
        print(f"⚠️ 时刻 {t*120}s: 开始漫滩 (最大水深 {max_depth:.2f}m)")
        break
```

### 示例3: 自适应时间步长

**`examples/hydrodynamics/adaptive_flood.py`**:
```python
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
```

### 示例4: GPU加速批量模拟

**`examples/hydrodynamics/gpu_benchmark.py`**:
```python
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
```

---

## 🔌 在HydroSIS配置中使用

### YAML配置示例

**`model_config_advanced.yaml`**:
```yaml
routing_models:
  # 矩形断面 + 固定步长
  - id: simple_routing
    model_type: hydrodynamic_1d
    parameters:
      section_type: rectangle
      width: 30
      length: 10000
      bed_slope: 0.001
      manning_n: 0.03
      num_sections: 25
      time_step: 300
  
  # 梯形断面 + 自适应步长
  - id: adaptive_routing
    model_type: hydrodynamic_1d
    parameters:
      section_type: trapezoid
      bottom_width: 12
      side_slope: 2.0
      length: 15000
      bed_slope: 0.0008
      manning_n: 0.035
      num_sections: 30
      time_step: 300
      adaptive_timestep: true  # 启用自适应
      min_dt: 60
      max_dt: 600
  
  # 复合断面（考虑漫滩）
  - id: floodplain_routing
    model_type: hydrodynamic_1d
    parameters:
      section_type: compound
      main_bottom_width: 8
      main_side_slope: 1.5
      floodplain_height: 2.5
      left_fp_width: 15
      right_fp_width: 15
      length: 8000
      bed_slope: 0.0005
      manning_n: 0.04
      num_sections: 20
      time_step: 180
  
  # GPU加速（大规模）
  - id: gpu_routing
    model_type: hydrodynamic_1d
    parameters:
      section_type: trapezoid
      bottom_width: 10
      side_slope: 2.5
      length: 25000
      bed_slope: 0.0012
      manning_n: 0.03
      num_sections: 100  # 大网格
      time_step: 120
      use_gpu: true      # 启用GPU

parameter_zones:
  - id: Z_headwater
    control_points: ["S1", "S2"]
    parameters:
      runoff_model: "curve"
      routing_model: "simple_routing"
  
  - id: Z_middle
    control_points: ["S3"]
    parameters:
      runoff_model: "reservoir"
      routing_model: "adaptive_routing"  # 中游使用自适应
  
  - id: Z_lowland
    control_points: ["S4"]
    parameters:
      runoff_model: "xin_an_jiang"
      routing_model: "floodplain_routing"  # 下游考虑漫滩
```

### Python API 使用

```python
from hydrosis import ModelConfig, HydroSISModel

# 加载配置
config = ModelConfig.from_yaml("model_config_advanced.yaml")

# 创建模型（自动使用高级水动力路由）
model = HydroSISModel.from_config(config)

# 运行模拟
forcing = {
    "S1": [10, 25, 45, 60, 50, 35, 20, 10],
    "S2": [8, 20, 40, 55, 48, 32, 18, 9],
    "S3": [5, 15, 30, 45, 40, 28, 15, 7],
    "S4": [0, 5, 15, 25, 22, 15, 8, 3]
}

local_flows = model.run(forcing)
aggregated = model.accumulate_discharge(local_flows)

# 查看使用了高级水动力路由的子流域结果
print("子流域S3 (自适应步长):")
print(f"  出口流量: {aggregated['S3']}")
```

---

## ⚠️ 重要注意事项

### 1. 依赖管理

**基础依赖** (`requirements.txt`):
```
numpy>=1.20.0
scipy>=1.7.0
```

**可选依赖** (`requirements-optional.txt`):
```
cupy-cuda11x>=10.0.0  # GPU加速（根据CUDA版本选择）
matplotlib>=3.3.0      # 可视化
```

安装命令:
```bash
# 基础安装
pip install -e .

# 完整安装（包含GPU）
pip install -e .[gpu]
pip install cupy-cuda11x  # CUDA 11.x
# 或
pip install cupy-cuda12x  # CUDA 12.x

# 开发环境
pip install -e .[dev]
```

**`setup.py` 配置**:
```python
setup(
    name="hydrosis",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "gpu": ["cupy-cuda11x>=10.0.0"],
        "viz": ["matplotlib>=3.3.0"],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0",
            "flake8>=3.9.0"
        ]
    }
)
```

### 2. 性能建议

| 场景 | 推荐配置 | 原因 |
|------|---------|------|
| 小流域(<5km²) | 固定步长 + CPU | GPU开销大于收益 |
| 中等流域(5-50km²) | 自适应步长 + CPU | 平衡效率与精度 |
| 大流域(>50km²) | 自适应步长 + GPU | 充分利用并行计算 |
| 参数率定/不确定性 | GPU批量模拟 | 并行运行多组参数 |

### 3. 向后兼容

- 旧代码使用 `routing.hydrodynamic_1d` 仍可运行
- 会显示弃用警告，建议迁移到新API
- 新功能（断面、自适应、GPU）仅在新模块中可用

---

## 📚 文档补充

**建议增加的文档文件**:

1. **`docs/hydrodynamics/user_guide.md`** - 用户指南
   - 快速入门
   - 断面类型选择
   - 参数配置指南
   - 常见问题

2. **`docs/hydrodynamics/api_reference.md`** - API参考
   - 类和函数文档
   - 参数说明
   - 返回值格式

3. **`docs/hydrodynamics/theory.md`** - 理论基础
   - 圣维南方程推导
   - 数值格式说明
   - 稳定性条件

4. **`docs/hydrodynamics/validation.md`** - 验证案例
   - 解析解对比
   - 实测数据验证
   - 基准测试结果

---

## ✅ 集成检查清单

完成以下步骤确保正确集成:

- [ ] 创建 `hydrosis/hydrodynamics/` 目录
- [ ] 放置 `geometry.py` 到该目录
- [ ] 放置 `adaptive_timestep.py` 到该目录
- [ ] 放置 `gpu_solver.py` 到该目录
- [ ] 重构原代码为 `core.py`
- [ ] 创建 `routing_interface.py`
- [ ] 编写 `__init__.py` 导出接口
- [ ] 创建示例代码到 `examples/hydrodynamics/`
- [ ] 编写单元测试到 `tests/hydrodynamics/`
- [ ] 更新 `setup.py` 添加可选依赖
- [ ] 更新主 `README.md` 添加新功能说明
- [ ] 运行测试确保无冲突: `pytest tests/`
- [ ] 构建文档: `sphinx-build docs build`
- [ ] 标记版本为 v1.1.0

---

## 🎯 总结

通过这样的组织方式:

✅ **模块化清晰**: 每个功能独立文件,易于维护  
✅ **向后兼容**: 旧代码可继续运行  
✅ **渐进增强**: 用户可选择性启用高级功能  
✅ **性能可扩展**: CPU/GPU自动适配  
✅ **易于测试**: 单元测试与示例并行  

这就是完整的模块组织方案！