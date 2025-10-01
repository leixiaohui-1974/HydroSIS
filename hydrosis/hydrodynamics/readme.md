# 一维水动力模型使用指南

## 📋 目录

1. [模型概述](#模型概述)
2. [快速开始](#快速开始)
3. [与 HydroSIS 集成](#与-hydrosis-集成)
4. [参数配置详解](#参数配置详解)
5. [耦合模式选择](#耦合模式选择)
6. [案例研究](#案例研究)

---

## 模型概述

### 核心特性

- **数值方法**: Preissmann 四点隐式差分格式
- **控制方程**: 完整圣维南方程组 (连续性 + 动量方程)
- **适用场景**: 
  - 中小河流洪水演算
  - 城市内河水位模拟
  - 水文-水力耦合研究

### 与 HydroSIS 的关系

```
┌─────────────────────────────────────────────────┐
│           HydroSIS 分布式水文模型                │
├─────────────────────────────────────────────────┤
│  产流模块 → 汇流模块 → [一维水动力路由]         │
│  (Runoff)   (Routing)   (可选替换/增强)         │
└─────────────────────────────────────────────────┘
```

**三种使用模式**:
1. **独立运行**: 接收外部流量数据,独立进行河道演算
2. **集成路由**: 作为 `RoutingModel` 替换 Lag/Muskingum
3. **双向耦合**: 河道水位反馈影响产流过程

---

## 快速开始

### 安装依赖

```bash
pip install numpy scipy

# 可选: 用于可视化
pip install matplotlib
```

### 最简示例

```python
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
```

---

## 与 HydroSIS 集成

### 方法1: 配置文件集成

在 `model_config.yaml` 中添加:

```yaml
routing_models:
  - id: hydrodynamic_main
    model_type: saint_venant_1d
    parameters:
      reach_id: main_channel
      length: 15000          # 河段长度 (m)
      bed_slope: 0.0005      # 河床坡度
      manning_n: 0.035       # 曼宁系数
      width: 40              # 河宽 (m)
      num_sections: 30       # 计算断面数
      time_step: 300         # 时间步长 (s)

parameter_zones:
  - id: Z1
    control_points: ["S1"]
    parameters:
      runoff_model: "curve"
      routing_model: "hydrodynamic_main"  # ← 使用水动力路由
```

### 方法2: Python API 集成

```python
from hydrosis import ModelConfig, HydroSISModel
from hydrodynamic_1d import HydrodynamicRoutingModel

# 加载配置
config = ModelConfig.from_yaml("model_config.yaml")

# 创建模型实例 (自动使用水动力路由)
model = HydroSISModel.from_config(config)

# 运行模拟
forcing = {
    "S1": [10, 20, 35, 40, 30, 20, 10],
    "S2": [5, 15, 25, 30, 25, 15, 8]
}

local_flows = model.run(forcing)
aggregated = model.accumulate_discharge(local_flows)
```

---

## 参数配置详解

### 河段几何参数

| 参数 | 说明 | 典型范围 | 默认值 |
|------|------|----------|--------|
| `length` | 河段长度 (m) | 1000 - 50000 | 10000 |
| `bed_slope` | 河床坡度 (无量纲) | 0.0001 - 0.01 | 0.001 |
| `width` | 河宽 (m) | 10 - 200 | 30 |
| `num_sections` | 计算断面数 | 10 - 100 | 20 |

**推荐配置**:
- **山区河流**: `bed_slope=0.005`, `manning_n=0.04`
- **平原河流**: `bed_slope=0.0002`, `manning_n=0.025`
- **城市河道**: `bed_slope=0.001`, `manning_n=0.035`

### 曼宁糙率系数选择

| 河道类型 | manning_n | 备注 |
|----------|-----------|------|
| 光滑混凝土渠道 | 0.012 - 0.015 | 城市排水渠 |
| 天然平直河流 | 0.025 - 0.035 | 少植被 |
| 弯曲杂草河道 | 0.035 - 0.050 | 常见中小河流 |
| 山区急流 | 0.040 - 0.060 | 卵石河床 |

### 数值计算参数

| 参数 | 说明 | 推荐值 | 影响 |
|------|------|--------|------|
| `time_step` | 时间步长 (s) | 60 - 600 | 步长越小精度越高,但计算慢 |
| `theta` | 时间权重因子 | 0.6 - 0.8 | 0.5=中心差分, 1.0=完全隐式 |
| `epsilon` | 收敛容差 | 1e-4 | 控制牛顿迭代精度 |

**稳定性准则** (Courant 数):
```
CFL = v * dt / dx < 1.0  (推荐 < 0.5)

其中:
- v: 洪峰流速 (通常 1-3 m/s)
- dt: 时间步长
- dx: 空间步长 = length / (num_sections - 1)
```

---

## 耦合模式选择

### 模式对比

| 耦合方式 | 计算成本 | 精度 | 适用场景 |
|----------|----------|------|----------|
| **松散耦合** | 低 | 中 | 快速评估、单向影响 |
| **集成耦合** | 中 | 高 | 标准水文模拟、替换传统路由 |
| **双向耦合** | 高 | 最高 | 平原河网、回水顶托显著区域 |

### 模式1: 松散耦合 (推荐新手)

**特点**: 水文和水动力独立运行,文件交换数据

**优点**:
- 模块完全解耦,易于调试
- 可使用不同时间步长
- 各模型可独立更新

**缺点**:
- 无法考虑反馈效应
- 需要手动管理数据传递

**适用**:
- 上游产流对下游水位不敏感
- 初步可行性研究

### 模式2: 集成耦合 (推荐常规使用)

**特点**: 水动力作为 HydroSIS 的 `RoutingModel`

**优点**:
- 无缝集成,一次运行获得全流域结果
- 利用 HydroSIS 的参数分区管理
- 支持情景对比评估

**缺点**:
- 计算时间较长
- 所有子流域使用相同时间步长

**适用**:
- 标准流域模拟
- 需要自动化批量计算

### 模式3: 双向耦合 (研究用)

**特点**: 河道水位影响产流系数

**物理机制**:
```
河道水位上升 → 地下水位抬升 → 土壤饱和区扩大 → 产流系数增加
```

**适用场景**:
- 平原河网密集区
- 下游顶托作用明显
- 洪涝风险评估

**实现要点**:
```python
# 在产流计算中加入反馈
adjusted_coeff = base_coeff * (1 + 0.1 * (river_stage - base_stage))
runoff = rainfall * adjusted_coeff * area
```

---

## 案例研究

### 案例1: 城市内河排涝能力评估

**背景**: 某城区排水河道,长5km,宽20m,评估50年一遇暴雨排水能力

**配置**:
```python
reach = RiverReach(
    id="urban_drainage",
    length=5000,
    bed_slope=0.0008,
    manning_n=0.025,  # 混凝土护岸
    width=20,
    num_sections=25
)

# 50年一遇设计暴雨产流 (芝加哥雨型)
peak_runoff = 45  # m³/s
inflow_hydrograph = generate_chicago_hyetograph(
    peak=peak_runoff, duration=180  # 3小时
)

# 下游闸门控制水位
downstream_stage = 3.5  # m (汛限水位)
```

**结果判读**:
- 若最大水深 < 设计堤高 - 0.5m → 安全
- 若出口流速 > 2.0 m/s → 需考虑护底
- 削峰率 = (入流峰值 - 出口峰值) / 入流峰值 × 100%

### 案例2: 水库下游河段冲刷风险

**背景**: 水库突然泄洪,评估下游10km河段冲刷风险

**关键指标**:
- 床面剪切应力: τ = ρ g R S_f
- 临界起动流速: v_c = 0.6 √(g d_50)  (d_50为中值粒径)

**配置要点**:
```python
# 上游边界: 泄洪流量过程
upstream_q = [10, 10, 150, 200, 180, 120, 80, 50, 30, 20]

# 计算沿程剪切应力
for section_depth, section_velocity in results:
    R = section_depth  # 简化: 宽浅河道
    Sf = (manning_n * section_velocity)**2 / R**(4/3)
    tau = 1000 * 9.81 * R * Sf
    
    if tau > tau_critical:
        print(f"⚠️ 断面 {i} 存在冲刷风险")
```

### 案例3: 平原河网双向耦合

**背景**: 太湖流域典型圩区,河网密布,排涝受下游水位顶托

**物理过程**:
1. 暴雨产流汇入河网
2. 下游水位上涨,排水不畅
3. 河网水位倒灌,农田积水
4. 饱和区扩大,产流系数从0.3升至0.6

**实现**:
```python
coupler = BidirectionalCoupler(reach, catchment_area=50)

for t, rainfall in enumerate(rainfall_series):
    # 根据当前水位调整产流
    result = coupler.coupled_timestep(
        rainfall, 
        base_coeff=0.3,
        bc=boundary_conditions,
        time_idx=t
    )
    
    # 监测反馈强度
    feedback_ratio = result['adjusted_coeff'] / 0.3
    if feedback_ratio > 1.5:
        print(f"⚠️ 时段{t}: 河网顶托导致产流系数增加{feedback_ratio:.1f}倍")
```

---

## 常见问题

### Q1: 模拟不收敛怎么办?

**可能原因**:
1. 时间步长过大 → 减小 `dt` 到 CFL < 0.5
2. 边界条件突变 → 使用过渡过程
3. 河床坡度过陡 → 检查 `bed_slope` 是否合理

**调试步骤**:
```python
solver = SaintVenantSolver(reach, dt=30)  # 减小步长
solver.theta = 0.8  # 增加隐式权重
solver.epsilon = 1e-3  # 放宽收敛标准
```

### Q2: 如何处理复杂断面?

当前版本支持矩形断面。对于复杂断面:

**方法1**: 等效矩形
```python
# 梯形断面 → 等效宽度
b = 底宽
m = 边坡系数
h = 水深
B_equiv = b + m * h  # 水面宽度作为等效宽度
```

**方法2**: 分段模拟
- 将河段按断面形式分段
- 每段使用单独的 `RiverReach` 实例

### Q3: 内存不足?

对于长河段:
```python
# 方案1: 减少断面数
reach.num_sections = 15  # 原30 → 15

# 方案2: 分段计算
reach_1 = RiverReach(length=5000, ...)  # 上游段
reach_2 = RiverReach(length=5000, ...)  # 下游段
# reach_1的出口作为reach_2的入口
```

---

## 技术支持

- **文档**: [完整API文档](./api_reference.md)
- **示例库**: `examples/` 目录
- **问题反馈**: GitHub Issues

**性能优化建议**:
- 空间步长与时间步长满足 CFL 条件
- 使用稀疏矩阵求解器 (已默认启用)
- 批量计算时使用多进程池

---

## 开发路线图

**v1.0 (当前)**
- ✅ 圣维南方程隐式求解
- ✅ 矩形断面
- ✅ HydroSIS 集成

**v1.1 (计划中)**
- 🔄 梯形/复合断面支持
- 🔄 自适应时间步长
- 🔄 GPU 加速

**v2.0 (研究方向)**
- 📋 二维浅水方程扩展
- 📋 泥沙输移模块
- 📋 水质模拟耦合

---

## 引用

如在学术研究中使用,请引用:

```
@software{hydrosis_hydrodynamic_1d,
  title={一维水动力模型 for HydroSIS},
  author={[Your Name]},
  year={2025},
  version={1.0}
}
```

让我继续完成核心功能总结和使用建议:

## 🎯 核心功能总结

### 1. **独立水动力模拟**
```python
# 可脱离 HydroSIS 单独使用
solver = SaintVenantSolver(reach, dt=60)
results = solver.run_simulation(bc, num_steps=100)
```

**应用场景**:
- 河道洪水演进模拟
- 水库泄洪影响分析
- 防洪标准复核

### 2. **作为 HydroSIS 路由模块**
```yaml
# 配置文件集成
routing_models:
  - id: hd_routing
    model_type: saint_venant_1d
    parameters:
      length: 10000
      manning_n: 0.03
```

**优势**:
- 替换简化的 Lag/Muskingum 路由
- 考虑水面线变化和回水效应
- 更精确的洪峰传播模拟

### 3. **双向耦合模拟**
```python
# 河道水位反馈影响产流
coupler = BidirectionalCoupler(reach, catchment_area)
result = coupler.coupled_timestep(rainfall, base_coeff, bc, t)
```

**物理机制**:
- 下游顶托 → 河网水位抬升
- 地下水位上涨 → 土壤饱和面积增加
- 产流系数动态调整 → 更真实的洪水过程

## 🚀 快速上手指南

### 步骤1: 准备河段数据

你需要收集以下信息:

| 数据项 | 获取方法 | 示例 |
|--------|----------|------|
| 河段长度 | 实测/地图量算 | 8000 m |
| 河床坡度 | 高程差/长度 | 0.0008 |
| 河宽 | 实测断面 | 35 m |
| 曼宁系数 | 查表/率定 | 0.035 |

**糙率快速估算**:
- 观察河床: 混凝土(0.015) < 泥沙(0.025) < 卵石(0.04)
- 植被覆盖: 每增加10%植被 → n 增加 0.005

### 步骤2: 选择耦合模式

**决策树**:
```
是否需要考虑河道水位反馈?
├─ 否 → 河道坡度 > 0.001?
│      ├─ 是 → 【集成耦合】(推荐)
│      └─ 否 → 需要精细模拟?
│             ├─ 是 → 【集成耦合】
│             └─ 否 → 【松散耦合】(更快)
└─ 是 → 【双向耦合】(平原河网)
```

### 步骤3: 运行模拟

**最简单的完整示例**:
```python
from hydrodynamic_1d import RiverReach, SaintVenantSolver, BoundaryCondition

# 定义河段
reach = RiverReach(
    id="my_river",
    length=5000,      # 5公里
    bed_slope=0.001,  # 1‰坡度
    manning_n=0.03,
    width=30
)

# 创建求解器
solver = SaintVenantSolver(reach, dt=60)

# 设置边界 (模拟洪水过程)
peak_discharge = 80  # m³/s
upstream_flow = [20, 40, 60, peak_discharge, 60, 40, 25, 15, 10]

bc = BoundaryCondition(
    upstream_type="discharge",
    upstream_values=upstream_flow,
    downstream_type="stage",
    downstream_values=[2.5] * len(upstream_flow)
)

# 运行
results = solver.run_simulation(bc, len(upstream_flow))

# 查看结果
print(f"入口峰值: {max(upstream_flow)} m³/s")
print(f"出口峰值: {max([d[-1] for d in results['discharge']])} m³/s")
print(f"最大水深: {max([max(d) for d in results['depth']])} m")
```

## 📊 结果分析技巧

### 关键指标提取

```python
# 提取出口断面时间序列
outlet_q = [discharge[-1] for discharge in results['discharge']]
outlet_h = [depth[-1] for depth in results['depth']]

# 计算特征值
peak_q = max(outlet_q)
peak_time = outlet_q.index(peak_q)
peak_depth = max(outlet_h)

# 洪峰传播时间
inflow_peak_time = upstream_flow.index(max(upstream_flow))
travel_time = (peak_time - inflow_peak_time) * dt / 60  # 转换为分钟

# 削峰率
attenuation = (max(upstream_flow) - peak_q) / max(upstream_flow) * 100

print(f"""
洪水演进特征:
- 出口峰值流量: {peak_q:.1f} m³/s
- 洪峰传播时间: {travel_time:.0f} 分钟
- 削峰率: {attenuation:.1f}%
- 最大水深: {peak_depth:.2f} m
""")
```

### 沿程水面线绘制

```python
import matplotlib.pyplot as plt

# 选择峰值时刻
peak_idx = outlet_q.index(peak_q)

# 提取沿程水深
x_coords = reach.x_coords
water_surface = results['depth'][peak_idx]
bed_elevation = [reach.bed_slope * x for x in x_coords]

# 绘图
plt.figure(figsize=(10, 5))
plt.fill_between(x_coords, 0, bed_elevation, color='brown', alpha=0.3, label='河床')
plt.fill_between(x_coords, bed_elevation, 
                 [b + h for b, h in zip(bed_elevation, water_surface)],
                 color='blue', alpha=0.5, label='水体')
plt.xlabel('距离 (m)')
plt.ylabel('高程 (m)')
plt.title(f'洪峰时刻水面线 (t={peak_idx * dt/60:.0f} min)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('water_surface_profile.png', dpi=150)
```

## ⚠️ 注意事项

### 1. **数值稳定性**

确保满足 CFL 条件:
```python
# 检查 CFL 数
v_max = 2.5  # 预估最大流速 (m/s)
dx = reach.length / (reach.num_sections - 1)
CFL = v_max * dt / dx

if CFL > 0.8:
    print(f"⚠️ CFL = {CFL:.2f} 过大,建议减小时间步长")
    recommended_dt = int(0.5 * dx / v_max)
    print(f"推荐 dt < {recommended_dt} 秒")
```

### 2. **边界条件设置**

**上游边界**:
- 流量边界: 适用于有实测流量站
- 水位边界: 适用于水库/湖泊出流

**下游边界**:
- 水位边界: 河口、感潮河段
- 流量边界: 自由出流、汇入大江大河
- 水位流量关系: 使用实测数据率定

```python
# 示例: 根据实测数据拟合水位-流量关系
import numpy as np

# 实测数据
observed_h = [1.5, 2.0, 2.5, 3.0, 3.5]  # m
observed_q = [10, 25, 45, 70, 100]      # m³/s

# 拟合 Q = a * H^b
log_h = np.log(observed_h)
log_q = np.log(observed_q)
b, log_a = np.polyfit(log_h, log_q, 1)
a = np.exp(log_a)

print(f"水位-流量关系: Q = {a:.2f} * H^{b:.2f}")

# 使用率定关系
bc = BoundaryCondition(
    downstream_type="rating_curve",
    downstream_rating=(a, b)
)
```

### 3. **侧向入流处理**

```python
# 情况1: 均匀分布 (适用于长河段多个支流汇入)
total_lateral = 15  # m³/s
lateral_per_m = total_lateral / reach.length
solver.set_lateral_inflow([lateral_per_m] * reach.num_sections)

# 情况2: 集中入流 (某个断面有大支流汇入)
lateral_inflow = [0] * reach.num_sections
tributary_section = 10  # 第10个断面
lateral_inflow[tributary_section] = 20 / reach.dx  # 集中20 m³/s
solver.set_lateral_inflow(lateral_inflow)

# 情况3: 来自水文模型产流 (HydroSIS耦合)
subbasin_runoff = 30  # m³/s (来自上游子流域)
lateral_per_section = subbasin_runoff / reach.num_sections
solver.set_lateral_inflow([lateral_per_section] * reach.num_sections)
```

## 🔧 高级功能

### 多河段串联模拟

```python
# 上游陡坡段
reach_upper = RiverReach(
    id="upper", length=3000, bed_slope=0.003,
    manning_n=0.04, width=20, num_sections=15
)
solver_upper = SaintVenantSolver(reach_upper, dt=60)

# 下游缓坡段
reach_lower = RiverReach(
    id="lower", length=7000, bed_slope=0.0005,
    manning_n=0.03, width=35, num_sections=25
)
solver_lower = SaintVenantSolver(reach_lower, dt=60)

# 串联计算
for t in range(num_steps):
    # 上游段计算
    bc_upper = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=inflow_series,
        downstream_type="stage",
        downstream_values=[transition_stage] * len(inflow_series)
    )
    solver_upper.solve_timestep(bc_upper, t)
    
    # 上游段出口作为下游段入口
    upper_outlet_q = solver_upper.state.discharge[-1]
    
    bc_lower = BoundaryCondition(
        upstream_type="discharge",
        upstream_values=[upper_outlet_q],
        downstream_type="stage",
        downstream_values=downstream_stage_series
    )
    solver_lower.solve_timestep(bc_lower, t)
```

### 实时预报应用

```python
# 滚动预报框架
current_time = 0
forecast_horizon = 12  # 预报12个时段

while current_time < total_time:
    # 获取实时观测数据
    observed_q = get_realtime_discharge(current_time)
    
    # 重新初始化模型状态
    solver.state.discharge[:] = observed_q
    solver.update_hydraulic_properties(solver.state)
    
    # 向前预报
    forecast_results = []
    for t in range(forecast_horizon):
        solver.solve_timestep(bc, current_time + t)
        forecast_results.append(solver.state.discharge[-1])
    
    # 发布预报
    publish_forecast(forecast_results, current_time)
    
    # 推进时间窗口
    current_time += 1
```

## 📚 相关资源

- **理论基础**: Chow, V.T. (1959). Open-Channel Hydraulics
- **数值方法**: Abbott, M.B. & Basco, D.R. (1989). Computational Fluid Dynamics
- **HydroSIS 文档**: 见仓库 `docs/` 目录

**推荐学习路径**:
1. 理解圣维南方程物理意义
2. 运行简单算例验证结果
3. 用实测数据率定参数
4. 尝试不同耦合模式
5. 根据需求扩展功能

---

希望这个实现能满足你的需求！主要特点包括:

✅ **完整的物理模型** - 基于圣维南方程组  
✅ **稳定的数值格式** - Preissmann 隐式格式  
✅ **灵活的耦合方式** - 三种模式适应不同场景  
✅ **与 HydroSIS 无缝集成** - 作为路由模块直接使用  
✅ **详细的文档和示例** - 快速上手

如果你需要:
- **二维水动力扩展** (考虑横向流动)
- **泥沙输移模块** (河床冲淤)
- **变断面几何** (梯形/天然断面)
- **性能优化** (GPU加速/并行计算)

请告诉我,我可以继续开发相应功能！