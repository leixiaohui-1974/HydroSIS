# 估计径流序列生成器设计

## 1. 目的

为参数敏感性分析和自动率定提供基准径流序列（"伪观测数据"）。

## 2. 算法原理

### 2.1 径流系数法（基础）

```
Q(t) = Rc × P(t) × A / 3.6
```

其中：
- Q(t): 时刻t的流量 (m³/s)
- Rc: 径流系数 (0.345为默认值，适用于中等植被覆盖)
- P(t): 时刻t的降雨强度 (mm/hr)
- A: 流域面积 (km²)
- 3.6: 单位转换系数 (mm·km²/hr → m³/s)

### 2.2 洪水演进修正

实际洪水过程受以下因素影响：

#### A. 时滞（Lag Time）
从降雨中心到汇水点的传播时间，导致洪峰延迟。

**实现方法**：时间序列平移
```python
Q_delayed(t) = Q(t - lag)
```

**参数**：
- `lag_hours`: 时滞小时数
  - 上游小流域：1-2小时
  - 中游：2-4小时
  - 下游主河道：4-8小时

#### B. 削峰（Attenuation）
洪水演进过程中的削峰填谷效应，由河道蓄水、漫滩等引起。

**实现方法1：移动平均**（简单坦化）
```python
Q_smooth(t) = α × Q(t) + (1-α) × Q(t-1)
```

**实现方法2：三角单位线**（更真实）
```python
# 三角形状的单位线卷积
Q_routed(t) = Σ [Q(τ) × UH(t-τ)]
```

**参数**：
- `attenuation_factor`: 削峰系数 (0-1)
  - 0.9: 轻微削峰（陡峭河道）
  - 0.7: 中等削峰（一般河道）
  - 0.5: 显著削峰（缓坡河道、有漫滩）

#### C. 基流（Baseflow）
旱季或前期降雨的地下水补给。

**实现方法**：指数衰减 + 最小流量
```python
Q_total(t) = Q_surface(t) + Q_base(t)
Q_base(t) = Q_base_min + (Q_base_0 - Q_base_min) × exp(-k × t)
```

**参数**：
- `base_flow_ratio`: 基流占总流量的比例 (0.05-0.20)

## 3. 分区参数配置

不同参数分区采用不同的修正参数：

| 参数分区 | 径流系数 | 时滞(hr) | 削峰系数 | 基流比例 | 说明 |
|---------|---------|---------|---------|---------|------|
| Zone 1 (上游) | 0.35 | 1.5 | 0.85 | 0.08 | 陡坡，快速汇流 |
| Zone 2 (中游) | 0.34 | 3.0 | 0.75 | 0.12 | 中等坡度 |
| Zone 3 (下游) | 0.33 | 5.0 | 0.65 | 0.15 | 缓坡，显著削峰 |
| Zone 4-6 | 0.345 | 3.5 | 0.70 | 0.10 | 默认值 |

## 4. 实现步骤

### Step 1: 加载面雨量数据
```python
# 读取各分区的面雨量时间序列
precip_df = pd.read_csv("subbasin_areal_precipitation.csv")
```

### Step 2: 按分区汇总降雨
```python
# 对每个参数分区，计算面积加权平均降雨
zone_precip = aggregate_precipitation_by_zone(precip_df, zone_id)
```

### Step 3: 生成径流序列
```python
generator = EstimatedRunoffGenerator(
    runoff_coefficient=0.345,
    lag_hours=2.0,
    attenuation_factor=0.75,
    base_flow_ratio=0.10
)

runoff = generator.generate(zone_precip, zone_area)
```

### Step 4: 保存为观测数据格式
```python
# 保存到 estimated_observations/
# 文件格式：zone_{id}_estimated_runoff.csv
# 列：datetime, discharge_m3s
```

## 5. 使用场景

### 5.1 参数敏感性分析
```python
# 使用估计径流作为"真值"
estimated_obs = load_estimated_runoff(zone_id)

# 运行模型
simulated = run_model(parameters)

# 计算指标
nse = nash_sutcliffe(simulated, estimated_obs)
rmse = root_mean_square_error(simulated, estimated_obs)
```

### 5.2 自动参数率定
```python
def objective_function(parameters):
    simulated = run_model(parameters)
    estimated = load_estimated_runoff()
    return -nash_sutcliffe(simulated, estimated)  # 最小化负NSE

# SCE-UA优化
best_params = sce_ua(objective_function, param_bounds)
```

### 5.3 不确定性分析
```python
# 蒙特卡洛采样
for i in range(1000):
    # 随机扰动径流系数
    rc = 0.345 + random.normal(0, 0.05)
    estimated = generate_runoff(precip, rc)
    # 分析不确定性范围
```

## 6. 验证方法

### 6.1 水量平衡检查
```python
total_precip = zone_precip.sum() * zone_area  # m³
total_runoff = estimated_runoff.sum() * 3600  # m³
actual_rc = total_runoff / total_precip

assert 0.20 < actual_rc < 0.50, "径流系数不合理"
```

### 6.2 峰值合理性
```python
# 单位面积峰值流量检查
peak_per_km2 = estimated_runoff.max() / zone_area

# 一般应在 0.1 - 2.0 m³/s/km² 范围内
assert 0.1 < peak_per_km2 < 2.0, "峰值流量不合理"
```

### 6.3 峰现时间检查
```python
precip_peak_time = precip.argmax()
runoff_peak_time = runoff.argmax()
lag_actual = runoff_peak_time - precip_peak_time

# 应接近设定的lag_hours
assert abs(lag_actual - lag_hours) < 2, "时滞不合理"
```

## 7. 文件结构

```
results/
  └── upper_truckee_complete_11steps/
      ├── step_08_areal_rainfall/
      │   └── 8.2_subbasin_areal_precipitation.csv
      └── estimated_observations/
          ├── zone_1_estimated_runoff.csv
          ├── zone_2_estimated_runoff.csv
          ├── zone_3_estimated_runoff.csv
          ├── generation_config.yaml          # 生成参数记录
          ├── generation_summary.txt          # 统计摘要
          └── comparison_with_simulated.png   # 对比图
```

## 8. 配置文件示例

```yaml
# estimated_runoff_config.yaml
version: "1.0"
description: "Estimated runoff generation for parameter calibration"

global_settings:
  default_runoff_coefficient: 0.345
  default_lag_hours: 2.5
  default_attenuation_factor: 0.75
  default_baseflow_ratio: 0.10

zone_specific:
  - zone_id: 1
    runoff_coefficient: 0.35
    lag_hours: 1.5
    attenuation_factor: 0.85
    baseflow_ratio: 0.08
    description: "Upper steep slopes"

  - zone_id: 2
    runoff_coefficient: 0.34
    lag_hours: 3.0
    attenuation_factor: 0.75
    baseflow_ratio: 0.12
    description: "Middle transition"

  - zone_id: 3
    runoff_coefficient: 0.33
    lag_hours: 5.0
    attenuation_factor: 0.65
    baseflow_ratio: 0.15
    description: "Lower gentle slopes"
```
