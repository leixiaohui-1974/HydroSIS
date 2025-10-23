# HydroSIS 参数管理系统设计

## 1. 系统架构

### 1.1 默认参数存储
默认参数存储在流域划分时生成的CSV文件中，分为两类：

**子流域参数** (`parameter_subbasins.csv` 扩展列):
```
..., hbv_fc, hbv_beta, hbv_k0, hbv_k1, hbv_k2, hbv_perc, hbv_tt, hbv_cfmax, ...
```

**河道参数** (`parameter_channels.csv` 扩展列):
```
..., muskingum_k, muskingum_x, muskingum_dt, ...
```

### 1.2 参数率定文件
使用YAML格式，支持按参数分区进行调整。

**文件位置**: `calibration/parameter_adjustments.yaml`

**文件结构**:
```yaml
# 参数率定配置文件
version: "1.0"
description: "HydroSIS parameter calibration for Upper Truckee River"

# 全局默认参数（可选，用于初始化）
global_defaults:
  hbv:
    FC: 150.0          # 最大土壤含水量 (mm)
    BETA: 1.0          # 形状系数 (-)
    K0: 0.30           # 快速响应系数 (1/hr)
    K1: 0.10           # 慢速响应系数 (1/hr)
    K2: 0.02           # 基流系数 (1/hr)
    PERC: 0.5          # 渗透率 (mm/hr)
    TT: 0.0            # 雪阈值温度 (°C)
    CFMAX: 3.5         # 度日因子 (mm/°C/day)
    initial_soil: 25.0 # 初始土壤含水量 (mm)
    initial_upper: 2.0 # 初始上层储水 (mm)
    initial_lower: 10.0# 初始下层储水 (mm)

  muskingum:
    K: 10.0            # 蓄量常数 (hr)
    x: 0.2             # 权重系数 (-)
    time_step: 1.0     # 时间步长 (hr)

# 按参数分区的调整
zone_adjustments:
  # Zone 1 (上游源头区)
  - zone_id: 1
    description: "Upper headwater zone"
    runoff_parameters:
      hbv:
        # 使用乘法因子（multiplier）
        FC:
          method: "multiply"
          value: 1.2        # FC * 1.2
        BETA:
          method: "multiply"
          value: 1.0
        K0:
          method: "multiply"
          value: 1.1        # 增加快速径流
        # 使用加法增量（additive）
        initial_soil:
          method: "add"
          value: 10.0       # initial_soil + 10

    routing_parameters:
      muskingum:
        K:
          method: "multiply"
          value: 0.8        # 缩短汇流时间
        x:
          method: "set"     # 直接设置值
          value: 0.25

  # Zone 2 (中游过渡区)
  - zone_id: 2
    description: "Middle transition zone"
    runoff_parameters:
      hbv:
        FC:
          method: "multiply"
          value: 1.0
        K1:
          method: "multiply"
          value: 1.15       # 增加慢速径流

    routing_parameters:
      muskingum:
        K:
          method: "multiply"
          value: 1.0

  # Zone 3 (下游主河道区)
  - zone_id: 3
    description: "Lower main channel zone"
    runoff_parameters:
      hbv:
        FC:
          method: "multiply"
          value: 0.9        # 减少田间持水能力
        BETA:
          method: "multiply"
          value: 1.2

    routing_parameters:
      muskingum:
        K:
          method: "multiply"
          value: 1.3        # 延长汇流时间
        x:
          method: "set"
          value: 0.15

# 可选：特定子流域的单独调整（覆盖分区调整）
subbasin_overrides:
  - subbasin_id: 112
    runoff_parameters:
      hbv:
        K0:
          method: "set"
          value: 0.40       # 直接设置为0.40
```

## 2. 参数调整方法

支持三种调整方法：

1. **multiply**: 乘法因子
   ```
   adjusted_value = default_value * factor
   ```

2. **add**: 加法增量
   ```
   adjusted_value = default_value + increment
   ```

3. **set**: 直接设置
   ```
   adjusted_value = new_value
   ```

## 3. 参数应用优先级

1. 全局默认参数 (global_defaults)
2. CSV文件中的子流域/河道参数
3. 参数分区调整 (zone_adjustments)
4. 特定子流域覆盖 (subbasin_overrides)

## 4. 实现步骤

### Step 1: 扩展CSV属性文件
修改流域划分代码，在生成CSV时添加默认参数列。

### Step 2: 创建参数率定模块
```python
# hydrosis/calibration.py
class ParameterCalibration:
    def load_adjustments(yaml_file)
    def apply_to_subbasin(subbasin_id, zone_id, defaults)
    def apply_to_channel(channel_id, zone_id, defaults)
```

### Step 3: 集成到Workflow
修改 `rerun_step09_10.py` 和主workflow，自动读取率定文件。

### Step 4: 创建率定工具
```bash
python calibrate_parameters.py --input calibration/parameter_adjustments.yaml --output results/calibrated_params/
```

## 5. 使用示例

### 5.1 初始化默认参数
```bash
# 流域划分时自动创建带默认参数的CSV
python run_delineation.py --add-default-parameters
```

### 5.2 创建率定文件
```bash
# 从模板创建
python create_calibration_template.py --output calibration/my_calibration.yaml
```

### 5.3 应用参数调整
```bash
# 运行模拟时自动读取
python rerun_step09_10.py --calibration calibration/parameter_adjustments.yaml
```

### 5.4 批量率定
```bash
# 测试多组参数
python batch_calibration.py --scenarios calibration/scenarios/*.yaml --observe data/observed_flow.csv
```

## 6. 验证与评估

参数率定后，自动计算评估指标：
- Nash-Sutcliffe效率系数 (NSE)
- 径流系数 (Rc)
- 峰值误差
- 峰现时间误差

输出对比报告：
```
calibration_results/
  ├── scenario_01_results.csv
  ├── scenario_02_results.csv
  ├── comparison_plots.png
  └── metrics_summary.csv
```
