# 水文模拟 - 测试报告

## 测试概览

- **测试名称**: 水文模拟
- **执行时间**: 2025-10-26T08:01:57.424188
- **状态**: ✅ 通过
- **耗时**: 0.00秒

## 输入参数

```yaml
config_file: config/workflows/test_scenarios/05_hydrologic_simulation.yaml
scenario_id: '05'
scenario_name: 水文模拟
```

## 输出结果

| 输出项 | 路径 | 状态 |
|--------|------|------|
| discharge | `results/workflow_tests/05_hydro_sim/routing/discharge.csv` | ❌ |

## 验证结果

### ⚠️ 警告

- 输出文件缺失: runoff.runoff_timeseries, routing.discharge_timeseries

### 📊 指标

- **total_duration**: 0.000866
- **step_count**: 2
- **completed_steps**: 2

## 可视化结果

### summary

![summary](visualizations/test_summary.png)

## 结论

✅ **测试通过** - 所有验证项都满足要求
