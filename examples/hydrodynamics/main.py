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