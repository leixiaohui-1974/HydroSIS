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