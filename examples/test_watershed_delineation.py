#!/usr/bin/env python3
"""测试流域划分和参数分区划分的正确性"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from hydrosis.config import IOConfig, ScenarioConfig, EvaluationConfig
from hydrosis.delineation.dem_delineator import DelineationConfig
from hydrosis.parameters.zone import ParameterZoneBuilder, ParameterZoneConfig
from hydrosis.io.inputs import load_forcing
from hydrosis import HydroSISModel, ModelConfig, run_workflow
from hydrosis.runoff.base import RunoffModelConfig
from hydrosis.routing.base import RoutingModelConfig


def load_test_config():
    """加载测试配置"""
    config_path = REPO_ROOT / "config" / "typical_watershed_test.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建配置对象
    delineation = DelineationConfig.from_dict(data["delineation"])
    runoff_models = [
        RunoffModelConfig.from_dict(cfg) for cfg in data.get("runoff_models", [])
    ]
    routing_models = [
        RoutingModelConfig.from_dict(cfg) for cfg in data.get("routing_models", [])
    ]
    parameter_zones = [
        ParameterZoneConfig.from_dict(cfg)
        for cfg in data.get("parameter_zones", [])
    ]
    io_config = IOConfig.from_dict(data["io"])
    
    return ModelConfig(
        delineation=delineation,
        runoff_models=runoff_models,
        routing_models=routing_models,
        parameter_zones=parameter_zones,
        io=io_config,
        scenarios=[ScenarioConfig(**cfg) for cfg in data.get("scenarios", [])],
        evaluation=EvaluationConfig(**data.get("evaluation", {}))
    )


def test_delineation():
    """测试流域划分"""
    print("=" * 50)
    print("测试流域划分")
    print("=" * 50)
    
    config = load_test_config()
    subbasins = config.delineation.to_subbasins()
    
    print(f"成功划分出 {len(subbasins)} 个子流域:")
    for sub in subbasins:
        print(f"  - {sub.id}: 面积 {sub.area_km2:.6f} km2, 下游: {sub.downstream or '无'}")
    
    # 验证流域划分的正确性
    # 1. 检查是否有重复的子流域ID
    sub_ids = [sub.id for sub in subbasins]
    if len(sub_ids) != len(set(sub_ids)):
        print("错误: 发现重复的子流域ID!")
        return False
    
    # 2. 检查下游关系是否正确
    sub_dict = {sub.id: sub for sub in subbasins}
    for sub in subbasins:
        if sub.downstream and sub.downstream not in sub_dict:
            print(f"错误: 子流域 {sub.id} 的下游 {sub.downstream} 不存在!")
            return False
    
    # 3. 检查是否有循环依赖
    visited = set()
    def check_cycle(sub_id, path):
        if sub_id in path:
            print(f"错误: 发现循环依赖 {' -> '.join(path)} -> {sub_id}")
            return False
        if sub_id in visited:
            return True
        visited.add(sub_id)
        sub = sub_dict.get(sub_id)
        if sub and sub.downstream:
            return check_cycle(sub.downstream, path + [sub_id])
        return True
    
    for sub in subbasins:
        if sub.id not in visited:
            if not check_cycle(sub.id, []):
                return False
    
    print("流域划分验证通过!")
    return True


def test_parameter_zones():
    """测试参数分区划分"""
    print("\n" + "=" * 50)
    print("测试参数分区划分")
    print("=" * 50)
    
    config = load_test_config()
    subbasins = config.delineation.to_subbasins()
    zones = ParameterZoneBuilder.from_config(config.parameter_zones, subbasins)
    
    print(f"成功创建 {len(zones)} 个参数分区:")
    for zone in zones:
        print(f"  - {zone.id}: 控制点 {', '.join(zone.controllers)}, 控制子流域 {', '.join(zone.controlled_subbasins)}")
    
    # 验证参数分区划分的正确性
    # 1. 检查每个子流域是否只属于一个参数分区
    sub_to_zone = {}
    for zone in zones:
        for sub_id in zone.controlled_subbasins:
            if sub_id in sub_to_zone:
                print(f"错误: 子流域 {sub_id} 同时属于参数分区 {sub_to_zone[sub_id]} 和 {zone.id}!")
                return False
            sub_to_zone[sub_id] = zone.id
    
    # 2. 检查所有子流域是否都被分配到参数分区
    sub_ids = {sub.id for sub in subbasins}
    assigned_subs = set(sub_to_zone.keys())
    unassigned = sub_ids - assigned_subs
    if unassigned:
        print(f"警告: 以下子流域未被分配到任何参数分区: {', '.join(unassigned)}")
    
    # 3. 检查控制点是否存在于子流域中
    sub_dict = {sub.id: sub for sub in subbasins}
    for zone in zones:
        for control in zone.controllers:
            if control not in sub_dict:
                print(f"错误: 控制点 {control} 不在任何子流域中!")
                return False
    
    print("参数分区划分验证通过!")
    return True


def test_model_simulation():
    """测试模型模拟"""
    print("\n" + "=" * 50)
    print("测试模型模拟")
    print("=" * 50)
    
    config = load_test_config()
    
    # 解析路径
    config.io.precipitation = REPO_ROOT / config.io.precipitation
    config.io.discharge_observations = REPO_ROOT / config.io.discharge_observations
    config.io.results_directory = REPO_ROOT / config.io.results_directory
    config.io.figures_directory = REPO_ROOT / config.io.figures_directory
    config.io.reports_directory = REPO_ROOT / config.io.reports_directory
    
    # 创建目录
    config.io.results_directory.mkdir(parents=True, exist_ok=True)
    config.io.figures_directory.mkdir(parents=True, exist_ok=True)
    config.io.reports_directory.mkdir(parents=True, exist_ok=True)
    
    # 加载 forcing 数据
    try:
        forcing = load_forcing(config.io.precipitation)
        print(f"成功加载 forcing 数据，包含 {len(forcing)} 个时间序列")
    except Exception as e:
        print(f"加载 forcing 数据失败: {e}")
        return False
    
    # 创建模型并运行模拟
    try:
        model = HydroSISModel.from_config(config)
        print("成功创建 HydroSIS 模型")
        
        # 运行模拟
        results = model.run(forcing)
        print("成功运行模型模拟")
        
        # 聚合结果
        aggregated = model.accumulate_discharge(results)
        print("成功聚合结果")
        
        # 显示部分结果
        for sub_id, series in aggregated.items():
            preview = ", ".join(f"{value:.2f}" for value in series[:5])
            print(f"  子流域 {sub_id} 前5个值: {preview}")
        
        return True
    except Exception as e:
        print(f"模型模拟失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_workflow():
    """测试完整工作流"""
    print("\n" + "=" * 50)
    print("测试完整工作流")
    print("=" * 50)
    
    config = load_test_config()
    
    # 解析路径
    config.io.precipitation = REPO_ROOT / config.io.precipitation
    config.io.discharge_observations = REPO_ROOT / config.io.discharge_observations
    config.io.results_directory = REPO_ROOT / config.io.results_directory
    config.io.figures_directory = REPO_ROOT / config.io.figures_directory
    config.io.reports_directory = REPO_ROOT / config.io.reports_directory
    
    # 创建目录
    config.io.results_directory.mkdir(parents=True, exist_ok=True)
    config.io.figures_directory.mkdir(parents=True, exist_ok=True)
    config.io.reports_directory.mkdir(parents=True, exist_ok=True)
    
    # 加载 forcing 数据
    try:
        forcing = load_forcing(config.io.precipitation)
    except Exception as e:
        print(f"加载 forcing 数据失败: {e}")
        return False
    
    # 运行工作流
    try:
        workflow_result = run_workflow(
            config,
            forcing,
            observations=None,
            persist_outputs=True,
            generate_report=False,
        )
        print("成功运行完整工作流")
        
        # 显示结果摘要
        if workflow_result.baseline.aggregated:
            print("基准情景结果:")
            for sub_id, series in workflow_result.baseline.aggregated.items():
                preview = ", ".join(f"{value:.2f}" for value in series[:3])
                print(f"  子流域 {sub_id} 前3个值: {preview}")
        
        if workflow_result.scenarios:
            print("情景结果:")
            for scenario_id, scenario in workflow_result.scenarios.items():
                if scenario.aggregated:
                    first_key = next(iter(scenario.aggregated))
                    preview = ", ".join(f"{value:.2f}" for value in scenario.aggregated[first_key][:3])
                    print(f"  {scenario_id}: {first_key} 前3个值: {preview}")
        
        return True
    except Exception as e:
        print(f"工作流运行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("开始测试流域划分和参数分区划分...")
    
    success = True
    
    # 测试流域划分
    if not test_delineation():
        success = False
    
    # 测试参数分区划分
    if not test_parameter_zones():
        success = False
    
    # 测试模型模拟
    if not test_model_simulation():
        success = False
    
    # 测试完整工作流
    if not test_workflow():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("所有测试通过!")
    else:
        print("部分测试失败，请检查错误信息!")
    print("=" * 50)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
