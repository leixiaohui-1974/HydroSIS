"""HydroSIS 模块化API使用示例

演示如何使用新的模块化API系统。
"""
from pathlib import Path

def example_1_simple_module():
    """示例1: 使用单个模块"""
    print("\n=== 示例1: 使用单个模块 ===\n")
    
    from hydrosis.modules import TerrainModule
    
    # 创建模块实例
    terrain = TerrainModule()
    
    # 执行模块
    output = terrain.run({
        "dem_path": "data/Upper_Truckee_River/terrain/elevation.tif",
        "method": "d8",
        "fill_depressions": True,
        "compute_slope": True,
        "output_dir": "results/modular_example/terrain"
    })
    
    print(f"✓ 地形处理完成")
    print(f"  流向文件: {output.flow_direction}")
    print(f"  流量累积: {output.flow_accumulation}")
    print(f"  坡度文件: {output.slope}")


def example_2_module_chain():
    """示例2: 手动组合多个模块"""
    print("\n=== 示例2: 手动组合多个模块 ===\n")
    
    from hydrosis.modules import TerrainModule, PourPointsModule
    
    # 步骤1: 地形处理
    print("步骤1: 地形处理...")
    terrain = TerrainModule()
    terrain_output = terrain.run({
        "dem_path": "data/Upper_Truckee_River/terrain/elevation.tif",
        "method": "d8",
        "output_dir": "results/modular_example/terrain"
    })
    print(f"✓ 地形处理完成")
    
    # 步骤2: 汇水点生成
    print("\n步骤2: 汇水点生成...")
    pour_points = PourPointsModule()
    pp_output = pour_points.run({
        "flow_accumulation": terrain_output.flow_accumulation,
        "method": "auto",
        "threshold": 1000.0,
        "output_dir": "results/modular_example/pour_points"
    })
    print(f"✓ 汇水点生成完成")
    print(f"  识别到 {len(pp_output.points)} 个汇水点")
    print(f"  输出文件: {pp_output.pour_points_geojson}")


def example_3_workflow_template():
    """示例3: 使用预定义工作流模板"""
    print("\n=== 示例3: 使用预定义工作流模板 ===\n")
    
    from hydrosis.workflow_engine import WorkflowEngine, WorkflowTemplates, WorkflowDefinition
    
    # 创建引擎
    engine = WorkflowEngine()
    
    # 加载模板
    template = WorkflowTemplates.get_template("pour_points_only")
    workflow = WorkflowDefinition.from_dict(template)
    
    print(f"工作流: {workflow.name}")
    print(f"步骤数: {len(workflow.steps)}")
    
    # 执行工作流
    print("\n开始执行工作流...\n")
    
    def progress_callback(run, step_result):
        """进度回调"""
        percent = run.progress_percent()
        print(f"[{percent:5.1f}%] {step_result.step_id:20s} -> {step_result.status}")
    
    run = engine.execute(
        workflow,
        parameters={
            "dem_path": "data/Upper_Truckee_River/terrain/elevation.tif",
            "output_dir": "results/modular_example/workflow",
            "threshold": 1500.0
        },
        progress_callback=progress_callback
    )
    
    print(f"\n✓ 工作流执行{run.status}")
    print(f"  运行ID: {run.run_id}")
    print(f"  耗时: {run.duration_seconds():.2f} 秒")
    print(f"  输出: {run.outputs}")


def example_4_workflow_from_yaml():
    """示例4: 从YAML文件加载工作流"""
    print("\n=== 示例4: 从YAML文件加载工作流 ===\n")
    
    from hydrosis.workflow_engine import WorkflowEngine, WorkflowDefinition
    
    # 检查配置文件是否存在
    config_file = Path("config/workflows/pour_points_only.yaml")
    if not config_file.exists():
        print(f"⚠ 配置文件不存在: {config_file}")
        return
    
    # 加载工作流
    workflow = WorkflowDefinition.from_yaml(config_file)
    
    print(f"工作流: {workflow.name}")
    print(f"描述: {workflow.description}")
    print(f"版本: {workflow.version}")
    
    # 执行
    engine = WorkflowEngine()
    run = engine.execute(workflow, parameters={
        "dem_path": "data/Upper_Truckee_River/terrain/elevation.tif",
        "output_dir": "results/modular_example/yaml_workflow"
    })
    
    print(f"\n✓ 工作流执行完成")
    print(f"  状态: {run.status}")


def example_5_module_registry():
    """示例5: 使用模块注册表"""
    print("\n=== 示例5: 使用模块注册表 ===\n")
    
    from hydrosis.modules.base import get_registry
    
    # 获取全局注册表
    registry = get_registry()
    
    # 列出所有模块
    modules = registry.list_modules()
    print(f"已注册模块 ({len(modules)}):")
    for module_id in modules:
        print(f"  - {module_id}")
    
    # 获取模块信息
    print("\n获取terrain模块信息:")
    terrain_module = registry.get_or_create_module("terrain")
    info = terrain_module.get_info()
    print(f"  模块ID: {info['module_id']}")
    print(f"  名称: {info['name']}")
    print(f"  描述: {info['description']}")
    print(f"  版本: {info['version']}")


def example_6_custom_workflow():
    """示例6: 动态构建自定义工作流"""
    print("\n=== 示例6: 动态构建自定义工作流 ===\n")
    
    from hydrosis.workflow_engine import WorkflowDefinition, WorkflowStep, WorkflowEngine
    
    # 动态构建工作流
    workflow = WorkflowDefinition(
        id="custom_demo",
        name="自定义演示工作流",
        version="1.0",
        description="动态构建的工作流示例",
        parameters={
            "dem_path": "data/Upper_Truckee_River/terrain/elevation.tif",
            "output_dir": "results/modular_example/custom"
        },
        steps=[
            WorkflowStep(
                id="terrain",
                module="terrain",
                inputs={
                    "dem_path": "${parameters.dem_path}",
                    "method": "d8",
                    "output_dir": "${parameters.output_dir}/terrain"
                }
            ),
            WorkflowStep(
                id="pour_points",
                module="pour_points",
                depends_on=["terrain"],
                inputs={
                    "flow_accumulation": "${steps.terrain.outputs.flow_accumulation}",
                    "method": "auto",
                    "threshold": 2000.0,
                    "output_dir": "${parameters.output_dir}/pour_points"
                }
            )
        ],
        outputs={
            "pour_points": "${steps.pour_points.outputs.pour_points_geojson}",
            "flow_acc": "${steps.terrain.outputs.flow_accumulation}"
        }
    )
    
    print(f"工作流: {workflow.name}")
    print(f"步骤: {[step.id for step in workflow.steps]}")
    
    # 执行
    engine = WorkflowEngine()
    run = engine.execute(workflow)
    
    print(f"\n✓ 自定义工作流执行完成")
    print(f"  状态: {run.status}")


def main():
    """主函数"""
    print("=" * 60)
    print("HydroSIS 模块化API示例")
    print("=" * 60)
    
    try:
        # 运行各个示例
        example_5_module_registry()
        
        # 注意: 以下示例需要实际的DEM数据文件
        # 如果数据文件不存在，这些示例会失败
        
        # example_1_simple_module()
        # example_2_module_chain()
        # example_3_workflow_template()
        # example_4_workflow_from_yaml()
        # example_6_custom_workflow()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ 示例运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
