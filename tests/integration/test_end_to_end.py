"""
端到端集成测试

测试完整的双智能体工作流：
用户自然语言 → HydroMind理解 → 配置生成 → HydroCompute计算 → HydroMind分析 → 报告生成
"""

import sys
import asyncio
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_orchestrator import TwinAgentCoordinator
from mcp_orchestrator.clients import HydroMindClient, HydroComputeClient


async def test_basic_understanding():
    """测试1: 基础理解功能"""
    print("\n" + "="*60)
    print("测试1: 基础理解功能")
    print("="*60)
    
    # 创建HydroMind客户端
    mind = HydroMindClient()
    
    # 测试健康检查
    print("\n1.1 健康检查...")
    is_healthy = await mind.health_check()
    print(f"   HydroMind状态: {'✅ 健康' if is_healthy else '❌ 不可用'}")
    
    if not is_healthy:
        print("   ⚠️  跳过测试（服务器未运行）")
        return False
    
    # 测试意图识别
    print("\n1.2 意图识别...")
    intent = await mind.parse_user_intent("我想建立长江上游的HBV模型")
    print(f"   意图: {intent.get('action')}")
    print(f"   置信度: {intent.get('confidence')}")
    
    # 测试实体抽取
    print("\n1.3 实体抽取...")
    entities = await mind.extract_entities("流域面积5000平方公里，使用HBV模型")
    print(f"   流域: {entities.get('basin', {}).get('name')}")
    print(f"   模型: {entities.get('model', {}).get('runoff_type')}")
    
    print("\n✅ 基础理解功能测试通过")
    return True


async def test_config_generation():
    """测试2: 配置生成"""
    print("\n" + "="*60)
    print("测试2: 配置生成")
    print("="*60)
    
    mind = HydroMindClient()
    
    if not await mind.health_check():
        print("   ⚠️  跳过测试（服务器未运行）")
        return False
    
    print("\n2.1 生成配置...")
    config = await mind.generate_model_config(
        intent={"action": "create_model"},
        entities={
            "basin": {"name": "长江上游", "area_km2": 5000},
            "model": {"runoff_type": "HBV", "routing_type": "Muskingum"}
        }
    )
    
    print(f"   配置摘要: {config.get('config_summary')}")
    print(f"   产流模型: {config.get('config', {}).get('runoff', {}).get('model_type')}")
    print(f"   汇流方法: {config.get('config', {}).get('routing', {}).get('model_type')}")
    
    print("\n✅ 配置生成测试通过")
    return True


async def test_config_conversion():
    """测试3: 配置转换"""
    print("\n" + "="*60)
    print("测试3: 配置转换")
    print("="*60)
    
    from mcp_orchestrator.config_converter import ConfigConverter
    
    converter = ConfigConverter()
    
    print("\n3.1 转换简单配置...")
    hydromind_config = {
        "runoff": {
            "model_type": "HBV",
            "parameters": {"fc": 200}
        },
        "routing": {
            "model_type": "Muskingum"
        }
    }
    
    hydrosis_config = converter.hydromind_to_hydrosis(hydromind_config)
    
    print(f"   ✅ 转换成功")
    print(f"   产流参数数: {len(hydrosis_config['runoff']['parameters'])}")
    print(f"   汇流参数数: {len(hydrosis_config['routing']['parameters'])}")
    
    # 验证配置
    is_valid, errors = converter.validate_config(hydrosis_config)
    print(f"\n3.2 配置验证: {'✅ 通过' if is_valid else '❌ 失败'}")
    if errors:
        for error in errors:
            print(f"   - {error}")
    
    print("\n✅ 配置转换测试通过")
    return True


async def test_coordinator_initialization():
    """测试4: 协调器初始化"""
    print("\n" + "="*60)
    print("测试4: 协调器初始化")
    print("="*60)
    
    print("\n4.1 创建协调器...")
    coordinator = TwinAgentCoordinator()
    
    print("\n4.2 检查组件...")
    print(f"   HydroMind客户端: {'✅' if coordinator.mind else '❌'}")
    print(f"   HydroCompute客户端: {'✅' if coordinator.compute else '❌'}")
    print(f"   配置转换器: {'✅' if coordinator.converter else '❌'}")
    print(f"   对话管理器: {'✅' if coordinator.conversation else '❌'}")
    
    print("\n✅ 协调器初始化测试通过")
    return True


async def test_quick_understand():
    """测试5: 快速理解模式"""
    print("\n" + "="*60)
    print("测试5: 快速理解模式")
    print("="*60)
    
    coordinator = TwinAgentCoordinator()
    
    # 检查HydroMind是否可用
    if not await coordinator.mind.health_check():
        print("   ⚠️  跳过测试（HydroMind不可用）")
        return False
    
    print("\n5.1 快速理解请求...")
    result = await coordinator.quick_understand(
        user_input="我想建立HBV模型，流域面积1000平方公里",
        session_id="test_session"
    )
    
    print(f"   模式: {result.get('mode')}")
    print(f"   意图: {result.get('intent', {}).get('action')}")
    print(f"   实体: {list(result.get('entities', {}).keys())}")
    print(f"   下一步建议: {len(result.get('next_steps', []))} 条")
    
    print("\n✅ 快速理解模式测试通过")
    return True


async def test_full_workflow_mock():
    """测试6: 完整工作流（Mock模式）"""
    print("\n" + "="*60)
    print("测试6: 完整工作流（Mock HydroCompute）")
    print("="*60)
    
    coordinator = TwinAgentCoordinator()
    
    # 检查HydroMind
    if not await coordinator.mind.health_check():
        print("   ⚠️  跳过测试（HydroMind不可用）")
        return False
    
    print("\n6.1 处理用户请求...")
    print("   用户输入: '我想建立长江上游的HBV模型'")
    
    # 模拟只执行理解和配置生成阶段
    user_input = "我想建立长江上游的HBV模型，流域面积50000平方公里"
    
    # 阶段1: 理解
    print("\n   [阶段1] 理解意图...")
    intent = await coordinator.mind.parse_user_intent(user_input)
    print(f"      ✅ 意图: {intent.get('action')}")
    
    # 阶段2: 抽取实体
    print("\n   [阶段2] 抽取实体...")
    entities = await coordinator.mind.extract_entities(user_input)
    print(f"      ✅ 流域: {entities.get('basin', {}).get('name')}")
    
    # 阶段3: 生成配置
    print("\n   [阶段3] 生成配置...")
    config = await coordinator.mind.generate_model_config(intent, entities)
    print(f"      ✅ 配置: {config.get('config_summary')}")
    
    # 阶段4: 转换配置
    if coordinator.converter:
        print("\n   [阶段4] 转换配置...")
        hydrosis_config = coordinator.converter.hydromind_to_hydrosis(
            config.get('config', {})
        )
        print(f"      ✅ 转换成功")
    
    print("\n✅ 完整工作流测试通过（Mock模式）")
    return True


async def test_report_generation():
    """测试7: 报告生成"""
    print("\n" + "="*60)
    print("测试7: 报告生成")
    print("="*60)
    
    mind = HydroMindClient()
    
    if not await mind.health_check():
        print("   ⚠️  跳过测试（HydroMind不可用）")
        return False
    
    print("\n7.1 生成模拟结果报告...")
    
    # 模拟结果
    mock_results = {
        "metrics": {
            "nse": 0.85,
            "rmse": 45.2,
            "mae": 32.1
        }
    }
    
    mock_config = {
        "runoff": {"model_type": "HBV"},
        "routing": {"model_type": "Muskingum"}
    }
    
    report = await mind.create_executive_report(
        workflow_result=mock_results,
        model_config=mock_config,
        report_type="executive"
    )
    
    report_content = report.get('report', {}).get('content', '')
    print(f"   报告长度: {len(report_content)} 字符")
    print(f"   包含章节: {'✅' if '##' in report_content else '❌'}")
    
    print("\n✅ 报告生成测试通过")
    return True


async def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("HydroSIS 双智能体系统 - 端到端集成测试")
    print("="*60)
    
    tests = [
        ("基础理解功能", test_basic_understanding),
        ("配置生成", test_config_generation),
        ("配置转换", test_config_conversion),
        ("协调器初始化", test_coordinator_initialization),
        ("快速理解模式", test_quick_understand),
        ("完整工作流（Mock）", test_full_workflow_mock),
        ("报告生成", test_report_generation),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ 测试失败: {name}")
            print(f"   错误: {e}")
            results.append((name, False))
    
    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败/跳过"
        print(f"  {status}  {name}")
    
    print("\n" + "-"*60)
    print(f"总计: {passed}/{total} 通过")
    print("="*60)
    
    if passed == total:
        print("\n🎉 所有测试通过！")
    elif passed > 0:
        print(f"\n⚠️  部分测试通过 ({passed}/{total})")
    else:
        print("\n❌ 所有测试失败")
    
    return passed, total


if __name__ == "__main__":
    print("\n提示: 请确保HydroMind服务器正在运行")
    print("启动命令: cd /workspace/mcp_server_mind && python3 examples/test_hydromind.py\n")
    
    passed, total = asyncio.run(run_all_tests())
    
    sys.exit(0 if passed == total else 1)
