"""
HydroMind Agent 测试脚本

演示12个认知工具的使用
"""

import sys
import os

# 添加父目录到路径
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import json
import asyncio

# 使用绝对导入
import cognitive_tools
CognitiveTools = cognitive_tools.CognitiveTools


async def test_understanding_layer():
    """测试理解层工具（工具1-3）"""
    print("\n" + "="*60)
    print("测试理解层工具（Understanding Layer）")
    print("="*60)
    
    tools = CognitiveTools()
    
    # 测试1: 意图识别
    print("\n[工具1] 意图识别")
    result = await tools.parse_user_intent({
        "user_input": "我想建立长江上游的HBV模型，使用2020年数据进行模拟",
        "conversation_history": []
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 测试2: 实体抽取
    print("\n[工具2] 实体抽取")
    result = await tools.extract_entities({
        "user_input": "流域面积5000平方公里，位于长江上游，使用2020-01-01到2020-12-31的数据"
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 测试3: 需求验证
    print("\n[工具3] 需求验证")
    result = await tools.validate_requirements({
        "requirements": {
            "basin": {"name": "长江上游", "area_km2": 5000},
            "model": {"runoff_type": "HBV"},
            "time_period": {"start": "2020-01-01", "end": "2020-12-31"}
        }
    })
    print(json.dumps(result, ensure_ascii=False, indent=2))


async def test_configuration_layer():
    """测试配置层工具（工具4-6）"""
    print("\n" + "="*60)
    print("测试配置层工具（Configuration Layer）")
    print("="*60)
    
    tools = CognitiveTools()
    
    # 测试4: 配置生成
    print("\n[工具4] 配置生成 ⭐")
    result = await tools.generate_model_config({
        "intent": {"action": "create_and_run_model"},
        "entities": {
            "basin": {"name": "长江上游", "area_km2": 5000},
            "model": {"runoff_type": "HBV", "routing_type": "Muskingum"}
        }
    })
    print("配置摘要:", result.get("config_summary"))
    print("配置内容:", json.dumps(result.get("config"), ensure_ascii=False, indent=2)[:500] + "...")
    
    # 测试5: 参数推荐
    print("\n[工具5] 参数推荐")
    result = await tools.suggest_parameters({
        "basin_features": {"climate": "humid", "area_km2": 5000},
        "model_type": "HBV"
    })
    print("推荐参数:")
    for param, info in result.get("suggested_parameters", {}).items():
        print(f"  {param}: {info['value']} (范围: {info['range']})")
    
    # 测试6: 情景设计
    print("\n[工具6] 情景设计")
    result = await tools.design_scenarios({
        "analysis_objective": "评估土地利用变化对径流的影响",
        "baseline_config": {}
    })
    print(f"设计了 {len(result.get('scenarios', []))} 个情景:")
    for scenario in result.get('scenarios', []):
        print(f"  - {scenario['name']}: {scenario['description']}")


async def test_analysis_layer():
    """测试分析层工具（工具7-9）"""
    print("\n" + "="*60)
    print("测试分析层工具（Analysis Layer）")
    print("="*60)
    
    tools = CognitiveTools()
    
    # 测试7: 结果解读
    print("\n[工具7] 结果解读")
    result = await tools.interpret_results({
        "simulation_results": {
            "metrics": {"nse": 0.85, "rmse": 45.2}
        },
        "model_config": {"runoff": {"model_type": "HBV"}},
        "user_objective": "洪峰模拟"
    })
    assessment = result.get("overall_assessment", {})
    print(f"性能评价: {assessment.get('performance_level')}")
    print(f"关键信息: {assessment.get('key_message')}")
    
    # 测试8: 问题诊断
    print("\n[工具8] 问题诊断")
    result = await tools.diagnose_issues({
        "poor_results": {
            "metrics": {"nse": 0.45, "peak_error": 0.25}
        },
        "model_config": {}
    })
    issues = result.get("identified_issues", [])
    print(f"识别了 {len(issues)} 个问题:")
    for issue in issues:
        print(f"  - {issue.get('issue_type')}: {issue.get('severity')}")
    
    # 测试9: 模型对比
    print("\n[工具9] 模型对比")
    result = await tools.compare_models({
        "models_results": {
            "HBV": {"metrics": {"nse": 0.85}},
            "SCS": {"metrics": {"nse": 0.72}},
            "XinAnJiang": {"metrics": {"nse": 0.80}}
        }
    })
    print("排名:")
    for item in result.get("ranking", []):
        print(f"  {item['rank']}. {item['model']}: {item['score']:.3f}")


async def test_reporting_layer():
    """测试报告层工具（工具10-12）"""
    print("\n" + "="*60)
    print("测试报告层工具（Reporting Layer）")
    print("="*60)
    
    tools = CognitiveTools()
    
    # 测试10: 叙述生成
    print("\n[工具10] 叙述生成")
    result = await tools.generate_narrative({
        "section": "executive_summary",
        "context": {
            "basin_name": "长江上游",
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "runoff_model": "HBV",
            "routing_model": "Muskingum",
            "metrics": {"nse": 0.85, "rmse": 45.2},
            "user_objective": "洪峰模拟"
        }
    })
    print("生成的叙述:")
    print(result.get("narrative")[:300] + "...")
    
    # 测试11: 执行报告生成
    print("\n[工具11] 执行报告 ⭐")
    result = await tools.create_executive_report({
        "workflow_result": {
            "metrics": {"nse": 0.85, "rmse": 45.2}
        },
        "model_config": {
            "runoff": {"model_type": "HBV"},
            "routing": {"model_type": "Muskingum"}
        },
        "report_type": "executive"
    })
    report = result.get("report", {})
    print(f"报告长度: {result.get('metadata', {}).get('word_count')} 字")
    print("报告内容预览:")
    print(report.get("content", "")[:400] + "...")
    
    # 测试12: 智能问答
    print("\n[工具12] 智能问答")
    result = await tools.answer_questions({
        "question": "为什么洪峰模拟偏小？",
        "context": {
            "simulation_results": {"peak_error": -0.15},
            "model_config": {"runoff": {"model_type": "HBV"}}
        }
    })
    print("回答:")
    print(result.get("answer")[:200] + "...")


async def test_end_to_end():
    """端到端测试"""
    print("\n" + "="*60)
    print("端到端测试：完整工作流")
    print("="*60)
    
    tools = CognitiveTools()
    
    user_input = "我想对长江上游三峡库区建立HBV模型，流域面积约5万平方公里，使用2020年数据模拟洪峰"
    
    print(f"\n用户输入: {user_input}")
    
    # 步骤1: 理解
    print("\n步骤1: 理解用户意图...")
    intent = await tools.parse_user_intent({"user_input": user_input})
    entities = await tools.extract_entities({"user_input": user_input})
    
    print(f"  意图: {intent.get('action')}")
    print(f"  实体: 流域={entities.get('basin', {}).get('name')}, 模型={entities.get('model', {}).get('runoff_type')}")
    
    # 步骤2: 配置
    print("\n步骤2: 生成配置...")
    config = await tools.generate_model_config({
        "intent": intent,
        "entities": entities
    })
    print(f"  配置: {config.get('config_summary')}")
    
    # 步骤3: 模拟（模拟结果）
    print("\n步骤3: 执行模拟...")
    print("  [模拟] HydroCompute执行中...")
    mock_results = {
        "metrics": {"nse": 0.87, "rmse": 42.3, "peak_error": 0.08},
        "peak_flow": 3250
    }
    print(f"  模拟完成: NSE={mock_results['metrics']['nse']:.3f}")
    
    # 步骤4: 解读
    print("\n步骤4: 解读结果...")
    interpretation = await tools.interpret_results({
        "simulation_results": mock_results,
        "model_config": config.get("config", {}),
        "user_objective": "洪峰模拟"
    })
    assessment = interpretation.get("overall_assessment", {})
    print(f"  评价: {assessment.get('key_message')}")
    
    # 步骤5: 报告
    print("\n步骤5: 生成报告...")
    report = await tools.create_executive_report({
        "workflow_result": mock_results,
        "model_config": config.get("config", {})
    })
    print(f"  报告生成完成")
    
    print("\n✅ 端到端测试完成!")


async def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("HydroMind Agent - 认知智能体测试")
    print("="*60)
    print("\n提示: 当前使用Mock LLM后端")
    print("设置环境变量 QWEN_API_KEY 可使用真实的千问模型\n")
    
    # 运行所有测试
    await test_understanding_layer()
    await test_configuration_layer()
    await test_analysis_layer()
    await test_reporting_layer()
    await test_end_to_end()
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
