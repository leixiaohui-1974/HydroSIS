"""
HydroMind MCP Server - 认知智能体服务器

提供12个认知工具的MCP接口
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import asyncio
from typing import Dict, Any
from pathlib import Path

# 复用HydroCompute的MCP Server基础设施
try:
    from mcp_server.server import MCPServer, MCPTool
except ImportError:
    # 如果没有，使用简化版
    print("⚠️  警告: 未找到mcp_server.server，使用简化MCP实现")
    
    class MCPTool:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    
    class MCPServer:
        def __init__(self, name: str, version: str):
            self.name = name
            self.version = version
            self.tools = {}
        
        def register_tool(self, name, func, description, category, schema=None):
            self.tools[name] = {
                "name": name,
                "func": func,
                "description": description,
                "category": category,
                "schema": schema or {}
            }
            print(f"  ✓ 注册工具: {name}")
        
        def list_tools(self):
            return [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "category": t["category"],
                    "inputSchema": t["schema"]
                }
                for t in self.tools.values()
            ]
        
        async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
            if tool_name not in self.tools:
                return {"error": f"工具不存在: {tool_name}"}
            
            func = self.tools[tool_name]["func"]
            try:
                result = await func(arguments)
                return {"result": result, "isError": False}
            except Exception as e:
                return {"error": str(e), "isError": True}


try:
    from .cognitive_tools import CognitiveTools
    from .llm_backend import create_default_backend
except ImportError:
    from cognitive_tools import CognitiveTools
    from llm_backend import create_default_backend


class HydroMindServer:
    """HydroMind MCP服务器"""
    
    def __init__(self):
        self.mcp = MCPServer(
            name="HydroMind",
            version="1.0.0"
        )
        
        # 初始化认知工具
        self.cognitive_tools = CognitiveTools(
            llm_backend=create_default_backend()
        )
        
        # 注册所有工具
        self._register_all_tools()
        
        print(f"\n✅ HydroMind服务器初始化完成")
        print(f"   已注册 {len(self.mcp.tools)} 个认知工具")
    
    def _register_all_tools(self):
        """注册所有12个认知工具"""
        
        print("\n正在注册认知工具...")
        
        # 理解层
        self.mcp.register_tool(
            name="parse_user_intent",
            func=self.cognitive_tools.parse_user_intent,
            description="解析用户意图，识别用户想要执行的操作",
            category="understanding",
            schema={
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "用户输入的自然语言"},
                    "conversation_history": {"type": "array", "description": "对话历史"}
                },
                "required": ["user_input"]
            }
        )
        
        self.mcp.register_tool(
            name="extract_entities",
            func=self.cognitive_tools.extract_entities,
            description="从用户输入中抽取流域、模型、时间等关键实体",
            category="understanding",
            schema={
                "type": "object",
                "properties": {
                    "user_input": {"type": "string", "description": "用户输入"}
                },
                "required": ["user_input"]
            }
        )
        
        self.mcp.register_tool(
            name="validate_requirements",
            func=self.cognitive_tools.validate_requirements,
            description="验证建模需求的完整性和可行性",
            category="understanding",
            schema={
                "type": "object",
                "properties": {
                    "requirements": {"type": "object", "description": "需求字典"}
                },
                "required": ["requirements"]
            }
        )
        
        # 配置层
        self.mcp.register_tool(
            name="generate_model_config",
            func=self.cognitive_tools.generate_model_config,
            description="生成完整的模型配置（核心工具）",
            category="configuration",
            schema={
                "type": "object",
                "properties": {
                    "intent": {"type": "object", "description": "用户意图"},
                    "entities": {"type": "object", "description": "抽取的实体"}
                },
                "required": ["entities"]
            }
        )
        
        self.mcp.register_tool(
            name="suggest_parameters",
            func=self.cognitive_tools.suggest_parameters,
            description="基于流域特征智能推荐模型参数",
            category="configuration",
            schema={
                "type": "object",
                "properties": {
                    "basin_features": {"type": "object", "description": "流域特征"},
                    "model_type": {"type": "string", "description": "模型类型"}
                },
                "required": ["model_type"]
            }
        )
        
        self.mcp.register_tool(
            name="design_scenarios",
            func=self.cognitive_tools.design_scenarios,
            description="根据分析目标设计对比情景",
            category="configuration",
            schema={
                "type": "object",
                "properties": {
                    "analysis_objective": {"type": "string", "description": "分析目标"},
                    "baseline_config": {"type": "object", "description": "基准配置"}
                },
                "required": ["analysis_objective"]
            }
        )
        
        # 分析层
        self.mcp.register_tool(
            name="interpret_results",
            func=self.cognitive_tools.interpret_results,
            description="智能解读模拟结果，提供专业分析",
            category="analysis",
            schema={
                "type": "object",
                "properties": {
                    "simulation_results": {"type": "object", "description": "模拟结果"},
                    "model_config": {"type": "object", "description": "模型配置"},
                    "user_objective": {"type": "string", "description": "用户目标"}
                },
                "required": ["simulation_results"]
            }
        )
        
        self.mcp.register_tool(
            name="diagnose_issues",
            func=self.cognitive_tools.diagnose_issues,
            description="诊断模型问题，提供改进建议",
            category="analysis",
            schema={
                "type": "object",
                "properties": {
                    "poor_results": {"type": "object", "description": "不佳的结果"},
                    "model_config": {"type": "object", "description": "模型配置"}
                },
                "required": ["poor_results"]
            }
        )
        
        self.mcp.register_tool(
            name="compare_models",
            func=self.cognitive_tools.compare_models,
            description="对比多个模型的性能表现",
            category="analysis",
            schema={
                "type": "object",
                "properties": {
                    "models_results": {"type": "object", "description": "多个模型的结果"}
                },
                "required": ["models_results"]
            }
        )
        
        # 报告层
        self.mcp.register_tool(
            name="generate_narrative",
            func=self.cognitive_tools.generate_narrative,
            description="生成特定章节的自然语言叙述",
            category="reporting",
            schema={
                "type": "object",
                "properties": {
                    "section": {"type": "string", "description": "章节名称"},
                    "context": {"type": "object", "description": "上下文数据"}
                },
                "required": ["section", "context"]
            }
        )
        
        self.mcp.register_tool(
            name="create_executive_report",
            func=self.cognitive_tools.create_executive_report,
            description="生成完整的执行报告（核心工具）",
            category="reporting",
            schema={
                "type": "object",
                "properties": {
                    "workflow_result": {"type": "object", "description": "工作流结果"},
                    "model_config": {"type": "object", "description": "模型配置"},
                    "report_type": {"type": "string", "description": "报告类型"}
                },
                "required": ["workflow_result", "model_config"]
            }
        )
        
        self.mcp.register_tool(
            name="answer_questions",
            func=self.cognitive_tools.answer_questions,
            description="回答用户关于模拟结果的问题",
            category="reporting",
            schema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "用户问题"},
                    "context": {"type": "object", "description": "模拟上下文"}
                },
                "required": ["question"]
            }
        )
    
    def list_tools(self) -> list:
        """列出所有工具"""
        return self.mcp.list_tools()
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """调用工具"""
        return await self.mcp.call_tool(tool_name, arguments)
    
    def start(self, host="0.0.0.0", port=8081):
        """启动服务器"""
        print(f"\n🚀 HydroMind服务器启动中...")
        print(f"   地址: http://{host}:{port}")
        print(f"   工具数: {len(self.mcp.tools)}")
        print(f"\n访问 http://{host}:{port}/tools 查看所有工具")
        print(f"访问 http://{host}:{port}/health 进行健康检查")
        
        # 这里可以集成FastAPI等Web框架
        # 目前仅打印信息
        print("\n提示: 实际部署时需要集成Web框架（FastAPI）")
        print("参考 mcp_server/main.py 的实现")


def create_app():
    """创建应用（供FastAPI使用）"""
    server = HydroMindServer()
    return server


if __name__ == "__main__":
    # 测试模式
    async def test_server():
        server = HydroMindServer()
        
        print("\n=== 工具列表 ===")
        tools = server.list_tools()
        for tool in tools:
            print(f"  [{tool['category']}] {tool['name']}: {tool['description']}")
        
        print("\n=== 测试工具调用 ===")
        result = await server.call_tool("parse_user_intent", {
            "user_input": "我想建立HBV模型",
            "conversation_history": []
        })
        
        if not result.get("isError"):
            print("✅ 调用成功:")
            print(json.dumps(result["result"], ensure_ascii=False, indent=2))
        else:
            print(f"❌ 调用失败: {result.get('error')}")
    
    asyncio.run(test_server())
