"""
HydroMind Client - HydroMind认知智能体客户端

提供对HydroMind 12个认知工具的访问
"""

from typing import Dict, Any, List, Optional
from .http_utils import MCPHttpClient


class HydroMindClient:
    """HydroMind MCP客户端"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        timeout: float = 60.0  # 认知任务可能较慢
    ):
        """
        初始化HydroMind客户端
        
        Args:
            base_url: HydroMind服务器地址
            timeout: 请求超时时间
        """
        self.base_url = base_url
        self.http = MCPHttpClient(base_url, timeout=timeout)
        self._tools_cache = None
    
    async def health_check(self) -> bool:
        """健康检查"""
        return await self.http.health_check()
    
    async def list_tools(self, force_refresh: bool = False) -> List[Dict[str, Any]]:
        """
        列出所有工具
        
        Args:
            force_refresh: 强制刷新缓存
            
        Returns:
            工具列表
        """
        if self._tools_cache is None or force_refresh:
            response = await self.http.get("/mcp/tools")
            self._tools_cache = response.get("tools", [])
        
        return self._tools_cache
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
            
        Returns:
            工具执行结果
            
        Raises:
            ValueError: 工具不存在或参数错误
            RuntimeError: 执行失败
        """
        endpoint = f"/mcp/tools/{tool_name}"
        
        try:
            response = await self.http.post(endpoint, json_data=arguments)
            
            # 检查是否有错误
            if response.get("isError"):
                error_msg = response.get("error", "未知错误")
                raise RuntimeError(f"工具执行失败: {error_msg}")
            
            # 返回结果
            return response.get("result", {})
        
        except ValueError as e:
            if "404" in str(e):
                raise ValueError(f"工具不存在: {tool_name}") from e
            raise
    
    # ============ 理解层工具 ============
    
    async def parse_user_intent(
        self,
        user_input: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        工具1: 解析用户意图
        
        Args:
            user_input: 用户输入的自然语言
            conversation_history: 对话历史
            
        Returns:
            意图识别结果
        """
        return await self.call_tool("parse_user_intent", {
            "user_input": user_input,
            "conversation_history": conversation_history or []
        })
    
    async def extract_entities(
        self,
        user_input: str
    ) -> Dict[str, Any]:
        """
        工具2: 提取实体
        
        Args:
            user_input: 用户输入
            
        Returns:
            提取的实体
        """
        return await self.call_tool("extract_entities", {
            "user_input": user_input
        })
    
    async def validate_requirements(
        self,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        工具3: 验证需求
        
        Args:
            requirements: 需求字典
            
        Returns:
            验证结果
        """
        return await self.call_tool("validate_requirements", {
            "requirements": requirements
        })
    
    # ============ 配置层工具 ============
    
    async def generate_model_config(
        self,
        intent: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        工具4: 生成模型配置 ⭐
        
        Args:
            intent: 用户意图
            entities: 提取的实体
            
        Returns:
            生成的配置
        """
        return await self.call_tool("generate_model_config", {
            "intent": intent,
            "entities": entities
        })
    
    async def suggest_parameters(
        self,
        basin_features: Dict[str, Any],
        model_type: str
    ) -> Dict[str, Any]:
        """
        工具5: 推荐参数
        
        Args:
            basin_features: 流域特征
            model_type: 模型类型
            
        Returns:
            推荐的参数
        """
        return await self.call_tool("suggest_parameters", {
            "basin_features": basin_features,
            "model_type": model_type
        })
    
    async def design_scenarios(
        self,
        analysis_objective: str,
        baseline_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        工具6: 设计情景
        
        Args:
            analysis_objective: 分析目标
            baseline_config: 基准配置
            
        Returns:
            设计的情景
        """
        return await self.call_tool("design_scenarios", {
            "analysis_objective": analysis_objective,
            "baseline_config": baseline_config
        })
    
    # ============ 分析层工具 ============
    
    async def interpret_results(
        self,
        simulation_results: Dict[str, Any],
        model_config: Dict[str, Any],
        user_objective: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        工具7: 解读结果
        
        Args:
            simulation_results: 模拟结果
            model_config: 模型配置
            user_objective: 用户目标
            
        Returns:
            结果解读
        """
        return await self.call_tool("interpret_results", {
            "simulation_results": simulation_results,
            "model_config": model_config,
            "user_objective": user_objective
        })
    
    async def diagnose_issues(
        self,
        poor_results: Dict[str, Any],
        model_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        工具8: 诊断问题
        
        Args:
            poor_results: 不佳的结果
            model_config: 模型配置
            
        Returns:
            诊断结果
        """
        return await self.call_tool("diagnose_issues", {
            "poor_results": poor_results,
            "model_config": model_config
        })
    
    async def compare_models(
        self,
        models_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        工具9: 对比模型
        
        Args:
            models_results: 多个模型的结果
            
        Returns:
            对比结果
        """
        return await self.call_tool("compare_models", {
            "models_results": models_results
        })
    
    # ============ 报告层工具 ============
    
    async def generate_narrative(
        self,
        section: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        工具10: 生成叙述
        
        Args:
            section: 章节名称
            context: 上下文数据
            
        Returns:
            生成的叙述
        """
        return await self.call_tool("generate_narrative", {
            "section": section,
            "context": context
        })
    
    async def create_executive_report(
        self,
        workflow_result: Dict[str, Any],
        model_config: Dict[str, Any],
        report_type: str = "executive"
    ) -> Dict[str, Any]:
        """
        工具11: 生成执行报告 ⭐
        
        Args:
            workflow_result: 工作流结果
            model_config: 模型配置
            report_type: 报告类型
            
        Returns:
            生成的报告
        """
        return await self.call_tool("create_executive_report", {
            "workflow_result": workflow_result,
            "model_config": model_config,
            "report_type": report_type
        })
    
    async def answer_questions(
        self,
        question: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        工具12: 智能问答
        
        Args:
            question: 用户问题
            context: 上下文
            
        Returns:
            回答
        """
        return await self.call_tool("answer_questions", {
            "question": question,
            "context": context
        })


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    async def test():
        print("=== HydroMind客户端测试 ===\n")
        
        client = HydroMindClient()
        
        # 测试健康检查
        print("1. 健康检查...")
        is_healthy = await client.health_check()
        print(f"   状态: {'✅ 健康' if is_healthy else '❌ 不可用'}\n")
        
        if not is_healthy:
            print("⚠️  服务器不可用，请先启动HydroMind服务器")
            return
        
        # 测试列出工具
        print("2. 列出工具...")
        tools = await client.list_tools()
        print(f"   找到 {len(tools)} 个工具:")
        for tool in tools[:3]:
            print(f"   - {tool['name']}: {tool['description']}")
        print()
        
        # 测试意图识别
        print("3. 测试意图识别...")
        try:
            result = await client.parse_user_intent(
                "我想建立HBV模型"
            )
            print(f"   意图: {result.get('action')}")
            print(f"   置信度: {result.get('confidence')}")
        except Exception as e:
            print(f"   ❌ 失败: {e}")
    
    # asyncio.run(test())
    print("HydroMind客户端已就绪")
