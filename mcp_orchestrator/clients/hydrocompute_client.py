"""
HydroCompute Client - HydroCompute机理智能体客户端

提供对HydroCompute 18个计算工具的访问
"""

from typing import Dict, Any, List, Optional
from .http_utils import MCPHttpClient


class HydroComputeClient:
    """HydroCompute MCP客户端"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 300.0  # 模拟可能很慢，5分钟超时
    ):
        """
        初始化HydroCompute客户端
        
        Args:
            base_url: HydroCompute服务器地址
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
    
    # ============ 项目管理 ============
    
    async def create_project(
        self,
        user_id: str,
        project_name: str,
        description: Optional[str] = None,
        template: str = "basic"
    ) -> Dict[str, Any]:
        """
        创建项目
        
        Args:
            user_id: 用户ID
            project_name: 项目名称
            description: 项目描述
            template: 模板类型
            
        Returns:
            项目信息
        """
        return await self.call_tool("create_project", {
            "user_id": user_id,
            "project_name": project_name,
            "description": description,
            "template": template
        })
    
    async def list_projects(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """
        列出项目
        
        Args:
            user_id: 用户ID
            
        Returns:
            项目列表
        """
        return await self.call_tool("list_projects", {
            "user_id": user_id
        })
    
    async def get_project(
        self,
        project_id: str
    ) -> Dict[str, Any]:
        """
        获取项目详情
        
        Args:
            project_id: 项目ID
            
        Returns:
            项目详情
        """
        return await self.call_tool("get_project", {
            "project_id": project_id
        })
    
    # ============ 模拟运行 ============
    
    async def run_simulation(
        self,
        project_id: str,
        scenario_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        generate_report: bool = False
    ) -> Dict[str, Any]:
        """
        运行模拟
        
        Args:
            project_id: 项目ID
            scenario_ids: 情景ID列表
            start_date: 开始日期
            end_date: 结束日期
            generate_report: 是否生成报告
            
        Returns:
            模拟结果
        """
        args = {
            "project_id": project_id,
            "generate_report": generate_report
        }
        
        if scenario_ids:
            args["scenario_ids"] = scenario_ids
        if start_date:
            args["start_date"] = start_date
        if end_date:
            args["end_date"] = end_date
        
        return await self.call_tool("run_simulation", args)
    
    # ============ 参数率定 ============
    
    async def calibrate_parameters(
        self,
        project_id: str,
        target_metrics: Optional[Dict[str, float]] = None,
        max_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        参数率定
        
        Args:
            project_id: 项目ID
            target_metrics: 目标指标
            max_iterations: 最大迭代次数
            
        Returns:
            率定结果
        """
        return await self.call_tool("calibrate_parameters", {
            "project_id": project_id,
            "target_metrics": target_metrics,
            "max_iterations": max_iterations
        })
    
    # ============ 结果分析 ============
    
    async def analyze_results(
        self,
        project_id: str,
        run_id: str
    ) -> Dict[str, Any]:
        """
        分析结果
        
        Args:
            project_id: 项目ID
            run_id: 运行ID
            
        Returns:
            分析结果
        """
        return await self.call_tool("analyze_results", {
            "project_id": project_id,
            "run_id": run_id
        })
    
    # ============ 配置管理 ============
    
    async def configure_runoff_model(
        self,
        project_id: str,
        model_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        配置产流模型
        
        Args:
            project_id: 项目ID
            model_type: 模型类型
            parameters: 参数
            
        Returns:
            配置结果
        """
        return await self.call_tool("configure_runoff_model", {
            "project_id": project_id,
            "model_type": model_type,
            "parameters": parameters
        })
    
    async def configure_routing(
        self,
        project_id: str,
        routing_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        配置汇流方法
        
        Args:
            project_id: 项目ID
            routing_type: 汇流类型
            parameters: 参数
            
        Returns:
            配置结果
        """
        return await self.call_tool("configure_routing", {
            "project_id": project_id,
            "routing_type": routing_type,
            "parameters": parameters
        })


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    async def test():
        print("=== HydroCompute客户端测试 ===\n")
        
        client = HydroComputeClient()
        
        # 测试健康检查
        print("1. 健康检查...")
        is_healthy = await client.health_check()
        print(f"   状态: {'✅ 健康' if is_healthy else '❌ 不可用'}\n")
        
        if not is_healthy:
            print("⚠️  服务器不可用，请先启动HydroCompute服务器")
            return
        
        # 测试列出工具
        print("2. 列出工具...")
        tools = await client.list_tools()
        print(f"   找到 {len(tools)} 个工具:")
        for tool in tools[:3]:
            print(f"   - {tool['name']}: {tool['description']}")
    
    # asyncio.run(test())
    print("HydroCompute客户端已就绪")
