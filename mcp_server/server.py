"""MCP服务器核心实现"""

from typing import Dict, List, Any, Optional, Callable
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import logging
import traceback
from datetime import datetime

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPTool(BaseModel):
    """MCP工具定义"""
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    inputSchema: Dict[str, Any] = Field(..., description="输入参数JSON Schema")
    category: Optional[str] = Field(None, description="工具分类")
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "create_project",
                "description": "创建新的水文模拟项目",
                "category": "project_management",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "project_name": {"type": "string"},
                        "description": {"type": "string"}
                    },
                    "required": ["user_id", "project_name"]
                }
            }
        }


class MCPContent(BaseModel):
    """MCP内容块"""
    type: str = Field(..., description="内容类型: text, image, resource")
    text: Optional[str] = Field(None, description="文本内容")
    data: Optional[str] = Field(None, description="Base64编码的数据")
    mimeType: Optional[str] = Field(None, description="MIME类型")


class MCPToolResult(BaseModel):
    """MCP工具执行结果"""
    content: List[MCPContent] = Field(..., description="结果内容列表")
    isError: bool = Field(False, description="是否为错误")
    metadata: Optional[Dict[str, Any]] = Field(None, description="额外元数据")


class MCPToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_schemas: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        func: Callable,
        schema: Dict[str, Any],
        description: str = "",
        category: str = "general"
    ):
        """注册工具"""
        self.tools[name] = {
            "func": func,
            "description": description,
            "category": category
        }
        self.tool_schemas[name] = schema
        logger.info(f"已注册工具: {name} (分类: {category})")
    
    def get_tool_definition(self, name: str) -> MCPTool:
        """获取工具定义"""
        if name not in self.tools:
            raise KeyError(f"工具不存在: {name}")
        
        tool_info = self.tools[name]
        return MCPTool(
            name=name,
            description=tool_info["description"],
            inputSchema=self.tool_schemas[name],
            category=tool_info.get("category", "general")
        )
    
    def list_tools(self) -> List[MCPTool]:
        """列出所有工具"""
        return [self.get_tool_definition(name) for name in self.tools.keys()]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        """调用工具"""
        if name not in self.tools:
            raise KeyError(f"工具不存在: {name}")
        
        func = self.tools[name]["func"]
        return await func(arguments)


class MCPServer:
    """MCP服务器实现"""
    
    def __init__(
        self,
        title: str = "HydroSIS MCP Server",
        version: str = "1.0.0",
        description: str = "分布式水文模拟MCP服务"
    ):
        self.app = FastAPI(
            title=title,
            version=version,
            description=description
        )
        self.registry = MCPToolRegistry()
        self._setup_middleware()
        self._register_routes()
    
    def _setup_middleware(self):
        """配置中间件"""
        # CORS配置
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # 生产环境应该配置具体域名
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # 请求日志中间件
        @self.app.middleware("http")
        async def log_requests(request: Request, call_next):
            start_time = datetime.now()
            response = await call_next(request)
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info(
                f"{request.method} {request.url.path} "
                f"- {response.status_code} - {duration:.3f}s"
            )
            return response
    
    def _register_routes(self):
        """注册MCP标准路由"""
        
        @self.app.get("/")
        async def root():
            """根路径"""
            return {
                "name": "HydroSIS MCP Server",
                "version": "1.0.0",
                "protocol": "MCP",
                "status": "running"
            }
        
        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "tools_count": len(self.registry.tools)
            }
        
        @self.app.get("/ready")
        async def readiness_check():
            """就绪检查"""
            return {
                "ready": True,
                "timestamp": datetime.now().isoformat()
            }
        
        @self.app.get("/mcp/tools", response_model=Dict[str, List[MCPTool]])
        async def list_tools():
            """
            列出所有可用工具
            
            返回格式符合MCP协议规范
            """
            try:
                tools = self.registry.list_tools()
                return {"tools": tools}
            except Exception as e:
                logger.error(f"列出工具失败: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/mcp/tools/categories")
        async def list_tool_categories():
            """列出工具分类"""
            categories = {}
            for name, info in self.registry.tools.items():
                category = info.get("category", "general")
                if category not in categories:
                    categories[category] = []
                categories[category].append(name)
            
            return {"categories": categories}
        
        @self.app.post(
            "/mcp/tools/{tool_name}",
            response_model=MCPToolResult
        )
        async def call_tool(
            tool_name: str,
            arguments: Dict[str, Any]
        ):
            """
            调用指定工具
            
            Args:
                tool_name: 工具名称
                arguments: 工具参数
            
            Returns:
                MCPToolResult: 工具执行结果
            """
            try:
                # 验证工具是否存在
                if tool_name not in self.registry.tools:
                    raise HTTPException(
                        status_code=404,
                        detail=f"工具不存在: {tool_name}"
                    )
                
                # 记录调用
                logger.info(f"调用工具: {tool_name}")
                logger.debug(f"参数: {json.dumps(arguments, ensure_ascii=False)}")
                
                # 执行工具
                result = await self.registry.call_tool(tool_name, arguments)
                
                # 包装结果
                if isinstance(result, MCPToolResult):
                    return result
                else:
                    # 自动包装为文本结果
                    return MCPToolResult(
                        content=[
                            MCPContent(
                                type="text",
                                text=json.dumps(result, ensure_ascii=False, indent=2)
                            )
                        ],
                        isError=False,
                        metadata={
                            "tool_name": tool_name,
                            "timestamp": datetime.now().isoformat()
                        }
                    )
            
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"工具执行失败: {tool_name}")
                logger.error(traceback.format_exc())
                
                return MCPToolResult(
                    content=[
                        MCPContent(
                            type="text",
                            text=f"错误: {str(e)}\n\n{traceback.format_exc()}"
                        )
                    ],
                    isError=True,
                    metadata={
                        "tool_name": tool_name,
                        "error_type": type(e).__name__,
                        "timestamp": datetime.now().isoformat()
                    }
                )
        
        @self.app.get("/mcp/tools/{tool_name}/schema")
        async def get_tool_schema(tool_name: str):
            """获取工具的JSON Schema"""
            try:
                tool_def = self.registry.get_tool_definition(tool_name)
                return {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "schema": tool_def.inputSchema
                }
            except KeyError:
                raise HTTPException(
                    status_code=404,
                    detail=f"工具不存在: {tool_name}"
                )
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """全局异常处理"""
            logger.error(f"未处理的异常: {exc}")
            logger.error(traceback.format_exc())
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "path": str(request.url)
                }
            )
    
    def register_tool(
        self,
        name: str,
        func: Callable,
        schema: Dict[str, Any],
        description: str = "",
        category: str = "general"
    ):
        """
        注册工具到MCP服务器
        
        Args:
            name: 工具名称
            func: 异步函数，接收Dict参数
            schema: JSON Schema定义
            description: 工具描述
            category: 工具分类
        """
        self.registry.register(name, func, schema, description, category)
    
    def get_app(self) -> FastAPI:
        """获取FastAPI应用实例"""
        return self.app


# 创建全局MCP服务器实例
mcp_server = MCPServer()
