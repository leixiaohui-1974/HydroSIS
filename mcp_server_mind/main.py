"""
HydroMind FastAPI服务器 - 主程序入口

提供HTTP API访问HydroMind的12个认知工具
"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, Any

# 检查是否有FastAPI
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    print("⚠️  警告: 未安装FastAPI，使用简化模式")
    print("   安装: pip install fastapi uvicorn")

try:
    from .server import HydroMindServer
    from .cognitive_tools import CognitiveTools
except ImportError:
    from server import HydroMindServer
    from cognitive_tools import CognitiveTools

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def create_app() -> "FastAPI":
    """创建FastAPI应用"""
    
    if not HAS_FASTAPI:
        raise RuntimeError("FastAPI未安装，无法创建应用")
    
    # 创建FastAPI实例
    app = FastAPI(
        title="HydroMind Agent API",
        description="水文认知智能体 - 基于LLM的水文建模辅助系统",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 初始化HydroMind服务器
    hydromind = HydroMindServer()
    tools = hydromind.cognitive_tools
    
    # ========== 健康检查 ==========
    
    @app.get("/")
    async def root():
        """根路径"""
        return {
            "service": "HydroMind Agent",
            "version": "1.0.0",
            "description": "水文认知智能体",
            "tools_count": len(tools.get_tool_list()),
            "docs": "/docs"
        }
    
    @app.get("/health")
    async def health():
        """健康检查"""
        return {
            "status": "healthy",
            "service": "HydroMind",
            "llm_backend": tools.llm.__class__.__name__,
            "llm_available": tools.llm.is_available()
        }
    
    # ========== 工具列表 ==========
    
    @app.get("/mcp/tools")
    async def list_tools():
        """列出所有工具"""
        tool_list = tools.get_tool_list()
        
        # 按类别分组
        by_category = {}
        for tool in tool_list:
            category = tool['category']
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(tool)
        
        return {
            "tools": tool_list,
            "count": len(tool_list),
            "categories": by_category
        }
    
    @app.get("/mcp/tools/categories")
    async def list_categories():
        """列出工具类别"""
        tool_list = tools.get_tool_list()
        categories = {}
        
        for tool in tool_list:
            cat = tool['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool['name'])
        
        return {
            "categories": categories,
            "count": len(categories)
        }
    
    # ========== 工具调用 ==========
    
    @app.post("/mcp/tools/{tool_name}")
    async def call_tool(tool_name: str, arguments: Dict[str, Any]):
        """
        调用工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数（JSON）
            
        Returns:
            工具执行结果
        """
        # 验证工具是否存在
        tool_names = [t['name'] for t in tools.get_tool_list()]
        if tool_name not in tool_names:
            raise HTTPException(
                status_code=404,
                detail=f"工具不存在: {tool_name}"
            )
        
        # 调用工具
        try:
            # 获取工具方法
            tool_method = getattr(tools, tool_name, None)
            if tool_method is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"工具方法未找到: {tool_name}"
                )
            
            # 执行工具
            result = await tool_method(arguments)
            
            return {
                "result": result,
                "isError": False,
                "metadata": {
                    "tool_name": tool_name,
                    "timestamp": __import__('datetime').datetime.now().isoformat()
                }
            }
        
        except Exception as e:
            logger.error(f"工具执行失败 {tool_name}: {e}")
            return {
                "error": str(e),
                "isError": True,
                "metadata": {
                    "tool_name": tool_name,
                    "timestamp": __import__('datetime').datetime.now().isoformat()
                }
            }
    
    # ========== 便捷端点 ==========
    
    @app.post("/understand")
    async def understand_request(user_input: str):
        """
        理解用户需求（快捷端点）
        
        组合调用：意图识别 + 实体抽取 + 需求验证
        """
        try:
            # 1. 意图识别
            intent = await tools.parse_user_intent({
                "user_input": user_input
            })
            
            # 2. 实体抽取
            entities = await tools.extract_entities({
                "user_input": user_input
            })
            
            # 3. 需求验证
            validation = await tools.validate_requirements({
                "requirements": {
                    "intent": intent,
                    "entities": entities
                }
            })
            
            return {
                "intent": intent,
                "entities": entities,
                "validation": validation,
                "is_valid": validation.get("is_valid", False)
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/generate_config")
    async def generate_config_endpoint(
        intent: Dict[str, Any],
        entities: Dict[str, Any]
    ):
        """
        生成配置（快捷端点）
        """
        try:
            config = await tools.generate_model_config({
                "intent": intent,
                "entities": entities
            })
            return config
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/analyze_results")
    async def analyze_results_endpoint(
        simulation_results: Dict[str, Any],
        model_config: Dict[str, Any]
    ):
        """
        分析结果（快捷端点）
        
        组合调用：解读结果 + 诊断问题（如果需要）
        """
        try:
            # 解读结果
            interpretation = await tools.interpret_results({
                "simulation_results": simulation_results,
                "model_config": model_config
            })
            
            # 如果表现不佳，进行诊断
            assessment = interpretation.get("overall_assessment", {})
            if assessment.get("performance_level") in ["fair", "poor"]:
                diagnosis = await tools.diagnose_issues({
                    "poor_results": simulation_results,
                    "model_config": model_config
                })
                interpretation["diagnosis"] = diagnosis
            
            return interpretation
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    logger.info("✅ HydroMind FastAPI应用创建成功")
    logger.info(f"   工具数: {len(tools.get_tool_list())}")
    logger.info(f"   LLM后端: {tools.llm.__class__.__name__}")
    
    return app


def main():
    """主函数"""
    if not HAS_FASTAPI:
        print("❌ FastAPI未安装")
        print("   请运行: pip install fastapi uvicorn")
        sys.exit(1)
    
    # 创建应用
    app = create_app()
    
    # 获取配置
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8081))
    
    # 启动服务器
    logger.info("=" * 60)
    logger.info("启动HydroMind服务器")
    logger.info(f"地址: http://{host}:{port}")
    logger.info(f"文档: http://{host}:{port}/docs")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
