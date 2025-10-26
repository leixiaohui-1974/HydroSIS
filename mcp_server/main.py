"""MCP服务器主程序入口"""

import os
import sys
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import uvicorn
from mcp_server.server import mcp_server
from mcp_server.hydrosis_tools import HydroSISTools
from mcp_server.tasks import task_manager
from mcp_server.auth import AuthMiddleware

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_server.log')
    ]
)

logger = logging.getLogger(__name__)


def init_server():
    """初始化MCP服务器"""
    logger.info("=" * 60)
    logger.info("初始化HydroSIS MCP服务器")
    logger.info("=" * 60)
    
    # 配置数据根目录
    data_root = os.environ.get('DATA_ROOT', '/data')
    logger.info(f"数据根目录: {data_root}")
    
    # 初始化HydroSIS工具集
    hydrosis_tools = HydroSISTools(mcp_server, data_root=data_root)
    
    # 添加认证中间件
    app = mcp_server.get_app()
    # app.add_middleware(AuthMiddleware)  # 可选：启用认证中间件
    
    # 添加任务管理路由
    register_task_routes(app)
    
    logger.info(f"已注册 {len(mcp_server.registry.tools)} 个MCP工具")
    logger.info("MCP服务器初始化完成")
    logger.info("=" * 60)
    
    return app


def register_task_routes(app):
    """注册任务管理相关路由"""
    from fastapi import HTTPException
    
    @app.post("/tasks/submit")
    async def submit_task(
        tool_name: str,
        arguments: dict,
        user_id: str,
        callback_url: str = None
    ):
        """提交异步任务"""
        try:
            # 验证工具是否存在
            if tool_name not in mcp_server.registry.tools:
                raise HTTPException(
                    status_code=404,
                    detail=f"工具不存在: {tool_name}"
                )
            
            # 创建任务
            task_id = await task_manager.create_task(
                tool_name=tool_name,
                arguments=arguments,
                user_id=user_id,
                callback_url=callback_url
            )
            
            # 提交任务执行
            async def executor(args, progress_reporter):
                return await mcp_server.registry.call_tool(tool_name, args)
            
            await task_manager.submit_task(task_id, executor)
            
            return {
                "task_id": task_id,
                "status": "submitted",
                "message": "任务已提交"
            }
        
        except Exception as e:
            logger.error(f"提交任务失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/tasks/{task_id}")
    async def get_task(task_id: str):
        """获取任务状态"""
        try:
            return await task_manager.get_task_status(task_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    
    @app.get("/tasks/user/{user_id}")
    async def list_user_tasks(user_id: str, status: str = None):
        """列出用户的任务"""
        from mcp_server.tasks import TaskStatus
        
        status_filter = None
        if status:
            try:
                status_filter = TaskStatus(status)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效的状态值: {status}"
                )
        
        tasks = await task_manager.list_user_tasks(user_id, status_filter)
        return {
            "user_id": user_id,
            "tasks": tasks,
            "count": len(tasks)
        }
    
    @app.delete("/tasks/{task_id}")
    async def cancel_task(task_id: str):
        """取消任务"""
        try:
            await task_manager.cancel_task(task_id)
            return {
                "task_id": task_id,
                "status": "cancelled",
                "message": "任务已取消"
            }
        except KeyError:
            raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")
    
    logger.info("任务管理路由已注册")


def main():
    """主函数"""
    # 初始化服务器
    app = init_server()
    
    # 获取配置
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 8080))
    workers = int(os.environ.get('WORKERS', 4))
    
    # 启动服务器
    logger.info(f"启动MCP服务器: http://{host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
