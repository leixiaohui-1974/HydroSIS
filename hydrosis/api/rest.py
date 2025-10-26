"""REST API接口

提供基于FastAPI的REST API，支持模块和工作流的调用。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hydrosis.modules.base import get_registry, ModuleMetadata
from hydrosis.workflow_engine import WorkflowEngine, WorkflowDefinition, WorkflowTemplates

logger = logging.getLogger(__name__)


# Pydantic模型定义
class ModuleExecuteRequest(BaseModel):
    """模块执行请求"""
    inputs: Dict[str, Any]
    task_id: str | None = None


class ModuleExecuteResponse(BaseModel):
    """模块执行响应"""
    task_id: str
    module_id: str
    status: str
    outputs: Dict[str, Any] | None = None
    error: str | None = None


class WorkflowExecuteRequest(BaseModel):
    """工作流执行请求"""
    parameters: Dict[str, Any] | None = None


class WorkflowExecuteResponse(BaseModel):
    """工作流执行响应"""
    run_id: str
    workflow_id: str
    status: str
    outputs: Dict[str, Any] | None = None
    error: str | None = None


def create_api_app() -> FastAPI:
    """创建REST API应用
    
    Returns:
        FastAPI应用实例
    """
    app = FastAPI(
        title="HydroSIS Modular API",
        description="模块化水文模拟系统API",
        version="1.0.0"
    )
    
    registry = get_registry()
    workflow_engine = WorkflowEngine(registry)
    
    @app.get("/")
    def root():
        """API根路径"""
        return {
            "name": "HydroSIS Modular API",
            "version": "1.0.0",
            "description": "模块化水文模拟系统",
            "endpoints": {
                "modules": "/api/v1/modules",
                "workflows": "/api/v1/workflows",
                "docs": "/docs"
            }
        }
    
    # ========== 模块相关端点 ==========
    
    @app.get("/api/v1/modules", tags=["modules"])
    def list_modules() -> Dict[str, List[str]]:
        """列出所有可用模块"""
        modules = registry.list_modules()
        return {"modules": modules}
    
    @app.get("/api/v1/modules/{module_id}", tags=["modules"])
    def get_module_info(module_id: str) -> Dict[str, Any]:
        """获取模块详细信息"""
        try:
            module = registry.get_or_create_module(module_id)
            return module.get_info()
        except KeyError:
            raise HTTPException(status_code=404, detail=f"模块不存在: {module_id}")
    
    @app.get("/api/v1/modules/{module_id}/metadata", tags=["modules"])
    def get_module_metadata(module_id: str) -> Dict[str, Any]:
        """获取模块元数据"""
        module_class = registry.get_module_class(module_id)
        if module_class is None:
            raise HTTPException(status_code=404, detail=f"模块不存在: {module_id}")
        
        metadata = module_class.metadata()
        return {
            "module_id": metadata.module_id,
            "name": metadata.name,
            "description": metadata.description,
            "version": metadata.version,
            "author": metadata.author,
            "input_schema": metadata.input_schema,
            "output_schema": metadata.output_schema
        }
    
    @app.post("/api/v1/modules/{module_id}/execute", tags=["modules"])
    def execute_module(
        module_id: str,
        request: ModuleExecuteRequest
    ) -> ModuleExecuteResponse:
        """执行模块"""
        try:
            module = registry.get_or_create_module(module_id)
            output = module.run(request.inputs, task_id=request.task_id)
            
            return ModuleExecuteResponse(
                task_id=request.task_id or "unknown",
                module_id=module_id,
                status="completed",
                outputs=output.to_dict() if hasattr(output, 'to_dict') else {}
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"模块不存在: {module_id}")
        except Exception as e:
            logger.error(f"模块执行失败: {e}")
            return ModuleExecuteResponse(
                task_id=request.task_id or "unknown",
                module_id=module_id,
                status="failed",
                error=str(e)
            )
    
    # ========== 工作流相关端点 ==========
    
    @app.get("/api/v1/workflows/templates", tags=["workflows"])
    def list_workflow_templates() -> Dict[str, List[str]]:
        """列出所有工作流模板"""
        templates = WorkflowTemplates.list_templates()
        return {"templates": templates}
    
    @app.get("/api/v1/workflows/templates/{template_id}", tags=["workflows"])
    def get_workflow_template(template_id: str) -> Dict[str, Any]:
        """获取工作流模板定义"""
        try:
            template = WorkflowTemplates.get_template(template_id)
            return template
        except KeyError:
            raise HTTPException(status_code=404, detail=f"模板不存在: {template_id}")
    
    @app.post("/api/v1/workflows/{workflow_id}/execute", tags=["workflows"])
    def execute_workflow(
        workflow_id: str,
        request: WorkflowExecuteRequest
    ) -> WorkflowExecuteResponse:
        """执行工作流"""
        try:
            # 加载工作流定义
            template = WorkflowTemplates.get_template(workflow_id)
            workflow = WorkflowDefinition.from_dict(template)
            
            # 执行工作流
            run = workflow_engine.execute(workflow, request.parameters)
            
            return WorkflowExecuteResponse(
                run_id=run.run_id,
                workflow_id=run.workflow_id,
                status=run.status,
                outputs=run.outputs,
                error=run.error
            )
        except KeyError:
            raise HTTPException(status_code=404, detail=f"工作流不存在: {workflow_id}")
        except Exception as e:
            logger.error(f"工作流执行失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/workflows/runs/{run_id}", tags=["workflows"])
    def get_workflow_run(run_id: str) -> Dict[str, Any]:
        """获取工作流执行状态"""
        run = workflow_engine.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"运行不存在: {run_id}")
        
        return {
            "run_id": run.run_id,
            "workflow_id": run.workflow_id,
            "status": run.status,
            "progress_percent": run.progress_percent(),
            "duration_seconds": run.duration_seconds(),
            "step_results": {
                step_id: {
                    "status": result.status,
                    "duration_seconds": result.duration_seconds(),
                    "error": result.error
                }
                for step_id, result in run.step_results.items()
            },
            "outputs": run.outputs,
            "error": run.error
        }
    
    return app


def main():
    """启动API服务器"""
    import uvicorn
    
    app = create_api_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
