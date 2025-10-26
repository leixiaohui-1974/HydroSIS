"""CLI命令行接口

提供hydrosis命令行工具，支持模块和工作流的执行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

from hydrosis.modules.base import get_registry
from hydrosis.workflow_engine import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowTemplates
)


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """HydroSIS 模块化水文模拟系统命令行工具"""
    pass


@cli.group()
def module():
    """模块管理命令"""
    pass


@module.command("list")
def module_list():
    """列出所有可用模块"""
    registry = get_registry()
    modules = registry.list_modules()
    
    click.echo("可用模块:")
    for module_id in modules:
        click.echo(f"  - {module_id}")


@module.command("info")
@click.argument("module_id")
def module_info(module_id: str):
    """显示模块详细信息"""
    registry = get_registry()
    
    try:
        module = registry.get_or_create_module(module_id)
        info = module.get_info()
        
        click.echo(f"模块ID: {info['module_id']}")
        click.echo(f"名称: {info['name']}")
        click.echo(f"描述: {info['description']}")
        click.echo(f"版本: {info['version']}")
        
        if info.get('author'):
            click.echo(f"作者: {info['author']}")
        
        click.echo("\n输入参数:")
        for prop_name, prop_schema in info['input_schema'].get('properties', {}).items():
            required = prop_name in info['input_schema'].get('required', [])
            req_mark = "*" if required else ""
            desc = prop_schema.get('description', '')
            click.echo(f"  - {prop_name}{req_mark}: {desc}")
        
    except KeyError:
        click.echo(f"错误: 模块 {module_id} 不存在", err=True)
        sys.exit(1)


@module.command("run")
@click.argument("module_id")
@click.option("--input", "-i", "input_file", type=click.Path(exists=True),
              help="输入参数JSON文件")
@click.option("--param", "-p", multiple=True,
              help="参数键值对，格式: key=value")
@click.option("--output-dir", "-o", type=click.Path(),
              help="输出目录")
def module_run(module_id: str, input_file: Optional[str], param: tuple, output_dir: Optional[str]):
    """执行模块
    
    示例:
      hydrosis module run terrain --param dem_path=data/dem.tif --output-dir results/
    """
    registry = get_registry()
    
    try:
        module = registry.get_or_create_module(module_id)
        
        # 构建输入参数
        inputs = {}
        
        # 从文件加载
        if input_file:
            with open(input_file) as f:
                inputs = json.load(f)
        
        # 从命令行参数覆盖
        for p in param:
            key, value = p.split('=', 1)
            # 尝试解析为JSON
            try:
                inputs[key] = json.loads(value)
            except json.JSONDecodeError:
                inputs[key] = value
        
        # 添加输出目录
        if output_dir:
            inputs['output_dir'] = output_dir
        
        # 执行模块
        click.echo(f"执行模块: {module_id}")
        click.echo(f"输入参数: {inputs}")
        
        output = module.run(inputs)
        
        click.echo(f"\n执行成功!")
        click.echo(f"输出: {json.dumps(output.to_dict(), indent=2, ensure_ascii=False)}")
        
    except KeyError:
        click.echo(f"错误: 模块 {module_id} 不存在", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


@cli.group()
def workflow():
    """工作流管理命令"""
    pass


@workflow.command("list")
def workflow_list():
    """列出所有可用工作流模板"""
    templates = WorkflowTemplates.list_templates()
    
    click.echo("可用工作流模板:")
    for template_id in templates:
        click.echo(f"  - {template_id}")


@workflow.command("info")
@click.argument("workflow_id")
def workflow_info(workflow_id: str):
    """显示工作流详细信息"""
    try:
        template = WorkflowTemplates.get_template(workflow_id)
        workflow_data = template.get('workflow', template)
        
        click.echo(f"工作流ID: {workflow_data['id']}")
        click.echo(f"名称: {workflow_data['name']}")
        click.echo(f"描述: {workflow_data.get('description', '')}")
        click.echo(f"版本: {workflow_data.get('version', '')}")
        
        click.echo(f"\n步骤 ({len(workflow_data.get('steps', []))}):")
        for step in workflow_data.get('steps', []):
            deps = f" (依赖: {', '.join(step.get('depends_on', []))})" if step.get('depends_on') else ""
            click.echo(f"  {step['id']}: {step['module']}{deps}")
        
    except KeyError:
        click.echo(f"错误: 工作流 {workflow_id} 不存在", err=True)
        sys.exit(1)


@workflow.command("run")
@click.argument("workflow_id")
@click.option("--config", "-c", "config_file", type=click.Path(exists=True),
              help="工作流配置文件（YAML）")
@click.option("--param", "-p", multiple=True,
              help="参数键值对，格式: key=value")
@click.option("--output-dir", "-o", type=click.Path(),
              help="输出目录")
def workflow_run(workflow_id: str, config_file: Optional[str], param: tuple, output_dir: Optional[str]):
    """执行工作流
    
    示例:
      hydrosis workflow run pour_points_only --param dem_path=data/dem.tif
    """
    registry = get_registry()
    engine = WorkflowEngine(registry)
    
    try:
        # 加载工作流定义
        if config_file:
            workflow = WorkflowDefinition.from_yaml(Path(config_file))
        else:
            template = WorkflowTemplates.get_template(workflow_id)
            workflow = WorkflowDefinition.from_dict(template)
        
        # 构建参数
        parameters = {}
        
        for p in param:
            key, value = p.split('=', 1)
            try:
                parameters[key] = json.loads(value)
            except json.JSONDecodeError:
                parameters[key] = value
        
        if output_dir:
            parameters['output_dir'] = output_dir
        
        # 执行工作流
        click.echo(f"执行工作流: {workflow.name}")
        click.echo(f"参数: {parameters}\n")
        
        def progress_callback(run, step_result):
            """进度回调"""
            percent = run.progress_percent()
            click.echo(f"[{percent:.1f}%] 步骤 {step_result.step_id}: {step_result.status}")
        
        run = engine.execute(workflow, parameters, progress_callback)
        
        click.echo(f"\n工作流执行{run.status}")
        click.echo(f"运行ID: {run.run_id}")
        click.echo(f"耗时: {run.duration_seconds():.2f} 秒")
        
        if run.status == "completed":
            click.echo(f"\n输出:")
            click.echo(json.dumps(run.outputs, indent=2, ensure_ascii=False))
        else:
            click.echo(f"\n错误: {run.error}", err=True)
            sys.exit(1)
        
    except KeyError:
        click.echo(f"错误: 工作流 {workflow_id} 不存在", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"错误: {e}", err=True)
        sys.exit(1)


@cli.group()
def config():
    """配置管理命令"""
    pass


@config.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
def config_validate(config_file: str):
    """验证配置文件"""
    try:
        workflow = WorkflowDefinition.from_yaml(Path(config_file))
        click.echo(f"配置文件有效")
        click.echo(f"工作流: {workflow.name}")
        click.echo(f"步骤数: {len(workflow.steps)}")
    except Exception as e:
        click.echo(f"配置文件无效: {e}", err=True)
        sys.exit(1)


def create_cli():
    """创建CLI应用"""
    return cli


def main():
    """CLI入口点"""
    cli()


if __name__ == "__main__":
    main()
