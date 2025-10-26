"""
测试MCP服务器
验证服务器能否正常启动和响应
"""

import sys
import time
import subprocess
import requests
from pathlib import Path

def test_mcp_server():
    """测试MCP服务器"""
    
    print("="*60)
    print("HydroSIS MCP服务器测试")
    print("="*60)
    
    # 测试导入
    print("\n1. 测试模块导入...")
    try:
        from mcp_server.server import mcp_server, MCPTool
        from mcp_server.hydrosis_tools import HydroSISTools
        from mcp_server.auth import Permission, Role
        from mcp_server.tasks import TaskManager
        print("✅ 所有模块导入成功")
    except Exception as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    
    # 测试工具注册
    print("\n2. 测试工具注册...")
    try:
        data_root = "/tmp/hydrosis-test-data"
        Path(data_root).mkdir(parents=True, exist_ok=True)
        
        hydrosis_tools = HydroSISTools(mcp_server, data_root=data_root)
        tools_count = len(mcp_server.registry.tools)
        
        print(f"✅ 已注册 {tools_count} 个工具")
        
        # 列出所有工具
        print("\n   工具列表:")
        for name in sorted(mcp_server.registry.tools.keys()):
            tool_info = mcp_server.registry.tools[name]
            print(f"   - {name}: {tool_info['description']}")
    
    except Exception as e:
        print(f"❌ 工具注册失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试工具调用
    print("\n3. 测试工具调用...")
    try:
        # 测试创建项目
        import asyncio
        
        async def test_create_project():
            result = await mcp_server.registry.call_tool("create_project", {
                "user_id": "test_user",
                "project_name": "测试项目",
                "description": "这是一个测试项目"
            })
            return result
        
        result = asyncio.run(test_create_project())
        print(f"✅ 工具调用成功")
        print(f"   项目ID: {result.get('project_id', 'N/A')}")
    
    except Exception as e:
        print(f"❌ 工具调用失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试权限系统
    print("\n4. 测试权限系统...")
    try:
        from mcp_server.auth import ROLE_PERMISSIONS, Role, Permission
        
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        viewer_perms = ROLE_PERMISSIONS[Role.VIEWER]
        
        print(f"✅ 权限系统正常")
        print(f"   管理员权限数: {len(admin_perms)}")
        print(f"   查看者权限数: {len(viewer_perms)}")
    
    except Exception as e:
        print(f"❌ 权限系统测试失败: {e}")
        return False
    
    # 测试任务管理器
    print("\n5. 测试任务管理器...")
    try:
        from mcp_server.tasks import TaskManager
        
        task_manager = TaskManager()
        
        async def test_task():
            task_id = await task_manager.create_task(
                tool_name="test_tool",
                arguments={"test": "value"},
                user_id="test_user"
            )
            
            status = await task_manager.get_task_status(task_id)
            return status
        
        status = asyncio.run(test_task())
        print(f"✅ 任务管理器正常")
        print(f"   任务ID: {status['task_id']}")
        print(f"   状态: {status['status']}")
    
    except Exception as e:
        print(f"❌ 任务管理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("✅ 所有测试通过!")
    print("="*60)
    print("\n可以启动服务器:")
    print("  python -m mcp_server.main")
    print("\n然后访问:")
    print("  http://localhost:8080/health")
    print("  http://localhost:8080/mcp/tools")
    print("="*60)
    
    return True


if __name__ == "__main__":
    success = test_mcp_server()
    sys.exit(0 if success else 1)
