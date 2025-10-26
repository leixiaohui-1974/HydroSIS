"""HydroSIS MCP服务器 - Model Context Protocol封装"""

from .server import MCPServer, MCPTool, MCPToolResult
from .hydrosis_tools import HydroSISTools
from .auth import verify_token, require_permission, Permission
from .tasks import TaskManager, ProgressReporter

__all__ = [
    "MCPServer",
    "MCPTool",
    "MCPToolResult",
    "HydroSISTools",
    "verify_token",
    "require_permission",
    "Permission",
    "TaskManager",
    "ProgressReporter",
]

__version__ = "1.0.0"
