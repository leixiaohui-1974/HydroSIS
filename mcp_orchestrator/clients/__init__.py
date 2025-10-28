"""
MCP Clients - 双智能体通信客户端

提供HydroMind和HydroCompute的HTTP客户端
"""

from .hydromind_client import HydroMindClient
from .hydrocompute_client import HydroComputeClient
from .http_utils import MCPHttpClient

__all__ = [
    "HydroMindClient",
    "HydroComputeClient",
    "MCPHttpClient",
]
