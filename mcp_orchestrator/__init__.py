"""
HydroSIS Twin-Agent Orchestrator

双智能体协调器，统一编排 HydroMind (认知) 和 HydroCompute (机理) 智能体
"""

__version__ = "1.0.0"

from .twin_agent_coordinator import TwinAgentCoordinator
from .conversation_manager import ConversationManager

__all__ = ["TwinAgentCoordinator", "ConversationManager"]
