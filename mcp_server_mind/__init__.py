"""
HydroMind Agent - 水文认知智能体

基于大语言模型的水文建模认知层，提供：
- 自然语言理解和实体抽取
- 智能配置生成和参数推荐
- 结果解读和问题诊断
- 自然语言报告生成

与HydroCompute Agent（机理智能体）协同工作，构成完整的双智能体系统。
"""

__version__ = "1.0.0"
__all__ = [
    "HydroMindTools",
    "LLMBackend",
    "QwenBackend",
    "MockLLMBackend",
    "PromptTemplateManager",
    "KnowledgeBase",
]
