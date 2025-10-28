"""
LLM Backend - 大语言模型后端接口层

支持多种LLM后端：
1. QwenBackend - 阿里千问
2. MockLLMBackend - 测试/开发用Mock
3. 可扩展其他模型（GPT、本地模型等）
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass


@dataclass
class LLMMessage:
    """LLM消息格式"""
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    """LLM响应格式"""
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None


class LLMBackend(ABC):
    """LLM后端抽象基类"""
    
    @abstractmethod
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """生成补全"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """检查后端是否可用"""
        pass


class QwenBackend(LLMBackend):
    """阿里千问后端"""
    
    DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-max",
        base_url: Optional[str] = None,
        timeout: float = 60.0
    ):
        """
        初始化千问后端
        
        Args:
            api_key: API密钥（如果不提供，从环境变量QWEN_API_KEY读取）
            model: 模型名称（qwen-max, qwen-plus, qwen-turbo等）
            base_url: API端点（可选）
            timeout: 请求超时时间
        """
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.model = model
        self.base_url = base_url or self.DEFAULT_ENDPOINT
        self.timeout = timeout
        
        if not self.api_key:
            print("⚠️  警告: 未设置QWEN_API_KEY，千问后端将不可用")
    
    def is_available(self) -> bool:
        """检查千问后端是否可用"""
        return self.api_key is not None
    
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        调用千问API生成补全
        
        Args:
            messages: 消息列表
            temperature: 温度参数（0-2）
            max_tokens: 最大token数
            **kwargs: 其他参数
            
        Returns:
            LLMResponse对象
        """
        if not self.is_available():
            raise RuntimeError("千问后端不可用，请设置QWEN_API_KEY环境变量")
        
        # 构建请求
        payload = {
            "model": self.model,
            "messages": [
                {"role": msg.role, "content": msg.content}
                for msg in messages
            ],
            "temperature": temperature,
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        # 添加额外参数
        payload.update(kwargs)
        
        # 发送请求
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw_body = response.read().decode("utf-8")
                body = json.loads(raw_body)
        except urllib.error.URLError as e:
            raise RuntimeError(f"千问API请求失败: {e}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"千问API响应解析失败: {e}")
        
        # 解析响应
        if "error" in body:
            error_msg = body["error"].get("message", "未知错误")
            raise RuntimeError(f"千问API错误: {error_msg}")
        
        choices = body.get("choices", [])
        if not choices:
            raise RuntimeError("千问API未返回choices")
        
        message = choices[0].get("message", {})
        content = message.get("content", "")
        
        usage = body.get("usage", {})
        
        return LLMResponse(
            content=content,
            model=self.model,
            usage=usage,
            metadata={"raw_response": body}
        )


class MockLLMBackend(LLMBackend):
    """
    Mock LLM后端 - 用于测试和开发
    
    返回预定义的响应或基于规则的简单回复
    """
    
    def __init__(self, mock_responses: Optional[Dict[str, str]] = None):
        """
        Args:
            mock_responses: 预定义响应字典，key为关键词，value为响应
        """
        self.mock_responses = mock_responses or {}
        self.call_count = 0
    
    def is_available(self) -> bool:
        """Mock后端始终可用"""
        return True
    
    async def complete(
        self,
        messages: List[LLMMessage],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        返回Mock响应
        
        策略：
        1. 如果消息中包含mock_responses的关键词，返回对应响应
        2. 否则根据消息类型返回通用响应
        """
        self.call_count += 1
        
        # 获取用户消息
        user_messages = [msg.content for msg in messages if msg.role == "user"]
        last_user_msg = user_messages[-1] if user_messages else ""
        
        # 检查是否有匹配的mock响应
        for keyword, response in self.mock_responses.items():
            if keyword.lower() in last_user_msg.lower():
                return LLMResponse(
                    content=response,
                    model="mock-llm",
                    usage={"total_tokens": len(response.split())}
                )
        
        # 生成基于规则的响应
        content = self._generate_rule_based_response(last_user_msg, messages)
        
        return LLMResponse(
            content=content,
            model="mock-llm",
            usage={"total_tokens": len(content.split())},
            metadata={"call_count": self.call_count}
        )
    
    def _generate_rule_based_response(
        self,
        user_input: str,
        messages: List[LLMMessage]
    ) -> str:
        """基于规则生成响应"""
        
        # 检查是否是JSON格式请求（工具调用）
        system_msg = next((m.content for m in messages if m.role == "system"), "")
        
        if "JSON" in system_msg or "json" in system_msg.lower():
            # 返回JSON格式响应
            if "intent" in system_msg.lower() or "意图" in user_input:
                return json.dumps({
                    "action": "create_and_run_model",
                    "sub_intents": ["create_project", "configure_model", "run_simulation"],
                    "confidence": 0.85,
                    "missing_info": [],
                    "clarification_needed": False
                }, ensure_ascii=False, indent=2)
            
            elif "entity" in system_msg.lower() or "实体" in user_input:
                return json.dumps({
                    "basin": {
                        "name": "测试流域",
                        "area_km2": 1000,
                        "location": {"lon": 105.0, "lat": 30.0}
                    },
                    "model": {
                        "runoff_type": "HBV",
                        "routing_type": "Muskingum"
                    },
                    "time_period": {
                        "start": "2020-01-01",
                        "end": "2020-12-31"
                    }
                }, ensure_ascii=False, indent=2)
            
            elif "config" in system_msg.lower() or "配置" in user_input:
                return json.dumps({
                    "delineation": {
                        "method": "automatic",
                        "pour_points": []
                    },
                    "runoff": {
                        "model_type": "HBV",
                        "parameters": {
                            "fc": 200.0,
                            "beta": 2.0,
                            "lp": 0.7
                        }
                    },
                    "routing": {
                        "model_type": "Muskingum",
                        "parameters": {
                            "k": 2.0,
                            "x": 0.2
                        }
                    }
                }, ensure_ascii=False, indent=2)
        
        # 返回自然语言响应
        return (
            f"[Mock LLM 响应]\n"
            f"这是一个模拟响应，用于测试和开发。\n"
            f"用户输入: {user_input[:100]}...\n"
            f"调用次数: {self.call_count}\n"
            f"\n"
            f"实际部署时，这里将返回千问大模型的智能响应。"
        )


def create_llm_backend(
    backend_type: str = "auto",
    **kwargs
) -> LLMBackend:
    """
    工厂函数：创建LLM后端
    
    Args:
        backend_type: 后端类型
            - "auto": 自动选择（优先千问，不可用则Mock）
            - "qwen": 千问
            - "mock": Mock
        **kwargs: 传递给后端的参数
        
    Returns:
        LLMBackend实例
    """
    if backend_type == "qwen":
        return QwenBackend(**kwargs)
    
    elif backend_type == "mock":
        return MockLLMBackend(**kwargs)
    
    elif backend_type == "auto":
        # 尝试创建千问后端
        qwen = QwenBackend(**kwargs)
        if qwen.is_available():
            print("✅ 使用千问后端")
            return qwen
        else:
            print("ℹ️  千问后端不可用，使用Mock后端")
            return MockLLMBackend()
    
    else:
        raise ValueError(f"不支持的后端类型: {backend_type}")


# 便捷函数：从环境变量创建后端
def create_default_backend() -> LLMBackend:
    """从环境变量创建默认后端"""
    backend_type = os.getenv("LLM_BACKEND", "auto")
    
    if backend_type in ["qwen", "auto"]:
        api_key = os.getenv("QWEN_API_KEY")
        model = os.getenv("QWEN_MODEL", "qwen-max")
        
        return create_llm_backend(
            backend_type=backend_type,
            api_key=api_key,
            model=model
        )
    
    return create_llm_backend(backend_type=backend_type)


if __name__ == "__main__":
    # 测试代码
    import asyncio
    
    async def test():
        print("=== 测试 Mock Backend ===")
        mock = MockLLMBackend()
        
        messages = [
            LLMMessage(role="system", content="你是水文专家"),
            LLMMessage(role="user", content="解析意图：我想建立HBV模型")
        ]
        
        response = await mock.complete(messages)
        print(f"响应: {response.content[:200]}...")
        print(f"模型: {response.model}")
        print(f"用量: {response.usage}")
        
        print("\n=== 测试自动选择 ===")
        backend = create_default_backend()
        print(f"选择的后端: {backend.__class__.__name__}")
        print(f"可用性: {backend.is_available()}")
    
    asyncio.run(test())
