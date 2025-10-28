"""
LLM Backend 单元测试
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import asyncio

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # 简单的pytest替代
    class pytest:
        class mark:
            @staticmethod
            def asyncio(func):
                return func

from llm_backend import (
    LLMBackend,
    LLMMessage,
    LLMResponse,
    QwenBackend,
    MockLLMBackend,
    create_llm_backend,
    create_default_backend
)


class TestMockLLMBackend:
    """测试Mock后端"""
    
    def test_initialization(self):
        """测试初始化"""
        backend = MockLLMBackend()
        assert backend is not None
        assert backend.is_available() == True
    
    def test_custom_responses(self):
        """测试自定义响应"""
        responses = {"HBV": "这是HBV模型的响应"}
        backend = MockLLMBackend(mock_responses=responses)
        assert backend.mock_responses == responses
    
    @pytest.mark.asyncio
    async def test_complete(self):
        """测试生成补全"""
        backend = MockLLMBackend()
        messages = [
            LLMMessage(role="user", content="测试消息")
        ]
        
        response = await backend.complete(messages)
        
        assert isinstance(response, LLMResponse)
        assert response.model == "mock-llm"
        assert response.content is not None
        assert len(response.content) > 0
    
    @pytest.mark.asyncio
    async def test_keyword_matching(self):
        """测试关键词匹配"""
        backend = MockLLMBackend(mock_responses={
            "HBV": "HBV模型响应"
        })
        
        messages = [
            LLMMessage(role="user", content="请介绍HBV模型")
        ]
        
        response = await backend.complete(messages)
        assert "HBV模型响应" in response.content
    
    @pytest.mark.asyncio
    async def test_call_count(self):
        """测试调用计数"""
        backend = MockLLMBackend()
        
        for i in range(3):
            await backend.complete([LLMMessage(role="user", content="test")])
        
        assert backend.call_count == 3


class TestQwenBackend:
    """测试千问后端"""
    
    def test_initialization_without_key(self):
        """测试无API key初始化"""
        # 清除环境变量
        old_key = os.environ.pop("QWEN_API_KEY", None)
        
        backend = QwenBackend()
        assert backend.is_available() == False
        
        # 恢复
        if old_key:
            os.environ["QWEN_API_KEY"] = old_key
    
    def test_initialization_with_key(self):
        """测试有API key初始化"""
        backend = QwenBackend(api_key="test-key")
        assert backend.is_available() == True
        assert backend.api_key == "test-key"
    
    def test_default_model(self):
        """测试默认模型"""
        backend = QwenBackend(api_key="test")
        assert backend.model == "qwen-max"
    
    def test_custom_model(self):
        """测试自定义模型"""
        backend = QwenBackend(api_key="test", model="qwen-plus")
        assert backend.model == "qwen-plus"


class TestBackendFactory:
    """测试后端工厂函数"""
    
    def test_create_mock(self):
        """测试创建Mock后端"""
        backend = create_llm_backend("mock")
        assert isinstance(backend, MockLLMBackend)
        assert backend.is_available() == True
    
    def test_create_qwen(self):
        """测试创建千问后端"""
        backend = create_llm_backend("qwen", api_key="test")
        assert isinstance(backend, QwenBackend)
    
    def test_create_auto_without_key(self):
        """测试auto模式（无API key）"""
        old_key = os.environ.pop("QWEN_API_KEY", None)
        
        backend = create_llm_backend("auto")
        assert isinstance(backend, MockLLMBackend)
        
        if old_key:
            os.environ["QWEN_API_KEY"] = old_key
    
    def test_create_default(self):
        """测试默认后端创建"""
        backend = create_default_backend()
        assert backend is not None
        assert backend.is_available() == True


if __name__ == "__main__":
    # 手动运行测试
    print("=== LLM Backend 单元测试 ===\n")
    
    # 测试Mock后端
    print("测试 MockLLMBackend...")
    test_mock = TestMockLLMBackend()
    test_mock.test_initialization()
    asyncio.run(test_mock.test_complete())
    print("  ✅ Mock后端测试通过\n")
    
    # 测试千问后端
    print("测试 QwenBackend...")
    test_qwen = TestQwenBackend()
    test_qwen.test_initialization_with_key()
    test_qwen.test_default_model()
    print("  ✅ 千问后端测试通过\n")
    
    # 测试工厂函数
    print("测试 Backend Factory...")
    test_factory = TestBackendFactory()
    test_factory.test_create_mock()
    test_factory.test_create_default()
    print("  ✅ 工厂函数测试通过\n")
    
    print("✅ 所有测试通过！")
