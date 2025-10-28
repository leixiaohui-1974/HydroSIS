"""
Conversation Manager - 对话管理器

管理多会话的对话历史和上下文
"""

from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict


class ConversationManager:
    """对话管理器"""
    
    def __init__(self, max_history: int = 20):
        """
        初始化对话管理器
        
        Args:
            max_history: 每个会话保留的最大消息数
        """
        self.max_history = max_history
        self.sessions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
    
    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Dict[str, Any] = None
    ) -> None:
        """
        添加消息到会话
        
        Args:
            session_id: 会话ID
            role: 角色 (user/assistant/system)
            content: 消息内容
            metadata: 额外的元数据
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.sessions[session_id].append(message)
        
        # 限制历史长度
        if len(self.sessions[session_id]) > self.max_history:
            self.sessions[session_id] = self.sessions[session_id][-self.max_history:]
        
        # 更新会话元数据
        if session_id not in self.session_metadata:
            self.session_metadata[session_id] = {
                "created_at": message["timestamp"],
                "message_count": 0
            }
        
        self.session_metadata[session_id]["message_count"] += 1
        self.session_metadata[session_id]["last_active"] = message["timestamp"]
    
    def get_history(
        self,
        session_id: str,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """
        获取会话历史
        
        Args:
            session_id: 会话ID
            limit: 返回最近N条消息
            
        Returns:
            消息列表
        """
        messages = self.sessions.get(session_id, [])
        
        if limit:
            return messages[-limit:]
        
        return messages
    
    def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        获取会话上下文
        
        Returns:
            包含历史、统计等信息的上下文
        """
        history = self.get_history(session_id)
        metadata = self.session_metadata.get(session_id, {})
        
        return {
            "session_id": session_id,
            "history": history,
            "message_count": metadata.get("message_count", 0),
            "created_at": metadata.get("created_at"),
            "last_active": metadata.get("last_active")
        }
    
    def clear_session(self, session_id: str) -> None:
        """清除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
        if session_id in self.session_metadata:
            del self.session_metadata[session_id]
    
    def list_sessions(self) -> List[str]:
        """列出所有会话ID"""
        return list(self.sessions.keys())
    
    def get_summary(self, session_id: str) -> str:
        """
        生成会话摘要
        
        Returns:
            摘要字符串
        """
        metadata = self.session_metadata.get(session_id, {})
        history = self.get_history(session_id)
        
        user_messages = [m for m in history if m["role"] == "user"]
        
        summary = f"会话 {session_id}\n"
        summary += f"  消息数: {metadata.get('message_count', 0)}\n"
        summary += f"  创建于: {metadata.get('created_at', 'N/A')}\n"
        summary += f"  最后活跃: {metadata.get('last_active', 'N/A')}\n"
        
        if user_messages:
            summary += f"  最近请求: {user_messages[-1]['content'][:50]}..."
        
        return summary


if __name__ == "__main__":
    # 测试
    manager = ConversationManager()
    
    session_id = "test_session_001"
    
    manager.add_message(session_id, "user", "我想建立HBV模型")
    manager.add_message(session_id, "assistant", "好的，请提供流域信息")
    manager.add_message(session_id, "user", "长江上游，面积5000平方公里")
    
    print("=== 对话历史 ===")
    for msg in manager.get_history(session_id):
        print(f"[{msg['role']}] {msg['content']}")
    
    print("\n=== 会话摘要 ===")
    print(manager.get_summary(session_id))
