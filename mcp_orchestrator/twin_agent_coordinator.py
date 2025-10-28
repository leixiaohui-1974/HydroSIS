"""
Twin-Agent Coordinator - 双智能体协调器

编排 HydroMind 和 HydroCompute 的协作工作流
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from .conversation_manager import ConversationManager
except ImportError:
    from conversation_manager import ConversationManager


class TwinAgentCoordinator:
    """双智能体协调器"""
    
    def __init__(
        self,
        hydromind_client,  # HydroMind MCP客户端
        hydrocompute_client,  # HydroCompute MCP客户端  
        conversation_manager: Optional[ConversationManager] = None
    ):
        """
        初始化协调器
        
        Args:
            hydromind_client: HydroMind客户端（认知智能体）
            hydrocompute_client: HydroCompute客户端（机理智能体）
            conversation_manager: 对话管理器
        """
        self.mind = hydromind_client
        self.compute = hydrocompute_client
        self.conversation = conversation_manager or ConversationManager()
        
        print("✅ 双智能体协调器初始化完成")
        print(f"   HydroMind (认知): {self.mind is not None}")
        print(f"   HydroCompute (机理): {self.compute is not None}")
    
    async def process_user_request(
        self,
        user_input: str,
        session_id: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        处理用户自然语言请求（完整流程）
        
        工作流：
        1. [HydroMind] 理解意图和实体
        2. [HydroMind] 验证需求
        3. [HydroMind] 生成配置
        4. [HydroCompute] 执行计算
        5. [HydroMind] 解读结果
        6. [HydroMind] 生成报告
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            **kwargs: 其他参数
            
        Returns:
            包含理解、配置、执行、解读、报告的完整结果
        """
        start_time = datetime.now()
        
        # 保存用户消息
        self.conversation.add_message(session_id, "user", user_input)
        
        result = {
            "status": "processing",
            "session_id": session_id,
            "user_input": user_input,
            "timestamp": start_time.isoformat()
        }
        
        try:
            # ========== 阶段1: 认知理解 (HydroMind) ==========
            print(f"\n[阶段1] 认知理解...")
            
            # 1.1 意图识别
            intent_result = await self.mind.call_tool("parse_user_intent", {
                "user_input": user_input,
                "conversation_history": self.conversation.get_history(session_id)
            })
            
            if intent_result.get("isError"):
                return self._error_response("意图识别失败", intent_result.get("error"))
            
            intent = intent_result.get("result", {})
            result["understanding"] = {"intent": intent}
            
            # 1.2 实体抽取
            entities_result = await self.mind.call_tool("extract_entities", {
                "user_input": user_input
            })
            
            if not entities_result.get("isError"):
                entities = entities_result.get("result", {})
                result["understanding"]["entities"] = entities
            else:
                entities = {}
            
            # 1.3 需求验证
            validation_result = await self.mind.call_tool("validate_requirements", {
                "requirements": {
                    "intent": intent,
                    "entities": entities
                }
            })
            
            if not validation_result.get("isError"):
                validation = validation_result.get("result", {})
                result["understanding"]["validation"] = validation
                
                # 如果需要澄清，直接返回
                if not validation.get("is_valid"):
                    result["status"] = "clarification_needed"
                    result["clarification"] = {
                        "question": self._generate_clarification_question(validation),
                        "missing_info": validation.get("issues", [])
                    }
                    return result
            
            # ========== 阶段2: 配置生成 (HydroMind) ==========
            print(f"[阶段2] 配置生成...")
            
            config_result = await self.mind.call_tool("generate_model_config", {
                "intent": intent,
                "entities": entities
            })
            
            if config_result.get("isError"):
                return self._error_response("配置生成失败", config_result.get("error"))
            
            config_data = config_result.get("result", {})
            result["configuration"] = config_data
            
            # ========== 阶段3: 机理计算 (HydroCompute) ==========
            print(f"[阶段3] 机理计算...")
            
            # 判断需要执行的计算任务
            action = intent.get("action", "")
            sub_intents = intent.get("sub_intents", [])
            
            compute_results = {}
            
            # 3.1 创建项目（如果需要）
            if "create_project" in sub_intents or "create" in action:
                project_result = await self.compute.call_tool("create_project", {
                    "user_id": session_id,
                    "project_name": entities.get("basin", {}).get("name", "项目"),
                    "description": f"基于自然语言创建: {user_input[:50]}"
                })
                
                if not project_result.get("isError"):
                    compute_results["project"] = project_result.get("result", {})
                    result["project_id"] = compute_results["project"].get("project_id")
            
            # 3.2 运行模拟（如果需要）
            if "run_simulation" in sub_intents or "simulate" in action:
                # 这里需要项目ID，如果没有就跳过
                project_id = result.get("project_id") or kwargs.get("project_id")
                
                if project_id:
                    simulation_result = await self.compute.call_tool("run_simulation", {
                        "project_id": project_id,
                        "generate_report": False  # 我们用HydroMind生成报告
                    })
                    
                    if not simulation_result.get("isError"):
                        compute_results["simulation"] = simulation_result.get("result", {})
                else:
                    compute_results["simulation"] = {
                        "status": "skipped",
                        "reason": "需要先创建项目"
                    }
            
            result["computation"] = compute_results
            
            # ========== 阶段4: 结果解读 (HydroMind) ==========
            print(f"[阶段4] 结果解读...")
            
            if "simulation" in compute_results and compute_results["simulation"].get("status") != "skipped":
                interpretation_result = await self.mind.call_tool("interpret_results", {
                    "simulation_results": compute_results["simulation"],
                    "model_config": config_data.get("config", {}),
                    "user_objective": entities.get("objectives", ["水文模拟"])[0] if entities.get("objectives") else "水文模拟"
                })
                
                if not interpretation_result.get("isError"):
                    result["interpretation"] = interpretation_result.get("result", {})
            
            # ========== 阶段5: 报告生成 (HydroMind) ==========
            print(f"[阶段5] 报告生成...")
            
            if "generate_report" in sub_intents or kwargs.get("generate_report", True):
                report_result = await self.mind.call_tool("create_executive_report", {
                    "workflow_result": compute_results.get("simulation", {}),
                    "model_config": config_data.get("config", {}),
                    "report_type": "executive"
                })
                
                if not report_result.get("isError"):
                    result["report"] = report_result.get("result", {})
            
            # ========== 完成 ==========
            result["status"] = "completed"
            result["duration_seconds"] = (datetime.now() - start_time).total_seconds()
            
            # 生成自然语言摘要
            result["natural_language_summary"] = self._generate_summary(result)
            
            # 保存助手回复
            self.conversation.add_message(
                session_id,
                "assistant",
                result["natural_language_summary"]
            )
            
            return result
            
        except Exception as e:
            return self._error_response("处理失败", str(e))
    
    async def quick_understand(
        self,
        user_input: str,
        session_id: str
    ) -> Dict[str, Any]:
        """
        快速理解模式（只执行认知层，不执行计算）
        
        用于：
        - 需求探索
        - 配置预览
        - 问答交互
        """
        intent_result = await self.mind.call_tool("parse_user_intent", {
            "user_input": user_input,
            "conversation_history": self.conversation.get_history(session_id)
        })
        
        entities_result = await self.mind.call_tool("extract_entities", {
            "user_input": user_input
        })
        
        return {
            "mode": "quick_understand",
            "intent": intent_result.get("result", {}),
            "entities": entities_result.get("result", {}),
            "next_steps": self._suggest_next_steps(
                intent_result.get("result", {}),
                entities_result.get("result", {})
            )
        }
    
    async def answer_question(
        self,
        question: str,
        context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        问答模式（使用HydroMind回答问题）
        
        Args:
            question: 用户问题
            context: 上下文（模拟结果、配置等）
            session_id: 会话ID
        """
        qa_result = await self.mind.call_tool("answer_questions", {
            "question": question,
            "context": context
        })
        
        if qa_result.get("isError"):
            return {"error": qa_result.get("error")}
        
        return qa_result.get("result", {})
    
    # ============ 辅助方法 ============
    
    def _error_response(self, message: str, error: str) -> Dict[str, Any]:
        """生成错误响应"""
        return {
            "status": "error",
            "message": message,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_clarification_question(self, validation: Dict[str, Any]) -> str:
        """生成澄清问题"""
        issues = validation.get("issues", [])
        if not issues:
            return "请提供更多信息"
        
        missing_fields = [
            issue["message"]
            for issue in issues
            if issue["type"] == "missing_data"
        ]
        
        if missing_fields:
            return f"为了更好地帮助您，请提供以下信息：\n" + "\n".join(f"- {field}" for field in missing_fields[:3])
        
        return "请提供更多详细信息"
    
    def _generate_summary(self, result: Dict[str, Any]) -> str:
        """生成自然语言摘要"""
        intent = result.get("understanding", {}).get("intent", {})
        action = intent.get("action", "处理请求")
        
        summary_parts = [f"已完成{action}。"]
        
        # 配置摘要
        if "configuration" in result:
            config_summary = result["configuration"].get("config_summary", "")
            if config_summary:
                summary_parts.append(config_summary)
        
        # 计算摘要
        if "computation" in result:
            if "project" in result["computation"]:
                summary_parts.append("已创建项目。")
            if "simulation" in result["computation"]:
                summary_parts.append("已完成模拟计算。")
        
        # 解读摘要
        if "interpretation" in result:
            assessment = result["interpretation"].get("overall_assessment", {})
            key_message = assessment.get("key_message", "")
            if key_message:
                summary_parts.append(key_message)
        
        # 报告摘要
        if "report" in result:
            summary_parts.append("已生成分析报告。")
        
        return " ".join(summary_parts)
    
    def _suggest_next_steps(
        self,
        intent: Dict[str, Any],
        entities: Dict[str, Any]
    ) -> List[str]:
        """建议下一步操作"""
        suggestions = []
        
        if not entities.get("model"):
            suggestions.append("指定产流模型类型（HBV/SCS/新安江）")
        
        if not entities.get("time_period"):
            suggestions.append("指定模拟时间范围")
        
        if not entities.get("data_requirements"):
            suggestions.append("准备必需的数据（DEM、降雨、观测）")
        
        if intent.get("confidence", 1.0) < 0.7:
            suggestions.append("明确您的具体需求")
        
        if not suggestions:
            suggestions.append("开始运行模拟")
        
        return suggestions


if __name__ == "__main__":
    # 测试代码
    print("=== 双智能体协调器 ===")
    print("协调HydroMind（认知）和HydroCompute（机理）的协作")
    print("\n需要连接到实际的MCP服务器才能运行测试")
