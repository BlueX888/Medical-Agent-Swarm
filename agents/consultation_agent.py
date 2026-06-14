"""
健康咨询Agent
支持 Skills 调用
"""
from typing import Dict, Any
from loguru import logger
import re

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin


class ConsultationAgent(BaseAgent, SkillRegistryMixin):
    """
    健康咨询Agent
    通过 Skills 调用底层工具
    """

    def __init__(self, config: Dict[str, Any] = None):
        default_config = {
            "model": "openai_compatible",
            "max_iterations": 5,
            "temperature": 0.8,
            "description": "健康咨询Agent，提供通用医疗咨询和健康建议"
        }

        config = config or default_config
        super().__init__(
            agent_id="consultation_agent",
            config=config
        )

        # 设置能力标签（Swarm 协作用）
        self.set_capabilities([
            "general_health_advice",
            "risk_assessment",
            "symptom_triage"
        ])

    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一位专业的医疗健康咨询顾问。你的职责是提供准确、专业的健康建议和疾病科普。

可用 Skills（5个）：
1. collect_clinical_context: 抽取问诊信息、识别缺失字段并生成追问
2. assess_risk: 评估症状风险等级（低/中/高/紧急）
3. analyze_symptoms: 分析症状模式和可能方向，避免确诊表达
4. recommend_lifestyle: 根据低风险问题提供生活方式建议（饮食、运动、睡眠、用药安全）
5. deep_research: 仅在用户明确要求指南、循证证据、文献或最新研究时进行深度研究（网络搜索+证据综合）

**自动注入信息**：
- 当前会话历史 recent_history 和相似历史案例 historical_cases 会由系统自动注入上下文/背景信息，无需手动调用 Skill
- 最终安全审查由 AgentLoop / SafetyGuard 自动执行，safety_check 不再是可手动调用的 Skill

**Skills 使用原则**：
- Skills 是可选的，不是必须的
- 对于简单的常识性问题，可以直接回答，无需使用 Skills
- 只在真正需要专业医学信息时才调用 Skills
- 调用 Skill 后，根据返回的结果给出最终答案
- 通常最多使用2-4个 Skills；不要尝试调用 safety_check，系统最终会强制兜底

工作流程建议：
1. 理解用户问题
2. 症状或健康风险问题先使用 collect_clinical_context 补全关键信息
3. 再使用 assess_risk 判断紧急程度
4. 需要解释症状时使用 analyze_symptoms；低风险生活方式问题再用 recommend_lifestyle
5. 只有指南、循证、文献或最新研究类问题才使用 deep_research；基础科普、急症分诊和普通症状分析默认不要使用
6. 最终回答前确保措辞谨慎；系统会在输出前强制执行 SafetyGuard

回答要求：
- 用通俗易懂的语言
- 提供实用的建议和注意事项
- 必要时建议就医
- 保持温和、专业的语气

**重要提醒**：
- 你不能做出明确的诊断
- 你不能替代医生的专业意见
- 对于严重或紧急情况，必须建议立即就医

在最终回答时，请按以下格式输出：

【回答】
[你的详细回答]

【核心建议】
1. 第一条建议
2. 第二条建议
...

【免责声明】
以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。
"""

    def register_tools(self):
        """注册所有 active Skills（共享实现，auto_execute/disabled 会自动跳过）"""
        self.register_all_skills()

    def format_user_input(self, input_data: Dict[str, Any]) -> str:
        """格式化用户输入"""
        question = input_data.get('question', '')
        session_id = input_data.get('session_id', '')

        # 构建消息
        parts = []

        # 添加session_id信息（如果有）
        if session_id:
            parts.append(f"[系统信息] 当前会话ID: {session_id}")

        # 添加上下文信息（如果有）
        context = input_data.get('context', {})
        if context:
            # 专门格式化历史案例（自动注入的长期记忆）
            historical_cases = context.get('historical_cases', [])
            if historical_cases:
                cases_lines = ["📋 相似历史案例（自动检索）："]
                for i, case in enumerate(historical_cases, 1):
                    summary = case.get('summary', '')
                    score = case.get('score', 0)
                    cases_lines.append(f"  案例{i}（相似度 {score:.0%}）：{summary}")
                parts.append("\n".join(cases_lines))

            # 格式化其他上下文
            other_context = {k: v for k, v in context.items() if k not in ('historical_cases', 'recent_history')}
            if other_context:
                context_str = "\n".join([f"{k}: {v}" for k, v in other_context.items()])
                parts.append(f"背景信息：\n{context_str}\n")

        # 添加用户问题
        parts.append(f"用户问题：{question}")

        return "\n".join(parts)

    async def post_process_result(
        self,
        result: Dict[str, Any],
        final_response: str
    ) -> Dict[str, Any]:
        """
        后处理：从最终响应中提取结构化信息
        """
        # 提取核心建议
        suggestions = []
        suggestion_pattern = r'【核心建议】\s*\n((?:\d+\.\s*.+\n?)+)'
        match = re.search(suggestion_pattern, final_response)

        if match:
            suggestion_text = match.group(1)
            suggestion_lines = re.findall(r'\d+\.\s*(.+)', suggestion_text)
            suggestions = [s.strip() for s in suggestion_lines if s.strip()]

        # 提取免责声明
        disclaimer_pattern = r'【免责声明】\s*\n(.+)'
        disclaimer_match = re.search(disclaimer_pattern, final_response)
        disclaimer = disclaimer_match.group(1) if disclaimer_match else \
            "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。"

        result.update({
            'suggestions': suggestions[:5],  # 最多5条
            'disclaimer': disclaimer
        })

        return result


# 便捷函数
async def consult(question: str, **kwargs) -> Dict[str, Any]:
    """快捷咨询函数"""
    agent = ConsultationAgent()
    return await agent.process({'question': question, **kwargs})
