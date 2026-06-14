"""
DiagnosticAgent：症状诊断推理 Agent

这是第一个 WorkerAgent 实现，展示如何：
1. 参与 Swarm 协作
2. 自主认领任务
3. 调用医疗工具
4. 将结果写入 SharedContext
"""
from typing import Dict, Any, Optional
from loguru import logger

from .base_agent import BaseAgent
from .skill_registry_mixin import SkillRegistryMixin
from core import LLMClient


class DiagnosticAgent(BaseAgent, SkillRegistryMixin):
    """
    诊断 Agent

    职责：
    - 复杂症状的鉴别诊断
    - 多系统关联分析
    - 诊断思路推理（类似医生的临床思维）

    能力标签：
    - symptom_analysis
    - differential_diagnosis
    - clinical_reasoning
    """

    def __init__(
        self,
        agent_id: str = "diagnostic_agent",
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None
    ):
        config = config or {}
        config.setdefault('max_iterations', 5)

        super().__init__(agent_id, config, llm_client)

        # 设置能力标签（Swarm 协作用）
        self.set_capabilities([
            "symptom_analysis",
            "differential_diagnosis",
            "clinical_reasoning",
            "multi_system_analysis"
        ])

    def register_tools(self):
        """注册所有 active Skills（共享实现，auto_execute/disabled 会自动跳过）"""
        self.register_all_skills()


    def get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是专业的诊断 Agent（DiagnosticAgent）。你的职责是：
1. 分析症状的模式和关联性
2. 生成鉴别诊断列表
3. 评估每个诊断的可能性

**诊断原则**：
- 使用医学推理方法（如 VINDICATE 框架）
- 考虑常见病优先，但不忽略危险疾病
- 明确需要进一步检查的项目
- 永远不做确诊，只提供诊断思路

**可用 Skills（5个）**：
1. collect_clinical_context: 抽取问诊信息、识别缺失字段并生成追问
2. assess_risk: 评估症状风险等级（低/中/高/紧急）
3. analyze_symptoms: 分析症状模式和可能方向，避免确诊表达
4. recommend_lifestyle: 低风险问题的生活方式建议和用药安全提醒
5. deep_research: 仅在用户明确要求指南、循证证据、文献或最新研究时进行深度研究

**自动注入信息**：
- 当前会话历史 recent_history 和相似历史案例 historical_cases 会由系统自动注入上下文/背景信息，无需手动调用 Skill
- 最终安全审查由 AgentLoop / SafetyGuard 自动执行，safety_check 不再是可手动调用的 Skill

**Skills 使用策略**：
- 首先使用 collect_clinical_context 整理问诊信息和缺失项
- 然后使用 assess_risk 评估风险
- 再使用 analyze_symptoms 分析模式和可能方向
- 低风险、生活方式相关问题可使用 recommend_lifestyle
- 只有指南、循证、文献或最新研究类问题才使用 deep_research；普通症状分析和急症分诊默认不要使用
- 最终回答前确保措辞谨慎；系统会在输出前强制执行 SafetyGuard
- 基于 Skill 结果进行诊断推理
- 通常最多2-4次 Skill 调用，然后给出诊断思路；不要尝试调用 safety_check，系统最终会强制兜底

**Swarm 协作模式**：
- 你可能从 SharedContext 读取其他 Agent 的评估结果
- 你的分析结果会被其他 Agent（如 ResearchAgent）使用
- 专注于你的专长：症状分析和诊断推理

**输出格式**：
【风险评估】
风险等级：...
紧急程度：...

【症状分析】
主要症状类别：...
症状关联性：...

【鉴别诊断】
1. 诊断A（可能性XX%）
   - 支持证据：...
   - 反对证据：...
2. 诊断B（可能性XX%）
   ...

【建议检查】
- 检查项目1
- 检查项目2

【推理过程】
简述诊断推理逻辑...
"""

    async def post_process_result(
        self,
        result: Dict[str, Any],
        final_response: str
    ) -> Dict[str, Any]:
        """
        结果后处理：提取结构化诊断信息

        这里可以添加更复杂的解析逻辑
        """
        # 尝试提取风险等级
        risk_level = "unknown"
        if "风险等级" in final_response:
            if "高" in final_response or "HIGH" in final_response:
                risk_level = "high"
            elif "中" in final_response or "MEDIUM" in final_response:
                risk_level = "medium"
            elif "低" in final_response or "LOW" in final_response:
                risk_level = "low"

        result.update({
            "risk_level": risk_level,
            "diagnosis_provided": True
        })

        return result


# 便捷函数
async def diagnose(question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    便捷函数：快速使用 DiagnosticAgent

    Args:
        question: 症状描述
        context: 额外上下文（年龄、既往史等）

    Returns:
        诊断结果
    """
    agent = DiagnosticAgent()
    input_data = {'question': question}
    if context:
        input_data['context'] = context

    return await agent.process(input_data)
