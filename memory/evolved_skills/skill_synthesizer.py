"""
SkillSynthesizer: 从成功的 Swarm 会话中提取可复用工作流

触发时机: 每次成功的 swarm 会话结束后自动触发
输入: SharedContext（任务分解、agent 贡献、事件流）
输出: 新的 EvolvedSkill 写入 SkillStore（或跳过如果重复）
"""
import re
from datetime import datetime
from typing import Dict, Any, Optional, List

import yaml
from loguru import logger

from .skill_store import SkillStore, EvolvedSkill
from .skill_matcher import SkillMatcher
from .usage_logger import UsageLogger


class SkillSynthesizer:
    """从 Swarm 协作会话中合成 Evolved Skills"""

    SYNTHESIS_PROMPT = """你是医疗智能体系统的知识工程师。请从以下成功的多Agent协作会话中提取一个可复用的工作流模式。

## 会话数据

**用户问题**: {question}

**任务分解**:
{task_decomposition}

**Agent 贡献**:
{agent_contributions}

**事件流**:
{events_summary}

**会话耗时**: {session_time:.1f} 秒
**参与 Agent**: {agents_involved}

## 输出要求

**第一部分：YAML 元数据**（用 ```yaml ``` 包裹）

```yaml
name: [简洁的工作流名称, 中文, 不超过20字]
category: [选择一个: diagnosis_workflow | treatment_plan | risk_assessment | research_workflow | consultation_guide]
trigger_keywords: [触发关键词列表, 逗号分隔, 3-8个]
target_agents: [适用的 agent ID 列表, 从 lead_agent/diagnostic_agent/research_agent/consultation_agent 中选]
tags: [英文标签, 逗号分隔]
embedding_text: [用于语义匹配的中文摘要, 50-100字, 覆盖核心症状和场景]
```

**第二部分：Markdown 工作流内容**（紧跟 YAML 之后）

# [工作流名称]

## 触发条件
[描述什么情况下应使用此工作流]

## 推荐工作流步骤
[编号步骤, 指明哪个 Agent 做什么, 用什么 tool]

## 工具调用模式
[列出推荐的 tool 调用序列]

## Agent 协作要点
[并行/串行建议, 预期耗时]

## 已知陷阱
[注意事项]

## 安全提示
[医疗安全相关]

## 质量标准
[好的回答应包含哪些要素]"""

    def __init__(
        self,
        skill_store: SkillStore,
        skill_matcher: SkillMatcher,
        usage_logger: UsageLogger,
        llm_client,
        duplicate_threshold: float = 0.85,
    ):
        self.skill_store = skill_store
        self.skill_matcher = skill_matcher
        self.usage_logger = usage_logger
        self.llm_client = llm_client
        self.duplicate_threshold = duplicate_threshold

    async def synthesize_from_session(
        self,
        session_id: str,
        question: str,
        shared_context,
        session_time: float = 0.0,
    ) -> Optional[str]:
        """
        从一次成功的 swarm 会话中合成 evolved skill

        Returns:
            新创建的 skill_id, 或 None（跳过/失败）
        """
        # 质量门控
        if not self._should_synthesize(shared_context):
            logger.debug("Skill synthesis skipped: quality gate not passed")
            return None

        # 提取结构化数据
        session_data = self._extract_session_data(question, shared_context, session_time)

        # 调用 LLM 生成 skill
        llm_output = await self._generate_skill_with_llm(session_data)
        if llm_output is None:
            self.usage_logger.log_synthesis(
                session_id=session_id,
                action="failed",
                reason="LLM generation failed",
            )
            return None

        # 语义去重检查
        embedding_text = llm_output.get("embedding_text", "")
        duplicate_id = self.skill_matcher.check_duplicate(
            embedding_text, threshold=self.duplicate_threshold
        )
        if duplicate_id:
            logger.info(f"Skill synthesis skipped: duplicate of {duplicate_id}")
            self.usage_logger.log_synthesis(
                session_id=session_id,
                action="skipped_duplicate",
                reason=f"Duplicate of {duplicate_id}",
            )
            return None

        # 创建并保存 skill
        skill = self._build_skill(session_id, llm_output)
        self.skill_store.save_skill(skill)
        self.skill_store.prune_if_needed()
        self.skill_matcher.invalidate_cache()

        self.usage_logger.log_synthesis(
            session_id=session_id,
            action="created",
            skill_id=skill.id,
            skill_name=skill.name,
            reason=f"Extracted from {len(shared_context.agent_contributions)}-agent swarm session",
        )
        logger.info(f"Synthesized evolved skill: {skill.id} - {skill.name}")
        return skill.id

    def _should_synthesize(self, shared_context) -> bool:
        """质量门控：判断会话是否值得提取 skill"""
        # 至少 2 个 agent 有贡献
        if len(shared_context.agent_contributions) < 2:
            return False
        # 至少 1 个子任务完成
        completed = shared_context.get_all_completed_subtasks()
        if len(completed) < 1:
            return False
        return True

    def _extract_session_data(
        self, question: str, shared_context, session_time: float
    ) -> Dict[str, str]:
        """从 SharedContext 提取 LLM prompt 所需的结构化数据"""
        # 任务分解
        task_lines = []
        for st_id, subtask in shared_context.task_decomposition.items():
            task_lines.append(
                f"- [{subtask.status.value}] {subtask.type}: {subtask.description} "
                f"(assigned to: {subtask.assigned_to})"
            )
        task_decomposition = "\n".join(task_lines) or "无"

        # Agent 贡献
        contrib_lines = []
        for agent_id, contributions in shared_context.agent_contributions.items():
            for c in contributions:
                result_str = str(c.result)
                if len(result_str) > 500:
                    result_str = result_str[:500] + "..."
                contrib_lines.append(
                    f"### {agent_id}\n"
                    f"子任务: {c.subtask_id}\n"
                    f"置信度: {c.confidence}\n"
                    f"结果摘要: {result_str}\n"
                )
        agent_contributions = "\n".join(contrib_lines) or "无"

        # 事件流摘要（只取关键事件）
        event_lines = []
        for event in shared_context.events:
            event_lines.append(
                f"- [{event.type.value}] {event.source_agent}: "
                f"{str(event.data)[:100]}"
            )
        events_summary = "\n".join(event_lines[-10:]) or "无"

        agents_involved = ", ".join(shared_context.agent_contributions.keys())

        return {
            "question": question,
            "task_decomposition": task_decomposition,
            "agent_contributions": agent_contributions,
            "events_summary": events_summary,
            "session_time": session_time,
            "agents_involved": agents_involved,
        }

    async def _generate_skill_with_llm(
        self, session_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """调用 LLM 生成结构化 skill"""
        prompt = self.SYNTHESIS_PROMPT.format(**session_data)

        try:
            response = await self.llm_client.chat([
                {"role": "user", "content": prompt}
            ])

            if not response:
                return None

            return self._parse_llm_output(response)

        except Exception as e:
            logger.error(f"Skill synthesis LLM call failed: {e}")
            return None

    def _parse_llm_output(self, response: str) -> Optional[Dict[str, Any]]:
        """解析 LLM 输出：提取 YAML 元数据 + Markdown body"""
        try:
            # 提取 YAML 块
            yaml_match = re.search(r"```yaml\s*\n(.*?)```", response, re.DOTALL)
            if not yaml_match:
                logger.warning("No YAML block found in LLM output")
                return None

            yaml_str = yaml_match.group(1).strip()
            meta = yaml.safe_load(yaml_str)
            if not meta or not isinstance(meta, dict):
                return None

            # 提取 YAML 块之后的 Markdown
            yaml_end = yaml_match.end()
            body = response[yaml_end:].strip()

            # 处理逗号分隔的列表字段
            for field in ["trigger_keywords", "target_agents", "tags"]:
                val = meta.get(field, [])
                if isinstance(val, str):
                    meta[field] = [v.strip() for v in val.split(",") if v.strip()]

            return {
                "name": meta.get("name", "未命名工作流"),
                "category": meta.get("category", "general"),
                "trigger_keywords": meta.get("trigger_keywords", []),
                "target_agents": meta.get("target_agents", []),
                "tags": meta.get("tags", []),
                "embedding_text": meta.get("embedding_text", ""),
                "body_markdown": body,
            }
        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}")
            return None

    def _build_skill(
        self, session_id: str, llm_output: Dict[str, Any]
    ) -> EvolvedSkill:
        """从 LLM 输出构造 EvolvedSkill 对象"""
        now = datetime.now()
        return EvolvedSkill(
            id=self.skill_store.generate_skill_id(),
            name=llm_output["name"],
            version=1,
            created_at=now,
            updated_at=now,
            source_sessions=[session_id],
            trigger_keywords=llm_output.get("trigger_keywords", []),
            category=llm_output.get("category", "general"),
            target_agents=llm_output.get("target_agents", []),
            performance={
                "times_matched": 0,
                "times_used": 0,
                "positive_outcomes": 0,
                "negative_outcomes": 0,
            },
            embedding_text=llm_output.get("embedding_text", ""),
            body_markdown=llm_output.get("body_markdown", ""),
        )
