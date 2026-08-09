"""Deep Orchestrator module: intent, decomposition, and Worker matching."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

from loguru import logger
from pydantic import ValidationError

from memory.evidence_cache import EvidenceMemory

from .agent_catalog import AgentCatalog
from .routing_models import (
    ExecutionMode,
    IntentType,
    PlannedTask,
    RiskLevel,
    RoutePlan,
    RouteSource,
)
from .routing_policy import (
    RoutePlanValidator,
    fallback_plan,
    medical_safety_precheck,
)


class Orchestrator:
    """Expose one stable planning operation and hide routing internals."""

    def __init__(
        self,
        llm_client: Any,
        agent_catalog: AgentCatalog,
        *,
        max_tasks: int = 5,
        evidence_memory: Optional[EvidenceMemory] = None,
    ):
        self.llm_client = llm_client
        self.agent_catalog = agent_catalog
        self.max_tasks = max_tasks
        self.evidence_memory = evidence_memory or EvidenceMemory()
        self.validator = RoutePlanValidator(agent_catalog, max_tasks=max_tasks)

    async def plan(self, question: str, context: Dict[str, Any]) -> RoutePlan:
        recent_user_messages = [
            str(message.get("content") or "")
            for message in (context or {}).get("recent_history", [])
            if isinstance(message, dict) and message.get("role") == "user"
        ][-3:]
        safety_text = "\n".join([*recent_user_messages, question])
        safety = medical_safety_precheck(safety_text)

        try:
            try:
                evidence_plan = self._plan_from_evidence(question)
            except Exception as exc:
                logger.warning(f"Evidence memory routing lookup failed: {exc}")
                evidence_plan = None
            if evidence_plan is not None:
                return self.validator.validate_and_repair(
                    evidence_plan, question, safety
                )

            raw = await self.llm_client.chat(
                [
                    {"role": "system", "content": self._system_prompt()},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"question": question, "context": context or {}},
                            ensure_ascii=False,
                            default=str,
                        ),
                    },
                ],
                temperature=0,
                response_format={"type": "json_object"},
                trace_name="orchestrator_plan",
            )
            plan = self._parse(raw).model_copy(update={"source": RouteSource.LLM})
            return self.validator.validate_and_repair(plan, question, safety)
        except Exception as exc:
            logger.error(f"Orchestrator planning failed; using safe fallback: {exc}")
            return fallback_plan(
                self.agent_catalog,
                question,
                safety,
                f"规划或校验失败：{type(exc).__name__}: {exc}",
            )

    def _parse(self, raw: str) -> RoutePlan:
        text = (raw or "").strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
        payload = json.loads(text)
        return RoutePlan.model_validate(payload)

    def _plan_from_evidence(self, question: str) -> Optional[RoutePlan]:
        hit = self.evidence_memory.lookup(question, min_score=0.9)
        if not hit:
            return None

        agent_id = str(hit.get("preferred_agent") or "research_agent")
        worker = self.agent_catalog.get_worker(agent_id)
        if worker is None:
            return None
        capabilities = list(worker.get_capabilities())
        if agent_id == "research_agent":
            required = [
                capability
                for capability in ["guideline_lookup", "evidence_synthesis"]
                if capability in capabilities
            ]
            intent = IntentType.EVIDENCE_RESEARCH
        else:
            required = [
                capability
                for capability in ["risk_assessment", "symptom_analysis"]
                if capability in capabilities
            ]
            intent = IntentType.SYMPTOM_TRIAGE
        required = required or capabilities[:1]

        return RoutePlan(
            intent_summary="使用高置信度本地循证记忆处理请求",
            intents=[intent],
            risk_level=RiskLevel.UNKNOWN,
            confidence=float(hit.get("match_score") or 0.9),
            tasks=[
                PlannedTask(
                    id="evidence_memory",
                    goal=(
                        f"结合本地循证记忆回答用户问题。用户问题：{question}\n"
                        f"本地证据：{str(hit.get('answer') or '')[:1200]}"
                    ),
                    required_capabilities=required,
                    assigned_agent=agent_id,
                    priority="high",
                )
            ],
            execution_mode=ExecutionMode.SINGLE,
            source=RouteSource.EVIDENCE_MEMORY,
            reasons=["命中高置信度本地 evidence memory"],
        )

    def _system_prompt(self) -> str:
        schema = RoutePlan.model_json_schema()
        return (
            "你是医疗 Orchestrator。一次完成意图识别、必要任务拆分、能力确定和 Worker 匹配。"
            "能由一个 Worker 完成时只创建一个任务；不要执行医学分析，不要生成最终回答，"
            "也绝不能在 goal 或 required_capabilities 中写具体 Skill 名称。"
            f"任务数最多 {self.max_tasks}。先确定 required_capabilities，再按运行时目录分配。"
            "独立且属于不同 Worker 的任务可 parallel；有依赖或同 Worker 多任务必须 sequential。"
            "\n运行时 Agent Catalog：\n"
            + json.dumps(self.agent_catalog.list_agents(), ensure_ascii=False)
            + "\n严格只输出满足下列 JSON Schema 的 JSON 对象：\n"
            + json.dumps(schema, ensure_ascii=False)
        )
