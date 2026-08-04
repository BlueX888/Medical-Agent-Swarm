"""Deep Orchestrator module: intent, decomposition, and Worker matching."""
from __future__ import annotations

import json
from typing import Any, Dict

from loguru import logger
from .agent_catalog import AgentCatalog
from .routing_models import RoutePlan, RouteSource
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
    ):
        self.llm_client = llm_client
        self.agent_catalog = agent_catalog
        self.max_tasks = max_tasks
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
