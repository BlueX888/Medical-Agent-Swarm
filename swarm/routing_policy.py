"""Deterministic medical routing safety and plan validation policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from constraints import ConstraintValidator
from core.medical_safety_rules import _has_emergency_signal

from .agent_catalog import AgentCatalog
from .routing_models import (
    ExecutionMode,
    IntentType,
    PlannedTask,
    RiskLevel,
    RoutePlan,
    RouteSource,
)


@dataclass
class SafetyAssessment:
    risk_level: RiskLevel
    reasons: List[str] = field(default_factory=list)


def medical_safety_precheck(question: str) -> SafetyAssessment:
    """Classify red flags without relying on generative output."""
    text = question or ""
    explicit_emergency_markers = ["严重过敏", "持续大量出血"]
    if _has_emergency_signal(text, "") or any(
        marker in text for marker in explicit_emergency_markers
    ):
        return SafetyAssessment(
            risk_level=RiskLevel.EMERGENCY,
            reasons=["确定性医疗安全预检命中急症/红旗信号"],
        )

    high_markers = [
        "越来越严重",
        "持续加重",
        "高热不退",
    ]
    if any(marker in text for marker in high_markers):
        return SafetyAssessment(
            risk_level=RiskLevel.HIGH,
            reasons=["确定性医疗安全预检命中高风险信号"],
        )
    return SafetyAssessment(risk_level=RiskLevel.UNKNOWN)


class RoutePlanValidator:
    """Validate catalog, safety, budget, decomposition, and execution semantics."""

    def __init__(
        self,
        catalog: AgentCatalog,
        max_tasks: int = 5,
        constraint_validator: ConstraintValidator | None = None,
    ):
        self.catalog = catalog
        self.max_tasks = max(1, max_tasks)
        self.constraint_validator = constraint_validator or ConstraintValidator()

    def validate_and_repair(
        self,
        plan: RoutePlan,
        question: str,
        safety: SafetyAssessment,
    ) -> RoutePlan:
        repairs: List[str] = []
        tasks: List[PlannedTask] = []
        seen_goals = set()

        for task in plan.tasks[: self.max_tasks]:
            forbidden_skills = {
                "collect_clinical_context",
                "assess_risk",
                "analyze_symptoms",
                "recommend_lifestyle",
                "deep_research",
            }
            task_text = f"{task.goal} {' '.join(task.required_capabilities)}"
            mentioned_skills = [
                skill for skill in forbidden_skills if skill in task_text
            ]
            if mentioned_skills:
                raise ValueError(
                    f"Orchestrator task {task.id} specifies concrete Skills: "
                    f"{sorted(mentioned_skills)}"
                )

            normalized_goal = " ".join(task.goal.lower().split())
            if normalized_goal in seen_goals:
                repairs.append(f"合并重复任务：{task.id}")
                continue
            seen_goals.add(normalized_goal)

            assigned = task.assigned_agent
            if not self.catalog.supports(assigned, task.required_capabilities):
                replacement = self.catalog.find_supporting(task.required_capabilities)
                if replacement:
                    repairs.append(f"按能力将任务 {task.id} 重新分配给 {replacement}")
                    assigned = replacement
                else:
                    raise ValueError(
                        f"no Worker supports task {task.id}: {task.required_capabilities}"
                    )
            tasks.append(task.model_copy(update={"assigned_agent": assigned}))

        if not tasks:
            raise ValueError("no executable tasks remain")
        if len(plan.tasks) > self.max_tasks:
            repairs.append(f"任务数截断到预算上限 {self.max_tasks}")

        risk = plan.risk_level
        source = plan.source
        effective_risk = (
            safety.risk_level
            if safety.risk_level in {RiskLevel.HIGH, RiskLevel.EMERGENCY}
            else plan.risk_level
        )
        high_risk = effective_risk in {RiskLevel.HIGH, RiskLevel.EMERGENCY}
        if high_risk:
            risk = effective_risk
            if safety.risk_level in {RiskLevel.HIGH, RiskLevel.EMERGENCY}:
                source = RouteSource.SAFETY_RULE
            triage = next(
                (
                    task
                    for task in tasks
                    if self.catalog.supports(
                        task.assigned_agent,
                        ["risk_assessment", "symptom_analysis"],
                    )
                ),
                None,
            )
            if triage is None:
                agent_id = self.catalog.find_supporting(
                    ["risk_assessment", "symptom_analysis"]
                )
                if agent_id is None:
                    raise ValueError("high-risk route has no triage-capable Worker")
                triage = PlannedTask(
                    id="urgent_triage",
                    goal=f"立即对用户当前症状进行风险分诊并给出急诊行动建议。用户原始描述：{question}",
                    required_capabilities=["risk_assessment", "symptom_analysis"],
                    assigned_agent=agent_id,
                    priority="critical",
                )
                tasks.insert(0, triage)
                repairs.append("安全规则补充高优先级风险分诊任务")
            else:
                index = tasks.index(triage)
                triage = triage.model_copy(update={"priority": "critical", "depends_on": []})
                tasks[index] = triage

            # Emergency care must not wait for any secondary task. A later request
            # can perform education, lifestyle advice, or research safely.
            if effective_risk == RiskLevel.EMERGENCY:
                if len(tasks) > 1:
                    repairs.append("急症场景延后非分诊任务，立即返回风险分诊")
                tasks = [triage]
            else:
                tasks = [
                    task
                    if task.id == triage.id
                    else task.model_copy(
                        update={
                            "depends_on": list(
                                dict.fromkeys([*task.depends_on, triage.id])
                            )
                        }
                    )
                    for task in tasks
                ]

        known_ids = {task.id for task in tasks}
        tasks = [
            task.model_copy(
                update={
                    "depends_on": [
                        dependency
                        for dependency in task.depends_on
                        if dependency in known_ids
                    ]
                }
            )
            for task in tasks
        ]

        decomposition = self.constraint_validator.validate_task_decomposition(
            question,
            [
                {
                    "id": task.id,
                    "description": task.goal,
                    "assigned_agent": task.assigned_agent,
                    "required_capabilities": task.required_capabilities,
                }
                for task in tasks
            ],
            agent_catalog=self.catalog,
            risk_level=risk.value,
        )
        if not decomposition["valid"]:
            raise ValueError("; ".join(decomposition["issues"]))

        mode = self._execution_mode(tasks)
        reasons = [
            *plan.reasons,
            *safety.reasons,
            *repairs,
        ]
        return RoutePlan(
            intent_summary=plan.intent_summary,
            intents=plan.intents,
            risk_level=risk,
            confidence=plan.confidence,
            tasks=tasks,
            execution_mode=mode,
            source=source,
            reasons=list(dict.fromkeys(reasons)),
            needs_clarification=plan.needs_clarification,
        )

    @staticmethod
    def _execution_mode(tasks: List[PlannedTask]) -> ExecutionMode:
        if len(tasks) == 1:
            return ExecutionMode.SINGLE
        if any(task.depends_on for task in tasks):
            return ExecutionMode.SEQUENTIAL
        if len({task.assigned_agent for task in tasks}) < len(tasks):
            return ExecutionMode.SEQUENTIAL
        return ExecutionMode.PARALLEL


def fallback_plan(
    catalog: AgentCatalog,
    question: str,
    safety: SafetyAssessment,
    reason: str,
) -> RoutePlan:
    high_risk = safety.risk_level in {RiskLevel.HIGH, RiskLevel.EMERGENCY}
    capabilities = (
        ["risk_assessment", "symptom_analysis"]
        if high_risk
        else ["general_health_advice"]
    )
    preferred = "diagnostic_agent" if high_risk else "consultation_agent"
    agent_id = (
        preferred
        if catalog.supports(preferred, capabilities)
        else catalog.find_supporting(capabilities)
    )
    if agent_id is None:
        # Last-resort catalog choice keeps failure observable but executable.
        agent_id = catalog.list_agents()[0]["agent_id"]
        capabilities = catalog.list_agents()[0]["capabilities"][:1] or ["general"]

    return RoutePlan(
        intent_summary="安全降级处理用户请求",
        intents=[
            IntentType.SYMPTOM_TRIAGE
            if high_risk
            else IntentType.GENERAL_CONSULTATION
        ],
        risk_level=safety.risk_level,
        confidence=0,
        tasks=[
            PlannedTask(
                id="fallback",
                goal=(
                    f"优先进行风险分诊并给出立即行动建议。用户原始描述：{question}"
                    if high_risk
                    else f"安全回答用户请求并在信息不足时提出必要追问。用户原始描述：{question}"
                ),
                required_capabilities=capabilities,
                assigned_agent=agent_id,
                priority="critical" if high_risk else "normal",
            )
        ],
        execution_mode=ExecutionMode.SINGLE,
        source=RouteSource.FALLBACK,
        reasons=[reason, *safety.reasons],
        needs_clarification=False,
    )
