import json

import pytest
from pydantic import ValidationError

from swarm.agent_catalog import AgentCatalog
from swarm.orchestrator import Orchestrator
from swarm.routing_models import (
    ExecutionMode,
    PlannedTask,
    RiskLevel,
    RoutePlan,
    RouteSource,
)


class FakeWorker:
    def __init__(self, agent_id, capabilities, description=""):
        self.agent_id = agent_id
        self._capabilities = capabilities
        self.config = {"description": description}

    def get_capabilities(self):
        return list(self._capabilities)


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        return self.payload


@pytest.fixture
def catalog():
    return AgentCatalog(
        [
            FakeWorker(
                "consultation_agent",
                ["general_health_advice", "risk_assessment", "symptom_triage"],
                "通用咨询和生活方式指导",
            ),
            FakeWorker(
                "diagnostic_agent",
                ["risk_assessment", "symptom_analysis", "clinical_reasoning"],
                "复杂症状分析和临床推理",
            ),
            FakeWorker(
                "research_agent",
                ["literature_search", "evidence_synthesis", "guideline_lookup"],
                "指南和循证研究",
            ),
        ]
    )


def payload_for(tasks, mode="single", risk="low"):
    return json.dumps(
        {
            "intent_summary": "测试计划",
            "intents": ["general_consultation"],
            "risk_level": risk,
            "confidence": 0.9,
            "tasks": tasks,
            "execution_mode": mode,
            "source": "llm",
            "reasons": ["测试"],
            "needs_clarification": False,
        },
        ensure_ascii=False,
    )


def task(task_id, goal, agent, capabilities, depends_on=None):
    return {
        "id": task_id,
        "goal": goal,
        "required_capabilities": capabilities,
        "assigned_agent": agent,
        "priority": "normal",
        "depends_on": depends_on or [],
    }


def test_route_plan_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        RoutePlan(
            intent_summary="x",
            intents=["general_consultation"],
            risk_level="low",
            confidence=1.1,
            tasks=[
                PlannedTask(
                    id="t1",
                    goal="回答问题",
                    required_capabilities=["general_health_advice"],
                    assigned_agent="consultation_agent",
                    priority="normal",
                )
            ],
            execution_mode="single",
            source="llm",
            reasons=[],
        )


def test_route_plan_rejects_cycles():
    with pytest.raises(ValidationError):
        RoutePlan(
            intent_summary="x",
            intents=["diagnostic_reasoning"],
            risk_level="medium",
            confidence=0.8,
            tasks=[
                PlannedTask(
                    id="a",
                    goal="先做 A",
                    required_capabilities=["clinical_reasoning"],
                    assigned_agent="diagnostic_agent",
                    priority="normal",
                    depends_on=["b"],
                ),
                PlannedTask(
                    id="b",
                    goal="再做 B",
                    required_capabilities=["general_health_advice"],
                    assigned_agent="consultation_agent",
                    priority="normal",
                    depends_on=["a"],
                ),
            ],
            execution_mode="sequential",
            source="llm",
            reasons=[],
        )


@pytest.mark.asyncio
async def test_general_question_uses_only_consultation_agent(catalog):
    llm = FakeLLM(
        payload_for(
            [
                task(
                    "consult",
                    "解释高血压的基本概念",
                    "consultation_agent",
                    ["general_health_advice"],
                )
            ]
        )
    )

    plan = await Orchestrator(llm, catalog).plan("什么是高血压？", {})

    assert [item.assigned_agent for item in plan.tasks] == ["consultation_agent"]
    assert plan.execution_mode == ExecutionMode.SINGLE
    assert llm.calls[0]["temperature"] == 0


@pytest.mark.asyncio
async def test_unknown_agent_is_reassigned_by_capability(catalog):
    llm = FakeLLM(
        payload_for(
            [
                task(
                    "research",
                    "检索高血压指南",
                    "invented_agent",
                    ["guideline_lookup"],
                )
            ]
        )
    )

    plan = await Orchestrator(llm, catalog).plan("请查高血压指南", {})

    assert plan.tasks[0].assigned_agent == "research_agent"
    assert any("重新分配" in reason for reason in plan.reasons)


@pytest.mark.asyncio
async def test_duplicate_tasks_are_merged(catalog):
    llm = FakeLLM(
        payload_for(
            [
                task(
                    "first",
                    "提供睡眠卫生建议",
                    "consultation_agent",
                    ["general_health_advice"],
                ),
                task(
                    "second",
                    "  提供睡眠卫生建议  ",
                    "consultation_agent",
                    ["general_health_advice"],
                ),
            ],
            mode="parallel",
        )
    )

    plan = await Orchestrator(llm, catalog).plan("如何改善睡眠？", {})

    assert [item.id for item in plan.tasks] == ["first"]
    assert plan.execution_mode == ExecutionMode.SINGLE


@pytest.mark.asyncio
async def test_invalid_json_uses_safe_fallback(catalog):
    plan = await Orchestrator(FakeLLM("not json"), catalog).plan("我想了解睡眠卫生", {})

    assert plan.source == RouteSource.FALLBACK
    assert plan.tasks[0].assigned_agent == "consultation_agent"
    assert plan.confidence == 0


@pytest.mark.asyncio
async def test_concrete_skill_names_in_tasks_are_rejected(catalog):
    llm = FakeLLM(
        payload_for(
            [
                task(
                    "unsafe-plan",
                    "调用 assess_risk 后回答",
                    "consultation_agent",
                    ["general_health_advice"],
                )
            ]
        )
    )

    plan = await Orchestrator(llm, catalog).plan("我有点不舒服", {})

    assert plan.source == RouteSource.FALLBACK
    assert plan.confidence == 0


@pytest.mark.asyncio
async def test_emergency_precheck_forces_diagnostic_triage_and_delays_research(catalog):
    llm = FakeLLM(
        payload_for(
            [
                task(
                    "research",
                    "先做深度文献研究",
                    "research_agent",
                    ["literature_search"],
                )
            ],
            risk="low",
        )
    )

    plan = await Orchestrator(llm, catalog).plan(
        "我现在胸痛、呼吸困难，还想了解最新研究",
        {},
    )

    assert plan.risk_level == RiskLevel.EMERGENCY
    assert plan.tasks[0].assigned_agent == "diagnostic_agent"
    assert "risk_assessment" in plan.tasks[0].required_capabilities
    assert all(task.assigned_agent != "research_agent" for task in plan.tasks)
    assert plan.source == RouteSource.SAFETY_RULE


@pytest.mark.asyncio
async def test_llm_declared_high_risk_still_requires_triage_capability(catalog):
    llm = FakeLLM(
        payload_for(
            [
                task(
                    "advice",
                    "提供一般处理建议",
                    "consultation_agent",
                    ["general_health_advice"],
                )
            ],
            risk="high",
        )
    )

    plan = await Orchestrator(llm, catalog).plan("症状比较复杂，请帮我判断", {})

    assert plan.risk_level == RiskLevel.HIGH
    assert plan.tasks[0].assigned_agent == "diagnostic_agent"
    assert plan.tasks[0].priority == "critical"
    assert plan.tasks[1].depends_on == [plan.tasks[0].id]
    assert plan.execution_mode == ExecutionMode.SEQUENTIAL


@pytest.mark.asyncio
async def test_recent_user_red_flag_is_included_in_deterministic_precheck(catalog):
    llm = FakeLLM(
        payload_for(
            [
                task(
                    "consult",
                    "回答用户追问",
                    "consultation_agent",
                    ["general_health_advice"],
                )
            ]
        )
    )

    plan = await Orchestrator(llm, catalog).plan(
        "那我现在怎么办？",
        {
            "recent_history": [
                {"role": "user", "content": "我胸痛而且呼吸困难"},
                {"role": "assistant", "content": "需要重视"},
            ]
        },
    )

    assert plan.risk_level == RiskLevel.EMERGENCY
    assert [item.assigned_agent for item in plan.tasks] == ["diagnostic_agent"]


@pytest.mark.parametrize("question", ["我出现严重过敏", "伤口持续大量出血"])
@pytest.mark.asyncio
async def test_explicit_emergency_markers_skip_secondary_work(
    question,
    catalog,
):
    llm = FakeLLM(
        payload_for(
            [
                task(
                    "research",
                    "检索资料",
                    "research_agent",
                    ["literature_search"],
                )
            ]
        )
    )

    plan = await Orchestrator(llm, catalog).plan(question, {})

    assert plan.risk_level == RiskLevel.EMERGENCY
    assert [item.assigned_agent for item in plan.tasks] == ["diagnostic_agent"]
