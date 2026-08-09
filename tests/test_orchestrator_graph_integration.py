import json

import pytest

from swarm.medical_swarm_graph import MedicalSwarmGraph


class FakeLLM:
    async def chat(self, messages, **kwargs):
        return json.dumps(
            {
                "intent_summary": "睡眠生活方式建议",
                "intents": ["lifestyle_guidance"],
                "risk_level": "low",
                "confidence": 0.93,
                "tasks": [
                    {
                        "id": "sleep",
                        "goal": "给出改善睡眠的生活方式建议",
                        "required_capabilities": ["general_health_advice"],
                        "assigned_agent": "consultation_agent",
                        "priority": "normal",
                        "depends_on": [],
                    }
                ],
                "execution_mode": "single",
                "source": "llm",
                "reasons": ["单 Worker 足够"],
                "needs_clarification": False,
            },
            ensure_ascii=False,
        )


class Worker:
    def __init__(self, agent_id, capabilities):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.config = {"description": agent_id}

    def get_capabilities(self):
        return self.capabilities

    def attach_shared_context(self, context):
        self.context = context

    async def process(self, input_data):
        return {
            "answer": "保持规律作息并减少睡前刺激。以上信息仅供参考，请咨询医生。",
            "suggestions": ["规律作息"],
            "disclaimer": "以上信息仅供参考，请咨询医生。",
        }


class Memory:
    async def load_context(self, session_id, max_turns=5):
        return []

    async def save_turn(self, **kwargs):
        return None


class DisabledLongTerm:
    enabled = False


@pytest.mark.asyncio
async def test_graph_uses_route_plan_and_keeps_public_result_shape():
    consultation = Worker(
        "consultation_agent",
        ["general_health_advice", "risk_assessment", "symptom_triage"],
    )
    diagnostic = Worker(
        "diagnostic_agent",
        ["risk_assessment", "symptom_analysis", "clinical_reasoning"],
    )
    research = Worker(
        "research_agent",
        ["guideline_lookup", "evidence_synthesis"],
    )
    graph = MedicalSwarmGraph(
        llm_client=FakeLLM(),
        worker_pool=[consultation, diagnostic, research],
        consultation_agent=consultation,
        diagnostic_agent=diagnostic,
        research_agent=research,
        short_term_memory=Memory(),
        long_term_memory=DisabledLongTerm(),
        session_manager=None,
        enable_long_term_memory=False,
    )
    graph.orchestrator.evidence_memory.lookup = lambda *args, **kwargs: None

    state = await graph.ainvoke(
        {
            "question": "如何改善睡眠？",
            "context": {},
            "session_id": "route-integration",
        }
    )
    result = state["result"]

    assert state["route"] == "single_task"
    assert result["swarm_enabled"] is False
    assert result["agents_involved"] == ["consultation_agent"]
    assert result["timeout_occurred"] is False
    assert result["safety_checked"] is True
    assert result["routing"] == {
        "intents": ["lifestyle_guidance"],
        "risk_level": "low",
        "confidence": 0.93,
        "execution_mode": "single",
        "source": "llm",
    }
