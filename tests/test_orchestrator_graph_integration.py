import json

import pytest

from knowledge import KnowledgeChunk, RetrievalBundle
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
        self.last_input = input_data
        return {
            "answer": "保持规律作息并减少睡前刺激 [K1]。以上信息仅供参考，请咨询医生。",
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


class FakeKnowledgeBase:
    def __init__(self):
        self.queries = []

    async def retrieve(self, query, filters=None, top_k=None):
        self.queries.append(query)
        chunk = KnowledgeChunk(
            chunk_id="sleep:v1:0",
            document_id="sleep",
            version="v1",
            title="睡眠卫生指南",
            section="生活方式",
            text="成年人应保持规律睡眠时间。",
            source_org="示例医学会",
            status="ready",
            citation_id="K1",
        )
        return RetrievalBundle(
            status="used",
            query=query,
            chunks=[chunk],
            context="[K1] 成年人应保持规律睡眠时间。",
            sources=[chunk.public_source()],
            candidate_count=1,
        )


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
    knowledge_base = FakeKnowledgeBase()
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
        knowledge_base=knowledge_base,
        enable_rag=True,
    )

    state = await graph.ainvoke(
        {
            "question": "如何改善睡眠？",
            "context": {},
            "session_id": "route-integration",
        }
    )
    result = state["result"]

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
    assert knowledge_base.queries == ["如何改善睡眠？"]
    assert consultation.last_input["context"]["knowledge_bundle"]["status"] == "used"
    assert result["sources"] == [
        {
            "citation_id": "K1",
            "title": "睡眠卫生指南",
            "source_org": "示例医学会",
            "version": "v1",
            "published_at": "",
            "section": "生活方式",
            "external_url": "",
        }
    ]


@pytest.mark.asyncio
async def test_graph_skips_rag_for_emergency_route():
    class EmergencyLLM(FakeLLM):
        async def chat(self, messages, **kwargs):
            value = json.loads(await super().chat(messages, **kwargs))
            value["risk_level"] = "emergency"
            value["intents"] = ["symptom_triage"]
            value["tasks"][0]["required_capabilities"] = ["risk_assessment"]
            value["tasks"][0]["assigned_agent"] = "diagnostic_agent"
            value["tasks"][0]["priority"] = "emergency"
            return json.dumps(value, ensure_ascii=False)

    consultation = Worker("consultation_agent", ["general_health_advice"])
    diagnostic = Worker("diagnostic_agent", ["risk_assessment", "symptom_analysis"])
    research = Worker("research_agent", ["guideline_lookup", "evidence_synthesis"])
    knowledge_base = FakeKnowledgeBase()
    graph = MedicalSwarmGraph(
        llm_client=EmergencyLLM(),
        worker_pool=[consultation, diagnostic, research],
        consultation_agent=consultation,
        diagnostic_agent=diagnostic,
        research_agent=research,
        short_term_memory=Memory(),
        long_term_memory=DisabledLongTerm(),
        session_manager=None,
        enable_long_term_memory=False,
        knowledge_base=knowledge_base,
        enable_rag=True,
    )

    state = await graph.ainvoke(
        {"question": "突然胸痛大汗", "context": {}, "session_id": "emergency-rag"}
    )

    assert knowledge_base.queries == []
    assert state["knowledge_bundle"]["status"] == "skipped"
    assert state["result"]["sources"] == []
