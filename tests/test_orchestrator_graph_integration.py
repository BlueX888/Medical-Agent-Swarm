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


MISSING = object()


class PlanLLM:
    def __init__(self, *, intents, knowledge_need=MISSING, needs_clarification=False, risk="low"):
        self.intents = intents
        self.knowledge_need = knowledge_need
        self.needs_clarification = needs_clarification
        self.risk = risk

    async def chat(self, messages, **kwargs):
        payload = {
                "intent_summary": "Assess a low-risk symptom and give care guidance",
                "intents": self.intents,
                "risk_level": self.risk,
                "confidence": 0.91,
                "tasks": [
                    {
                        "id": "triage",
                        "goal": "Assess symptoms and advise on care and escalation timing",
                        "required_capabilities": ["risk_assessment", "symptom_analysis"],
                        "assigned_agent": "diagnostic_agent",
                        "priority": "normal",
                        "depends_on": [],
                    }
                ],
                "execution_mode": "single",
                "source": "llm",
                "reasons": ["Medical claims need evidence"],
                "needs_clarification": self.needs_clarification,
            }
        if self.knowledge_need is not MISSING:
            payload["knowledge_need"] = self.knowledge_need
        return json.dumps(payload)


def build_graph_for_rag_policy(llm, knowledge_base):
    consultation = Worker("consultation_agent", ["general_health_advice"])
    diagnostic = Worker(
        "diagnostic_agent",
        ["risk_assessment", "symptom_analysis", "clinical_reasoning"],
    )
    research = Worker("research_agent", ["guideline_lookup", "evidence_synthesis"])
    return MedicalSwarmGraph(
        llm_client=llm,
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


@pytest.mark.asyncio
async def test_low_risk_symptom_advice_retrieves_medical_evidence():
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(
        PlanLLM(intents=["symptom_triage"], knowledge_need="required"),
        knowledge_base,
    )
    question = "I have had a cough for two weeks. What should I monitor and when seek care?"

    state = await graph.ainvoke(
        {
            "question": question,
            "context": {},
            "session_id": "low-risk-triage-rag",
        }
    )

    assert knowledge_base.queries == [question]
    assert state["knowledge_bundle"]["status"] == "used"
    assert state["result"]["routing"]["intents"] == ["symptom_triage"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "needs_clarification"),
    [
        ("Please provide the missing symptom duration.", True),
        ("How do I upload a document to the application?", False),
    ],
)
async def test_requests_without_medical_evidence_need_skip_rag(
    question,
    needs_clarification,
):
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(
        PlanLLM(
            intents=["general_consultation"],
            knowledge_need="none",
            needs_clarification=needs_clarification,
        ),
        knowledge_base,
    )

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": "no-rag-needed"}
    )

    assert knowledge_base.queries == []
    assert state["knowledge_bundle"]["status"] == "skipped"
    assert state["knowledge_bundle"]["error"] == "no_medical_evidence_needed"


@pytest.mark.asyncio
async def test_clarification_does_not_block_evidence_needed_by_the_answer():
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(
        PlanLLM(
            intents=["diagnostic_reasoning"],
            knowledge_need="required",
            needs_clarification=True,
        ),
        knowledge_base,
    )
    question = "What can cause this symptom, and how long has it been present?"

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": "clarify-with-evidence"}
    )

    assert knowledge_base.queries == [question]
    assert state["knowledge_bundle"]["status"] == "used"


@pytest.mark.asyncio
async def test_general_consultation_can_clarify_and_retrieve_background_evidence():
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(
        PlanLLM(
            intents=["general_consultation"],
            knowledge_need="required",
            needs_clarification=True,
        ),
        knowledge_base,
    )
    question = "My blood pressure is high. What should I do and what details do you need?"

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": "clarify-general-evidence"}
    )

    assert knowledge_base.queries == [question]
    assert state["knowledge_bundle"]["status"] == "used"


@pytest.mark.asyncio
async def test_diagnostic_answer_requires_evidence_even_if_planner_marks_none():
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(
        PlanLLM(intents=["diagnostic_reasoning"], knowledge_need="none"),
        knowledge_base,
    )
    question = "What diagnoses could explain persistent swelling?"

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": "diagnosis-needs-rag"}
    )

    assert knowledge_base.queries == [question]
    assert state["knowledge_bundle"]["status"] == "used"


@pytest.mark.asyncio
async def test_system_operation_skips_rag_when_planner_falls_back():
    class BrokenLLM:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("planner unavailable")

    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(BrokenLLM(), knowledge_base)
    question = "How do I upload a document to the application?"

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": "fallback-system-operation"}
    )

    assert knowledge_base.queries == []
    assert state["knowledge_bundle"]["error"] == "no_medical_evidence_needed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    ["How are you?", "Tell me a joke.", "Hello!", "Open my application profile."],
)
async def test_unmistakable_chitchat_and_app_fallbacks_skip_rag(question):
    class BrokenLLM:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("planner unavailable")

    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(BrokenLLM(), knowledge_base)

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": "fallback-no-evidence"}
    )

    assert knowledge_base.queries == []
    assert state["knowledge_bundle"]["error"] == "no_medical_evidence_needed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent",
    ["treatment_guidance", "medication_guidance", "prognosis_guidance"],
)
async def test_treatment_medication_and_prognosis_require_evidence(intent):
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(
        PlanLLM(intents=[intent], knowledge_need="none"),
        knowledge_base,
    )
    question = f"Please provide {intent.replace('_', ' ')}."

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": f"mandatory-{intent}"}
    )

    assert knowledge_base.queries == [question]
    assert state["result"]["routing"]["intents"] == [intent]


@pytest.mark.asyncio
async def test_missing_knowledge_need_skips_pure_clarification():
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(
        PlanLLM(intents=["general_consultation"], needs_clarification=True),
        knowledge_base,
    )

    state = await graph.ainvoke(
        {
            "question": "Please clarify how long the symptom has lasted.",
            "context": {},
            "session_id": "legacy-pure-clarification",
        }
    )

    assert knowledge_base.queries == []
    assert state["knowledge_bundle"]["error"] == "no_medical_evidence_needed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "question"),
    [
        ("non_medical", "Tell me a joke."),
        ("system_operation", "Open my application profile."),
    ],
)
async def test_typed_nonmedical_requests_skip_rag_when_field_is_omitted(
    intent,
    question,
):
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(PlanLLM(intents=[intent]), knowledge_base)

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": f"typed-{intent}"}
    )

    assert knowledge_base.queries == []
    assert state["result"]["routing"]["intents"] == [intent]
    assert state["knowledge_bundle"]["error"] == "no_medical_evidence_needed"


@pytest.mark.asyncio
async def test_system_operation_does_not_veto_combined_medical_evidence_need():
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(
        PlanLLM(
            intents=["system_operation", "general_consultation"],
            knowledge_need="required",
        ),
        knowledge_base,
    )
    question = "How do I upload a file, and what does this blood-pressure result mean?"

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": "mixed-system-medical"}
    )

    assert knowledge_base.queries == [question]
    assert state["knowledge_bundle"]["status"] == "used"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "What causes rapid heartbeat?",
        "免疫系统疾病如何治疗？",
    ],
)
async def test_fallback_medical_words_do_not_look_like_system_operations(question):
    class BrokenLLM:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("planner unavailable")

    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(BrokenLLM(), knowledge_base)

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": "fallback-medical-query"}
    )

    assert knowledge_base.queries == [question]
    assert state["knowledge_bundle"]["status"] == "used"


@pytest.mark.asyncio
async def test_medium_risk_symptom_handling_retrieves_evidence():
    knowledge_base = FakeKnowledgeBase()
    graph = build_graph_for_rag_policy(
        PlanLLM(
            intents=["symptom_triage"],
            knowledge_need="required",
            risk="medium",
        ),
        knowledge_base,
    )
    question = "How should I handle persistent moderate nausea at home?"

    state = await graph.ainvoke(
        {"question": question, "context": {}, "session_id": "medium-triage-rag"}
    )

    assert knowledge_base.queries == [question]
    assert state["knowledge_bundle"]["status"] == "used"
