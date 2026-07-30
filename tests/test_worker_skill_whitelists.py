import pytest

from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent


@pytest.mark.parametrize(
    ("agent_type", "allowed"),
    [
        (
            ConsultationAgent,
            {
                "collect_clinical_context",
                "assess_risk",
                "analyze_symptoms",
                "recommend_lifestyle",
            },
        ),
        (
            DiagnosticAgent,
            {"collect_clinical_context", "assess_risk", "analyze_symptoms"},
        ),
        (
            ResearchAgent,
            {"deep_research", "collect_clinical_context"},
        ),
    ],
)
def test_worker_exposes_only_its_allowed_skills(monkeypatch, agent_type, allowed):
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_MODEL", "test")
    agent = agent_type()

    exposed = {
        tool["function"]["name"]
        for tool in agent.get_tools_for_llm()
    }

    assert exposed == allowed
