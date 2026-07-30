from agents.base_agent import BaseAgent


def test_default_worker_input_preserves_context_risk_and_dependencies():
    formatted = BaseAgent.format_user_input(
        object(),
        {
            "question": "分析当前子任务",
            "context": {"original_user_question": "我头痛并恶心"},
            "risk_level": "high",
            "priority": "critical",
            "dependency_results": {"triage": {"answer": "建议尽快就医"}},
        }
    )

    assert "我头痛并恶心" in formatted
    assert "high" in formatted
    assert "critical" in formatted
    assert "建议尽快就医" in formatted
