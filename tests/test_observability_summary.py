from datetime import datetime, timedelta

from debug.models import DebugEvent, DebugRun
from debug.observability_summary import summarize_debug_run


def test_debug_run_summary_matches_root_observability_shape():
    started = datetime(2026, 7, 30, 8, 0, 0)
    run = DebugRun(
        run_id="run-1",
        session_id="secret-session",
        question="private medical question",
        started_at=started,
        ended_at=started + timedelta(milliseconds=1250),
        route="swarm",
        status="success",
        final_answer="private answer",
    )
    events = [
        DebugEvent(
            timestamp=started,
            stage="llm_call",
            agent_id="diagnostic_agent",
            metadata={"usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}},
        ),
        DebugEvent(
            timestamp=started,
            stage="skill_call",
            agent_id="diagnostic_agent",
            skill_name="assess_risk",
            status="success",
        ),
        DebugEvent(
            timestamp=started,
            stage="constraint_check",
            agent_id="research_agent",
            skill_name="web_search",
            name="tool_policy",
            status="failed",
        ),
        DebugEvent(
            timestamp=started,
            stage="safety_check",
            status="success",
            output={"safety_checked": True, "safety_passed": True},
        ),
    ]

    summary = summarize_debug_run(run, events)

    assert summary == {
        "status": "success",
        "route": "swarm",
        "duration_ms": 1250.0,
        "agent_count": 2,
        "llm_call_count": 1,
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "tool_requested": 2,
        "tool_executed": 1,
        "tool_call_count": 2,
        "tool_success_count": 1,
        "tool_failed": 0,
        "tool_blocked": 1,
        "safety_checked": True,
        "safety_passed": True,
        "safety_error": False,
        "answer_length": len("private answer"),
    }
