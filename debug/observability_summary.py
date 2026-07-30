"""Deterministic observability summaries derived from local debug events."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from .models import DebugEvent, DebugRun


def summarize_debug_run(
    run: DebugRun,
    events: Iterable[DebugEvent],
) -> Dict[str, Any]:
    """Return the local fact-source summary mirrored by the root trace."""
    event_list = list(events)
    tool_events = [event for event in event_list if event.stage == "skill_call"]
    blocked_events = [
        event
        for event in event_list
        if event.stage == "constraint_check"
        and event.name == "tool_policy"
        and event.status == "failed"
    ]
    llm_events = [event for event in event_list if event.stage == "llm_call"]
    safety_events = [event for event in event_list if event.stage == "safety_check"]

    token_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for event in llm_events:
        usage = _usage_from_event(event)
        token_totals["input_tokens"] += _integer(
            usage.get("input_tokens", usage.get("prompt_tokens", 0))
        )
        token_totals["output_tokens"] += _integer(
            usage.get("output_tokens", usage.get("completion_tokens", 0))
        )
        total = usage.get("total_tokens")
        token_totals["total_tokens"] += (
            _integer(total)
            if total is not None
            else _integer(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
            + _integer(usage.get("output_tokens", usage.get("completion_tokens", 0)))
        )

    safety_checked = False
    safety_passed = False
    safety_error = False
    if safety_events:
        final_safety = safety_events[-1]
        output = final_safety.output if isinstance(final_safety.output, Mapping) else {}
        safety_checked = bool(output.get("safety_checked"))
        safety_passed = bool(output.get("safety_passed"))
        safety_error = not safety_checked or final_safety.status not in {"success", "failed"}

    executed_success = sum(event.status == "success" for event in tool_events)
    executed_failed = sum(event.status != "success" for event in tool_events)
    agent_ids = {event.agent_id for event in event_list if event.agent_id}
    duration_ms = None
    if run.ended_at is not None:
        duration_ms = round((run.ended_at - run.started_at).total_seconds() * 1000, 3)

    return {
        "status": run.status,
        "route": run.route or "unknown",
        "duration_ms": duration_ms,
        "agent_count": len(agent_ids),
        "llm_call_count": len(llm_events),
        **token_totals,
        "tool_requested": len(tool_events) + len(blocked_events),
        "tool_executed": len(tool_events),
        "tool_call_count": len(tool_events) + len(blocked_events),
        "tool_success_count": executed_success,
        "tool_failed": executed_failed,
        "tool_blocked": len(blocked_events),
        "safety_checked": safety_checked,
        "safety_passed": safety_passed,
        "safety_error": safety_error,
        "answer_length": len(run.final_answer or ""),
    }


def _usage_from_event(event: DebugEvent) -> Mapping[str, Any]:
    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    usage = metadata.get("usage")
    if isinstance(usage, Mapping):
        return usage
    output = event.output if isinstance(event.output, Mapping) else {}
    usage = output.get("usage")
    return usage if isinstance(usage, Mapping) else {}


def _integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
