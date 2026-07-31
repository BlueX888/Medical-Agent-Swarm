from typing import Any

import pytest

from core.agent_loop import AgentLoop
from core.llm_client import LLMResponse, ToolCall


class SequencedLLM:
    def __init__(self, tool_name: str, tool_arguments=None):
        self.tool_name = tool_name
        self.tool_arguments = tool_arguments or {
            "symptoms": "private chest pain text"
        }
        self.calls = 0

    async def chat_with_tools(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name=self.tool_name,
                        arguments=self.tool_arguments,
                    )
                ],
                finish_reason="tool_calls",
            )
        return LLMResponse(content="done", tool_calls=[], finish_reason="stop")

    def create_tool_message(self, *, tool_call_id, tool_name, result):
        return {"role": "tool", "tool_call_id": tool_call_id, "content": str(result)}


class ToolAgent:
    agent_id = "diagnostic_agent"
    config = {"temperature": 0}

    def __init__(
        self,
        tool_name="assess_risk",
        tool_result=None,
        tool_arguments=None,
    ):
        self.llm_client = SequencedLLM(tool_name, tool_arguments)
        self.executed = []
        self.tool_result = tool_result or {
            "success": True,
            "risk_level": "low",
            "evidence": "private result",
        }

    def get_system_prompt(self):
        return "system"

    def format_user_input(self, input_data):
        return input_data["question"]

    def get_tools_for_llm(self, **kwargs):
        return [
            {
                "type": "function",
                "function": {
                    "name": "assess_risk",
                    "parameters": {
                        "type": "object",
                        "properties": {"symptoms": {"type": "string"}},
                    },
                },
            }
        ]

    async def execute_tool(self, *, tool_name, arguments):
        self.executed.append((tool_name, arguments))
        return self.tool_result


@pytest.mark.asyncio
async def test_successful_tool_call_emits_one_redacted_tool_span(monkeypatch):
    spans: list[dict[str, Any]] = []

    async def capture_trace(**kwargs):
        result = await kwargs["func"]()
        spans.append(
            {
                "name": kwargs["name"],
                "run_type": kwargs["run_type"],
                "inputs": kwargs.get("inputs"),
                "metadata": kwargs.get("metadata"),
                "output": kwargs["output_mapper"](result)
                if kwargs.get("output_mapper")
                else result,
            }
        )
        return result

    monkeypatch.setattr("core.agent_loop.trace_async", capture_trace)
    agent = ToolAgent()

    result = await AgentLoop(max_iterations=2).run(
        agent,
        {"question": "private question", "tool_policy": {"allow_tools": ["assess_risk"]}},
    )

    tool_spans = [span for span in spans if span["run_type"] == "tool"]
    state_spans = [span for span in spans if span["name"].startswith("state.")]
    assert result["answer"] == "done"
    assert len(tool_spans) == 1
    assert {span["metadata"]["agent.state_stage"] for span in state_spans} == {
        "after_llm",
        "after_tool",
        "final_output",
    }
    assert tool_spans[0]["name"] == "tool.assess_risk"
    assert tool_spans[0]["inputs"] == {
        "tool_name": "assess_risk",
        "argument_keys": ["symptoms"],
        "argument_count": 1,
    }
    assert tool_spans[0]["output"]["tool.outcome"] == "success"
    assert "private" not in repr(tool_spans[0])


@pytest.mark.asyncio
async def test_policy_blocked_tool_still_emits_tool_span(monkeypatch):
    spans = []

    async def capture_trace(**kwargs):
        result = await kwargs["func"]()
        spans.append((kwargs, kwargs["output_mapper"](result)))
        return result

    monkeypatch.setattr("core.agent_loop.trace_async", capture_trace)
    agent = ToolAgent()

    await AgentLoop(max_iterations=2).run(
        agent,
        {
            "question": "private question",
            "tool_policy": {
                "allow_tools": ["another_tool"],
                "reason": "policy",
            },
        },
    )

    tool_spans = [item for item in spans if item[0]["run_type"] == "tool"]
    assert len(tool_spans) == 1
    assert tool_spans[0][1]["tool.outcome"] == "blocked"
    assert agent.executed == []


@pytest.mark.asyncio
async def test_failed_tool_result_marks_span_failed_without_result_text(monkeypatch):
    spans = []

    async def capture_trace(**kwargs):
        result = await kwargs["func"]()
        output = (
            kwargs["output_mapper"](result)
            if kwargs.get("output_mapper")
            else result
        )
        spans.append((kwargs, output))
        return result

    monkeypatch.setattr("core.agent_loop.trace_async", capture_trace)
    agent = ToolAgent(
        tool_result={
            "success": False,
            "error_code": "provider_error",
            "error": "private patient result",
        }
    )

    await AgentLoop(max_iterations=2).run(
        agent,
        {"question": "private question"},
    )

    tool_span = next(item for item in spans if item[0]["run_type"] == "tool")
    assert tool_span[1]["status"] == "failed"
    assert tool_span[1]["tool.outcome"] == "error"
    assert "private patient result" not in repr(tool_span)


@pytest.mark.asyncio
async def test_unknown_model_supplied_tool_name_is_canonicalized(monkeypatch):
    spans = []

    async def capture_trace(**kwargs):
        result = await kwargs["func"]()
        spans.append(kwargs)
        return result

    monkeypatch.setattr("core.agent_loop.trace_async", capture_trace)
    agent = ToolAgent(tool_name="Alice chest pain 13800138000")

    await AgentLoop(max_iterations=2).run(
        agent,
        {
            "question": "private question",
            "tool_policy": {"allow_tools": ["assess_risk"]},
        },
    )

    tool_span = next(span for span in spans if span["run_type"] == "tool")
    assert tool_span["name"] == "tool.unknown"
    assert tool_span["metadata"]["tool.name"] == "unknown"
    assert "Alice" not in repr(tool_span)


@pytest.mark.asyncio
async def test_model_supplied_argument_key_is_canonicalized_to_schema(monkeypatch):
    spans = []

    async def capture_trace(**kwargs):
        result = await kwargs["func"]()
        spans.append(kwargs)
        return result

    monkeypatch.setattr("core.agent_loop.trace_async", capture_trace)
    agent = ToolAgent(
        tool_arguments={
            "symptoms": "private value",
            "Alice chest pain 13800138000": "private value",
        }
    )

    await AgentLoop(max_iterations=2).run(
        agent,
        {"question": "private question"},
    )

    tool_span = next(span for span in spans if span["run_type"] == "tool")
    assert tool_span["inputs"]["argument_keys"] == ["symptoms", "unknown"]
    assert "Alice" not in repr(tool_span)


@pytest.mark.asyncio
async def test_tool_retry_is_recorded_on_the_tool_span(monkeypatch):
    spans = []

    async def capture_trace(**kwargs):
        result = await kwargs["func"]()
        output = kwargs["output_mapper"](result)
        spans.append((kwargs, output))
        return result

    monkeypatch.setattr("core.agent_loop.trace_async", capture_trace)
    agent = ToolAgent()
    calls = 0

    async def flaky_execute_tool(*, tool_name, arguments):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("transient")
        return agent.tool_result

    agent.execute_tool = flaky_execute_tool
    result = await AgentLoop(max_iterations=2, max_tool_retries=1).run(
        agent,
        {"question": "private question"},
    )

    tool_span = next(item for item in spans if item[0]["run_type"] == "tool")
    assert result["answer"] == "done"
    assert calls == 2
    assert tool_span[1]["tool.retry_count"] == 1
