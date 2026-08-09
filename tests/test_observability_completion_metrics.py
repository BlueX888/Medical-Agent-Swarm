import asyncio
from types import SimpleNamespace

import pytest

from core.llm_client import LLMClient, LLMResponse
from core.observability import trace_async


class _AsyncStream:
    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        self._iterator = iter(self.chunks)
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration as exc:
            raise StopAsyncIteration from exc


class _StreamingCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["stream"] is True
        return _AsyncStream(
            [
                SimpleNamespace(
                    id="stream-1",
                    model="test-model",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(role="assistant", content=None),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                ),
                SimpleNamespace(
                    id="stream-1",
                    model="test-model",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="Hel"),
                            finish_reason=None,
                        )
                    ],
                    usage=None,
                ),
                SimpleNamespace(
                    id="stream-1",
                    model="test-model",
                    choices=[
                        SimpleNamespace(
                            delta=SimpleNamespace(content="lo"),
                            finish_reason="stop",
                        )
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=4,
                        completion_tokens=2,
                        total_tokens=6,
                    ),
                ),
            ]
        )


def _client_with_completions(completions):
    client = LLMClient.__new__(LLMClient)
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )
    client.model_name = "test-model"
    client.model_type = "openai_compatible"
    client.temperature = 0.0
    client.max_tokens = 32
    client.streaming = True
    return client


@pytest.mark.asyncio
async def test_streaming_llm_collects_ttft_and_usage(monkeypatch):
    captured = []

    async def capture_trace(**kwargs):
        result = await kwargs["func"]()
        mapped = kwargs["output_mapper"](result)
        captured.append(mapped)
        return result

    monkeypatch.setattr("core.llm_client.trace_async", capture_trace)
    completions = _StreamingCompletions()
    client = _client_with_completions(completions)

    response = await client.chat_with_tools(
        messages=[{"role": "user", "content": "hello"}],
        trace_name="diagnostic_agent.iteration",
        stream=True,
    )

    assert response.content == "Hello"
    assert response.finish_reason == "stop"
    assert response.usage["total_tokens"] == 6
    assert response.ttft_ms is not None
    assert response.duration_ms is not None
    assert response.streamed is True
    assert captured[0]["llm.ttft_ms"] >= 0
    assert captured[0]["llm.input_tokens"] == 4
    assert completions.calls[0]["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_root_trace_aggregates_ttft_retries_durations_and_exceptions(monkeypatch):
    captured = []

    def fake_traceable(**span):
        def decorate(func):
            async def wrapped(payload):
                try:
                    output = await func(payload)
                except Exception as error:
                    captured.append({**span, "error": str(error)})
                    raise
                captured.append({**span, "output": output})
                return output

            return wrapped

        return decorate

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setattr("core.observability._load_traceable", lambda: fake_traceable)

    async def request():
        await trace_async(
            name="llm.diagnostic_agent.iteration",
            run_type="llm",
            func=lambda: asyncio.sleep(0, result="ok"),
            metadata={"llm.retry_count": 1},
            output_mapper=lambda _: {
                "status": "success",
                "llm.input_tokens": 5,
                "llm.output_tokens": 2,
                "llm.total_tokens": 7,
                "llm.ttft_ms": 12.5,
                "duration_ms": 80.0,
                "llm.outcome": "success",
            },
        )
        try:
            await trace_async(
                name="tool.assess_risk",
                run_type="tool",
                func=lambda: asyncio.sleep(0, result=None),
                    metadata={"tool.retry_count": 0},
                output_mapper=lambda _: {
                    "status": "failed",
                    "tool.outcome": "timeout",
                    "tool.retry_count": 1,
                    "duration_ms": 30.0,
                },
            )
        except TimeoutError:
            pass
        return {"status": "degraded"}

    await trace_async(
        name="medical_swarm_request",
        run_type="chain",
        func=request,
        output_mapper=lambda value: {
            "status": value["status"],
            "route": "safe_fallback",
        },
    )

    root = next(item for item in captured if item["name"] == "medical_swarm_request")
    output = root["output"]
    assert output["retry_count"] == 2
    assert output["retry_success_count"] == 1
    assert output["retry_exhausted_count"] == 1
    assert output["exception_count"] == 0
    assert output["llm_ttft_ms_count"] == 1
    assert output["llm_ttft_ms_avg"] == 12.5
    assert output["llm_duration_ms_total"] == 80.0
    assert output["tool_duration_ms_total"] == 30.0


@pytest.mark.asyncio
async def test_failed_tool_retry_is_aggregated_from_exception_metadata(monkeypatch):
    captured = []

    def fake_traceable(**span):
        def decorate(func):
            async def wrapped(payload):
                try:
                    output = await func(payload)
                except Exception:
                    captured.append({**span, "failed": True})
                    raise
                captured.append({**span, "output": output})
                return output

            return wrapped

        return decorate

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setattr("core.observability._load_traceable", lambda: fake_traceable)

    async def request():
        async def fail_tool():
            raise TimeoutError("transient provider")

        try:
            await trace_async(
                name="tool.deep_research",
                run_type="tool",
                func=fail_tool,
                metadata={
                    "tool.retry_count": 2,
                    "tool.retry_exhausted": True,
                },
                output_mapper=lambda _: {
                    "status": "failed",
                    "tool.outcome": "timeout",
                },
            )
        except Exception:
            pass
        return {"status": "degraded"}

    await trace_async(
        name="medical_swarm_request",
        run_type="chain",
        func=request,
        output_mapper=lambda value: value,
    )

    root = next(item for item in captured if item["name"] == "medical_swarm_request")
    assert root["output"]["retry_count"] == 1
    assert root["output"]["retry_exhausted_count"] == 1


@pytest.mark.asyncio
async def test_chat_with_tools_retry_records_attempt_and_reason(monkeypatch):
    client = LLMClient.__new__(LLMClient)
    attempts = []

    async def fake_chat(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise TimeoutError("provider timeout")
        return LLMResponse(content="ok", tool_calls=[], finish_reason="stop")

    async def no_sleep(_):
        return None

    client.chat_with_tools = fake_chat
    monkeypatch.setattr("core.llm_client.asyncio.sleep", no_sleep)

    result = await client.chat_with_tools_retry(
        messages=[{"role": "user", "content": "hello"}],
        max_retries=1,
    )

    assert result.content == "ok"
    assert [attempt["retry_count"] for attempt in attempts] == [0, 1]
    assert attempts[1]["retry_reason"] == "timeout"


def test_structured_observability_log_contains_correlation_fields(monkeypatch):
    events = []

    class BoundLogger:
        def bind(self, **fields):
            events.append(fields)
            return self

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    monkeypatch.setattr("core.observability.logger", BoundLogger())
    from core.observability import log_observability_event

    log_observability_event(
        "span.completed",
        name="tool.assess_risk",
        run_type="tool",
        run_id="run-1",
        duration_ms=12.0,
        retry_count=0,
        status="success",
    )

    assert events[0]["observability.event"] == "span.completed"
    assert events[0]["run_id"] == "run-1"
    assert events[0]["duration_ms"] == 12.0


@pytest.mark.asyncio
async def test_untraced_logging_reports_mapped_failure(monkeypatch):
    events = []

    class BoundLogger:
        def bind(self, **fields):
            events.append(fields)
            return self

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr("core.observability.logger", BoundLogger())

    result = await trace_async(
        name="tool.assess_risk",
        run_type="tool",
        func=lambda: asyncio.sleep(0, result={"success": False}),
        metadata={"run_id": "run-2", "tool.retry_count": 1},
        output_mapper=lambda _: {
            "status": "failed",
            "tool.outcome": "error",
        },
    )

    assert result == {"success": False}
    assert [event["observability.event"] for event in events] == [
        "span.started",
        "span.failed",
    ]
    assert events[-1]["run_id"] == "run-2"
    assert events[-1]["retry_count"] == 1


@pytest.mark.asyncio
async def test_untraced_root_log_contains_aggregated_metrics(monkeypatch):
    events = []

    class BoundLogger:
        def bind(self, **fields):
            events.append(fields)
            return self

        def info(self, *args, **kwargs):
            return None

        def warning(self, *args, **kwargs):
            return None

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setattr("core.observability.logger", BoundLogger())

    async def request():
        await trace_async(
            name="llm.diagnostic_agent.iteration",
            run_type="llm",
            func=lambda: asyncio.sleep(0, result="ok"),
            output_mapper=lambda _: {
                "status": "success",
                "llm.input_tokens": 2,
                "llm.output_tokens": 3,
                "llm.total_tokens": 5,
                "llm.ttft_ms": 4,
                "duration_ms": 6,
            },
        )
        return "ok"

    await trace_async(
        name="medical_swarm_request",
        run_type="chain",
        func=request,
        output_mapper=lambda _: {"status": "success", "route": "safe_fallback"},
    )

    root_events = [
        event
        for event in events
        if event.get("span.name") == "medical_swarm_request"
    ]
    assert root_events[-1]["llm_call_count"] == 1
    assert root_events[-1]["total_tokens"] == 5
    assert root_events[-1]["llm_ttft_ms_avg"] == 4
