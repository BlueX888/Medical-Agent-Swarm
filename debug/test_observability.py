import asyncio

from core.observability import sanitize_for_langsmith, trace_async


def test_sanitize_redacts_medical_text_by_default(monkeypatch):
    monkeypatch.delenv("LANGSMITH_REDACT_MEDICAL_TEXT", raising=False)

    payload = {
        "question": "Patient has chest pain and shortness of breath",
        "messages": [{"role": "user", "content": "My blood pressure is high"}],
        "api_key": "secret",
        "model": "test-model",
    }

    safe = sanitize_for_langsmith(payload)

    assert safe["question"].startswith("[redacted text")
    assert safe["messages"][0]["content"].startswith("[redacted text")
    assert safe["api_key"] == "[redacted]"
    assert safe["model"] == "test-model"


def test_trace_async_is_noop_when_langsmith_disabled(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    calls = 0

    async def run():
        nonlocal calls
        calls += 1
        return {"answer": "ok"}

    result = asyncio.run(
        trace_async(
            name="test",
            run_type="chain",
            func=run,
            inputs={"question": "hello"},
        )
    )

    assert calls == 1
    assert result == {"answer": "ok"}


def test_trace_async_is_noop_without_langsmith_api_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    calls = 0

    async def run():
        nonlocal calls
        calls += 1
        return {"answer": "ok"}

    result = asyncio.run(
        trace_async(
            name="test",
            run_type="chain",
            func=run,
            inputs={"question": "hello"},
        )
    )

    assert calls == 1
    assert result == {"answer": "ok"}
