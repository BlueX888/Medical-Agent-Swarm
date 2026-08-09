import asyncio

import pytest

from core.observability import (
    TELEMETRY_SCHEMA_VERSION,
    build_observability_metadata,
    classify_tool_result,
    normalize_error,
    session_reference,
    summarize_tool_input,
    summarize_tool_output,
    trace_async,
)


def test_common_metadata_uses_stable_low_cardinality_schema(monkeypatch):
    monkeypatch.setenv("OBSERVABILITY_ENVIRONMENT", "staging")
    monkeypatch.setenv("OBSERVABILITY_ENTRYPOINT", "cli")
    monkeypatch.setenv("APP_VERSION", "abc123")
    monkeypatch.setenv("OBSERVABILITY_HASH_KEY", "test-key")

    metadata = build_observability_metadata(
        run_id="9b812865-50c9-4561-b8e4-e1f63f4d2d52",
        session_id="patient-session-42",
        route="multiple_tasks",
        status="success",
    )

    assert metadata == {
        "app.name": "medical-agent-swarm",
        "app.version": "abc123",
        "deployment.environment": "staging",
        "telemetry.schema_version": TELEMETRY_SCHEMA_VERSION,
        "entrypoint": "cli",
        "run_id": "9b812865-50c9-4561-b8e4-e1f63f4d2d52",
        "session_ref": session_reference("patient-session-42"),
        "route": "multiple_tasks",
        "status": "success",
    }
    assert "patient-session-42" not in metadata["session_ref"]


def test_session_reference_is_omitted_without_hmac_key(monkeypatch):
    monkeypatch.delenv("OBSERVABILITY_HASH_KEY", raising=False)

    assert session_reference("patient-session-42") is None
    assert "session_ref" not in build_observability_metadata(
        session_id="patient-session-42"
    )


def test_error_normalization_never_includes_exception_message():
    secret = "Alice has chest pain and takes 20mg medication"

    normalized = normalize_error(TimeoutError(secret))

    assert normalized == {"error.type": "TimeoutError", "error.code": "timeout"}
    assert secret not in repr(normalized)


def test_tool_summaries_are_allowlisted_and_contain_no_payload_text():
    secret = "Alice has chest pain and takes 20mg medication"

    inputs = summarize_tool_input("assess_risk", {"symptoms": secret, "age": 42})
    outputs = summarize_tool_output({"success": True, "evidence": secret})

    assert inputs == {
        "tool_name": "assess_risk",
        "argument_keys": ["age", "symptoms"],
        "argument_count": 2,
    }
    assert outputs["success"] is True
    assert outputs["result_kind"] == "object"
    assert outputs["result_size"] > 0
    assert secret not in repr(inputs)
    assert secret not in repr(outputs)


def test_tool_result_classifies_stable_outcomes():
    assert classify_tool_result({"success": True, "data": []}) == "success"
    assert classify_tool_result({"success": False, "error_code": "denied"}) == "error"
    assert classify_tool_result([]) == "empty"
    assert classify_tool_result(None) == "empty"


@pytest.mark.parametrize(
    "field",
    [
        "question",
        "answer",
        "messages",
        "content",
        "context",
        "arguments",
        "result",
        "error",
        "name",
        "phone",
        "id_card",
        "patient_id",
        "symptoms",
        "history",
        "medical_history",
        "medication",
        "dose",
        "evidence",
    ],
)
def test_medical_text_regression_fields_never_export_plaintext(monkeypatch, field):
    from core.observability import sanitize_for_langsmith

    monkeypatch.delenv("LANGSMITH_REDACT_MEDICAL_TEXT", raising=False)
    secret = "Alice 13800138000 chest pain aspirin 20mg"

    sanitized = sanitize_for_langsmith({"nested": {field: secret}})

    assert secret not in repr(sanitized)
    assert sanitized["nested"][field].startswith("[redacted text")


def test_traced_failure_rethrows_original_exception_without_exporting_message(monkeypatch):
    exported = {}
    secret = "Alice has chest pain and takes 20mg medication"
    failure = RuntimeError(secret)

    def fake_traceable(**span):
        exported["span"] = span

        def decorate(func):
            async def wrapped(payload):
                exported["payload"] = payload
                try:
                    exported["output"] = await func(payload)
                    return exported["output"]
                except Exception as error:
                    exported["error"] = str(error)
                    raise

            return wrapped

        return decorate

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setattr("core.observability._load_traceable", lambda: fake_traceable)

    async def fail():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(
            trace_async(
                name="failure",
                run_type="chain",
                func=fail,
                inputs={"question": secret},
            )
        )

    assert caught.value is failure
    assert exported["error"] == "observability outcome: failed"
    assert secret not in repr(exported)


def test_exporter_failure_does_not_fail_or_repeat_business_call(monkeypatch):
    calls = 0

    def broken_traceable(**span):
        def decorate(func):
            async def wrapped(payload):
                await func(payload)
                raise OSError("exporter unavailable")

            return wrapped

        return decorate

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setattr("core.observability._load_traceable", lambda: broken_traceable)

    async def business_call():
        nonlocal calls
        calls += 1
        return {"answer": "still works"}

    result = asyncio.run(
        trace_async(name="request", run_type="chain", func=business_call)
    )

    assert result == {"answer": "still works"}
    assert calls == 1
