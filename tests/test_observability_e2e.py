import asyncio

from core.observability import trace_async


def test_nested_spans_roll_up_into_one_root_summary(monkeypatch):
    captured = []

    def fake_traceable(**span):
        def decorate(func):
            async def wrapped(payload):
                output = await func(payload)
                captured.append({**span, "inputs": payload, "output": output})
                return output

            return wrapped

        return decorate

    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "test")
    monkeypatch.setenv("OBSERVABILITY_HASH_KEY", "test-hmac-key")
    monkeypatch.setattr("core.observability._load_traceable", lambda: fake_traceable)

    async def agent(agent_id):
        async def work():
            await trace_async(
                name=f"llm.{agent_id}.iteration",
                run_type="llm",
                func=lambda: asyncio.sleep(0, result="provider-response"),
                output_mapper=lambda _: {
                    "llm.input_tokens": 5,
                    "llm.output_tokens": 2,
                    "llm.total_tokens": 7,
                    "llm.tool_calls_requested": 1,
                    "llm.outcome": "success",
                },
            )
            await trace_async(
                name="tool.assess_risk",
                run_type="tool",
                func=lambda: asyncio.sleep(0, result={"success": True}),
                output_mapper=lambda _: {
                    "success": True,
                    "result_kind": "object",
                    "result_size": 16,
                    "tool.outcome": "success",
                },
            )
            return {"answer": "private"}

        return await trace_async(
            name=f"agent.{agent_id}",
            run_type="chain",
            func=work,
            metadata={"agent_id": agent_id},
            output_mapper=lambda _: {"status": "success"},
        )

    async def request():
        await asyncio.gather(agent("diagnostic_agent"), agent("research_agent"))
        await trace_async(
            name="safety.runtime_guard",
            run_type="chain",
            func=lambda: asyncio.sleep(0, result=True),
            output_mapper=lambda _: {
                "safety.executed": True,
                "safety.passed": True,
                "safety.outcome": "success",
            },
        )
        return {"answer": "private final"}

    asyncio.run(
        trace_async(
            name="medical_swarm_request",
            run_type="chain",
            func=request,
            metadata={"session_id": "private-session"},
            output_mapper=lambda result: {
                "status": "success",
                "route": "swarm",
                "answer_length": len(result["answer"]),
            },
        )
    )

    roots = [span for span in captured if span["name"] == "medical_swarm_request"]
    assert len(roots) == 1
    run_ids = {span["metadata"]["run_id"] for span in captured}
    session_refs = {span["metadata"]["session_ref"] for span in captured}
    assert len(run_ids) == 1
    assert len(session_refs) == 1
    assert "private-session" not in repr(captured)
    assert roots[0]["output"] == {
        "status": "success",
        "route": "swarm",
        "answer_length": len("private final"),
        "agent_count": 2,
        "llm_call_count": 2,
        "input_tokens": 10,
        "output_tokens": 4,
        "total_tokens": 14,
        "tool_call_count": 2,
        "tool_success_count": 2,
        "tool_blocked": 0,
        "tool_failed": 0,
        "safety_checked": True,
        "safety_passed": True,
        "safety_error": False,
    }
    assert "private" not in repr(captured)


def test_llm_timeout_is_safe_and_rolls_up_as_failed_call(monkeypatch):
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
    private_message = "Alice chest pain and medication history"

    async def request():
        async def timeout():
            raise TimeoutError(private_message)

        try:
            await trace_async(
                name="llm.diagnostic_agent.iteration",
                run_type="llm",
                func=timeout,
            )
        except TimeoutError:
            pass
        return {"status": "degraded"}

    asyncio.run(
        trace_async(
            name="medical_swarm_request",
            run_type="chain",
            func=request,
            output_mapper=lambda value: {
                "status": value["status"],
                "route": "fallback",
            },
        )
    )

    llm_span = next(span for span in captured if span["run_type"] == "llm")
    root_span = next(
        span for span in captured if span["name"] == "medical_swarm_request"
    )
    assert llm_span["error"] == "observability outcome: timeout"
    assert root_span["output"]["llm_call_count"] == 1
    assert root_span["output"]["status"] == "degraded"
    assert private_message not in repr(captured)
