import asyncio
import time
from contextlib import asynccontextmanager

from fastapi.testclient import TestClient
import pytest

import api.server as server
from core.checkpointing import CheckpointSnapshot
from debug import DebugTraceCollector
from memory import ShortTermMemory, ShortTermMemoryUnavailable


def test_lifespan_closes_memory_when_checkpointer_startup_fails(
    monkeypatch,
    short_term_memory_factory,
):
    memory = short_term_memory_factory()
    closed = False
    original_close = memory.close

    async def tracked_close():
        nonlocal closed
        closed = True
        await original_close()

    async def create_application_short_term_memory():
        return memory

    @asynccontextmanager
    async def fail_to_open_checkpointer(*args, **kwargs):
        raise RuntimeError("checkpoint startup failed")
        yield

    memory.close = tracked_close
    monkeypatch.setattr(
        server,
        "create_short_term_memory",
        create_application_short_term_memory,
    )
    monkeypatch.setattr(server, "open_checkpointer", fail_to_open_checkpointer)

    with pytest.raises(RuntimeError, match="checkpoint startup failed"):
        with TestClient(server.app):
            pass

    assert closed is True


class UnavailableShortTermMemoryAdapter:
    backend_name = "redis"

    async def load_messages(self, session_id, message_limit):
        raise ShortTermMemoryUnavailable("Redis is unavailable")

    async def save_messages(self, session_id, messages):
        raise ShortTermMemoryUnavailable("Redis is unavailable")

    async def clear_session(self, session_id):
        raise ShortTermMemoryUnavailable("Redis is unavailable")

    async def get_session_ttl(self, session_id):
        raise ShortTermMemoryUnavailable("Redis is unavailable")

    async def health(self):
        return {"backend": "redis", "status": "degraded"}

    async def close(self):
        return None


def test_memory_api_uses_application_short_term_memory_and_can_clear_a_session(
    monkeypatch,
    short_term_memory_factory,
):
    memory = short_term_memory_factory(ttl_seconds=60)

    async def create_application_short_term_memory():
        return memory

    async def save_without_calling_a_model(
        coordinator,
        question,
        context=None,
        session_id=None,
        **kwargs,
    ):
        assert coordinator.short_term_memory is memory
        assert coordinator.medical_graph.checkpointer is server.app.state.checkpointer
        assert kwargs["run_id"]
        await coordinator.short_term_memory.save_turn(
            session_id,
            question,
            "回答",
        )
        return {"answer": "回答", "session_id": session_id}

    monkeypatch.setattr(
        server.SwarmCoordinator,
        "process",
        save_without_calling_a_model,
    )
    monkeypatch.setattr(
        server,
        "create_short_term_memory",
        create_application_short_term_memory,
    )

    with TestClient(server.app) as client:
        server.app.state.short_term_memory = memory

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["memory"] == {
            "backend": "redis",
            "status": "ok",
        }

        created = client.post(
            "/api/runs",
            json={
                "question": "问题",
                "session_id": "session-api",
                "enable_memory": True,
            },
        )
        assert created.status_code == 200

        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if asyncio.run(memory.load_context("session-api")):
                break
            time.sleep(0.01)

        response = client.get("/api/sessions/session-api/memory")
        assert response.status_code == 200
        payload = response.json()
        assert payload["backend"] == "redis"
        assert payload["ttl_seconds"] > 0
        assert [message["content"] for message in payload["recent_history"]] == [
            "问题",
            "回答",
        ]

        deleted = client.delete("/api/sessions/session-api/memory")
        assert deleted.status_code == 200
        assert deleted.json() == {
            "session_id": "session-api",
            "cleared": True,
        }

        assert client.get("/api/sessions/session-api/memory").json()[
            "recent_history"
        ] == []


def test_checkpoint_api_lists_history_and_accepts_resume(
    monkeypatch,
    short_term_memory_factory,
):
    memory = short_term_memory_factory()
    resumed = asyncio.Event()
    admin_headers = {"X-Checkpoint-Admin-Token": "test-admin-token"}
    monkeypatch.setenv("CHECKPOINT_ADMIN_TOKEN", "test-admin-token")

    async def create_application_short_term_memory():
        return memory

    class CheckpointCoordinatorProbe:
        def __init__(self, *, short_term_memory, checkpointer, **kwargs):
            assert short_term_memory is memory
            assert checkpointer is server.app.state.checkpointer

        async def get_checkpoint(self, run_id, checkpoint_id=None):
            return CheckpointSnapshot(
                run_id=run_id,
                checkpoint_id=checkpoint_id or "checkpoint-latest",
                values={
                    "question": "resume question",
                    "context": {"age": 30},
                    "session_id": "session-a",
                },
                next_nodes=("finish",),
                metadata={"step": 2},
                created_at="2026-08-02T00:00:00+00:00",
                parent_checkpoint_id="checkpoint-parent",
                status="pending",
            )

        async def list_checkpoints(self, run_id, *, limit=None):
            checkpoint = await self.get_checkpoint(run_id)
            return [checkpoint]

        async def resume(self, run_id, checkpoint_id=None, **kwargs):
            resumed.set()
            return {"answer": "resumed", "run_id": run_id}

    monkeypatch.setattr(
        server,
        "create_short_term_memory",
        create_application_short_term_memory,
    )
    monkeypatch.setattr(server, "SwarmCoordinator", CheckpointCoordinatorProbe)

    with TestClient(server.app) as client:
        forbidden = client.get("/api/runs/run-a/checkpoints")
        history = client.get(
            "/api/runs/run-a/checkpoints?include_values=true",
            headers=admin_headers,
        )
        resumed_response = client.post(
            "/api/runs/run-a/resume",
            json={"checkpoint_id": "checkpoint-a"},
            headers=admin_headers,
        )

        assert forbidden.status_code == 403
        assert history.status_code == 200
        assert history.json()["checkpoints"][0] == {
            "run_id": "run-a",
            "checkpoint_id": "checkpoint-latest",
            "values": {
                "question": "resume question",
                "context": {"age": 30},
                "session_id": "session-a",
            },
            "next_nodes": ["finish"],
            "metadata": {"step": 2},
            "created_at": "2026-08-02T00:00:00+00:00",
            "parent_checkpoint_id": "checkpoint-parent",
            "status": "pending",
        }
        assert resumed_response.status_code == 200
        assert resumed_response.json()["run_id"] == "run-a"

        deadline = time.monotonic() + 1
        while time.monotonic() < deadline and not resumed.is_set():
            time.sleep(0.01)
        assert resumed.is_set()


def test_debug_api_reads_durable_audit_only_for_admin(monkeypatch):
    monkeypatch.setenv("CHECKPOINT_ADMIN_TOKEN", "test-admin-token")
    collector = DebugTraceCollector(
        question="sensitive question",
        context={},
        session_id="session-a",
        run_id="durable-audit-run",
    )
    collector.record_event(
        "planning",
        name="plan",
        output={"answer": "sensitive answer"},
    )
    server.RUN_STORE.add(collector)

    with TestClient(server.app) as client:
        asyncio.run(
            server.app.state.audit_store.save_attempt(
                "durable-audit-fallback",
                "attempt-a",
                collector.to_dict(),
            )
        )
        asyncio.run(
            server.app.state.audit_store.claim_effect(
                "durable-audit-fallback", "long_term_memory"
            )
        )
        asyncio.run(
            server.app.state.audit_store.complete_effect(
                "durable-audit-fallback", "long_term_memory", "unknown"
            )
        )

        forbidden_list = client.get("/api/runs")
        forbidden = client.get("/api/runs/durable-audit-run/events")
        allowed = client.get(
            "/api/runs/durable-audit-fallback/events",
            headers={"X-Checkpoint-Admin-Token": "test-admin-token"},
        )
        effects = client.get(
            "/api/runs/durable-audit-fallback/effects",
            headers={"X-Checkpoint-Admin-Token": "test-admin-token"},
        )
        reconciled = client.patch(
            "/api/runs/durable-audit-fallback/effects/long_term_memory",
            json={"resolution": "completed"},
            headers={"X-Checkpoint-Admin-Token": "test-admin-token"},
        )

    assert forbidden_list.status_code == 403
    assert forbidden.status_code == 403
    assert allowed.status_code == 200
    assert effects.status_code == 200
    assert effects.json()["effects"][0]["status"] == "unknown"
    assert reconciled.status_code == 200
    assert reconciled.json()["status"] == "completed"
    assert allowed.json()["events"][0]["metadata"]["audit_attempt_id"] == "attempt-a"


def test_public_consultation_snapshot_is_session_scoped_and_sanitized(
    monkeypatch,
    short_term_memory_factory,
):
    memory = short_term_memory_factory()

    async def create_application_short_term_memory():
        return memory

    monkeypatch.setattr(
        server,
        "create_short_term_memory",
        create_application_short_term_memory,
    )
    collector = DebugTraceCollector(
        question="sensitive symptom description",
        context={"medications": "sensitive medication"},
        session_id="session-public-a",
        run_id="public-consultation-success",
        metadata={"source": "consultation_api"},
    )
    collector.record_event(
        "planning",
        name="route_plan",
        output={"private_reasoning": "must never be public"},
    )
    collector.record_event(
        "agent_loop",
        name="agent_loop",
        agent_id="diagnostic_agent",
        output={"private_analysis": "must never be public"},
    )
    collector.record_event(
        "safety_check",
        name="runtime_safety_guard",
        output={
            "safety_checked": True,
            "safety_passed": True,
            "private_review": "must never be public",
        },
    )
    collector.finish_success(
        result_json={
            "answer": "请尽快安排线下评估。",
            "risk_level": "high",
            "suggestions": ["今天联系医生"],
            "disclaimer": "以上信息不能替代医生诊断。",
            "agents_involved": ["diagnostic_agent"],
        },
        final_answer="请尽快安排线下评估。",
    )
    server.RUN_STORE.add(collector)

    with TestClient(server.app) as client:
        allowed = client.get(
            "/api/consultations/public-consultation-success",
            headers={"X-Session-ID": "session-public-a"},
        )
        wrong_session = client.get(
            "/api/consultations/public-consultation-success",
            headers={"X-Session-ID": "session-public-b"},
        )
        missing_session = client.get(
            "/api/consultations/public-consultation-success",
        )

    assert allowed.status_code == 200
    payload = allowed.json()
    analysis_steps = payload["progress"]["analysis_steps"]
    payload_without_analysis = {
        **payload,
        "progress": {
            key: value
            for key, value in payload["progress"].items()
            if key != "analysis_steps"
        },
    }
    assert payload_without_analysis == {
        "consultation_id": "public-consultation-success",
        "status": "success",
        "progress": {
            "current_phase": "finalizing",
            "completed_phases": [
                "understanding",
                "planning",
                "consulting",
                "safety_review",
                "finalizing",
            ],
            "participants": [
                {
                    "id": "symptom_analysis",
                    "label": "风险与症状分析",
                    "state": "done",
                }
            ],
            "safety_checked": True,
        },
        "result": {
            "answer": "请尽快安排线下评估。",
            "risk_level": "high",
            "suggestions": ["今天联系医生"],
            "disclaimer": "以上信息不能替代医生诊断。",
                "participants": ["风险与症状分析"],
                "sources": [],
            },
        "failure": None,
    }
    assert analysis_steps[0] == {
        "id": "risk",
        "label": "风险预检",
        "summary": "已识别需要优先处理的高风险信号，将先给出就医时机建议。",
        "state": "attention",
    }
    assert analysis_steps[-1] == {
        "id": "safety",
        "label": "安全复核",
        "summary": "已检查急症提醒、过度诊断和用药风险。",
        "state": "done",
    }
    assert wrong_session.status_code == 404
    assert missing_session.status_code == 404
    serialized = repr(payload)
    assert "sensitive" not in serialized
    assert "private" not in serialized
    assert "diagnostic_agent" not in serialized


def test_public_consultation_exposes_a_safe_structured_analysis_summary(
    monkeypatch,
    short_term_memory_factory,
):
    memory = short_term_memory_factory()

    async def create_application_short_term_memory():
        return memory

    monkeypatch.setattr(
        server,
        "create_short_term_memory",
        create_application_short_term_memory,
    )
    collector = DebugTraceCollector(
        question="private patient question",
        session_id="session-analysis-summary",
        run_id="public-analysis-summary",
        metadata={"source": "consultation_api"},
    )
    collector.record_event(
        "planning",
        name="route_plan",
        output={
            "intents": ["lifestyle_guidance", "symptom_triage"],
            "risk_level": "low",
            "private_reasoning": "never expose this chain of thought",
            "tasks": [
                {
                    "assigned_agent": "consultation_agent",
                    "goal": "private task instructions",
                },
                {
                    "assigned_agent": "research_agent",
                    "goal": "private task instructions",
                },
            ],
        },
    )
    collector.record_event(
        "knowledge_retrieval",
        name="retrieve_knowledge",
        output={
            "status": "used",
            "candidate_count": 12,
            "source_count": 3,
            "private_chunks": ["never expose retrieved text"],
        },
    )
    collector.record_event("routing", name="route_by_subtasks")
    server.RUN_STORE.add(collector)

    with TestClient(server.app) as client:
        response = client.get(
            "/api/consultations/public-analysis-summary",
            headers={"X-Session-ID": "session-analysis-summary"},
        )

    assert response.status_code == 200
    steps = response.json()["progress"]["analysis_steps"]
    assert steps == [
        {
            "id": "risk",
            "label": "风险预检",
            "summary": "当前信息未触发高风险路径，仍会保留症状加重时的就医提醒。",
            "state": "done",
        },
        {
            "id": "focus",
            "label": "本次重点",
            "summary": "本次重点：可执行的生活调整、风险与就医时机。",
            "state": "done",
        },
        {
            "id": "evidence",
            "label": "资料核对",
            "summary": "已核对 3 条本地医学资料，并保留可引用来源。",
            "state": "done",
        },
        {
            "id": "collaboration",
            "label": "协作分工",
            "summary": "已安排 2 个分析角色：健康咨询、医学证据检索。",
            "state": "active",
        },
        {
            "id": "safety",
            "label": "安全复核",
            "summary": "回答生成后将检查急症提醒、过度诊断和用药风险。",
            "state": "pending",
        },
    ]
    serialized = repr(response.json())
    assert "private" not in serialized
    assert "chain of thought" not in serialized
    assert "retrieved text" not in serialized


def test_public_consultation_can_be_created_without_exposing_debug_run(
    monkeypatch,
    short_term_memory_factory,
):
    memory = short_term_memory_factory()

    async def create_application_short_term_memory():
        return memory

    async def finish_without_calling_a_model(
        coordinator,
        question,
        context=None,
        session_id=None,
        **kwargs,
    ):
        return {
            "answer": "保持休息并观察症状。",
            "risk_level": "low",
            "session_id": session_id,
        }

    monkeypatch.setattr(
        server,
        "create_short_term_memory",
        create_application_short_term_memory,
    )
    monkeypatch.setattr(
        server.SwarmCoordinator,
        "process",
        finish_without_calling_a_model,
    )

    with TestClient(server.app) as client:
        response = client.post(
            "/api/consultations",
            json={
                "question": "我有些头痛",
                "context": {"age": "32"},
                "session_id": "session-create-public",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {"consultation_id", "status"}
    assert payload["consultation_id"]
    assert payload["status"] in {"queued", "running"}


def test_public_consultation_reports_running_roles_and_safe_failure(
    monkeypatch,
    short_term_memory_factory,
):
    memory = short_term_memory_factory()

    async def create_application_short_term_memory():
        return memory

    monkeypatch.setattr(
        server,
        "create_short_term_memory",
        create_application_short_term_memory,
    )
    running = DebugTraceCollector(
        question="running question",
        session_id="session-running",
        run_id="public-consultation-running",
        metadata={"source": "consultation_api"},
    )
    running.record_event("memory", name="load_memory", output={"loaded": True})
    running.record_event(
        "memory",
        name="initialize_messages",
        agent_id="consultation_agent",
        output={"message_count": 2},
    )
    running.record_event("planning", name="route_plan", output={"route": "parallel"})
    running.record_event(
        "planning",
        name="subtasks_created",
        output=[
            {"assigned_agent": "consultation_agent", "description": "private task"},
            {"assigned_agent": "research_agent", "description": "private task"},
        ],
    )
    running.record_event("routing", name="route_by_subtasks")
    server.RUN_STORE.add(running)

    single = DebugTraceCollector(
        question="single-agent question",
        session_id="session-single",
        run_id="public-consultation-single",
        metadata={"source": "consultation_api"},
    )
    single.record_event(
        "planning",
        name="route_plan",
        output={
            "execution_mode": "single",
            "tasks": [{"assigned_agent": "diagnostic_agent", "goal": "private task"}],
        },
    )
    single.record_event("routing", name="route_by_subtasks")
    server.RUN_STORE.add(single)

    failed = DebugTraceCollector(
        question="failed question",
        session_id="session-failed",
        run_id="public-consultation-failed",
        metadata={"source": "consultation_api"},
    )
    failed.record_event(
        "planning",
        name="route_plan",
        output={"tasks": [{"assigned_agent": "consultation_agent"}]},
    )
    failed.record_event("routing", name="route_by_subtasks")
    failed.record_event(
        "agent_loop",
        name="run_single_agent",
        status="failed",
        error="private worker failure",
    )
    failed.finish_failed("raw provider credential and stack trace")
    server.RUN_STORE.add(failed)

    timed_out = DebugTraceCollector(
        question="timeout question",
        session_id="session-timeout",
        run_id="public-consultation-timeout",
        metadata={"source": "consultation_api"},
    )
    timed_out.record_event(
        "safety_check",
        name="runtime_safety_guard",
        output={"safety_checked": False, "error": "private safety error"},
        status="failed",
    )
    timed_out.finish_success(
        result_json={"answer": "partial private answer"},
        timeout=True,
    )
    server.RUN_STORE.add(timed_out)

    with TestClient(server.app) as client:
        running_response = client.get(
            "/api/consultations/public-consultation-running",
            headers={"X-Session-ID": "session-running"},
        )
        single_response = client.get(
            "/api/consultations/public-consultation-single",
            headers={"X-Session-ID": "session-single"},
        )
        failed_response = client.get(
            "/api/consultations/public-consultation-failed",
            headers={"X-Session-ID": "session-failed"},
        )
        timeout_response = client.get(
            "/api/consultations/public-consultation-timeout",
            headers={"X-Session-ID": "session-timeout"},
        )

    assert running_response.status_code == 200
    running_progress = running_response.json()["progress"]
    running_analysis = running_progress["analysis_steps"]
    assert {
        key: value
        for key, value in running_progress.items()
        if key != "analysis_steps"
    } == {
        "current_phase": "consulting",
        "completed_phases": ["understanding", "planning"],
        "participants": [
            {"id": "health_consultation", "label": "健康咨询", "state": "active"},
            {"id": "evidence_research", "label": "医学证据检索", "state": "active"},
        ],
        "safety_checked": False,
    }
    assert next(step for step in running_analysis if step["id"] == "collaboration") == {
        "id": "collaboration",
        "label": "协作分工",
        "summary": "已安排 2 个分析角色：健康咨询、医学证据检索。",
        "state": "active",
    }
    assert single_response.status_code == 200
    assert single_response.json()["progress"]["participants"] == [
        {"id": "symptom_analysis", "label": "风险与症状分析", "state": "active"}
    ]
    failure = failed_response.json()
    assert failure["status"] == "failed"
    assert {
        key: value
        for key, value in failure["progress"].items()
        if key != "analysis_steps"
    } == {
        "current_phase": "consulting",
        "completed_phases": ["understanding", "planning"],
        "participants": [
            {"id": "health_consultation", "label": "健康咨询", "state": "failed"}
        ],
        "safety_checked": False,
    }
    assert failure["result"] is None
    assert failure["failure"] == {
        "code": "analysis_failed",
        "message": "本次分析未能完成，请重新尝试；如症状严重或正在加重，请及时线下就医。",
        "retryable": True,
    }
    assert "credential" not in repr(failure)
    assert "stack trace" not in repr(failure)
    assert "worker failure" not in repr(failure)
    timeout = timeout_response.json()
    assert timeout["status"] == "timeout"
    assert timeout["progress"]["current_phase"] == "safety_review"
    assert timeout["progress"]["completed_phases"] == [
        "understanding",
        "planning",
        "consulting",
    ]
    assert timeout["progress"]["safety_checked"] is False
    assert timeout["result"] is None
    assert timeout["failure"]["code"] == "analysis_timeout"
    assert timeout["failure"]["retryable"] is True
    assert "partial private answer" not in repr(timeout)
    assert "private safety error" not in repr(timeout)


def test_public_consultation_route_rejects_debug_runs(
    monkeypatch,
    short_term_memory_factory,
):
    memory = short_term_memory_factory()

    async def create_application_short_term_memory():
        return memory

    monkeypatch.setattr(
        server,
        "create_short_term_memory",
        create_application_short_term_memory,
    )
    debug_run = DebugTraceCollector(
        question="administrator-only input",
        session_id="shared-session-id",
        run_id="administrator-debug-run",
        metadata={"source": "api"},
    )
    debug_run.finish_success(result_json={"answer": "administrator-only output"})
    server.RUN_STORE.add(debug_run)

    with TestClient(server.app) as client:
        response = client.get(
            "/api/consultations/administrator-debug-run",
            headers={"X-Session-ID": "shared-session-id"},
        )

    assert response.status_code == 404


def test_memory_api_returns_service_unavailable_when_redis_cannot_be_read(
    monkeypatch,
):
    memory = ShortTermMemory(adapter=UnavailableShortTermMemoryAdapter())

    async def create_application_short_term_memory():
        return memory

    monkeypatch.setattr(
        server,
        "create_short_term_memory",
        create_application_short_term_memory,
    )

    with TestClient(server.app) as client:
        server.app.state.short_term_memory = memory

        loaded = client.get("/api/sessions/session-api/memory")
        deleted = client.delete("/api/sessions/session-api/memory")

    assert loaded.status_code == 503
    assert loaded.json()["detail"] == "Short-term memory backend unavailable"
    assert deleted.status_code == 503
    assert deleted.json()["detail"] == "Short-term memory backend unavailable"
