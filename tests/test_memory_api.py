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

        forbidden_list = client.get("/api/runs")
        forbidden = client.get("/api/runs/durable-audit-run/events")
        allowed = client.get(
            "/api/runs/durable-audit-fallback/events",
            headers={"X-Checkpoint-Admin-Token": "test-admin-token"},
        )

    assert forbidden_list.status_code == 403
    assert forbidden.status_code == 403
    assert allowed.status_code == 200
    assert allowed.json()["events"][0]["metadata"]["audit_attempt_id"] == "attempt-a"


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
