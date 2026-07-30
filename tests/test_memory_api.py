import asyncio
import time

from fastapi.testclient import TestClient

import api.server as server
from memory import ShortTermMemory, ShortTermMemoryUnavailable


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
