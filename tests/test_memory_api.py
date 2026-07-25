import asyncio

from fastapi.testclient import TestClient

from api.server import app
from memory import ShortTermMemory


def test_memory_api_uses_application_memory_and_can_clear_a_session():
    memory = ShortTermMemory(storage_type="memory", ttl_seconds=60)
    asyncio.run(memory.save_turn("session-api", "问题", "回答"))

    with TestClient(app) as client:
        app.state.short_term_memory = memory

        health = client.get("/api/health")
        assert health.status_code == 200
        assert health.json()["memory"] == {
            "backend": "memory",
            "status": "ok",
        }

        response = client.get("/api/sessions/session-api/memory")
        assert response.status_code == 200
        payload = response.json()
        assert payload["backend"] == "memory"
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
