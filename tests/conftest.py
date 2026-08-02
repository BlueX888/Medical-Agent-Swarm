import copy
import math
import time

import pytest

from memory import ShortTermMemory


@pytest.fixture(autouse=True)
def use_ephemeral_checkpoints_in_tests(monkeypatch):
    """Application tests opt into memory; persistence tests configure storage."""
    monkeypatch.setenv("CHECKPOINT_BACKEND", "memory")


class FakeRedisShortTermMemoryAdapter:
    """Redis protocol test double; never used by application code."""

    backend_name = "redis"

    def __init__(self, *, ttl_seconds: int, max_messages: int):
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self._sessions = {}
        self._idempotency_keys = set()

    async def load_messages(self, session_id, message_limit):
        self._purge_expired(session_id)
        session = self._sessions.get(session_id)
        if session is None:
            return []
        return copy.deepcopy(session["messages"][-message_limit:])

    async def save_messages(self, session_id, messages, idempotency_key=None):
        if idempotency_key and idempotency_key in self._idempotency_keys:
            return False
        self._purge_expired(session_id)
        session = self._sessions.get(session_id, {"messages": []})
        session["messages"].extend(copy.deepcopy(messages))
        session["messages"] = session["messages"][-self.max_messages :]
        session["expires_at"] = time.monotonic() + self.ttl_seconds
        self._sessions[session_id] = session
        if idempotency_key:
            self._idempotency_keys.add(idempotency_key)
        return True

    async def clear_session(self, session_id):
        self._purge_expired(session_id)
        return self._sessions.pop(session_id, None) is not None

    async def get_session_ttl(self, session_id):
        self._purge_expired(session_id)
        session = self._sessions.get(session_id)
        if session is None:
            return -2
        return max(1, math.ceil(session["expires_at"] - time.monotonic()))

    async def health(self):
        return {"backend": self.backend_name, "status": "ok"}

    async def close(self):
        return None

    def _purge_expired(self, session_id):
        session = self._sessions.get(session_id)
        if session and session["expires_at"] <= time.monotonic():
            self._sessions.pop(session_id, None)


@pytest.fixture
def short_term_memory_factory():
    def create(*, ttl_seconds=24 * 60 * 60, max_messages=40):
        adapter = FakeRedisShortTermMemoryAdapter(
            ttl_seconds=ttl_seconds,
            max_messages=max_messages,
        )
        return ShortTermMemory(
            ttl_seconds=ttl_seconds,
            max_messages=max_messages,
            adapter=adapter,
        )

    return create
