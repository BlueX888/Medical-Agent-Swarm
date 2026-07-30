import asyncio
from unittest.mock import ANY

import pytest

from memory import ShortTermMemory, ShortTermMemoryUnavailable, create_short_term_memory


@pytest.mark.asyncio
async def test_saved_turn_is_loaded_in_conversation_order(short_term_memory_factory):
    memory = short_term_memory_factory()

    await memory.save_turn(
        session_id="session-a",
        user_message="我最近一直头痛",
        assistant_message="请先留意是否伴随高热或神经系统症状。",
    )

    assert await memory.load_context("session-a") == [
        {
            "role": "user",
            "content": "我最近一直头痛",
            "timestamp": ANY,
        },
        {
            "role": "assistant",
            "content": "请先留意是否伴随高热或神经系统症状。",
            "timestamp": ANY,
        },
    ]


@pytest.mark.asyncio
async def test_sessions_are_isolated_and_history_is_limited_by_complete_turns(
    short_term_memory_factory,
):
    memory = short_term_memory_factory(max_messages=4)

    for index in range(3):
        await memory.save_turn(
            "session-a",
            f"问题 {index}",
            f"回答 {index}",
        )
    await memory.save_turn("session-b", "另一会话", "另一回答")

    assert [
        (message["role"], message["content"])
        for message in await memory.load_context("session-a", max_turns=10)
    ] == [
        ("user", "问题 1"),
        ("assistant", "回答 1"),
        ("user", "问题 2"),
        ("assistant", "回答 2"),
    ]
    assert [
        message["content"]
        for message in await memory.load_context("session-b", max_turns=10)
    ] == ["另一会话", "另一回答"]


@pytest.mark.asyncio
async def test_session_can_be_cleared_and_expires_after_inactivity(
    short_term_memory_factory,
):
    memory = short_term_memory_factory(ttl_seconds=1)
    await memory.save_turn("clear-me", "问题", "回答")

    assert await memory.get_session_ttl("clear-me") > 0
    assert await memory.clear_session("clear-me") is True
    assert await memory.load_context("clear-me") == []
    assert await memory.clear_session("clear-me") is False

    await memory.save_turn("expire-me", "问题", "回答")
    await asyncio.sleep(1.05)

    assert await memory.load_context("expire-me") == []
    assert await memory.get_session_ttl("expire-me") == -2


@pytest.mark.asyncio
async def test_health_reports_the_active_backend(short_term_memory_factory):
    memory = short_term_memory_factory()

    assert await memory.health() == {
        "backend": "redis",
        "status": "ok",
    }


@pytest.mark.asyncio
async def test_assistant_display_metadata_round_trips_with_the_visible_turn(
    short_term_memory_factory,
):
    memory = short_term_memory_factory()
    metadata = {
        "risk_level": "high",
        "suggestions": ["Seek in-person care today"],
        "disclaimer": "For reference only.",
        "agents_involved": ["diagnostic"],
    }

    await memory.save_turn(
        session_id="session-with-metadata",
        user_message="I have a severe headache.",
        assistant_message="Please arrange an in-person assessment.",
        assistant_metadata=metadata,
    )

    messages = await memory.load_context("session-with-metadata")

    assert "metadata" not in messages[0]
    assert messages[1]["metadata"] == metadata


@pytest.mark.asyncio
async def test_loaded_metadata_is_isolated_from_adapter_storage(
    short_term_memory_factory,
):
    memory = short_term_memory_factory()
    await memory.save_turn(
        session_id="session-with-isolated-metadata",
        user_message="问题",
        assistant_message="回答",
        assistant_metadata={"suggestions": ["最初建议"]},
    )

    loaded = await memory.load_context("session-with-isolated-metadata")
    loaded[1]["metadata"]["suggestions"].append("调用方修改")

    loaded_again = await memory.load_context("session-with-isolated-metadata")
    assert loaded_again[1]["metadata"]["suggestions"] == ["最初建议"]


@pytest.mark.asyncio
async def test_configured_redis_never_falls_back_when_unavailable(monkeypatch):
    monkeypatch.setenv("SHORT_TERM_MEMORY_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:6399/0")
    monkeypatch.setenv("SHORT_TERM_MEMORY_ALLOW_FALLBACK", "true")

    with pytest.raises(ShortTermMemoryUnavailable, match="required but unavailable"):
        await create_short_term_memory()


@pytest.mark.asyncio
async def test_memory_backend_configuration_is_rejected(monkeypatch):
    monkeypatch.setenv("SHORT_TERM_MEMORY_BACKEND", "memory")

    with pytest.raises(ValueError, match="only Redis is supported"):
        await create_short_term_memory()


def test_storage_type_selection_is_no_longer_supported():
    with pytest.raises(TypeError, match="storage_type"):
        ShortTermMemory(storage_type="memory")


def test_memory_named_adapter_is_rejected():
    class MemoryNamedAdapter:
        backend_name = "memory"

    with pytest.raises(ValueError, match="must use the Redis backend"):
        ShortTermMemory(adapter=MemoryNamedAdapter())
