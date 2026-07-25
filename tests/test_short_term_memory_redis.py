import os
import uuid

import pytest

from memory import ShortTermMemory


REDIS_TEST_URL = os.getenv("REDIS_TEST_URL")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif(
    not REDIS_TEST_URL,
    reason="Set REDIS_TEST_URL to run Redis integration tests",
)
async def test_redis_backend_persists_trims_refreshes_and_clears():
    session_id = f"integration-{uuid.uuid4()}"
    memory = ShortTermMemory(
        storage_type="redis",
        redis_url=REDIS_TEST_URL,
        ttl_seconds=60,
        max_messages=4,
    )
    second_process_memory = ShortTermMemory(
        storage_type="redis",
        redis_url=REDIS_TEST_URL,
        ttl_seconds=60,
        max_messages=4,
    )

    try:
        assert await memory.health() == {
            "backend": "redis",
            "status": "ok",
        }

        for index in range(3):
            await memory.save_turn(session_id, f"问题 {index}", f"回答 {index}")

        assert [
            message["content"]
            for message in await second_process_memory.load_context(
                session_id,
                max_turns=10,
            )
        ] == ["问题 1", "回答 1", "问题 2", "回答 2"]
        assert 0 < await memory.get_session_ttl(session_id) <= 60
        assert await memory.clear_session(session_id) is True
        assert await memory.load_context(session_id) == []
    finally:
        await memory.clear_session(session_id)
        await memory.close()
        await second_process_memory.close()
