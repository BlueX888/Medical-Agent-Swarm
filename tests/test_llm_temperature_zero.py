from types import SimpleNamespace

import pytest

from core.llm_client import LLMClient


@pytest.mark.asyncio
async def test_chat_preserves_explicit_zero_temperature():
    captured = {}

    async def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok"),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

    client = LLMClient.__new__(LLMClient)
    client.model_type = "openai_compatible"
    client.model_name = "fake"
    client.temperature = 0.7
    client.max_tokens = 10
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )

    await client.chat([{"role": "user", "content": "x"}], temperature=0)

    assert captured["temperature"] == 0
