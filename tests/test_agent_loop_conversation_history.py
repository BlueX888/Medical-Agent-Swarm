from typing import Any, Dict, List

import pytest

from core.agent_loop import AgentLoop
from core.llm_client import LLMResponse


class RecordingLLMClient:
    def __init__(self):
        self.messages: List[Dict[str, Any]] = []

    async def chat_with_tools(self, *, messages, **kwargs):
        self.messages = list(messages)
        return LLMResponse(
            content="当前回答",
            tool_calls=[],
            finish_reason="stop",
        )


class ConversationAgent:
    agent_id = "conversation-agent"
    config = {"temperature": 0.1}

    def __init__(self):
        self.llm_client = RecordingLLMClient()

    def get_system_prompt(self):
        return "系统提示"

    def format_user_input(self, input_data):
        return input_data["question"]

    def get_tools_for_llm(self, **kwargs):
        return []

    async def post_process_result(self, result, final_response):
        return result


@pytest.mark.asyncio
async def test_agent_loop_consumes_history_passed_with_the_request():
    agent = ConversationAgent()
    loop = AgentLoop(max_iterations=1)

    result = await loop.run(
        agent,
        {
            "question": "当前问题",
            "conversation_history": [
                {"role": "user", "content": "上一轮问题", "timestamp": "ignored"},
                {"role": "assistant", "content": "上一轮回答", "timestamp": "ignored"},
                {"role": "tool", "content": "不应进入上下文", "timestamp": "ignored"},
            ],
        },
        session_id="session-a",
    )

    assert result["answer"] == "当前回答"
    assert agent.llm_client.messages == [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "上一轮问题"},
        {"role": "assistant", "content": "上一轮回答"},
        {"role": "user", "content": "当前问题"},
    ]
