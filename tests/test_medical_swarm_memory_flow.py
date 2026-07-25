from datetime import datetime

import pytest

from swarm.medical_swarm_graph import MedicalSwarmGraph


class RecordingShortTermMemory:
    def __init__(self):
        self.saved_turns = []

    async def load_context(self, session_id, max_turns=5):
        assert session_id == "session-a"
        assert max_turns == 5
        return [
            {
                "role": "user",
                "content": "上一轮问题",
                "timestamp": "2026-07-25T00:00:00+00:00",
            },
            {
                "role": "assistant",
                "content": "上一轮回答",
                "timestamp": "2026-07-25T00:01:00+00:00",
            },
        ]

    async def save_turn(self, session_id, user_message, assistant_message):
        self.saved_turns.append((session_id, user_message, assistant_message))


class DisabledLongTermMemory:
    enabled = False

    def search_similar_sessions(self, *args, **kwargs):
        raise AssertionError("long-term memory is disabled")

    def add_session_summary(self, *args, **kwargs):
        raise AssertionError("long-term memory is disabled")


def make_graph_for_memory_nodes():
    graph = MedicalSwarmGraph.__new__(MedicalSwarmGraph)
    graph.short_term_memory = RecordingShortTermMemory()
    graph.long_term_memory = DisabledLongTermMemory()
    graph.enable_short_term_memory = True
    graph.enable_long_term_memory = False
    graph.session_manager = None
    return graph


@pytest.mark.asyncio
async def test_graph_loads_short_term_history_independently_from_long_term_memory():
    graph = make_graph_for_memory_nodes()

    result = await graph.load_memory(
        {
            "question": "当前问题",
            "session_id": "session-a",
            "context": {"age": 30},
        }
    )

    assert [message["content"] for message in result["recent_history"]] == [
        "上一轮问题",
        "上一轮回答",
    ]
    assert result["enhanced_context"]["recent_history"] == result["recent_history"]
    assert result["historical_cases"] == []


@pytest.mark.asyncio
async def test_graph_saves_exactly_one_user_visible_turn():
    graph = make_graph_for_memory_nodes()

    await graph.save_memory(
        {
            "session_id": "session-a",
            "question": "当前问题",
            "result": {"answer": "最终回答"},
            "start_time": datetime.now(),
            "mode": "single_agent",
        }
    )

    assert graph.short_term_memory.saved_turns == [
        ("session-a", "当前问题", "最终回答")
    ]
