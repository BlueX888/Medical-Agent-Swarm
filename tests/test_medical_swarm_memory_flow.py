from datetime import datetime

import pytest

from memory import ShortTermMemory
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


class FakeWorker:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.shared_context = None

    def attach_shared_context(self, shared_context):
        self.shared_context = shared_context

    async def process_subtask(self, subtask, debug_collector=None):
        return {"answer": f"{self.agent_id} 完成 {subtask.type}"}


class SwarmAcceptanceGraph(MedicalSwarmGraph):
    async def plan_and_decompose(self, state):
        subtasks = [
            {
                "type": "consultation",
                "description": "咨询分析",
                "assigned_agent": "consultation_agent",
            },
            {
                "type": "diagnostic",
                "description": "症状分析",
                "assigned_agent": "diagnostic_agent",
            },
        ]
        return {
            "assessment": {"subtasks": subtasks, "reason": "acceptance test"},
            "subtasks": subtasks,
        }

    async def _synthesize_results(self, *args, **kwargs):
        return "这是两个 Agent 汇总后的最终回答。"


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


@pytest.mark.asyncio
async def test_full_swarm_workflow_persists_only_one_completed_turn():
    memory = ShortTermMemory(storage_type="memory")
    consultation = FakeWorker("consultation_agent")
    diagnostic = FakeWorker("diagnostic_agent")
    graph = SwarmAcceptanceGraph(
        llm_client=object(),
        worker_pool=[consultation, diagnostic],
        consultation_agent=consultation,
        diagnostic_agent=diagnostic,
        research_agent=FakeWorker("research_agent"),
        short_term_memory=memory,
        long_term_memory=DisabledLongTermMemory(),
        session_manager=None,
        enable_swarm=True,
        enable_short_term_memory=True,
        enable_long_term_memory=False,
    )

    state = await graph.ainvoke(
        {
            "question": "需要多角度分析的问题",
            "session_id": "swarm-session",
            "context": {},
        }
    )

    assert state["result"]["swarm_enabled"] is True
    history = await memory.load_context("swarm-session", max_turns=10)
    assert [(message["role"], message["content"]) for message in history] == [
        ("user", "需要多角度分析的问题"),
        ("assistant", state["result"]["answer"]),
    ]
