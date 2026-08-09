import asyncio
from datetime import datetime

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from memory import LongTermMemoryWriteUnknown, ShortTermMemoryUnavailable
from debug import DebugTraceCollector
from core.checkpointing import CheckpointSnapshot
from core.audit import MemoryAuditStore
import swarm.swarm_coordinator as coordinator_module
from swarm.medical_swarm_graph import MedicalSwarmGraph
from swarm.medical_swarm_state import MedicalSwarmState
from swarm.swarm_coordinator import SwarmCoordinator


class RecordingShortTermMemory:
    def __init__(self):
        self.saved_turns = []
        self.idempotency_keys = set()

    async def load_context(self, session_id, max_turns=5):
        assert session_id == "session-a"
        assert max_turns == 10
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

    async def save_turn(
        self,
        session_id,
        user_message,
        assistant_message,
        assistant_metadata=None,
        idempotency_key=None,
    ):
        assert isinstance(assistant_metadata, dict)
        if idempotency_key and idempotency_key in self.idempotency_keys:
            return False
        if idempotency_key:
            self.idempotency_keys.add(idempotency_key)
        self.saved_turns.append((session_id, user_message, assistant_message))
        return True


class DisabledLongTermMemory:
    enabled = False

    def search_similar_sessions(self, *args, **kwargs):
        raise AssertionError("long-term memory is disabled")

    def add_session_summary(self, *args, **kwargs):
        raise AssertionError("long-term memory is disabled")


class UnavailableShortTermMemory:
    async def load_context(self, session_id, max_turns=5):
        raise ShortTermMemoryUnavailable("Redis is unavailable")


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


class ConcurrentRunProbe:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.first_started = asyncio.Event()
        self.completed = []
        self.last_state = None

    async def ainvoke(self, state):
        self.last_state = state
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if state["question"] == "first":
            self.first_started.set()
            await asyncio.sleep(0.05)
        self.completed.append(state["question"])
        self.active -= 1
        return {"result": {"answer": state["question"]}}


class ResumeProbe:
    def __init__(self):
        self.resumed = []

    async def get_checkpoint(self, run_id, checkpoint_id=None):
        return CheckpointSnapshot(
            run_id=run_id,
            checkpoint_id=checkpoint_id or "latest",
            values={"session_id": "resume-session"},
            next_nodes=("finish",),
            metadata={},
            created_at=None,
            parent_checkpoint_id=None,
            status="pending",
        )

    async def resume(self, run_id, checkpoint_id=None, **kwargs):
        self.resumed.append((run_id, checkpoint_id))
        return {"result": {"answer": "resumed", "run_id": run_id}}


def make_coordinator_for_concurrency(memory, graph):
    coordinator = SwarmCoordinator.__new__(SwarmCoordinator)
    coordinator.enable_swarm = False
    coordinator.enable_short_term_memory = True
    coordinator.enable_long_term_memory = False
    coordinator.swarm_timeout_s = 120.0
    coordinator.short_term_memory = memory
    coordinator.medical_graph = graph
    coordinator.worker_pool = []
    return coordinator


@pytest.mark.asyncio
async def test_coordinator_propagates_explicit_run_id_to_graph(
    short_term_memory_factory,
):
    graph = ConcurrentRunProbe()
    coordinator = make_coordinator_for_concurrency(
        short_term_memory_factory(),
        graph,
    )

    await coordinator.process(
        "question",
        session_id="session-a",
        run_id="durable-run-a",
    )

    assert graph.completed == ["question"]
    assert graph.last_state["run_id"] == "durable-run-a"
    assert graph.last_state["enable_swarm"] is False
    assert graph.last_state["enable_short_term_memory"] is True
    assert graph.last_state["enable_long_term_memory"] is False


@pytest.mark.asyncio
async def test_coordinator_resumes_checkpoint_inside_session_scope(
    short_term_memory_factory,
):
    graph = ResumeProbe()
    coordinator = make_coordinator_for_concurrency(
        short_term_memory_factory(),
        graph,
    )

    result = await coordinator.resume("durable-run-a", "checkpoint-a")

    assert result == {"answer": "resumed", "run_id": "durable-run-a"}
    assert graph.resumed == [("durable-run-a", "checkpoint-a")]


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
async def test_graph_continues_without_history_when_short_term_memory_is_unavailable():
    graph = make_graph_for_memory_nodes()
    graph.short_term_memory = UnavailableShortTermMemory()

    result = await graph.load_memory(
        {
            "question": "当前问题",
            "session_id": "session-a",
            "context": {"age": 30},
        }
    )

    assert result["recent_history"] == []
    assert result["enhanced_context"] == {"age": 30}
    assert result["short_term_memory_error"] == "Redis is unavailable"


@pytest.mark.asyncio
async def test_coordinators_serialize_concurrent_runs_for_the_same_session(
    short_term_memory_factory,
):
    graph = ConcurrentRunProbe()
    first_coordinator = make_coordinator_for_concurrency(
        short_term_memory_factory(),
        graph,
    )
    second_coordinator = make_coordinator_for_concurrency(
        short_term_memory_factory(),
        graph,
    )

    first = asyncio.create_task(
        first_coordinator.process("first", session_id="shared-session")
    )
    await graph.first_started.wait()
    second = asyncio.create_task(
        second_coordinator.process("second", session_id="shared-session")
    )
    await asyncio.gather(first, second)

    assert graph.max_active == 1
    assert graph.completed == ["first", "second"]


def test_process_with_swarm_initializes_default_memory_once_per_event_loop(
    monkeypatch,
    short_term_memory_factory,
):
    create_calls = 0

    async def create_memory():
        nonlocal create_calls
        create_calls += 1
        await asyncio.sleep(0.05)
        return short_term_memory_factory()

    class CoordinatorProbe:
        def __init__(
            self,
            *,
            enable_swarm,
            enable_memory,
            enable_short_term_memory,
            enable_long_term_memory,
            short_term_memory,
            **kwargs,
        ):
            self.enable_swarm = enable_swarm
            self.enable_short_term_memory = enable_short_term_memory
            self.enable_long_term_memory = enable_long_term_memory
            self.short_term_memory = short_term_memory

        async def process(self, question, context, **kwargs):
            return {"answer": question}

    monkeypatch.setattr(coordinator_module, "create_short_term_memory", create_memory)
    monkeypatch.setattr(coordinator_module, "SwarmCoordinator", CoordinatorProbe)

    async def run_concurrently():
        await asyncio.gather(
            coordinator_module.process_with_swarm(
                "first",
                session_id="shared-session",
            ),
            coordinator_module.process_with_swarm(
                "second",
                session_id="shared-session",
            ),
        )

    asyncio.run(run_concurrently())
    asyncio.run(run_concurrently())

    assert create_calls == 2


@pytest.mark.asyncio
async def test_graph_saves_exactly_one_user_visible_turn():
    graph = make_graph_for_memory_nodes()

    await graph.save_memory(
        {
            "session_id": "session-a",
            "question": "当前问题",
            "result": {"answer": "最终回答"},
            "start_time": datetime.now(),
            "mode": "single_task",
            "run_id": "run-a",
        }
    )

    await graph.save_memory(
        {
            "session_id": "session-a",
            "question": "当前问题",
            "result": {"answer": "最终回答"},
            "start_time": datetime.now(),
            "mode": "single_task",
            "run_id": "run-a",
        }
    )

    assert graph.short_term_memory.saved_turns == [
        ("session-a", "当前问题", "最终回答")
    ]


@pytest.mark.asyncio
async def test_graph_claims_non_transactional_long_term_write_once_per_run():
    class RecordingLongTermMemory:
        enabled = True

        def __init__(self):
            self.calls = []

        def add_session_summary(self, **kwargs):
            self.calls.append(kwargs)
            return "memory-a"

    graph = make_graph_for_memory_nodes()
    graph.enable_long_term_memory = True
    graph.long_term_memory = RecordingLongTermMemory()
    graph.audit_store = MemoryAuditStore()
    state = {
        "session_id": "session-a",
        "question": "current question",
        "result": {"answer": "final answer"},
        "start_time": datetime.now(),
        "mode": "single_task",
        "run_id": "durable-run-a",
    }

    await graph.save_memory(state)
    await graph.save_memory(state)

    assert len(graph.long_term_memory.calls) == 1
    assert graph.long_term_memory.calls[0]["metadata"]["run_id"] == "durable-run-a"


@pytest.mark.asyncio
async def test_graph_retries_failed_long_term_outbox_effect():
    class FlakyLongTermMemory:
        enabled = True

        def __init__(self):
            self.calls = 0

        def add_session_summary(self, **kwargs):
            self.calls += 1
            return None if self.calls == 1 else "memory-a"

    graph = make_graph_for_memory_nodes()
    graph.enable_long_term_memory = True
    graph.long_term_memory = FlakyLongTermMemory()
    graph.audit_store = MemoryAuditStore()
    state = {
        "session_id": "session-a",
        "question": "current question",
        "result": {"answer": "final answer"},
        "start_time": datetime.now(),
        "mode": "single_task",
        "run_id": "retry-run-a",
    }

    with pytest.raises(RuntimeError, match="provider rejected"):
        await graph.save_memory(state)
    await graph.save_memory(state)

    assert graph.long_term_memory.calls == 2


@pytest.mark.asyncio
async def test_checkpoint_keeps_failed_effect_node_resumable():
    class FlakyLongTermMemory:
        enabled = True

        def __init__(self):
            self.calls = 0

        def add_session_summary(self, **kwargs):
            self.calls += 1
            return None if self.calls == 1 else "memory-a"

    node_owner = make_graph_for_memory_nodes()
    node_owner.enable_short_term_memory = False
    node_owner.enable_long_term_memory = True
    node_owner.long_term_memory = FlakyLongTermMemory()
    node_owner.audit_store = MemoryAuditStore()
    saver = InMemorySaver()
    state_graph = StateGraph(MedicalSwarmState)
    state_graph.add_node("save_memory", node_owner.save_memory)
    state_graph.set_entry_point("save_memory")
    state_graph.add_edge("save_memory", END)
    compiled = state_graph.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "effect-recovery-run"}}
    state = {
        "session_id": "session-a",
        "question": "current question",
        "result": {"answer": "final answer"},
        "start_time": datetime.now(),
        "mode": "single_task",
        "run_id": "effect-recovery-run",
    }

    with pytest.raises(RuntimeError, match="provider rejected"):
        await compiled.ainvoke(state, config=config)
    failed_snapshot = await compiled.aget_state(config)
    assert failed_snapshot.next == ("save_memory",)

    await compiled.ainvoke(None, config=config)
    completed_snapshot = await compiled.aget_state(config)
    assert completed_snapshot.next == ()
    assert node_owner.long_term_memory.calls == 2


@pytest.mark.asyncio
async def test_unknown_provider_outcome_is_not_automatically_retried():
    class UncertainLongTermMemory:
        enabled = True

        def __init__(self):
            self.calls = 0

        def add_session_summary(self, **kwargs):
            self.calls += 1
            raise LongTermMemoryWriteUnknown("response was lost")

    graph = make_graph_for_memory_nodes()
    graph.enable_long_term_memory = True
    graph.long_term_memory = UncertainLongTermMemory()
    graph.audit_store = MemoryAuditStore()
    state = {
        "session_id": "session-a",
        "question": "current question",
        "result": {"answer": "final answer"},
        "start_time": datetime.now(),
        "mode": "single_task",
        "run_id": "unknown-run-a",
    }

    with pytest.raises(RuntimeError, match="response was lost"):
        await graph.save_memory(state)
    await graph.save_memory(state)

    assert graph.long_term_memory.calls == 1
    effects = await graph.audit_store.get_effects("unknown-run-a")
    assert effects[0]["status"] == "unknown"


@pytest.mark.asyncio
async def test_full_swarm_workflow_persists_only_one_completed_turn(
    short_term_memory_factory,
):
    memory = short_term_memory_factory()
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

    assert state["route"] == "multiple_tasks"
    assert state["result"]["swarm_enabled"] is True
    history = await memory.load_context("swarm-session", max_turns=10)
    assert [(message["role"], message["content"]) for message in history] == [
        ("user", "需要多角度分析的问题"),
        ("assistant", state["result"]["answer"]),
    ]


@pytest.mark.asyncio
async def test_debug_swarm_records_validated_tasks_without_crashing(
    short_term_memory_factory,
):
    memory = short_term_memory_factory()
    consultation = FakeWorker("consultation_agent")
    diagnostic = FakeWorker("diagnostic_agent")
    checkpointer = InMemorySaver()
    audit_store = MemoryAuditStore()
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
        enable_short_term_memory=False,
        enable_long_term_memory=False,
        checkpointer=checkpointer,
        audit_store=audit_store,
    )
    collector = DebugTraceCollector(
        question="需要多角度分析的问题",
        context={},
        session_id="debug-swarm",
    )

    state = await graph.ainvoke(
        {
            "question": "需要多角度分析的问题",
            "session_id": "debug-swarm",
            "context": {},
            "debug_collector": collector,
        }
    )

    assert state["result"]["swarm_enabled"] is True
    assert any(
        event.name == "subtasks_created"
        for event in collector.get_events()
    )
    checkpoint = await graph.get_checkpoint(collector.run_id)
    assert checkpoint is not None
    assert "debug_collector" not in checkpoint.values
    assert isinstance(checkpoint.values["shared_context"], dict)
    audit_attempts = await audit_store.get_attempts(collector.run_id)
    assert audit_attempts[-1]["run"]["status"] == "success"
    assert any(
        event["name"] == "subtasks_created"
        for event in audit_attempts[-1]["events"]
    )
