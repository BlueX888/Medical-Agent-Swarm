import asyncio

import pytest

from swarm.agent_catalog import AgentCatalog
from swarm.route_executor import RouteExecutor
from swarm.routing_models import ExecutionMode, PlannedTask, RiskLevel, RoutePlan
from swarm.shared_context import SharedContext, TaskStatus


class RecordingWorker:
    def __init__(self, agent_id, capabilities, recorder):
        self.agent_id = agent_id
        self._capabilities = capabilities
        self.recorder = recorder
        self.active = 0
        self.max_active = 0

    def get_capabilities(self):
        return self._capabilities

    def attach_shared_context(self, context):
        self.context = context

    async def process_subtask(self, subtask, debug_collector=None):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.recorder.append(("start", subtask.id))
        await asyncio.sleep(0.01)
        dependency_results = subtask.metadata.get("dependency_results", {})
        self.recorder.append(("end", subtask.id))
        self.active -= 1
        return {"answer": subtask.description, "dependencies": dependency_results}


def make_plan(tasks, mode):
    return RoutePlan(
        intent_summary="execute",
        intents=["general_consultation"],
        risk_level=RiskLevel.LOW,
        confidence=1,
        tasks=tasks,
        execution_mode=mode,
        source="llm",
        reasons=[],
    )


def planned(task_id, agent, depends_on=None):
    return PlannedTask(
        id=task_id,
        goal=f"完成 {task_id}",
        required_capabilities=["cap"],
        assigned_agent=agent,
        priority="normal",
        depends_on=depends_on or [],
    )


@pytest.mark.asyncio
async def test_independent_tasks_for_different_workers_run_in_parallel():
    recorder = []
    first = RecordingWorker("first", ["cap"], recorder)
    second = RecordingWorker("second", ["cap"], recorder)
    executor = RouteExecutor(AgentCatalog([first, second]))
    context = SharedContext("parallel")

    await executor.execute(
        make_plan([planned("a", "first"), planned("b", "second")], ExecutionMode.PARALLEL),
        context,
        {},
    )

    assert recorder[:2] == [("start", "a"), ("start", "b")]


@pytest.mark.asyncio
async def test_dependencies_execute_in_order_and_receive_prior_results():
    recorder = []
    first = RecordingWorker("first", ["cap"], recorder)
    second = RecordingWorker("second", ["cap"], recorder)
    context = SharedContext("sequential")

    await RouteExecutor(AgentCatalog([first, second])).execute(
        make_plan(
            [planned("a", "first"), planned("b", "second", ["a"])],
            ExecutionMode.SEQUENTIAL,
        ),
        context,
        {},
    )

    assert recorder == [
        ("start", "a"),
        ("end", "a"),
        ("start", "b"),
        ("end", "b"),
    ]
    assert context.get_subtask("b").result["dependencies"]["a"]["answer"] == "完成 a"


@pytest.mark.asyncio
async def test_tasks_for_same_worker_are_never_concurrent():
    recorder = []
    worker = RecordingWorker("only", ["cap"], recorder)
    context = SharedContext("one-worker")

    await RouteExecutor(AgentCatalog([worker])).execute(
        make_plan(
            [planned("a", "only"), planned("b", "only")],
            ExecutionMode.SEQUENTIAL,
        ),
        context,
        {},
    )

    assert worker.max_active == 1
    assert all(task.status == TaskStatus.COMPLETED for task in context.task_decomposition.values())

