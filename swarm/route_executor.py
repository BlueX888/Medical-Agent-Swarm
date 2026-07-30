"""Dependency-aware Worker execution for validated RoutePlans."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from .agent_catalog import AgentCatalog
from .routing_models import RoutePlan
from .shared_context import SharedContext, SubTask, TaskStatus


class RouteExecutor:
    """Execute ready tasks in waves, with at most one call per Worker."""

    def __init__(self, catalog: AgentCatalog):
        self.catalog = catalog

    async def execute(
        self,
        plan: RoutePlan,
        shared_context: SharedContext,
        context: Dict[str, Any],
        debug_collector: Optional[Any] = None,
    ) -> SharedContext:
        for worker_info in self.catalog.list_agents():
            worker = self.catalog.get_worker(worker_info["agent_id"])
            worker.attach_shared_context(shared_context)

        for task in plan.tasks:
            shared_context.add_subtask(
                SubTask(
                    id=task.id,
                    type=f"{task.assigned_agent}_task",
                    description=task.goal,
                    assigned_agent=task.assigned_agent,
                    dependencies=list(task.depends_on),
                    metadata={
                        "priority": task.priority,
                        "risk_level": plan.risk_level.value,
                        "conversation_history": list(context.get("recent_history") or []),
                        "context": context,
                        "tool_strategy": "Worker autonomously selects allowed Skills via function calling",
                    },
                )
            )

        while True:
            pending = [
                task
                for task in shared_context.task_decomposition.values()
                if task.status == TaskStatus.PENDING
            ]
            if not pending:
                break

            ready = []
            for task in pending:
                dependencies = [
                    shared_context.get_subtask(dependency)
                    for dependency in task.dependencies
                ]
                if any(
                    dependency is not None
                    and dependency.status == TaskStatus.FAILED
                    for dependency in dependencies
                ):
                    task.fail("dependency_failed")
                    continue
                if all(
                    dependency is not None
                    and dependency.status == TaskStatus.COMPLETED
                    for dependency in dependencies
                ):
                    ready.append(task)

            if not ready:
                for task in pending:
                    if task.status == TaskStatus.PENDING:
                        task.fail("unresolvable_dependencies")
                break

            # One ready task per Worker per wave prevents unsafe concurrent calls
            # on stateful AgentLoop instances. Different Workers still run together.
            wave = {}
            for task in ready:
                wave.setdefault(task.assigned_agent, task)

            await asyncio.gather(
                *[
                    self._execute_one(
                        task,
                        shared_context,
                        debug_collector,
                    )
                    for task in wave.values()
                ],
                return_exceptions=True,
            )
        return shared_context

    async def _execute_one(
        self,
        task: SubTask,
        shared_context: SharedContext,
        debug_collector: Optional[Any],
    ) -> None:
        worker = self.catalog.get_worker(task.assigned_agent)
        if worker is None:
            task.fail("assigned Worker is unavailable")
            return

        task.metadata["dependency_results"] = {
            dependency_id: shared_context.get_subtask(dependency_id).result
            for dependency_id in task.dependencies
        }
        if not shared_context.start_subtask(task.id):
            return
        try:
            result = await worker.process_subtask(
                task,
                debug_collector=debug_collector,
            )
            shared_context.complete_subtask(task.id, worker.agent_id, result)
        except Exception as exc:
            task.fail(str(exc))

