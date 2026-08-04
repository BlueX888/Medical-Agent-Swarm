"""
SwarmCoordinator: public entry and dependency assembly for the medical swarm.

The runtime workflow is executed by MedicalSwarmGraph. This class keeps the
external API stable while wiring LLM, Worker Agents, memory, and the graph.
"""
import asyncio
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from core import LLMClient
from core.checkpointing import (
    CheckpointSettings,
    RunLeaseManager,
    open_checkpointer,
    open_run_lease,
)
from core.audit import AuditStore, open_audit_store
from core.observability import trace_async
from debug import DebugTraceCollector, summarize_debug_run
from memory import (
    LongTermMemory,
    SessionSummaryManager,
    ShortTermMemory,
    create_short_term_memory,
)
from knowledge import KnowledgeBase, KnowledgeRuntime, create_knowledge_runtime

from .medical_swarm_graph import MedicalSwarmGraph
from .agent_catalog import AgentCatalog

@dataclass
class _LoopDefaults:
    init_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    short_term_memory: Optional[ShortTermMemory] = None
    knowledge_runtime: Optional[KnowledgeRuntime] = None


_LOOP_DEFAULTS_ATTRIBUTE = "_medical_swarm_defaults"


def _get_loop_defaults() -> _LoopDefaults:
    loop = asyncio.get_running_loop()
    defaults = getattr(loop, _LOOP_DEFAULTS_ATTRIBUTE, None)
    if defaults is None:
        defaults = _LoopDefaults()
        setattr(loop, _LOOP_DEFAULTS_ATTRIBUTE, defaults)
    return defaults


class SwarmCoordinator:
    """Stable public coordinator for medical swarm processing."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enable_swarm: bool = True,
        enable_memory: bool = True,
        enable_short_term_memory: Optional[bool] = None,
        enable_long_term_memory: Optional[bool] = None,
        swarm_timeout_s: float = 120.0,
        short_term_memory: Optional[ShortTermMemory] = None,
        checkpointer: Optional[Any] = None,
        run_lease: Optional[RunLeaseManager] = None,
        audit_store: Optional[AuditStore] = None,
        enable_rag: Optional[bool] = None,
        knowledge_base: Optional[KnowledgeBase] = None,
    ):
        self.llm_client = llm_client or LLMClient()
        self.enable_swarm = enable_swarm
        enable_short_term_memory, enable_long_term_memory = _resolve_memory_flags(
            enable_memory,
            enable_short_term_memory,
            enable_long_term_memory,
        )
        self.enable_short_term_memory = enable_short_term_memory
        self.enable_long_term_memory = enable_long_term_memory
        self.swarm_timeout_s = swarm_timeout_s
        knowledge_runtime = None
        if knowledge_base is None:
            knowledge_runtime = create_knowledge_runtime()
            knowledge_base = knowledge_runtime.knowledge_base
        self.knowledge_base = knowledge_base
        self.enable_rag = (
            knowledge_runtime.settings.enabled
            if enable_rag is None and knowledge_runtime is not None
            else bool(enable_rag)
        )

        self.consultation_agent = ConsultationAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.research_agent = ResearchAgent()

        self.worker_pool: List[Any] = [
            self.consultation_agent,
            self.diagnostic_agent,
            self.research_agent,
        ]
        self.agent_catalog = AgentCatalog(self.worker_pool)

        self.session_manager = SessionSummaryManager()
        self.short_term_memory = short_term_memory or ShortTermMemory()
        self.long_term_memory = LongTermMemory()

        self.medical_graph = MedicalSwarmGraph(
            llm_client=self.llm_client,
            worker_pool=self.worker_pool,
            consultation_agent=self.consultation_agent,
            diagnostic_agent=self.diagnostic_agent,
            research_agent=self.research_agent,
            short_term_memory=self.short_term_memory,
            long_term_memory=self.long_term_memory,
            session_manager=self.session_manager,
            enable_swarm=self.enable_swarm,
            enable_short_term_memory=self.enable_short_term_memory,
            enable_long_term_memory=self.enable_long_term_memory,
            swarm_timeout=self.swarm_timeout_s,
            agent_catalog=self.agent_catalog,
            checkpointer=checkpointer,
            run_lease=run_lease,
            audit_store=audit_store,
            knowledge_base=self.knowledge_base,
            enable_rag=self.enable_rag,
        )

        logger.info(f"SwarmCoordinator initialized with {len(self.worker_pool)} workers")
        logger.info(
            "Memory system: "
            f"short_term={'enabled' if self.enable_short_term_memory else 'disabled'}, "
            f"long_term={'enabled' if self.enable_long_term_memory else 'disabled'}"
        )

    def _get_agent_by_id(self, agent_id: str):
        """Compatibility helper for tests or internal callers."""
        return self.medical_graph._get_agent_by_id(agent_id)

    async def process(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        debug_collector: Optional[DebugTraceCollector] = None,
        swarm_timeout_s: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user question.

        Args:
            question: User question.
            context: Optional context such as demographics or medical history.
            session_id: Optional session id. A new id is generated when omitted.

        Returns:
            API-compatible result dictionary.
        """
        logger.info(f"Processing question (session={session_id or 'new'}): {question[:50]}...")
        initial_state = {
            "question": question,
            "context": context or {},
            "session_id": session_id,
            "enable_swarm": self.enable_swarm,
            "enable_short_term_memory": self.enable_short_term_memory,
            "enable_long_term_memory": self.enable_long_term_memory,
            "enable_rag": bool(getattr(self, "enable_rag", False)),
            "swarm_timeout_s": (
                swarm_timeout_s
                if swarm_timeout_s is not None
                else self.swarm_timeout_s
            ),
        }
        effective_run_id = run_id or getattr(debug_collector, "run_id", None)
        if effective_run_id:
            initial_state["run_id"] = effective_run_id
        if debug_collector:
            initial_state["debug_collector"] = debug_collector

        async def invoke_graph() -> Dict[str, Any]:
            return await trace_async(
                name="medical_swarm_request",
                run_type="chain",
                func=lambda: self.medical_graph.ainvoke(initial_state),
                inputs={
                    "question": question,
                    "context": context or {},
                    "session_id": session_id,
                    "enable_swarm": self.enable_swarm,
                    "enable_short_term_memory": self.enable_short_term_memory,
                    "enable_long_term_memory": self.enable_long_term_memory,
                    "enable_rag": bool(getattr(self, "enable_rag", False)),
                    "swarm_timeout_s": swarm_timeout_s or self.swarm_timeout_s,
                },
                metadata={
                    "run_id": getattr(debug_collector, "run_id", None),
                    "session_id": session_id,
                    "entrypoint": (
                        (debug_collector.get_run().metadata or {}).get("source")
                        if debug_collector
                        else None
                    ) or None,
                    "enable_swarm": self.enable_swarm,
                    "enable_short_term_memory": self.enable_short_term_memory,
                    "enable_long_term_memory": self.enable_long_term_memory,
                    "enable_rag": bool(getattr(self, "enable_rag", False)),
                    "swarm_timeout_s": swarm_timeout_s or self.swarm_timeout_s,
                    "worker_count": len(self.worker_pool),
                },
                tags=["medical-agent-swarm", "request"],
                output_mapper=lambda state: self._trace_result_summary(
                    state,
                    debug_collector,
                ),
            )

        if self.enable_short_term_memory and session_id:
            async with self.short_term_memory.session_scope(session_id):
                state = await invoke_graph()
        else:
            state = await invoke_graph()
        return state["result"]

    async def resume(
        self,
        run_id: str,
        checkpoint_id: Optional[str] = None,
        debug_collector: Optional[DebugTraceCollector] = None,
    ) -> Dict[str, Any]:
        """Resume a previously checkpointed workflow run."""
        checkpoint = await self.medical_graph.get_checkpoint(run_id, checkpoint_id)
        if checkpoint is None:
            raise LookupError(f"Checkpoint run not found: {run_id}")

        session_id = checkpoint.values.get("session_id")

        async def invoke_resume() -> Dict[str, Any]:
            state = await self.medical_graph.resume(
                run_id,
                checkpoint_id,
                debug_collector=debug_collector,
            )
            return state["result"]

        if self.enable_short_term_memory and session_id:
            async with self.short_term_memory.session_scope(str(session_id)):
                return await invoke_resume()
        return await invoke_resume()

    async def get_checkpoint(self, run_id: str, checkpoint_id: Optional[str] = None):
        return await self.medical_graph.get_checkpoint(run_id, checkpoint_id)

    async def list_checkpoints(self, run_id: str, *, limit: Optional[int] = None):
        return await self.medical_graph.list_checkpoints(run_id, limit=limit)

    def _trace_result_summary(
        self,
        state: Dict[str, Any],
        debug_collector: Optional[DebugTraceCollector] = None,
    ) -> Dict[str, Any]:
        if debug_collector:
            summary = summarize_debug_run(
                debug_collector.get_run(),
                debug_collector.get_events(),
            )
            return summary
        result = state.get("result") or {}
        route = state.get("route") or state.get("mode") or "unknown"
        if route not in {"single_agent", "swarm", "fallback"}:
            route = "unknown"
        timeout = bool(result.get("timeout_occurred"))
        agents = result.get("agents_involved", []) or []
        return {
            "status": "timeout" if timeout else "success",
            "route": route,
            "agent_count": len(set(agents)),
            "llm_call_count": result.get("llm_call_count", 0),
            "tool_call_count": result.get("tool_call_count", 0),
            "tool_success_count": result.get("tool_success_count", 0),
            "safety_checked": result.get("safety_checked"),
            "safety_passed": result.get("safety_passed"),
            "answer_length": len(result.get("answer", "") or ""),
        }


async def process_with_swarm(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    enable_swarm: bool = True,
    session_id: Optional[str] = None,
    debug: bool = False,
    enable_memory: bool = True,
    enable_short_term_memory: Optional[bool] = None,
    enable_long_term_memory: Optional[bool] = None,
    enable_rag: Optional[bool] = None,
    swarm_timeout_s: Optional[float] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function for processing a question through the swarm entry.
    """
    short_term_enabled, long_term_enabled = _resolve_memory_flags(
        enable_memory,
        enable_short_term_memory,
        enable_long_term_memory,
    )
    effective_run_id = run_id or str(uuid.uuid4())
    debug_collector = (
        DebugTraceCollector(
            question=question,
            context=context or {},
            session_id=session_id,
            run_id=effective_run_id,
            metadata={
                "source": "process_with_swarm",
                "enable_swarm": enable_swarm,
                "enable_memory": enable_memory,
                "enable_short_term_memory": short_term_enabled,
                "enable_long_term_memory": long_term_enabled,
                "enable_rag": enable_rag,
            },
        )
        if debug
        else None
    )

    defaults = _get_loop_defaults()
    async with defaults.init_lock:
        if defaults.short_term_memory is None:
            defaults.short_term_memory = await create_short_term_memory()
        if defaults.knowledge_runtime is None:
            defaults.knowledge_runtime = create_knowledge_runtime()

    async with _open_checkpointed_coordinator(
        short_term_memory=defaults.short_term_memory,
        enable_swarm=enable_swarm,
        enable_memory=enable_memory,
        enable_short_term_memory=short_term_enabled,
        enable_long_term_memory=long_term_enabled,
        enable_rag=enable_rag,
        knowledge_runtime=defaults.knowledge_runtime,
        swarm_timeout_s=swarm_timeout_s or 120.0,
    ) as coordinator:
        try:
            result = await coordinator.process(
                question,
                context,
                session_id=session_id,
                debug_collector=debug_collector,
                swarm_timeout_s=swarm_timeout_s,
                run_id=effective_run_id,
            )
        except Exception as exc:
            if debug_collector:
                debug_collector.finish_failed(exc)
            raise

    result = dict(result)
    result.setdefault("run_id", effective_run_id)
    if debug_collector:
        return {
            "result": result,
            "debug_run": debug_collector.get_run().to_dict(),
            "debug_events": [event.to_dict() for event in debug_collector.get_events()],
        }

    return result


async def resume_with_swarm(
    run_id: str,
    checkpoint_id: Optional[str] = None,
    *,
    debug: bool = False,
) -> Dict[str, Any]:
    """Resume a run created by ``process_with_swarm`` or the HTTP API."""
    defaults = _get_loop_defaults()
    async with defaults.init_lock:
        if defaults.short_term_memory is None:
            defaults.short_term_memory = await create_short_term_memory()
        if defaults.knowledge_runtime is None:
            defaults.knowledge_runtime = create_knowledge_runtime()

    async with _open_checkpointed_coordinator(
        short_term_memory=defaults.short_term_memory,
        knowledge_runtime=defaults.knowledge_runtime,
    ) as coordinator:
        checkpoint = await coordinator.get_checkpoint(run_id, checkpoint_id)
        if checkpoint is None:
            raise LookupError(f"Checkpoint run not found: {run_id}")
        values = checkpoint.values
        coordinator.enable_swarm = bool(values.get("enable_swarm", True))
        coordinator.enable_short_term_memory = bool(
            values.get("enable_short_term_memory", True)
        )
        coordinator.enable_long_term_memory = bool(
            values.get("enable_long_term_memory", True)
        )
        coordinator.enable_rag = bool(values.get("enable_rag", False))
        coordinator.medical_graph.enable_swarm = coordinator.enable_swarm
        coordinator.medical_graph.enable_short_term_memory = (
            coordinator.enable_short_term_memory
        )
        coordinator.medical_graph.enable_long_term_memory = (
            coordinator.enable_long_term_memory
        )
        coordinator.medical_graph.enable_rag = coordinator.enable_rag
        collector = (
            DebugTraceCollector(
                question=str(values.get("question") or ""),
                context=dict(values.get("context") or {}),
                session_id=values.get("session_id"),
                run_id=run_id,
                metadata={
                    "source": "resume_with_swarm",
                    "checkpoint_id": checkpoint_id or checkpoint.checkpoint_id,
                },
            )
            if debug
            else None
        )
        result = await coordinator.resume(
            run_id,
            checkpoint_id,
            debug_collector=collector,
        )

    if collector:
        return {
            "result": result,
            "debug_run": collector.get_run().to_dict(),
            "debug_events": [event.to_dict() for event in collector.get_events()],
        }
    return result


@asynccontextmanager
async def _open_checkpointed_coordinator(
    *,
    short_term_memory: ShortTermMemory,
    enable_swarm: bool = True,
    enable_memory: bool = True,
    enable_short_term_memory: Optional[bool] = None,
    enable_long_term_memory: Optional[bool] = None,
    swarm_timeout_s: float = 120.0,
    enable_rag: Optional[bool] = None,
    knowledge_runtime: Optional[KnowledgeRuntime] = None,
):
    settings = CheckpointSettings.from_env()
    effective_knowledge_runtime = knowledge_runtime or create_knowledge_runtime()
    effective_enable_rag = (
        effective_knowledge_runtime.settings.enabled
        if enable_rag is None
        else bool(enable_rag)
    )
    async with AsyncExitStack() as stack:
        checkpointer = await stack.enter_async_context(open_checkpointer(settings))
        run_lease = await stack.enter_async_context(open_run_lease(settings))
        audit_store = await stack.enter_async_context(open_audit_store(settings))
        yield SwarmCoordinator(
            enable_swarm=enable_swarm,
            enable_memory=enable_memory,
            enable_short_term_memory=enable_short_term_memory,
            enable_long_term_memory=enable_long_term_memory,
            swarm_timeout_s=swarm_timeout_s,
            short_term_memory=short_term_memory,
            checkpointer=checkpointer,
            run_lease=run_lease,
            audit_store=audit_store,
            enable_rag=effective_enable_rag,
            knowledge_base=effective_knowledge_runtime.knowledge_base,
        )


def _resolve_memory_flags(
    enable_memory: bool,
    enable_short_term_memory: Optional[bool],
    enable_long_term_memory: Optional[bool],
) -> tuple[bool, bool]:
    """Resolve legacy combined configuration into independent memory flags."""
    return (
        enable_memory
        if enable_short_term_memory is None
        else enable_short_term_memory,
        enable_memory
        if enable_long_term_memory is None
        else enable_long_term_memory,
    )
