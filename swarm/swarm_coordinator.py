"""
SwarmCoordinator: public entry and dependency assembly for the medical swarm.

The runtime workflow is executed by MedicalSwarmGraph. This class keeps the
external API stable while wiring LLM, Worker Agents, memory, and the graph.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from core import LLMClient
from core.observability import trace_async
from debug import DebugTraceCollector
from memory import (
    LongTermMemory,
    SessionSummaryManager,
    ShortTermMemory,
    create_short_term_memory,
)

from .medical_swarm_graph import MedicalSwarmGraph

@dataclass
class _LoopDefaults:
    init_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    coordinator: Optional["SwarmCoordinator"] = None
    short_term_memory: Optional[ShortTermMemory] = None


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

        self.consultation_agent = ConsultationAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.research_agent = ResearchAgent()

        self.worker_pool: List[Any] = [
            self.consultation_agent,
            self.diagnostic_agent,
            self.research_agent,
        ]

        self.session_manager = SessionSummaryManager()
        self.short_term_memory = short_term_memory or ShortTermMemory(storage_type="memory")
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
        }
        if swarm_timeout_s is not None:
            initial_state["swarm_timeout_s"] = swarm_timeout_s
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
                    "swarm_timeout_s": swarm_timeout_s or self.swarm_timeout_s,
                },
                metadata={
                    "session_id": session_id,
                    "enable_swarm": self.enable_swarm,
                    "enable_short_term_memory": self.enable_short_term_memory,
                    "enable_long_term_memory": self.enable_long_term_memory,
                    "swarm_timeout_s": swarm_timeout_s or self.swarm_timeout_s,
                    "worker_count": len(self.worker_pool),
                },
                tags=["medical-agent-swarm", "request"],
                output_mapper=self._trace_result_summary,
            )

        if self.enable_short_term_memory and session_id:
            async with self.short_term_memory.session_scope(session_id):
                state = await invoke_graph()
        else:
            state = await invoke_graph()
        return state["result"]

    def _trace_result_summary(self, state: Dict[str, Any]) -> Dict[str, Any]:
        result = state.get("result") or {}
        return {
            "session_id": state.get("session_id") or result.get("session_id"),
            "route": state.get("route") or state.get("mode"),
            "swarm_enabled": result.get("swarm_enabled"),
            "timeout_occurred": result.get("timeout_occurred"),
            "agents_involved": result.get("agents_involved", []),
            "answer": result.get("answer"),
            "safety_checked": result.get("safety_checked"),
            "safety_passed": result.get("safety_passed"),
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
    swarm_timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience function for processing a question through the swarm entry.
    """
    short_term_enabled, long_term_enabled = _resolve_memory_flags(
        enable_memory,
        enable_short_term_memory,
        enable_long_term_memory,
    )
    debug_collector = (
        DebugTraceCollector(
            question=question,
            context=context or {},
            session_id=session_id,
            metadata={
                "source": "process_with_swarm",
                "enable_swarm": enable_swarm,
                "enable_memory": enable_memory,
                "enable_short_term_memory": short_term_enabled,
                "enable_long_term_memory": long_term_enabled,
            },
        )
        if debug
        else None
    )

    defaults = _get_loop_defaults()
    async with defaults.init_lock:
        # Reuse cached coordinator when called with default params.
        if (
            defaults.coordinator is not None
            and defaults.coordinator.enable_swarm == enable_swarm
            and defaults.coordinator.enable_short_term_memory == short_term_enabled
            and defaults.coordinator.enable_long_term_memory == long_term_enabled
        ):
            coordinator = defaults.coordinator
        else:
            if defaults.short_term_memory is None:
                defaults.short_term_memory = await create_short_term_memory()
            coordinator = SwarmCoordinator(
                enable_swarm=enable_swarm,
                enable_memory=enable_memory,
                enable_short_term_memory=short_term_enabled,
                enable_long_term_memory=long_term_enabled,
                short_term_memory=defaults.short_term_memory,
            )
            if enable_swarm and enable_memory:
                defaults.coordinator = coordinator

    try:
        result = await coordinator.process(
            question,
            context,
            session_id=session_id,
            debug_collector=debug_collector,
            swarm_timeout_s=swarm_timeout_s,
        )
    except Exception as exc:
        if debug_collector:
            debug_collector.finish_failed(exc)
        raise

    if debug_collector:
        return {
            "result": result,
            "debug_run": debug_collector.get_run().to_dict(),
            "debug_events": [event.to_dict() for event in debug_collector.get_events()],
        }

    return result


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
