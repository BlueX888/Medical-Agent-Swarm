"""
SwarmCoordinator: public entry and dependency assembly for the medical swarm.

The runtime workflow is executed by MedicalSwarmGraph. This class keeps the
external API stable while wiring LLM, Worker Agents, memory, and the graph.
"""
from typing import Any, Dict, List, Optional

from loguru import logger

from agents import ConsultationAgent, DiagnosticAgent, ResearchAgent
from core import LLMClient
from debug import DebugTraceCollector
from memory import LongTermMemory, SessionSummaryManager, ShortTermMemory

from .medical_swarm_graph import MedicalSwarmGraph


class SwarmCoordinator:
    """Stable public coordinator for medical swarm processing."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        enable_swarm: bool = True,
        enable_memory: bool = True,
        swarm_timeout_s: float = 120.0,
    ):
        self.llm_client = llm_client or LLMClient()
        self.enable_swarm = enable_swarm
        self.enable_memory = enable_memory
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
        self.short_term_memory = ShortTermMemory(storage_type="memory")
        self.long_term_memory = LongTermMemory()

        for worker in self.worker_pool:
            if hasattr(worker, "loop"):
                worker.loop.short_term_memory = self.short_term_memory if enable_memory else None

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
            enable_memory=self.enable_memory,
            swarm_timeout=self.swarm_timeout_s,
        )

        logger.info(f"SwarmCoordinator initialized with {len(self.worker_pool)} workers")
        logger.info(
            "Memory system: "
            f"short_term={self.short_term_memory.storage_type}, "
            f"long_term={'enabled' if self.long_term_memory.enabled else 'disabled'}"
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

        state = await self.medical_graph.ainvoke(initial_state)
        return state["result"]


async def process_with_swarm(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    enable_swarm: bool = True,
    session_id: Optional[str] = None,
    debug: bool = False,
    enable_memory: bool = True,
    swarm_timeout_s: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Convenience function for processing a question through the swarm entry.
    """
    debug_collector = (
        DebugTraceCollector(
            question=question,
            context=context or {},
            session_id=session_id,
            metadata={
                "source": "process_with_swarm",
                "enable_swarm": enable_swarm,
                "enable_memory": enable_memory,
            },
        )
        if debug
        else None
    )
    coordinator = SwarmCoordinator(enable_swarm=enable_swarm, enable_memory=enable_memory)

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
