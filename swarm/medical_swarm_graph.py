"""
LangGraph orchestration layer for the medical swarm workflow.

SwarmCoordinator owns dependency assembly. This graph owns runtime flow:
memory injection, planning, conditional routing, worker execution, synthesis,
memory persistence, and response shaping.
"""
import asyncio
import json
import re
import uuid
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph
from loguru import logger

from core import LLMClient
from core.observability import trace_async
from debug import DebugTraceCollector
from memory import (
    LongTermMemory,
    SessionSummary,
    SessionSummaryManager,
    ShortTermMemory,
    ShortTermMemoryError,
)
from memory.evidence_cache import EvidenceMemory

from .medical_swarm_state import MedicalSwarmState
from .shared_context import SharedContext, SubTask


class MedicalSwarmGraph:
    """Executable LangGraph workflow for medical multi-agent processing."""

    def __init__(
        self,
        llm_client: Optional[LLMClient],
        worker_pool: List[Any],
        consultation_agent: Any,
        diagnostic_agent: Any,
        research_agent: Any,
        short_term_memory: ShortTermMemory,
        long_term_memory: LongTermMemory,
        session_manager: SessionSummaryManager,
        enable_swarm: bool = True,
        enable_short_term_memory: bool = True,
        enable_long_term_memory: bool = True,
        swarm_timeout: float = 120.0,
        swarm_timeout_grace_s: float = 10.0,
    ):
        self.llm_client = llm_client or LLMClient()
        self.worker_pool = worker_pool
        self.consultation_agent = consultation_agent
        self.diagnostic_agent = diagnostic_agent
        self.research_agent = research_agent
        self.short_term_memory = short_term_memory
        self.long_term_memory = long_term_memory
        self.session_manager = session_manager
        self.enable_swarm = enable_swarm
        self.enable_short_term_memory = enable_short_term_memory
        self.enable_long_term_memory = enable_long_term_memory
        self.swarm_timeout = swarm_timeout
        self.swarm_timeout_grace_s = swarm_timeout_grace_s

        self._compiled_graph = self.build_graph()

    def build_graph(self):
        """Build the executable LangGraph graph."""
        graph = StateGraph(MedicalSwarmState)

        graph.add_node("load_memory", self._trace_node("load_memory", "load_memory", self.load_memory))
        graph.add_node("plan_and_decompose", self._trace_node("planning", "plan_and_decompose", self.plan_and_decompose))
        graph.add_node("route_by_subtasks", self._trace_node("routing", "route_by_subtasks", self.route_by_subtasks))
        graph.add_node("run_single_agent", self._trace_node("agent_loop", "run_single_agent", self.run_single_agent))
        graph.add_node("run_swarm", self._trace_node("agent_loop", "run_swarm", self.run_swarm))
        graph.add_node("run_fallback", self._trace_node("agent_loop", "run_fallback", self.run_fallback))
        graph.add_node("save_memory", self._trace_node("save_memory", "save_memory", self.save_memory))
        graph.add_node("build_response", self._trace_node("safety_check", "build_response", self.build_response))

        graph.set_entry_point("load_memory")
        graph.add_edge("load_memory", "plan_and_decompose")
        graph.add_edge("plan_and_decompose", "route_by_subtasks")
        graph.add_conditional_edges(
            "route_by_subtasks",
            self._select_route,
            {
                "single_agent": "run_single_agent",
                "swarm": "run_swarm",
                "fallback": "run_fallback",
                "disabled_swarm": "run_fallback",
            },
        )
        graph.add_edge("run_single_agent", "build_response")
        graph.add_edge("run_swarm", "build_response")
        graph.add_edge("run_fallback", "build_response")
        graph.add_edge("build_response", "save_memory")
        graph.add_edge("save_memory", END)

        return graph.compile()

    async def ainvoke(self, initial_state: Dict[str, Any]) -> MedicalSwarmState:
        """Invoke the compiled graph with API-compatible defaults."""
        state: MedicalSwarmState = dict(initial_state)
        start_time = state.get("start_time") or datetime.now()
        state["start_time"] = start_time
        state["context"] = state.get("context") or {}

        if not state.get("session_id"):
            state["session_id"] = (
                f"{start_time.strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:8]}"
            )

        collector = self._get_debug_collector(state)
        if collector:
            collector.update_run(
                session_id=state["session_id"],
                question=state.get("question", ""),
                context=state.get("context") or {},
                metadata={
                    **(collector.get_run().metadata or {}),
                    "enable_swarm": self.enable_swarm,
                    "enable_short_term_memory": self.enable_short_term_memory,
                    "enable_long_term_memory": self.enable_long_term_memory,
                    "swarm_timeout": self.swarm_timeout,
                    "worker_count": len(self.worker_pool),
                    "source": "api",
                },
            )

        try:
            final_state = await self._compiled_graph.ainvoke(state)
        except Exception as exc:
            if collector:
                collector.finish_failed(exc)
            raise

        collector = self._get_debug_collector(final_state) or collector
        if collector:
            result = final_state.get("result") or {}
            collector.finish_success(
                result_json=result,
                route=final_state.get("route") or final_state.get("mode"),
                final_answer=result.get("answer") or final_state.get("final_answer", ""),
                timeout=bool(final_state.get("timeout_occurred", False)),
            )

        return final_state

    async def load_memory(self, state: MedicalSwarmState) -> Dict[str, Any]:
        """Inject recent session history and similar historical cases."""
        question = state["question"]
        session_id = state["session_id"]
        context = dict(state.get("context") or {})

        if not self.enable_short_term_memory and not self.enable_long_term_memory:
            collector = self._get_debug_collector(state)
            if collector:
                collector.record_event(
                    "memory",
                    name="load_memory",
                    input={"session_id": session_id, "question": question},
                    output={
                        "enabled": False,
                        "recent_history_count": 0,
                        "historical_cases_count": 0,
                    },
                    metadata={
                        "long_term_enabled": bool(getattr(self.long_term_memory, "enabled", False)),
                    },
                )
            return {
                "recent_history": [],
                "historical_cases": [],
                "enhanced_context": context,
            }

        short_term_memory_error = None
        if self.enable_short_term_memory:
            try:
                recent_history = await self.short_term_memory.load_context(
                    session_id=session_id,
                    max_turns=5,
                )
            except ShortTermMemoryError as exc:
                short_term_memory_error = str(exc)
                recent_history = []
                logger.error(
                    "Short-term memory unavailable while loading context "
                    f"(session={session_id}): {exc}"
                )
        else:
            recent_history = []
        similar_memories = (
            self.long_term_memory.search_similar_sessions(
                query=question,
                limit=3,
            )
            if self.enable_long_term_memory
            else []
        )

        enhanced_context = dict(context)
        if recent_history:
            enhanced_context["recent_history"] = recent_history
            logger.info(
                f"Loaded {len(recent_history)} recent messages from short-term memory"
            )

        if similar_memories:
            enhanced_context["historical_cases"] = [
                {
                    "summary": mem["content"],
                    "score": mem["score"],
                }
                for mem in similar_memories
            ]
            logger.info(
                f"Found {len(similar_memories)} similar historical cases from long-term memory"
            )

        collector = self._get_debug_collector(state)
        if collector:
            collector.record_event(
                "memory",
                name="load_memory",
                input={
                    "session_id": session_id,
                    "question": question,
                    "context": context,
                },
                output={
                    "enabled": True,
                    "recent_history_count": len(recent_history),
                    "historical_cases_count": len(similar_memories),
                    "recent_history": recent_history,
                    "historical_cases": similar_memories,
                    "enhanced_context": enhanced_context,
                    "short_term_memory_error": short_term_memory_error,
                },
                metadata={
                    "long_term_enabled": bool(getattr(self.long_term_memory, "enabled", False)),
                },
            )

        return {
            "recent_history": recent_history,
            "historical_cases": similar_memories,
            "enhanced_context": enhanced_context,
            "short_term_memory_error": short_term_memory_error,
        }

    async def plan_and_decompose(self, state: MedicalSwarmState) -> Dict[str, Any]:
        """Use the planning node to decide worker subtasks."""
        question = state["question"]
        context = state.get("enhanced_context") or {}

        memory_assessment = self._assessment_from_evidence_memory(question)
        if memory_assessment:
            subtasks = memory_assessment.get("subtasks", [])
            collector = self._get_debug_collector(state)
            if collector:
                collector.record_event(
                    "planning",
                    name="evidence_memory_plan",
                    input={"question": question},
                    output=memory_assessment,
                    metadata={
                        "subtasks_count": len(subtasks),
                        "source": "evidence_memory",
                    },
                )
                collector.record_event(
                    "constraint_check",
                    name="validate_task_decomposition",
                    input={"question": question, "subtasks": subtasks},
                    output={
                        "valid": True,
                        "issues": [],
                        "recommendations": [],
                        "note": "Evidence memory shortcut generated a single-agent plan.",
                    },
                    metadata={"subtasks_count": len(subtasks)},
                )
            logger.info(
                "MedicalSwarmGraph used evidence memory plan "
                f"({memory_assessment.get('memory_id')}, {len(subtasks)} subtasks)"
            )
            return {"assessment": memory_assessment, "subtasks": subtasks}

        messages = [
            {"role": "system", "content": self._get_planning_prompt()},
            {"role": "user", "content": f"问题：{question}\n\n背景：{context or '无'}"},
        ]

        try:
            content = await self._call_llm_with_retry(messages, state)
            logger.debug(f"MedicalSwarmGraph planning response: {content[:200]}...")
            assessment = self._parse_planning_response(content)
        except Exception as exc:
            logger.error(f"MedicalSwarmGraph planning error after retries: {exc}")
            assessment = {
                "subtasks": [],
                "reason": f"规划失败：{exc}",
            }

        subtasks = assessment.get("subtasks", [])
        if not isinstance(subtasks, list):
            subtasks = []
            assessment["subtasks"] = subtasks

        collector = self._get_debug_collector(state)
        if collector:
            collector.record_event(
                "constraint_check",
                name="validate_task_decomposition",
                input={
                    "question": question,
                    "subtasks": subtasks,
                },
                output={
                    "valid": True,
                    "issues": [],
                    "recommendations": [],
                    "note": "No swarm-level validator is wired in this graph yet.",
                },
                metadata={
                    "subtasks_count": len(subtasks),
                },
            )

        logger.info(f"MedicalSwarmGraph planned {len(subtasks)} subtasks")
        return {"assessment": assessment, "subtasks": subtasks}

    async def route_by_subtasks(self, state: MedicalSwarmState) -> Dict[str, Any]:
        """Choose single-agent, swarm, or fallback route."""
        subtasks = state.get("subtasks") or []

        if len(subtasks) == 1:
            route = "single_agent"
        elif len(subtasks) >= 2 and self.enable_swarm:
            route = "swarm"
        elif len(subtasks) == 0:
            route = "fallback"
        else:
            route = "disabled_swarm"

        logger.info(f"MedicalSwarmGraph route: {route} ({len(subtasks)} subtasks)")
        return {"route": route, "mode": route}

    async def run_single_agent(self, state: MedicalSwarmState) -> Dict[str, Any]:
        """Run the selected worker directly for a one-subtask plan."""
        task = (state.get("subtasks") or [{}])[0]
        agent_id = task.get("assigned_agent")
        agent = self._get_agent_by_id(agent_id)
        if agent is None:
            logger.warning(f"Unknown agent_id: {agent_id}, fallback to ConsultationAgent")
            agent = self.consultation_agent
            agent_id = agent.agent_id
        return await self._run_agent_exec(
            agent=agent,
            question=state["question"],
            enhanced_context=state.get("enhanced_context") or {},
            session_id=state["session_id"],
            collector=self._get_debug_collector(state),
            route_reason=f"单任务路由到 {agent_id}",
            task=task,
        )

    async def run_swarm(self, state: MedicalSwarmState) -> Dict[str, Any]:
        """Run multi-worker collaboration inside a LangGraph node."""
        question = state["question"]
        session_id = state["session_id"]
        assessment = state.get("assessment") or {}
        shared_context = SharedContext(session_id=session_id)
        enhanced_context = state.get("enhanced_context") or {}

        for worker in self.worker_pool:
            worker.attach_shared_context(shared_context)

        subtasks = self._create_subtasks(
            assessment,
            shared_context,
            question=question,
            context=enhanced_context,
        )
        logger.info(f"Created {len(subtasks)} subtasks")
        collector = self._get_debug_collector(state)
        if collector:
            collector.record_event(
                "planning",
                name="subtasks_created",
                input=assessment,
                output=[
                    {
                        "id": subtask.id,
                        "type": subtask.type,
                        "description": subtask.description,
                        "assigned_agent": subtask.assigned_agent,
                        "tool_policy": subtask.metadata.get("tool_policy", {}),
                    }
                    for subtask in subtasks
                ],
            )

        tasks = [
            asyncio.create_task(
                self._worker_execute_assigned_tasks(worker, shared_context, collector)
            )
            for worker in self.worker_pool
        ]

        timeout_occurred = False
        effective_swarm_timeout = self._effective_swarm_timeout(state)
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=effective_swarm_timeout,
            )
        except asyncio.TimeoutError:
            timeout_occurred = True
            logger.warning(f"Swarm execution timeout ({effective_swarm_timeout}s)")
            completed_agents = list(shared_context.agent_contributions.keys())
            unfinished_tasks = [
                (subtask.assigned_agent, subtask.type, subtask.status.value)
                for subtask in shared_context.task_decomposition.values()
                if subtask.status.value in {"pending", "claimed", "in_progress"}
            ]
            logger.info(f"Completed agents: {completed_agents}")
            logger.info(f"Timed out tasks: {unfinished_tasks}")

        final_answer = await self._synthesize_results(
            question=question,
            shared_context=shared_context,
            timeout_occurred=timeout_occurred,
            debug_collector=collector,
        )
        if collector:
            collector.record_event(
                "swarm_context",
                name="shared_context_snapshot",
                input={
                    "timeout_occurred": timeout_occurred,
                    "swarm_timeout": effective_swarm_timeout,
                    "requested_swarm_timeout_s": state.get("swarm_timeout_s"),
                },
                output=self._shared_context_debug_snapshot(shared_context),
                status="failed" if timeout_occurred else "success",
                error="swarm_timeout" if timeout_occurred else None,
                metadata={
                    "total_subtasks": len(shared_context.task_decomposition),
                    "completed_subtasks": len(shared_context.get_all_completed_subtasks()),
                    "contribution_count": len(shared_context.get_contributions()),
                    "event_count": len(shared_context.events),
                },
            )

        return {
            "shared_context": shared_context,
            "final_answer": final_answer,
            "timeout_occurred": timeout_occurred,
            "effective_swarm_timeout_s": effective_swarm_timeout,
        }

    async def run_fallback(self, state: MedicalSwarmState) -> Dict[str, Any]:
        """Fallback to ConsultationAgent for empty plans or disabled swarm."""
        return await self._run_agent_exec(
            agent=self.consultation_agent,
            question=state["question"],
            enhanced_context=state.get("enhanced_context") or {},
            session_id=state["session_id"],
            collector=self._get_debug_collector(state),
            route_reason="fallback",
            task=None,
        )

    async def _run_agent_exec(
        self,
        agent: Any,
        question: str,
        enhanced_context: Dict[str, Any],
        session_id: str,
        collector: Any,
        route_reason: str = "",
        task: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute an agent with standard boilerplate (single_agent + fallback)."""
        tool_policy = self._tool_policy_for_request(
            question=question,
            context=enhanced_context,
            assessment={},
            task=task or {},
        )
        result = await agent.process(
            {
                "question": question,
                "context": enhanced_context,
                "conversation_history": enhanced_context.get("recent_history", []),
                "session_id": session_id,
                "tool_policy": tool_policy,
                "debug_collector": collector,
            }
        )
        final_answer = result.get("answer", "")
        result.update(
            {
                "swarm_enabled": False,
                "session_id": session_id,
                "route_reason": route_reason,
            }
        )
        result.setdefault(
            "disclaimer",
            "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。",
        )
        result.setdefault("suggestions", [])
        return {"result": result, "final_answer": final_answer}

    async def save_memory(self, state: MedicalSwarmState) -> Dict[str, Any]:
        """Persist one visible turn and optional long-term artifacts."""
        end_time = datetime.now()
        try:
            mode = state.get("mode") or state.get("route") or "unknown"
            final_answer = (state.get("result") or {}).get("answer") or state.get("final_answer", "")
            shared_context = state.get("shared_context")
            collector = self._get_debug_collector(state)

            if not self.enable_short_term_memory and not self.enable_long_term_memory:
                if collector:
                    collector.record_event(
                        "memory",
                        name="save_memory",
                        input={"session_id": state["session_id"], "mode": mode},
                        output={"enabled": False, "saved": False},
                    )
                return {"end_time": end_time}

            short_term_saved = False
            short_term_error = None
            if self.enable_short_term_memory and final_answer:
                try:
                    result = dict(state.get("result") or {})
                    assistant_metadata = {
                        "risk_level": self._extract_risk_level_from_state(state),
                        "suggestions": result.get("suggestions", []),
                        "disclaimer": result.get("disclaimer", ""),
                        "agents_involved": result.get("agents_involved", []),
                    }
                    await self.short_term_memory.save_turn(
                        session_id=state["session_id"],
                        user_message=state["question"],
                        assistant_message=final_answer,
                        assistant_metadata=assistant_metadata,
                    )
                    short_term_saved = True
                    logger.info(
                        "Saved completed turn to short-term memory "
                        f"(session={state['session_id']})"
                    )
                except Exception as exc:
                    short_term_error = str(exc)
                    logger.error(f"Failed to save to short-term memory: {exc}")

            summary_saved = False
            if self.enable_long_term_memory and mode == "swarm" and shared_context:
                try:
                    summary = SessionSummary.from_shared_context(
                        session_id=state["session_id"],
                        question=state["question"],
                        shared_context=shared_context,
                        final_answer=final_answer,
                        start_time=state["start_time"],
                        end_time=end_time,
                    )
                    self.session_manager.save_summary(summary)
                    summary_saved = True
                except Exception as exc:
                    logger.error(f"Failed to generate session summary: {exc}")

            long_term_saved = False
            long_term_error = None
            if self.enable_long_term_memory:
                try:
                    metadata = {
                        "mode": mode,
                        "total_time": (end_time - state["start_time"]).total_seconds(),
                    }
                    if mode == "swarm" and shared_context:
                        metadata.update(
                            {
                                "agents_count": len(shared_context.agent_contributions),
                                "timeout_occurred": state.get("timeout_occurred", False),
                            }
                        )
                    else:
                        metadata["subtasks_count"] = len(state.get("subtasks") or [])

                    memory_id = self.long_term_memory.add_session_summary(
                        session_id=state["session_id"],
                        question=state["question"],
                        answer=final_answer,
                        metadata=metadata,
                    )
                    long_term_saved = bool(memory_id)
                    logger.info(
                        f"Processed long-term memory save "
                        f"(session={state['session_id']}, mode={mode})"
                    )
                except Exception as exc:
                    long_term_error = str(exc)
                    logger.error(f"Failed to save to long-term memory: {exc}")

            if collector:
                errors = [
                    error
                    for error in (short_term_error, long_term_error)
                    if error is not None
                ]
                collector.record_event(
                    "memory",
                    name="save_memory",
                    input={
                        "session_id": state["session_id"],
                        "mode": mode,
                        "final_answer": final_answer,
                    },
                    output={
                        "enabled": True,
                        "short_term_saved": short_term_saved,
                        "short_term_error": short_term_error,
                        "summary_saved": summary_saved,
                        "long_term_saved": long_term_saved,
                        "long_term_error": long_term_error,
                    },
                    status="success" if not errors else "failed",
                    error="; ".join(errors) if errors else None,
                    metadata={
                        "short_term_enabled": self.enable_short_term_memory,
                        "long_term_requested": self.enable_long_term_memory,
                        "long_term_enabled": bool(getattr(self.long_term_memory, "enabled", False)),
                    },
                )

        except Exception as exc:
            logger.error(f"save_memory failed (non-critical, response already built): {exc}")

        return {"end_time": end_time}

    async def build_response(self, state: MedicalSwarmState) -> Dict[str, Any]:
        """Build the API-compatible final result."""
        mode = state.get("mode") or state.get("route") or "fallback"
        result = dict(state.get("result") or {})
        final_answer = state.get("final_answer") or result.get("answer", "")

        if mode == "swarm":
            shared_context = state.get("shared_context")
            completed_agents = (
                list(shared_context.agent_contributions.keys()) if shared_context else []
            )
            total_time = (
                (state.get("end_time") or datetime.now()) - state["start_time"]
            ).total_seconds()
            timeout_occurred = bool(state.get("timeout_occurred", False))

            result = {
                "answer": final_answer,
                "swarm_enabled": True,
                "session_id": state["session_id"],
                "agents_involved": completed_agents,
                "subtasks_completed": (
                    len(shared_context.get_all_completed_subtasks()) if shared_context else 0
                ),
                "total_time": total_time,
                "swarm_metadata": shared_context.get_summary() if shared_context else {},
                "timeout_occurred": timeout_occurred,
                "requested_swarm_timeout_s": state.get("swarm_timeout_s"),
                "effective_swarm_timeout_s": state.get("effective_swarm_timeout_s"),
                "suggestions": self._extract_suggestions(final_answer),
            }

            if timeout_occurred and not completed_agents:
                result["disclaimer"] = (
                    "由于系统超时，未能提供完整分析。建议简化问题重试，或在紧急情况下立即就医。"
                )
            elif timeout_occurred:
                result["disclaimer"] = (
                    f"以上分析基于 {len(completed_agents)} 个 Agent 的部分协作结果"
                    "（部分分析模块超时未完成），仅供参考，不能替代医生诊断。"
                )
            else:
                result["disclaimer"] = (
                    "以上分析基于多个专业 Agent 的协作，仅供参考，不能替代医生诊断。"
                )
        else:
            result.setdefault("answer", final_answer)
            result.setdefault("swarm_enabled", False)
            result.setdefault("session_id", state["session_id"])
            result["suggestions"] = self._extract_suggestions(final_answer)
            result.setdefault(
                "disclaimer",
                "⚠️ 以上信息仅供参考，不能替代专业医生的诊断和治疗。如有疑虑，请及时就医。",
            )

        result = await self._ensure_runtime_safety(state, result)
        return {"result": result}

    async def _call_llm_with_retry(
        self,
        messages: List[Dict[str, Any]],
        state: MedicalSwarmState,
        max_retries: int = 2,
    ) -> str:
        """Call LLM with retry logic for transient failures."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self.llm_client.chat(
                    messages,
                    debug_collector=self._get_debug_collector(state),
                    trace_name="plan_and_decompose",
                )
            except Exception as exc:
                last_error = exc
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(f"LLM call failed (attempt {attempt+1}), retrying in {wait}s: {exc}")
                    await asyncio.sleep(wait)
        raise last_error  # type: ignore

    def _trace_node(
        self,
        stage: str,
        name: str,
        handler: Callable[[MedicalSwarmState], Any],
    ):
        async def wrapped(state: MedicalSwarmState) -> Dict[str, Any]:
            async def execute_node() -> Dict[str, Any]:
                collector = self._get_debug_collector(state)
                if not collector:
                    return await handler(state)

                timer = collector.time_event(
                    stage,
                    input=self._debug_state_snapshot(state),
                    name=name,
                )
                try:
                    output = await handler(state)
                    timer.finish(output=output)
                    return output
                except Exception as exc:
                    timer.finish(status="failed", error=str(exc), output={"error": str(exc)})
                    raise

            return await trace_async(
                name=f"graph.{name}",
                run_type="chain",
                func=execute_node,
                inputs=self._debug_state_snapshot(state),
                metadata={
                    "stage": stage,
                    "node": name,
                    "session_id": state.get("session_id"),
                    "route": state.get("route") or state.get("mode"),
                },
                tags=["medical-agent-swarm", "langgraph-node", stage],
            )

        return wrapped

    def _get_debug_collector(
        self,
        state: MedicalSwarmState,
    ) -> Optional[DebugTraceCollector]:
        collector = state.get("debug_collector")
        return collector if isinstance(collector, DebugTraceCollector) else None

    def _debug_state_snapshot(self, state: MedicalSwarmState) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for key in (
            "question",
            "context",
            "enhanced_context",
            "session_id",
            "assessment",
            "subtasks",
            "route",
            "mode",
            "final_answer",
            "result",
            "timeout_occurred",
        ):
            if key in state:
                snapshot[key] = state[key]

        shared_context = state.get("shared_context")
        if shared_context and hasattr(shared_context, "get_summary"):
            snapshot["shared_context"] = shared_context.get_summary()
        return snapshot

    def _shared_context_debug_snapshot(self, shared_context: SharedContext) -> Dict[str, Any]:
        """Return a full JSON-safe snapshot of the shared swarm blackboard."""
        return {
            "summary": shared_context.get_summary(),
            "subtasks": [
                {
                    "id": subtask.id,
                    "type": subtask.type,
                    "description": subtask.description,
                    "assigned_agent": subtask.assigned_agent,
                    "status": subtask.status.value,
                    "result": subtask.result,
                    "created_at": subtask.created_at.isoformat(),
                    "started_at": subtask.started_at.isoformat() if subtask.started_at else None,
                    "completed_at": subtask.completed_at.isoformat() if subtask.completed_at else None,
                    "dependencies": subtask.dependencies,
                    "metadata": subtask.metadata,
                }
                for subtask in shared_context.task_decomposition.values()
            ],
            "contributions": [
                {
                    "agent_id": contribution.agent_id,
                    "subtask_id": contribution.subtask_id,
                    "result": contribution.result,
                    "timestamp": contribution.timestamp.isoformat(),
                    "confidence": contribution.confidence,
                    "metadata": contribution.metadata,
                }
                for contribution in shared_context.get_contributions()
            ],
            "events": [event.to_dict() for event in shared_context.events],
            "data": shared_context.data,
            "memory_pool": shared_context.memory_pool,
        }

    def _select_route(self, state: MedicalSwarmState) -> str:
        return state.get("route") or "fallback"

    def _get_agent_by_id(self, agent_id: Optional[str]):
        mapping = {
            "consultation_agent": self.consultation_agent,
            "diagnostic_agent": self.diagnostic_agent,
            "research_agent": self.research_agent,
        }
        return mapping.get(agent_id or "")

    def _effective_swarm_timeout(self, state: MedicalSwarmState) -> float:
        """Return the worker-execution timeout for this request."""
        requested = state.get("swarm_timeout_s")
        if requested is None:
            return float(self.swarm_timeout)

        try:
            requested_timeout = float(requested)
        except (TypeError, ValueError):
            return float(self.swarm_timeout)

        if requested_timeout <= 0:
            return float(self.swarm_timeout)

        start_time = state.get("start_time")
        if isinstance(start_time, datetime):
            elapsed_s = (datetime.now() - start_time).total_seconds()
            remaining_s = requested_timeout - elapsed_s - self.swarm_timeout_grace_s
            return max(1.0, remaining_s)

        return requested_timeout

    def _assessment_from_evidence_memory(self, question: str) -> Optional[Dict[str, Any]]:
        """Create a single-agent plan when local evidence memory has a strong hit."""
        try:
            hit = EvidenceMemory().lookup(question, min_score=0.9)
        except Exception as exc:
            logger.warning(f"Evidence memory planning lookup failed: {exc}")
            return None

        if not hit:
            return None

        memory_id = str(hit.get("id") or "")
        # Data-driven routing: read preferred_agent from evidence entry
        agent_id = hit.get("preferred_agent", "research_agent")
        task_type = f"{agent_id}_task"
        summary = str(hit.get("answer") or "")[:900]
        description = (
            "Use the local evidence memory below to answer the user's medical question. "
            "Keep standard medical safety boundaries, avoid diagnosis certainty, and "
            "recommend urgent care when red flags are present.\n\n"
            f"User question: {question}\n\n"
            f"Evidence memory ({memory_id}, score={hit.get('match_score')}):\n{summary}"
        )

        return {
            "subtasks": [
                {
                    "type": task_type,
                    "description": description,
                    "assigned_agent": agent_id,
                }
            ],
            "reason": "High-confidence local evidence memory hit; using single-agent plan to avoid redundant live research.",
            "memory_id": memory_id,
            "memory_source": hit.get("source", "evidence_memory"),
            "memory_match_score": hit.get("match_score"),
        }

    def _tool_policy_for_request(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        assessment: Optional[Dict[str, Any]] = None,
        task: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
        """Build per-request tool policy. Only deny deep_research for urgent/emergency topics."""
        text = " ".join(
            str(part or "")
            for part in [
                question,
                (task or {}).get("type"),
                (task or {}).get("description"),
                (task or {}).get("assigned_agent"),
                (assessment or {}).get("reason"),
            ]
        ).lower()

        emergency_markers = [
            "急症", "急诊", "急救", "立即就医", "拨打120", "120",
            "胸痛", "卒中", "中风", "喘不上气", "嘴唇发紫", "过敏性休克",
        ]
        if any(marker in text for marker in emergency_markers):
            return {
                "deny_tools": ["deep_research"],
                "allow_deep_research": False,
                "reason": "急症场景优先使用基础技能，跳过深度研究避免响应延迟。",
            }

        # Allow Agent (especially ResearchAgent) to autonomously decide
        return {
            "deny_tools": [],
            "allow_deep_research": True,
            "reason": "Agent 自主决定是否需要深度研究。",
        }

    def _context_category(self, context: Dict[str, Any]) -> str:
        for key in ("category", "case_category", "evaluation_category"):
            if context.get(key):
                return str(context[key])

        evaluation = context.get("evaluation")
        if isinstance(evaluation, dict) and evaluation.get("category"):
            return str(evaluation["category"])

        return ""

    def _parse_planning_response(self, content: str) -> Dict[str, Any]:
        json_match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            if isinstance(result, dict):
                return result

        return {
            "subtasks": [
                {
                    "type": "consultation_agent_task",
                    "description": "回答用户问题",
                    "assigned_agent": "consultation_agent",
                }
            ],
            "reason": "无法解析 LLM 响应，默认使用 ConsultationAgent",
        }

    def _create_subtasks(
        self,
        decomposition_result: Dict[str, Any],
        shared_context: SharedContext,
        question: str = "",
        context: Optional[Dict[str, Any]] = None,
    ) -> List[SubTask]:
        subtasks_data = decomposition_result.get("subtasks", [])
        subtasks = []

        for data in subtasks_data:
            assigned_agent = data["assigned_agent"]
            inferred_type = data.get("type") or f"{assigned_agent}_task"

            subtask = SubTask(
                id=str(uuid.uuid4()),
                type=inferred_type,
                description=data["description"],
                assigned_agent=assigned_agent,
                metadata={
                    "tool_policy": self._tool_policy_for_request(
                        question=question,
                        context=context or {},
                        assessment=decomposition_result,
                        task=data,
                    ),
                    "conversation_history": list(
                        (context or {}).get("recent_history") or []
                    ),
                },
            )

            shared_context.add_subtask(subtask)
            subtasks.append(subtask)

            logger.info(
                f"Created SubTask: {subtask.type} "
                f"(assigned to: {subtask.assigned_agent})"
            )

        return subtasks

    async def _worker_execute_assigned_tasks(
        self,
        worker: Any,
        shared_context: SharedContext,
        debug_collector: Optional[DebugTraceCollector] = None,
    ):
        try:
            assigned_tasks = shared_context.get_subtasks_for_agent(worker.agent_id)

            if not assigned_tasks:
                logger.debug(f"{worker.agent_id}: No assigned tasks")
                return

            tasks = []
            for subtask in assigned_tasks:
                logger.info(f"{worker.agent_id}: Starting {subtask.type}")
                shared_context.start_subtask(subtask.id)
                tasks.append(
                    asyncio.create_task(
                        self._execute_single_subtask(
                            worker,
                            subtask,
                            shared_context,
                            debug_collector,
                        )
                    )
                )

            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as exc:
            logger.error(f"{worker.agent_id}: Error processing subtask: {exc}")

    async def _execute_single_subtask(
        self,
        worker: Any,
        subtask: SubTask,
        shared_context: SharedContext,
        debug_collector: Optional[DebugTraceCollector] = None,
    ):
        timer = None
        if debug_collector:
            timer = debug_collector.time_event(
                "agent_loop",
                agent_id=worker.agent_id,
                input={
                    "subtask_id": subtask.id,
                    "type": subtask.type,
                    "description": subtask.description,
                    "assigned_agent": subtask.assigned_agent,
                },
                name="subtask_execution",
            )
        try:
            result = await worker.process_subtask(subtask, debug_collector=debug_collector)
            shared_context.complete_subtask(subtask.id, worker.agent_id, result)
            logger.info(f"{worker.agent_id}: Completed {subtask.type}")
            if timer:
                timer.finish(output=result)
        except Exception as exc:
            subtask.fail(str(exc))
            logger.error(f"{worker.agent_id}: Error in {subtask.type}: {exc}")
            if timer:
                timer.finish(status="failed", error=str(exc), output={"error": str(exc)})

    async def _synthesize_results(
        self,
        question: str,
        shared_context: SharedContext,
        timeout_occurred: bool = False,
        debug_collector: Optional[DebugTraceCollector] = None,
    ) -> str:
        all_contributions = shared_context.get_contributions()

        if not all_contributions:
            if timeout_occurred:
                return """抱歉，由于系统响应超时，所有 Agent 均未能在规定时间内完成分析。

【建议】：
- 您的问题可能比较复杂，建议简化问题后重试
- 或者将问题拆分为多个小问题分别咨询

【紧急情况】：
如果您的症状严重或紧急，请立即就医或拨打急救电话，不要依赖在线咨询。"""
            return "抱歉，Swarm 未能提供有效分析结果。"

        contributions_text = []
        completed_agents = []
        for contrib in all_contributions:
            subtask = shared_context.get_subtask(contrib.subtask_id)
            contributions_text.append(
                f"**{contrib.agent_id}** ({subtask.type if subtask else '未知'}):\n"
                f"{contrib.result}"
            )
            completed_agents.append(contrib.agent_id)

        timeout_note = ""
        if timeout_occurred:
            incomplete_tasks = [
                subtask.type
                for subtask in shared_context.task_decomposition.values()
                if subtask.status.value in {"pending", "claimed", "in_progress"}
            ]
            if incomplete_tasks:
                timeout_note = f"""

**注意**：由于系统响应超时，以下分析模块未能完成：{', '.join(incomplete_tasks)}
以下是基于已完成的 {len(completed_agents)} 个 Agent 的部分分析结果。"""

        synthesis_prompt = f"""你是医疗 Swarm 的结果综合节点，负责汇总多个专业 Worker Agent 的分析结果。

**用户问题**：{question}

**Agent 贡献**：
{chr(10).join(contributions_text)}{timeout_note}

**任务**：
整合以上所有分析，生成一个全面、专业的最终答案。

**要求**：
1. 综合所有 Agent 的观点
2. 突出多角度分析的价值，但不要暴露不必要的内部实现细节
3. 保持医疗建议的严谨性
4. 包含【风险评估】【诊断分析】【医学证据】等模块（如果相关 Agent 提供了）
5. 给出【核心建议】
6. 添加【免责声明】
{"7. 如果有分析模块未完成，在答案中明确说明" if timeout_occurred else ""}

**输出格式**：
【风险评估】 (如果有)
...

【诊断分析】 (如果有)
...

【医学证据】 (如果有)
...

【核心建议】
1. ...
2. ...

【免责声明】
...
"""

        try:
            return await self.llm_client.chat(
                [{"role": "user", "content": synthesis_prompt}],
                debug_collector=debug_collector,
                trace_name="synthesize_results",
            )
        except Exception as exc:
            logger.error(f"Synthesis error: {exc}")
            return f"汇总结果时出错：{exc}"

    async def _ensure_runtime_safety(
        self,
        state: MedicalSwarmState,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run mandatory safety review on the final answer before returning."""
        from core.safety_guard import SafetyGuard

        collector = self._get_debug_collector(state)
        risk_level = self._extract_risk_level_from_state(state)

        timer = None
        if collector:
            timer = collector.time_event(
                "safety_check",
                input={
                    "answer": result.get("answer", "") or "",
                    "question": state.get("question", "") or "",
                    "risk_level": risk_level,
                },
                name="runtime_safety_guard",
            )

        safety_guard = SafetyGuard()
        safety_result = await safety_guard.review(
            response=result.get("answer", "") or "",
            original_question=state.get("question", "") or "",
            risk_level=risk_level,
        )
        if timer:
            timer.finish(
                output=safety_result,
                status="success" if safety_result.get("safety_passed") else "failed",
                error=None if safety_result.get("safety_checked") else "safety_check_failed",
            )

        if not safety_result.get("safety_passed"):
            logger.warning(
                f"MedicalSwarmGraph safety guard found issues: "
                f"{safety_result.get('safety_issues')}"
            )

        result.update(
            {
                "answer": safety_result["answer"],
                "safety_checked": safety_result["safety_checked"],
                "safety_passed": safety_result["safety_passed"],
                "safety_issues": safety_result["safety_issues"],
            }
        )
        return result

    def _extract_risk_level_from_state(self, state: MedicalSwarmState) -> str:
        candidates: List[Any] = [
            state.get("assessment"),
            state.get("result"),
        ]

        shared_context = state.get("shared_context")
        if shared_context:
            candidates.extend(contrib.result for contrib in shared_context.get_contributions())

        for candidate in candidates:
            risk_level = self._find_key_recursive(candidate, "risk_level")
            if risk_level:
                return str(risk_level)
        return ""

    def _find_key_recursive(self, value: Any, key: str) -> Optional[Any]:
        if isinstance(value, dict):
            if value.get(key):
                return value[key]
            for child in value.values():
                found = self._find_key_recursive(child, key)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = self._find_key_recursive(child, key)
                if found:
                    return found
        return None

    def _extract_suggestions(self, final_answer: str) -> List[str]:
        suggestions = []

        if "【核心建议】" in final_answer:
            start_idx = final_answer.find("【核心建议】")
            end_idx = final_answer.find("【", start_idx + 1)
            if end_idx == -1:
                end_idx = len(final_answer)

            suggestions_text = final_answer[start_idx:end_idx]
            matches = re.findall(r"\d+\.\s*([^\n]+)", suggestions_text)
            suggestions = matches[:5]

        return suggestions or ["请遵循医嘱，注意休息和营养"]

    def _get_planning_prompt(self) -> str:
        return """你是医疗 Swarm 的任务规划节点。你的职责是**分析问题并分配给合适的 Worker Agent**。

**核心原则**：
1. **尽量少分配任务**：能用 1 个 Agent 解决的，不要用 2 个；能用 2 个的，不要用 3 个
2. **优先使用 ConsultationAgent**：对于常见病症（感冒、发烧、咳嗽等）、健康科普，单独使用 ConsultationAgent 就足够
3. 你**只负责分配 Agent**，不决定具体使用哪些工具/技能（Worker Agent 会自己选择）
4. 子任务应该相对独立，可以并行执行

---

## 可用的 Worker Agents

### 1. ConsultationAgent（健康咨询专家）
**擅长**：
- 常见疾病科普和健康建议
- 症状初步评估和风险分级
- 生活方式指导（饮食、运动、睡眠）
- 日常健康管理

**适用场景**：
- 简单症状咨询（"我感冒了""头痛怎么办"）
- 健康科普（"什么是高血压""多喝水的好处"）
- 预防建议（"如何预防感冒"）
- 生活方式指导（"高血压患者饮食注意什么"）

---

### 2. DiagnosticAgent（诊断推理专家）
**擅长**：
- 症状模式分析和关联性评估
- 鉴别诊断推理
- 复杂症状的风险评估

**适用场景**：
- 复杂症状分析（"头痛+恶心+视力模糊"）
- 多系统问题（"胸闷气短冒冷汗，严重吗"）
- 症状持续加重（"头痛一周了越来越严重"）
- 需要鉴别诊断的情况

---

### 3. ResearchAgent（循证医学专家）
**擅长**：
- 临床指南和诊疗规范检索
- 最新医学研究和证据综合
- 权威治疗方案查询
- 文献支持和证据等级评估

**适用场景**：
- 需要权威指南（"高血压最新诊疗指南"）
- 询问标准治疗方案（"糖尿病如何治疗"）
- 需要最新医学进展
- 需要循证医学证据支持

---

## 任务分配策略

### 策略 1：简单问题 → 1 个 Agent（ConsultationAgent）
**问题特征**：
- 单一常见症状（感冒、发烧、头痛、咳嗽）
- 健康科普和预防建议
- 一般性健康咨询

**示例**：
- "我感冒了怎么办？" → ConsultationAgent
- "什么是高血压？" → ConsultationAgent
- "如何预防流感？" → ConsultationAgent
- "糖尿病患者饮食注意什么？" → ConsultationAgent

---

### 策略 2：复杂症状 → 2 个 Agents（DiagnosticAgent + ConsultationAgent）
**问题特征**：
- 多个症状组合（3个以上不同症状）
- 症状持续时间长或加重
- 明确询问严重程度或是否需要就医
- 有既往病史或用药史

**示例**：
- "头痛一周了越来越严重，需要就医吗？" → DiagnosticAgent (评估风险) + ConsultationAgent (处理建议)
- "胸闷气短冒冷汗，严重吗？" → DiagnosticAgent (症状分析) + ConsultationAgent (建议)

---

### 策略 3：需要权威指南 → 2-3 个 Agents
**问题特征**：
- 询问疾病治疗方案
- 需要标准诊疗规范
- 需要权威指南和生活建议的综合方案

**示例**：
- "高血压如何治疗？" → ResearchAgent (指南) + ConsultationAgent (生活建议)
- "糖尿病最新诊疗指南是什么？" → ResearchAgent

---

## 输出格式（JSON）

**重要**：输出中**不需要 `type` 字段**，只需要 `description`（任务描述）和 `assigned_agent`（分配的 Agent）

### 示例 1：简单问题（1 个 Agent）
```json
{
  "subtasks": [
    {
      "description": "回答用户关于感冒的咨询，提供处理建议和注意事项",
      "assigned_agent": "consultation_agent"
    }
  ]
}
```

### 示例 2：复杂症状（2 个 Agents）
```json
{
  "subtasks": [
    {
      "description": "评估头痛症状的风险等级、紧急程度，分析症状模式和可能原因",
      "assigned_agent": "diagnostic_agent"
    },
    {
      "description": "提供头痛的处理建议、缓解方法和注意事项",
      "assigned_agent": "consultation_agent"
    }
  ]
}
```

### 示例 3：需要指南（2 个 Agents）
```json
{
  "subtasks": [
    {
      "description": "检索高血压的最新临床诊疗指南和标准治疗方案",
      "assigned_agent": "research_agent"
    },
    {
      "description": "提供高血压患者的日常生活管理建议（饮食、运动、用药）",
      "assigned_agent": "consultation_agent"
    }
  ]
}
```

---

## 关键要点

1. **只写 `description` 和 `assigned_agent`**
2. **description 要具体**：明确说明这个 Agent 需要做什么
3. **Agent 会自主选择工具**：你不需要指定使用哪个工具/技能
4. **尽量少分配**：1 个 Agent 能搞定的，不要分配 2 个
5. **任务要独立**：各个 Agent 的任务应该可以并行执行
"""
