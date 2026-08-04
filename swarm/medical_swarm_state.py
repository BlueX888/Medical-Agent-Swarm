"""
LangGraph state for the medical swarm workflow.
"""
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.channels import UntrackedValue


class MedicalSwarmState(TypedDict, total=False):
    """State passed through the medical swarm graph."""

    schema_version: int
    graph_version: str
    run_id: str
    enable_swarm: bool
    enable_short_term_memory: bool
    enable_long_term_memory: bool
    enable_rag: bool
    question: str
    context: Dict[str, Any]
    enhanced_context: Dict[str, Any]
    session_id: str
    start_time: datetime
    end_time: datetime

    recent_history: List[Dict[str, Any]]
    historical_cases: List[Dict[str, Any]]
    knowledge_bundle: Dict[str, Any]
    rag_status: str
    grounded_sources: List[Dict[str, Any]]
    short_term_memory_error: Optional[str]

    assessment: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    route_plan: Dict[str, Any]
    route: str
    mode: str

    shared_context: Dict[str, Any]
    final_answer: str
    result: Dict[str, Any]

    timeout_occurred: bool
    swarm_timeout_s: Optional[float]
    error: Optional[str]
    # Observability objects are process-local dependencies, not durable state.
    debug_collector: Annotated[Any, UntrackedValue]
