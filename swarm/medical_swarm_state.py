"""
LangGraph state for the medical swarm workflow.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict


class MedicalSwarmState(TypedDict, total=False):
    """State passed through the medical swarm graph."""

    question: str
    context: Dict[str, Any]
    enhanced_context: Dict[str, Any]
    session_id: str
    start_time: datetime
    end_time: datetime

    recent_history: List[Dict[str, Any]]
    historical_cases: List[Dict[str, Any]]

    assessment: Dict[str, Any]
    subtasks: List[Dict[str, Any]]
    route: str
    mode: str

    shared_context: Any
    final_answer: str
    result: Dict[str, Any]

    timeout_occurred: bool
    swarm_timeout_s: Optional[float]
    error: Optional[str]
    debug_collector: Any
