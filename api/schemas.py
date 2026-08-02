"""FastAPI request and response schemas for the debug console."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RunCreateRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = None
    enable_swarm: bool = True
    enable_memory: bool = True
    enable_short_term_memory: Optional[bool] = None
    enable_long_term_memory: Optional[bool] = None


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    run: Dict[str, Any]


class RunResumeRequest(BaseModel):
    checkpoint_id: Optional[str] = None


class EffectReconcileRequest(BaseModel):
    resolution: Literal["completed", "failed"]


class DebugRunResponse(BaseModel):
    run: Dict[str, Any]


class DebugEventsResponse(BaseModel):
    events: List[Dict[str, Any]]


class SkillInfo(BaseModel):
    name: str
    function_name: str
    script_name: str
    description: str
    active: bool
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentInfo(BaseModel):
    agent_id: str
    class_name: str
    capabilities: List[str] = Field(default_factory=list)
    config: Dict[str, Any] = Field(default_factory=dict)
    skills: List[Dict[str, Any]] = Field(default_factory=list)


class MemoryResponse(BaseModel):
    session_id: str
    backend: str
    ttl_seconds: int
    recent_history: List[Dict[str, Any]]
    historical_cases: List[Dict[str, Any]]
    long_term_enabled: bool


class MemoryClearResponse(BaseModel):
    session_id: str
    cleared: bool
