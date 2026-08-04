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
    enable_rag: Optional[bool] = None


class RunCreateResponse(BaseModel):
    run_id: str
    status: str
    run: Dict[str, Any]


class ConsultationCreateRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    context: Dict[str, Any] = Field(default_factory=dict)
    session_id: str = Field(..., min_length=1, max_length=200)


class ConsultationCreateResponse(BaseModel):
    consultation_id: str
    status: Literal["queued", "running", "success", "failed", "timeout"]


class ConsultationParticipant(BaseModel):
    id: str
    label: str
    state: Literal["waiting", "active", "done", "failed"]


class ConsultationAnalysisStep(BaseModel):
    id: Literal["risk", "focus", "evidence", "collaboration", "safety"]
    label: str
    summary: str
    state: Literal["pending", "active", "done", "skipped", "attention"]


class ConsultationProgress(BaseModel):
    current_phase: Literal[
        "understanding",
        "planning",
        "consulting",
        "safety_review",
        "finalizing",
    ]
    completed_phases: List[str] = Field(default_factory=list)
    participants: List[ConsultationParticipant] = Field(default_factory=list)
    analysis_steps: List[ConsultationAnalysisStep] = Field(default_factory=list)
    safety_checked: bool = False


class CitationSource(BaseModel):
    citation_id: str = ""
    title: str = ""
    source_org: str = ""
    version: str = ""
    published_at: str = ""
    section: str = ""
    external_url: str = ""


class ConsultationResult(BaseModel):
    answer: str
    risk_level: Optional[Literal["low", "medium", "high", "emergency"]] = None
    suggestions: List[str] = Field(default_factory=list)
    disclaimer: str = ""
    participants: List[str] = Field(default_factory=list)
    sources: List[CitationSource] = Field(default_factory=list)


class ConsultationFailure(BaseModel):
    code: Literal["analysis_failed", "analysis_timeout"]
    message: str
    retryable: bool = True


class ConsultationSnapshot(BaseModel):
    consultation_id: str
    status: Literal["queued", "running", "success", "failed", "timeout"]
    progress: ConsultationProgress
    result: Optional[ConsultationResult] = None
    failure: Optional[ConsultationFailure] = None


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
