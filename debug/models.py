"""Data models for one structured debug run."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


DebugRoute = str
DebugRunStatus = str
DebugStage = str
DebugEventStatus = str


@dataclass
class DebugEvent:
    """One timestamped step in an agent request trace."""

    timestamp: datetime
    stage: DebugStage
    sequence: int = 0
    agent_id: Optional[str] = None
    skill_name: Optional[str] = None
    input: Any = None
    output: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None
    status: DebugEventStatus = "success"
    error: Optional[str] = None
    event_id: Optional[str] = None
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "timestamp": self.timestamp.isoformat(),
            "stage": self.stage,
            "name": self.name,
            "agent_id": self.agent_id,
            "skill_name": self.skill_name,
            "input": self.input,
            "output": self.output,
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class DebugRun:
    """Structured record for one end-to-end request execution."""

    run_id: str
    session_id: Optional[str]
    question: str
    context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    route: Optional[DebugRoute] = None
    status: DebugRunStatus = "running"
    final_answer: str = ""
    result_json: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "question": self.question,
            "context": self.context,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "route": self.route,
            "status": self.status,
            "final_answer": self.final_answer,
            "result_json": self.result_json,
            "metadata": self.metadata,
        }


@dataclass
class DebugRunSnapshot:
    """Convenience payload used by APIs that return run plus event timeline."""

    run: DebugRun
    events: List[DebugEvent]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": self.run.to_dict(),
            "events": [event.to_dict() for event in self.events],
        }
