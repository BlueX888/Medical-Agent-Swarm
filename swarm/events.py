"""
Event system - kept for backward compatibility, currently unused.

EventType and Event remain defined so that existing references
in shared_context.py and other modules continue to import cleanly.
New code should not use this module.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class EventType(Enum):
    """Kept for backward compatibility."""
    TASK_DECOMPOSED = "task_decomposed"
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_COMPLETED = "subtask_completed"
    CONTEXT_UPDATED = "context_updated"
    AGENT_QUESTION = "agent_question"
    AGENT_ANSWER = "agent_answer"
    SWARM_STARTED = "swarm_started"
    SWARM_COMPLETED = "swarm_completed"


@dataclass
class Event:
    """Kept for backward compatibility."""
    type: EventType
    source_agent: str
    data: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    target_agents: Optional[List[str]] = None

    def is_for_agent(self, agent_id: str) -> bool:
        if self.target_agents is None:
            return True
        return agent_id in self.target_agents

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "source_agent": self.source_agent,
            "timestamp": self.timestamp.isoformat(),
            "target_agents": self.target_agents,
            "data": self.data,
        }
