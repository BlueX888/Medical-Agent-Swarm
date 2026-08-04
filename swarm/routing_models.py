"""Strongly typed models for orchestrator routing decisions."""
from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IntentType(str, Enum):
    NON_MEDICAL = "non_medical"
    SYSTEM_OPERATION = "system_operation"
    GENERAL_CONSULTATION = "general_consultation"
    SYMPTOM_TRIAGE = "symptom_triage"
    DIAGNOSTIC_REASONING = "diagnostic_reasoning"
    TREATMENT_GUIDANCE = "treatment_guidance"
    MEDICATION_GUIDANCE = "medication_guidance"
    PROGNOSIS_GUIDANCE = "prognosis_guidance"
    LIFESTYLE_GUIDANCE = "lifestyle_guidance"
    EVIDENCE_RESEARCH = "evidence_research"


class RiskLevel(str, Enum):
    UNKNOWN = "unknown"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EMERGENCY = "emergency"


class KnowledgeNeed(str, Enum):
    """Whether the planned answer will make evidence-backed medical claims."""

    REQUIRED = "required"
    NONE = "none"


class ExecutionMode(str, Enum):
    SINGLE = "single"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


class RouteSource(str, Enum):
    SAFETY_RULE = "safety_rule"
    EVIDENCE_MEMORY = "evidence_memory"
    LLM = "llm"
    FALLBACK = "fallback"


class PlannedTask(BaseModel):
    """One independently executable Worker assignment."""

    model_config = ConfigDict(extra="forbid")

    id: str
    goal: str
    required_capabilities: List[str]
    assigned_agent: str
    priority: str
    depends_on: List[str] = Field(default_factory=list)

    @field_validator("id", "goal", "assigned_agent", "priority")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("required_capabilities")
    @classmethod
    def _normalize_capabilities(cls, values: List[str]) -> List[str]:
        normalized = list(dict.fromkeys(value.strip() for value in values if value.strip()))
        if not normalized:
            raise ValueError("required_capabilities must not be empty")
        return normalized

    @field_validator("depends_on")
    @classmethod
    def _normalize_dependencies(cls, values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class RoutePlan(BaseModel):
    """Validated output of the Orchestrator."""

    model_config = ConfigDict(extra="forbid")

    intent_summary: str
    intents: List[IntentType]
    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1)
    tasks: List[PlannedTask]
    execution_mode: ExecutionMode
    source: RouteSource
    reasons: List[str]
    needs_clarification: bool = False
    knowledge_need: KnowledgeNeed | None = Field(
        default=None,
        description=(
            "Use required when the answer will contain medical facts, causes, diagnosis, "
            "treatment, medication, prognosis, lifestyle advice, monitoring, or care timing. "
            "Use none only for emergency routing, pure clarification, non-medical conversation, "
            "or application/system operations."
        ),
    )

    @field_validator("intent_summary")
    @classmethod
    def _summary_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("intent_summary must not be empty")
        return value

    @field_validator("intents")
    @classmethod
    def _intents_not_empty(cls, values: List[IntentType]) -> List[IntentType]:
        if not values:
            raise ValueError("intents must not be empty")
        return list(dict.fromkeys(values))

    @field_validator("tasks")
    @classmethod
    def _tasks_not_empty(cls, values: List[PlannedTask]) -> List[PlannedTask]:
        if not values:
            raise ValueError("tasks must not be empty")
        return values

    @model_validator(mode="after")
    def _validate_task_graph(self) -> "RoutePlan":
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task ids must be unique")

        known = set(ids)
        for task in self.tasks:
            unknown = set(task.depends_on) - known
            if unknown:
                raise ValueError(f"task {task.id} has unknown dependencies: {sorted(unknown)}")
            if task.id in task.depends_on:
                raise ValueError(f"task {task.id} cannot depend on itself")

        visiting = set()
        visited = set()
        graph = {task.id: task.depends_on for task in self.tasks}

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("cyclic task dependency")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in graph[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in ids:
            visit(task_id)
        return self
