"""Deterministic policy for deciding whether a routed answer needs local evidence."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from .routing_models import IntentType, KnowledgeNeed, RiskLevel


MANDATORY_EVIDENCE_INTENTS = {
    IntentType.DIAGNOSTIC_REASONING,
    IntentType.TREATMENT_GUIDANCE,
    IntentType.MEDICATION_GUIDANCE,
    IntentType.PROGNOSIS_GUIDANCE,
    IntentType.LIFESTYLE_GUIDANCE,
    IntentType.EVIDENCE_RESEARCH,
}

NO_EVIDENCE_INTENTS = {
    IntentType.NON_MEDICAL,
    IntentType.SYSTEM_OPERATION,
}

_SOCIAL_MESSAGES = {
    "hello",
    "hi",
    "how are you",
    "tell me a joke",
    "thanks",
    "thank you",
    "你好",
    "您好",
    "谢谢",
    "你是谁",
}

_SYSTEM_OPERATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(?:upload|import)\s+(?:a\s+)?(?:document|file)\b",
        r"\b(?:log\s*in|sign\s*in|log\s*out|sign\s*out)\b",
        r"\b(?:open|show|edit|update)\s+(?:my\s+)?(?:app|application|account)\s+(?:profile|settings)\b",
        r"\b(?:api|knowledge base|user interface)\b",
        r"(?:上传|导入).{0,6}(?:文档|文件|知识库)",
        r"(?:登录|注册|退出).{0,6}(?:账号|账户|系统|应用)",
        r"(?:系统|应用|界面|知识库).{0,6}(?:怎么用|如何使用|如何操作)",
    )
)


@dataclass(frozen=True)
class RagRouteDecision:
    status: str
    reason: str | None = None


def normalize_knowledge_need(
    *,
    intents: Sequence[IntentType],
    risk_level: RiskLevel,
    declared_need: KnowledgeNeed | None,
    needs_clarification: bool,
    question: str,
) -> KnowledgeNeed:
    """Normalize model output with deterministic safety and evidence invariants."""
    if risk_level in {RiskLevel.HIGH, RiskLevel.EMERGENCY}:
        return KnowledgeNeed.NONE
    if set(intents) & MANDATORY_EVIDENCE_INTENTS:
        return KnowledgeNeed.REQUIRED
    intent_set = set(intents)
    if intent_set and intent_set <= NO_EVIDENCE_INTENTS:
        return KnowledgeNeed.NONE
    if declared_need is not None:
        return declared_need
    if needs_clarification and set(intents) == {IntentType.GENERAL_CONSULTATION}:
        return KnowledgeNeed.NONE
    if IntentType.SYMPTOM_TRIAGE in intents:
        return KnowledgeNeed.REQUIRED
    return infer_fallback_knowledge_need(question)


def infer_fallback_knowledge_need(question: str) -> KnowledgeNeed:
    """Classify only unmistakable social or application-operation fallbacks as no-RAG."""
    normalized = " ".join((question or "").casefold().split())
    social_normalized = " ".join(re.sub(r"[^\w]+", " ", normalized).split())
    if social_normalized in _SOCIAL_MESSAGES:
        return KnowledgeNeed.NONE
    if any(pattern.search(normalized) for pattern in _SYSTEM_OPERATION_PATTERNS):
        return KnowledgeNeed.NONE
    return KnowledgeNeed.REQUIRED


def decide_rag_route(
    *,
    enabled: bool,
    intents: Sequence[IntentType],
    risk_level: RiskLevel,
    declared_need: KnowledgeNeed | None,
    needs_clarification: bool,
    question: str,
) -> RagRouteDecision:
    if not enabled:
        return RagRouteDecision(status="disabled")
    if risk_level in {RiskLevel.HIGH, RiskLevel.EMERGENCY}:
        return RagRouteDecision(status="skipped", reason="urgent_request")
    need = normalize_knowledge_need(
        intents=intents,
        risk_level=risk_level,
        declared_need=declared_need,
        needs_clarification=needs_clarification,
        question=question,
    )
    if need == KnowledgeNeed.NONE:
        return RagRouteDecision(
            status="skipped",
            reason="no_medical_evidence_needed",
        )
    return RagRouteDecision(status="retrieve")
