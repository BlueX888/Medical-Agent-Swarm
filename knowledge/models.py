from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KnowledgeChunk:
    chunk_id: str
    document_id: str
    version: str
    title: str
    section: str
    text: str
    source_org: str = ""
    published_at: str = ""
    external_url: str = ""
    language: str = ""
    status: str = "ready"
    citation_id: str = ""
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "KnowledgeChunk":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: payload.get(key) for key in allowed if key in payload})

    def public_source(self) -> Dict[str, str]:
        return {
            "citation_id": self.citation_id,
            "title": self.title,
            "source_org": self.source_org,
            "version": self.version,
            "published_at": self.published_at,
            "section": self.section,
            "external_url": self.external_url,
        }


@dataclass
class RetrievalBundle:
    status: str
    query: str
    chunks: List[KnowledgeChunk] = field(default_factory=list)
    context: str = ""
    sources: List[Dict[str, str]] = field(default_factory=list)
    candidate_count: int = 0
    embedding_model: str = ""
    reranker_model: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "query": self.query,
            "chunks": [chunk.to_payload() for chunk in self.chunks],
            "context": self.context,
            "sources": list(self.sources),
            "candidate_count": self.candidate_count,
            "embedding_model": self.embedding_model,
            "reranker_model": self.reranker_model,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "RetrievalBundle":
        return cls(
            status=str(value.get("status") or "skipped"),
            query=str(value.get("query") or ""),
            chunks=[KnowledgeChunk.from_payload(item) for item in value.get("chunks") or []],
            context=str(value.get("context") or ""),
            sources=list(value.get("sources") or []),
            candidate_count=int(value.get("candidate_count") or 0),
            embedding_model=str(value.get("embedding_model") or ""),
            reranker_model=str(value.get("reranker_model") or ""),
            error=value.get("error"),
        )


@dataclass
class VectorMatch:
    chunk: KnowledgeChunk
    score: float
