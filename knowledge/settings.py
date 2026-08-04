from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KnowledgeSettings:
    enabled: bool = False
    qdrant_url: str = "http://127.0.0.1:6333"
    collection_alias: str = "medical_knowledge_current"
    embedding_model: str = "intfloat/multilingual-e5-base"
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    candidate_count: int = 12
    top_k: int = 5
    score_threshold: float = 0.55
    context_token_budget: int = 3000
    chunk_tokens: int = 420
    chunk_overlap: int = 60
    max_file_bytes: int = 25 * 1024 * 1024
    catalog_path: Path = Path(".data/knowledge/catalog.sqlite3")
    documents_path: Path = Path(".data/knowledge/documents")
    admin_token: str = ""
    medquad_source_path: Path = Path(".data/sources/medquad")
    medquad_revision: str = "577bd37b96c02d1833b2c9eed2de9f96964e96cb"
    medquad_batch_size: int = 128

    @classmethod
    def from_env(cls) -> "KnowledgeSettings":
        return cls(
            enabled=_env_bool("RAG_ENABLED", False),
            qdrant_url=os.getenv("QDRANT_URL", cls.qdrant_url),
            collection_alias=os.getenv("RAG_COLLECTION_ALIAS", cls.collection_alias),
            embedding_model=os.getenv("RAG_EMBEDDING_MODEL", cls.embedding_model),
            reranker_model=os.getenv("RAG_RERANKER_MODEL", cls.reranker_model),
            candidate_count=int(os.getenv("RAG_CANDIDATE_COUNT", str(cls.candidate_count))),
            top_k=int(os.getenv("RAG_TOP_K", str(cls.top_k))),
            score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD", str(cls.score_threshold))),
            context_token_budget=int(os.getenv("RAG_CONTEXT_TOKEN_BUDGET", str(cls.context_token_budget))),
            chunk_tokens=int(os.getenv("RAG_CHUNK_TOKENS", str(cls.chunk_tokens))),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", str(cls.chunk_overlap))),
            max_file_bytes=int(os.getenv("RAG_MAX_FILE_BYTES", str(cls.max_file_bytes))),
            catalog_path=Path(os.getenv("RAG_CATALOG_PATH", str(cls.catalog_path))),
            documents_path=Path(os.getenv("RAG_DOCUMENTS_PATH", str(cls.documents_path))),
            admin_token=os.getenv("RAG_ADMIN_TOKEN", ""),
            medquad_source_path=Path(
                os.getenv("MEDQUAD_SOURCE_PATH", str(cls.medquad_source_path))
            ),
            medquad_revision=os.getenv("MEDQUAD_REVISION", cls.medquad_revision),
            medquad_batch_size=int(
                os.getenv("MEDQUAD_BATCH_SIZE", str(cls.medquad_batch_size))
            ),
        )
