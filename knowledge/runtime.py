from __future__ import annotations

from dataclasses import dataclass

from .adapters import CrossEncoderReranker, QdrantVectorStore, SentenceTransformerEmbedder
from .base import KnowledgeBase
from .manager import KnowledgeManager
from .settings import KnowledgeSettings


@dataclass
class KnowledgeRuntime:
    settings: KnowledgeSettings
    knowledge_base: KnowledgeBase
    manager: KnowledgeManager


def create_knowledge_runtime(settings: KnowledgeSettings | None = None) -> KnowledgeRuntime:
    settings = settings or KnowledgeSettings.from_env()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)
    reranker = CrossEncoderReranker(settings.reranker_model)
    store = QdrantVectorStore(settings.qdrant_url, settings.collection_alias)
    return KnowledgeRuntime(
        settings=settings,
        knowledge_base=KnowledgeBase(
            settings=settings,
            vector_store=store,
            embedder=embedder,
            reranker=reranker,
        ),
        manager=KnowledgeManager(settings=settings, vector_store=store, embedder=embedder),
    )
