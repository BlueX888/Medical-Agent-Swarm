from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .models import KnowledgeChunk, VectorMatch


class VectorStore(Protocol):
    async def upsert(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> None: ...
    async def search(self, vector: Sequence[float], limit: int, filters: Optional[Dict[str, Any]] = None) -> List[VectorMatch]: ...
    async def delete_document(self, document_id: str, version: Optional[str] = None) -> None: ...
    async def activate_version(
        self, document_id: str, version: str, previous_version: Optional[str]
    ) -> None: ...
    async def replace_all(self, chunks: Sequence[KnowledgeChunk], vectors: Sequence[Sequence[float]]) -> None: ...
    async def start_rebuild(self, vector_size: int) -> str: ...
    async def upsert_rebuild(
        self,
        rebuild_id: str,
        chunks: Sequence[KnowledgeChunk],
        vectors: Sequence[Sequence[float]],
    ) -> None: ...
    async def finish_rebuild(self, rebuild_id: str) -> None: ...
    async def abort_rebuild(self, rebuild_id: str) -> None: ...
    async def health(self) -> Dict[str, Any]: ...


class InMemoryVectorStore:
    """Deterministic Adapter used by tests and local fakes."""

    def __init__(self):
        self._values: Dict[str, tuple[KnowledgeChunk, List[float]]] = {}
        self._rebuilds: Dict[str, Dict[str, tuple[KnowledgeChunk, List[float]]]] = {}

    async def upsert(self, chunks, vectors) -> None:
        for chunk, vector in zip(chunks, vectors):
            self._values[chunk.chunk_id] = (chunk, list(vector))

    async def search(self, vector, limit, filters=None) -> List[VectorMatch]:
        matches = []
        for chunk, candidate in self._values.values():
            if chunk.status != "ready":
                continue
            if filters and any(getattr(chunk, key, None) != value for key, value in filters.items()):
                continue
            matches.append(VectorMatch(chunk=chunk, score=_cosine(vector, candidate)))
        return sorted(matches, key=lambda match: match.score, reverse=True)[:limit]

    async def delete_document(self, document_id: str, version=None) -> None:
        self._values = {
            key: value
            for key, value in self._values.items()
            if not (
                value[0].document_id == document_id
                and (version is None or value[0].version == version)
            )
        }

    async def activate_version(self, document_id, version, previous_version=None) -> None:
        for chunk, _vector in self._values.values():
            if chunk.document_id != document_id:
                continue
            if previous_version and chunk.version == previous_version:
                chunk.status = "retired"
            elif chunk.version == version:
                chunk.status = "ready"

    async def replace_all(self, chunks, vectors) -> None:
        self._values = {}
        await self.upsert(chunks, vectors)

    async def start_rebuild(self, vector_size: int) -> str:
        rebuild_id = f"memory-{len(self._rebuilds) + 1}"
        self._rebuilds[rebuild_id] = {}
        return rebuild_id

    async def upsert_rebuild(self, rebuild_id, chunks, vectors) -> None:
        staged = self._rebuilds[rebuild_id]
        for chunk, vector in zip(chunks, vectors):
            staged[chunk.chunk_id] = (chunk, list(vector))

    async def finish_rebuild(self, rebuild_id: str) -> None:
        self._values = self._rebuilds.pop(rebuild_id)

    async def abort_rebuild(self, rebuild_id: str) -> None:
        self._rebuilds.pop(rebuild_id, None)

    async def health(self) -> Dict[str, Any]:
        return {"status": "ok", "backend": "memory", "vectors": len(self._values)}


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right))
    denominator = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return numerator / denominator if denominator else 0.0
