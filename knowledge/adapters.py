from __future__ import annotations

import asyncio
import importlib.util
import math
import threading
import uuid
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from .models import KnowledgeChunk, VectorMatch


REBUILD_UPSERT_BATCH_SIZE = 256


class SentenceTransformerEmbedder:
    """Lazy local multilingual embedding Adapter."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise RuntimeError("sentence_transformers_not_installed") from exc
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_query(self, text: str) -> List[float]:
        model = self._load()
        return model.encode(f"query: {text}", normalize_embeddings=True).tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        model = self._load()
        values = model.encode(
            [f"passage: {text}" for text in texts],
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=False,
        )
        return values.tolist()

    def count_tokens(self, text: str) -> int:
        return len(
            self._load().tokenizer.encode(
                text,
                add_special_tokens=False,
                verbose=False,
            )
        )

    def split_tokens(self, text: str, *, size: int, overlap: int) -> List[str]:
        tokenizer = self._load().tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        step = max(1, size - overlap)
        return [
            tokenizer.decode(tokens[start : start + size], skip_special_tokens=True)
            for start in range(0, len(tokens), step)
        ]

    def split_prefixed_tokens(
        self,
        prefix: str,
        text: str,
        *,
        size: int,
        overlap: int,
    ) -> List[str]:
        """Split text while keeping every rendered chunk inside the total budget."""
        tokenizer = self._load().tokenizer
        header = f"{prefix}\nAnswer:"
        header_tokens = tokenizer.encode(
            header,
            add_special_tokens=False,
            verbose=False,
        )
        # Reserve some room for the answer even if source metadata is unusually long.
        header_tokens = header_tokens[: max(1, size - min(64, size // 2))]
        answer_tokens = tokenizer.encode(
            text,
            add_special_tokens=False,
            verbose=False,
        )
        answer_size = max(1, size - len(header_tokens))
        answer_overlap = min(overlap, max(0, answer_size - 1))
        step = max(1, answer_size - answer_overlap)
        chunks: List[str] = []
        for start in range(0, len(answer_tokens), step):
            piece_tokens = answer_tokens[start : start + answer_size]
            combined = header_tokens + piece_tokens
            rendered = tokenizer.decode(combined, skip_special_tokens=True).strip()
            # Tokenizer decode/encode is not guaranteed to be perfectly idempotent.
            while piece_tokens and self.count_tokens(rendered) > size:
                piece_tokens = piece_tokens[:-1]
                rendered = tokenizer.decode(
                    header_tokens + piece_tokens,
                    skip_special_tokens=True,
                ).strip()
            if rendered:
                chunks.append(rendered)
        return chunks

    def health(self) -> Dict[str, str]:
        if importlib.util.find_spec("sentence_transformers") is None:
            return {"status": "unavailable", "model": self.model_name}
        return {
            "status": "ready" if self._model is not None else "lazy",
            "model": self.model_name,
        }


class CrossEncoderReranker:
    """Lazy cross-encoder Adapter for reranking a small candidate set."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError as exc:
                        raise RuntimeError("sentence_transformers_not_installed") from exc
                    self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, chunks: Sequence[KnowledgeChunk]):
        if not chunks:
            return []
        scores = self._load().predict([(query, chunk.text) for chunk in chunks])
        return sorted(
            [
                (chunk, 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, float(score))))))
                for chunk, score in zip(chunks, scores)
            ],
            key=lambda item: item[1],
            reverse=True,
        )

    def health(self) -> Dict[str, str]:
        if importlib.util.find_spec("sentence_transformers") is None:
            return {"status": "unavailable", "model": self.model_name}
        return {
            "status": "ready" if self._model is not None else "lazy",
            "model": self.model_name,
        }


class QdrantVectorStore:
    """Lazy Qdrant Adapter; importing the project does not require qdrant-client."""

    def __init__(self, url: str, collection_alias: str, client: Any = None):
        self.url = url
        self.collection_alias = collection_alias
        self._client = client
        self._collection_name: Optional[str] = None
        self._lock = asyncio.Lock()

    def _get_client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("qdrant_client_not_installed") from exc
            self._client = QdrantClient(url=self.url, timeout=20)
        return self._client

    async def _ensure_collection(self, vector_size: int) -> str:
        async with self._lock:
            if self._collection_name:
                return self._collection_name
            self._collection_name = await asyncio.to_thread(self._ensure_collection_sync, vector_size)
            return self._collection_name

    def _ensure_collection_sync(self, vector_size: int) -> str:
        from qdrant_client import models

        client = self._get_client()
        aliases = client.get_aliases().aliases
        for alias in aliases:
            if alias.alias_name == self.collection_alias:
                return alias.collection_name
        collection_name = f"{self.collection_alias}_{uuid.uuid4().hex[:8]}"
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        client.update_collection_aliases(
            change_aliases_operations=[
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=collection_name,
                        alias_name=self.collection_alias,
                    )
                )
            ]
        )
        return collection_name

    async def upsert(self, chunks, vectors) -> None:
        if not chunks:
            return
        collection = await self._ensure_collection(len(vectors[0]))
        await asyncio.to_thread(self._upsert_sync, collection, chunks, vectors)

    def _upsert_sync(self, collection, chunks, vectors) -> None:
        from qdrant_client import models

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                vector=list(vector),
                payload=chunk.to_payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self._get_client().upsert(collection_name=collection, points=points, wait=True)

    async def search(self, vector, limit, filters=None) -> List[VectorMatch]:
        collection = await self._resolve_alias()
        if collection is None:
            return []
        return await asyncio.to_thread(self._search_sync, collection, vector, limit, filters)

    def _search_sync(self, collection, vector, limit, filters):
        query_filter = self._filter({"status": "ready", **(filters or {})})
        response = self._get_client().query_points(
            collection_name=collection,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        return [
            VectorMatch(KnowledgeChunk.from_payload(point.payload or {}), float(point.score))
            for point in response.points
        ]

    async def delete_document(self, document_id: str, version=None) -> None:
        collection = await self._resolve_alias()
        if collection is None:
            return
        values = {"document_id": document_id}
        if version is not None:
            values["version"] = version
        await asyncio.to_thread(self._delete_sync, collection, values)

    async def activate_version(self, document_id, version, previous_version=None) -> None:
        collection = await self._resolve_alias()
        if collection is None:
            raise RuntimeError("knowledge_collection_missing")
        await asyncio.to_thread(
            self._activate_version_sync,
            collection,
            document_id,
            version,
            previous_version,
        )

    def _activate_version_sync(self, collection, document_id, version, previous_version) -> None:
        try:
            self._set_status_sync(collection, document_id, version, "ready")
            if previous_version:
                self._set_status_sync(collection, document_id, previous_version, "retired")
        except Exception:
            self._set_status_sync(collection, document_id, version, "staged")
            if previous_version:
                self._set_status_sync(collection, document_id, previous_version, "ready")
            raise

    async def start_rebuild(self, vector_size: int) -> str:
        return await asyncio.to_thread(self._start_rebuild_sync, vector_size)

    def _start_rebuild_sync(self, vector_size: int) -> str:
        from qdrant_client import models

        collection = f"{self.collection_alias}_{uuid.uuid4().hex[:8]}"
        self._get_client().create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        return collection

    async def upsert_rebuild(self, rebuild_id: str, chunks, vectors) -> None:
        await asyncio.to_thread(self._upsert_sync, rebuild_id, chunks, vectors)

    async def finish_rebuild(self, rebuild_id: str) -> None:
        await asyncio.to_thread(self._finish_rebuild_sync, rebuild_id)

    def _finish_rebuild_sync(self, rebuild_id: str) -> None:
        from qdrant_client import models

        client = self._get_client()
        old_collection = None
        for alias in client.get_aliases().aliases:
            if alias.alias_name == self.collection_alias:
                old_collection = alias.collection_name
                break
        operations = []
        if old_collection:
            operations.append(
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.collection_alias)
                )
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=rebuild_id,
                    alias_name=self.collection_alias,
                )
            )
        )
        client.update_collection_aliases(change_aliases_operations=operations)
        self._collection_name = rebuild_id
        if old_collection and old_collection != rebuild_id:
            try:
                client.delete_collection(old_collection)
            except Exception as cleanup_error:
                logger.warning(
                    "Old Qdrant collection cleanup failed: {}",
                    type(cleanup_error).__name__,
                )

    async def abort_rebuild(self, rebuild_id: str) -> None:
        try:
            await asyncio.to_thread(self._get_client().delete_collection, rebuild_id)
        except Exception:
            pass

    def _set_status_sync(self, collection, document_id, version, status) -> None:
        self._get_client().set_payload(
            collection_name=collection,
            payload={"status": status},
            points=self._filter({"document_id": document_id, "version": version}),
            wait=True,
        )

    async def replace_all(self, chunks, vectors) -> None:
        if not chunks:
            await asyncio.to_thread(self._replace_all_sync, chunks, vectors)
            return
        rebuild_id = await self.start_rebuild(len(vectors[0]))
        try:
            for start in range(0, len(chunks), REBUILD_UPSERT_BATCH_SIZE):
                await self.upsert_rebuild(
                    rebuild_id,
                    chunks[start : start + REBUILD_UPSERT_BATCH_SIZE],
                    vectors[start : start + REBUILD_UPSERT_BATCH_SIZE],
                )
            await self.finish_rebuild(rebuild_id)
        except Exception:
            await self.abort_rebuild(rebuild_id)
            raise

    def _replace_all_sync(self, chunks, vectors) -> None:
        from qdrant_client import models

        client = self._get_client()
        old_collection = None
        for alias in client.get_aliases().aliases:
            if alias.alias_name == self.collection_alias:
                old_collection = alias.collection_name
                break
        if not chunks:
            if old_collection:
                client.update_collection_aliases(
                    change_aliases_operations=[
                        models.DeleteAliasOperation(
                            delete_alias=models.DeleteAlias(alias_name=self.collection_alias)
                        )
                    ]
                )
                client.delete_collection(old_collection)
            self._collection_name = None
            return
        # Non-empty replacement is implemented by the streaming rebuild methods.

    def _delete_sync(self, collection: str, values: Dict[str, Any]):
        from qdrant_client import models

        self._get_client().delete(
            collection_name=collection,
            points_selector=models.FilterSelector(filter=self._filter(values)),
            wait=True,
        )

    async def health(self) -> Dict[str, Any]:
        try:
            collection = await self._resolve_alias()
            return {"status": "ok", "backend": "qdrant", "collection": collection}
        except Exception:
            return {"status": "degraded", "backend": "qdrant", "error": "qdrant_unavailable"}

    async def _resolve_alias(self) -> Optional[str]:
        if self._collection_name:
            return self._collection_name
        aliases = await asyncio.to_thread(lambda: self._get_client().get_aliases().aliases)
        for alias in aliases:
            if alias.alias_name == self.collection_alias:
                self._collection_name = alias.collection_name
                return self._collection_name
        return None

    @staticmethod
    def _filter(values: Dict[str, Any]):
        from qdrant_client import models

        return models.Filter(
            must=[
                models.FieldCondition(key=key, match=models.MatchValue(value=value))
                for key, value in values.items()
            ]
        )
