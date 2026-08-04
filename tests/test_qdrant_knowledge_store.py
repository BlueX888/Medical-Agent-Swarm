from __future__ import annotations

import pytest
from qdrant_client import QdrantClient

from knowledge import KnowledgeChunk, QdrantVectorStore


@pytest.mark.asyncio
async def test_qdrant_adapter_atomically_replaces_the_search_collection():
    store = QdrantVectorStore(
        "http://unused",
        "test_knowledge_alias",
        client=QdrantClient(":memory:"),
    )
    old = KnowledgeChunk(
        chunk_id="old:v1:0",
        document_id="old",
        version="v1",
        title="旧指南",
        section="正文",
        text="旧内容",
    )
    new = KnowledgeChunk(
        chunk_id="new:v1:0",
        document_id="new",
        version="v1",
        title="新指南",
        section="正文",
        text="新内容",
    )
    await store.upsert([old], [[1.0, 0.0]])

    await store.replace_all([new], [[0.0, 1.0]])

    assert await store.search([1.0, 0.0], 5, {"document_id": "old"}) == []
    matches = await store.search([0.0, 1.0], 5)
    assert [match.chunk.document_id for match in matches] == ["new"]


@pytest.mark.asyncio
async def test_qdrant_adapter_switches_document_versions_before_cleanup():
    store = QdrantVectorStore(
        "http://unused",
        "test_version_alias",
        client=QdrantClient(":memory:"),
    )
    old = KnowledgeChunk(
        chunk_id="guide:v1:0",
        document_id="guide",
        version="v1",
        title="旧版",
        section="正文",
        text="旧内容",
        status="ready",
    )
    new = KnowledgeChunk(
        chunk_id="guide:v2:0",
        document_id="guide",
        version="v2",
        title="新版",
        section="正文",
        text="新内容",
        status="staged",
    )
    await store.upsert([old], [[1.0, 0.0]])
    await store.upsert([new], [[0.0, 1.0]])

    await store.activate_version("guide", "v2", "v1")

    assert await store.search([1.0, 0.0], 5, {"version": "v1"}) == []
    matches = await store.search([0.0, 1.0], 5, {"version": "v2"})
    assert [match.chunk.version for match in matches] == ["v2"]
