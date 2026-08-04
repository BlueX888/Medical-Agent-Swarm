from __future__ import annotations

import pytest

from knowledge import (
    CitationValidator,
    InMemoryVectorStore,
    KnowledgeBase,
    KnowledgeChunk,
    KnowledgeManager,
    KnowledgeSettings,
)


class FakeEmbedder:
    model_name = "fake-multilingual"

    def embed_documents(self, texts):
        return [[float("肾" in text), float("心" in text)] for text in texts]

    def embed_query(self, text):
        return [float("肾" in text), float("心" in text)]

    def count_tokens(self, text):
        return len(text)


class FakeReranker:
    model_name = "fake-reranker"

    def rerank(self, query, chunks):
        return sorted(
            [(chunk, 0.95 if "权威结论" in chunk.text else 0.70) for chunk in chunks],
            key=lambda item: item[1],
            reverse=True,
        )


@pytest.mark.asyncio
async def test_retrieve_returns_ranked_ready_evidence_with_stable_citations():
    store = InMemoryVectorStore()
    await store.upsert(
        [
            KnowledgeChunk(
                chunk_id="doc-1:v1:0",
                document_id="doc-1",
                version="v1",
                title="慢性肾病指南",
                section="治疗目标",
                text="权威结论：慢性肾病管理需要个体化。",
                source_org="示例医学会",
                published_at="2026-01-01",
                external_url="https://example.test/ckd",
                status="ready",
            ),
            KnowledgeChunk(
                chunk_id="doc-2:v1:0",
                document_id="doc-2",
                version="v1",
                title="待发布文档",
                section="草稿",
                text="权威结论：这段草稿不能参与检索。",
                status="indexing",
            ),
        ],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    base = KnowledgeBase(
        settings=KnowledgeSettings(
            enabled=True,
            candidate_count=12,
            top_k=5,
            score_threshold=0.55,
            context_token_budget=3000,
        ),
        vector_store=store,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )

    bundle = await base.retrieve("肾病治疗证据")

    assert bundle.status == "used"
    assert bundle.candidate_count == 1
    assert [chunk.citation_id for chunk in bundle.chunks] == ["K1"]
    assert bundle.chunks[0].title == "慢性肾病指南"
    assert bundle.sources == [
        {
            "citation_id": "K1",
            "title": "慢性肾病指南",
            "source_org": "示例医学会",
            "version": "v1",
            "published_at": "2026-01-01",
            "section": "治疗目标",
            "external_url": "https://example.test/ckd",
        }
    ]
    assert "[K1]" in bundle.context


@pytest.mark.asyncio
async def test_retrieve_degrades_without_leaking_backend_error():
    class BrokenStore(InMemoryVectorStore):
        async def search(self, vector, limit, filters=None):
            raise RuntimeError("qdrant-secret-host")

    base = KnowledgeBase(
        settings=KnowledgeSettings(enabled=True),
        vector_store=BrokenStore(),
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )

    bundle = await base.retrieve("肾病治疗证据")

    assert bundle.status == "degraded"
    assert bundle.chunks == []
    assert bundle.sources == []
    assert bundle.error == "retrieval_unavailable"


@pytest.mark.asyncio
async def test_retrieve_reranks_all_vector_candidates_before_applying_threshold():
    class LowVectorScoreStore(InMemoryVectorStore):
        async def search(self, vector, limit, filters=None):
            matches = await super().search(vector, limit, filters)
            for match in matches:
                match.score = 0.10
            return matches

    store = LowVectorScoreStore()
    chunk = KnowledgeChunk(
        chunk_id="doc-1:v1:0",
        document_id="doc-1",
        version="v1",
        title="指南",
        section="正文",
        text="权威结论：应进行个体化管理。",
    )
    await store.upsert([chunk], [[1.0, 0.0]])
    base = KnowledgeBase(
        settings=KnowledgeSettings(enabled=True, score_threshold=0.55),
        vector_store=store,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )

    bundle = await base.retrieve("肾病证据")

    assert bundle.status == "used"
    assert bundle.candidate_count == 1


@pytest.mark.asyncio
async def test_retrieve_never_exceeds_context_budget_for_an_oversized_first_chunk():
    store = InMemoryVectorStore()
    chunk = KnowledgeChunk(
        chunk_id="doc-1:v1:0",
        document_id="doc-1",
        version="v1",
        title="指南",
        section="正文",
        text="权威结论" * 20,
    )
    await store.upsert([chunk], [[1.0, 0.0]])
    base = KnowledgeBase(
        settings=KnowledgeSettings(enabled=True, context_token_budget=10),
        vector_store=store,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
    )

    bundle = await base.retrieve("肾病证据")

    assert bundle.status == "empty"
    assert bundle.context == ""


def test_citation_validator_removes_unknown_references_and_returns_used_sources():
    chunks = [
        KnowledgeChunk(
            chunk_id="doc-1:v1:0",
            document_id="doc-1",
            version="v1",
            title="慢性肾病指南",
            section="治疗目标",
            text="证据正文",
            status="ready",
            citation_id="K1",
        )
    ]

    answer, sources = CitationValidator.validate(
        "应结合患者情况调整目标 [K1]，不存在的来源 [K99]。",
        chunks,
    )

    assert answer == "应结合患者情况调整目标 [K1]，不存在的来源。"
    assert [source["citation_id"] for source in sources] == ["K1"]


@pytest.mark.asyncio
async def test_document_lifecycle_indexes_deduplicates_and_deletes(tmp_path):
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
        chunk_tokens=80,
        chunk_overlap=10,
    )
    store = InMemoryVectorStore()
    manager = KnowledgeManager(settings=settings, vector_store=store, embedder=FakeEmbedder())

    submitted = await manager.submit_document(
        filename="ckd.md",
        content="# 慢性肾病\n\n权威结论：慢性肾病患者需要按肾功能分期管理。".encode(),
        metadata={"title": "慢性肾病指南", "source_org": "示例医学会", "version": "2026"},
    )
    assert submitted["status"] == "queued"

    completed = await manager.process_job(submitted["job_id"])
    assert completed["status"] == "ready"
    documents = await manager.list_documents()
    assert documents[0]["title"] == "慢性肾病指南"
    assert documents[0]["status"] == "ready"

    duplicate = await manager.submit_document(
        filename="renamed.md",
        content="# 慢性肾病\n\n权威结论：慢性肾病患者需要按肾功能分期管理。".encode(),
        metadata={"title": "重复文档"},
    )
    assert duplicate == {
        "document_id": submitted["document_id"],
        "job_id": submitted["job_id"],
        "status": "ready",
        "duplicate": True,
    }

    reindex = await manager.reindex_all()
    rebuilt = await manager.process_job(reindex["job_id"])
    assert rebuilt["status"] == "ready"
    assert rebuilt["chunk_count"] == 1

    deletion = await manager.submit_delete(submitted["document_id"])
    deleted = await manager.process_job(deletion["job_id"])
    assert deleted["status"] == "deleted"
    assert (await manager.get_job(deletion["job_id"]))["status"] == "ready"
    assert await manager.list_documents() == []
    assert (await store.health())["vectors"] == 0


@pytest.mark.asyncio
async def test_failed_replacement_keeps_the_previous_ready_version(tmp_path):
    class FailingReplacementStore(InMemoryVectorStore):
        fail = False

        async def upsert(self, chunks, vectors):
            if self.fail:
                raise RuntimeError("replacement failed")
            await super().upsert(chunks, vectors)

    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
    )
    store = FailingReplacementStore()
    manager = KnowledgeManager(settings=settings, vector_store=store, embedder=FakeEmbedder())
    first = await manager.submit_document(
        filename="guide.md",
        content=b"# Guide\n\nrenal guidance version one",
        metadata={"title": "Renal guide", "version": "2026"},
    )
    await manager.process_job(first["job_id"])

    replacement = await manager.submit_document(
        document_id=first["document_id"],
        filename="guide.md",
        content=b"# Guide\n\nrenal guidance version two",
        metadata={"title": "Renal guide", "version": "2027"},
    )
    store.fail = True
    failed = await manager.process_job(replacement["job_id"])

    assert failed["status"] == "failed"
    document = (await manager.list_documents())[0]
    assert document["version"] == "2026"
    assert document["status"] == "ready"
    assert {item["version"]: item["status"] for item in document["versions"]} == {
        "2026": "ready",
        "2027": "failed",
    }
    assert (await store.health())["vectors"] == 1


@pytest.mark.asyncio
async def test_replacement_rejects_reusing_any_historical_version(tmp_path):
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
    )
    manager = KnowledgeManager(
        settings=settings,
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
    )
    first = await manager.submit_document(
        filename="guide.md",
        content=b"# Guide\n\nversion one",
        metadata={"version": "v1"},
    )
    await manager.process_job(first["job_id"])
    second = await manager.submit_document(
        document_id=first["document_id"],
        filename="guide.md",
        content=b"# Guide\n\nversion two",
        metadata={"version": "v2"},
    )
    await manager.process_job(second["job_id"])
    document = (await manager.list_documents())[0]
    assert {item["version"]: item["status"] for item in document["versions"]} == {
        "v1": "retired",
        "v2": "ready",
    }
    assert await manager.vector_store.search([1.0, 0.0], 5, {"version": "v1"}) == []

    with pytest.raises(ValueError, match="version_already_exists"):
        await manager.submit_document(
            document_id=first["document_id"],
            filename="guide.md",
            content=b"# Guide\n\nreplacement content",
            metadata={"version": "v1"},
        )

    duplicate = await manager.submit_document(
        filename="old-copy.md",
        content=b"# Guide\n\nversion one",
        metadata={"version": "copy"},
    )
    assert duplicate["duplicate"] is True
    assert duplicate["document_id"] == first["document_id"]


@pytest.mark.asyncio
async def test_recovery_includes_interrupted_delete_and_reindex_jobs(tmp_path):
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
    )
    manager = KnowledgeManager(
        settings=settings,
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
    )
    await manager.initialize()
    await manager.catalog.create_job("delete-job", "doc-1", "delete", {})
    await manager.catalog.update_job("delete-job", status="deleting")
    await manager.catalog.create_job("reindex-job", None, "reindex", {})
    await manager.catalog.update_job("reindex-job", status="indexing")

    jobs = await manager.recover_jobs()

    assert {job["id"] for job in jobs} == {"delete-job", "reindex-job"}


@pytest.mark.asyncio
async def test_reindex_uses_only_current_ready_documents(tmp_path):
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
    )
    store = InMemoryVectorStore()
    manager = KnowledgeManager(settings=settings, vector_store=store, embedder=FakeEmbedder())
    ready = await manager.submit_document(
        filename="ready.md",
        content=b"# Ready\n\nrenal ready guidance",
        metadata={"version": "v1"},
    )
    await manager.process_job(ready["job_id"])
    failed = await manager.submit_document(
        filename="failed.md",
        content=b"# Failed\n\nrenal failed guidance",
        metadata={"version": "v1"},
    )
    await manager.catalog.update_document(failed["document_id"], status="failed")

    reindex = await manager.reindex_all()
    result = await manager.process_job(reindex["job_id"])

    assert result["chunk_count"] == 1
