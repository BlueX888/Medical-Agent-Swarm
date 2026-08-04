from __future__ import annotations

import pytest

from knowledge import InMemoryVectorStore, KnowledgeManager, KnowledgeSettings
from knowledge.adapters import SentenceTransformerEmbedder


class FakeEmbedder:
    model_name = "fake-medquad-embedder"

    def __init__(self):
        self.batch_sizes = []

    def embed_documents(self, texts):
        self.batch_sizes.append(len(texts))
        return [[1.0, float("treatment" in text.lower())] for text in texts]

    def embed_query(self, text):
        return [1.0, float("treatment" in text.lower())]

    def count_tokens(self, text):
        return len(text.split())


def _write_medquad_document(path, *, answer: str, qid: str = "1-1") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<Document id="1" source="NIDDK" url="https://www.niddk.nih.gov/example">
  <Focus>Chronic kidney disease</Focus>
  <FocusAnnotations>
    <Category>Disease</Category>
    <UMLS><CUIs><CUI>C1561643</CUI></CUIs></UMLS>
    <Synonyms><Synonym>CKD</Synonym></Synonyms>
  </FocusAnnotations>
  <QAPairs>
    <QAPair pid="1">
      <Question qid="{qid}" qtype="treatment">How is CKD treated?</Question>
      <Answer>{answer}</Answer>
    </QAPair>
  </QAPairs>
</Document>
""",
        encoding="utf-8",
    )


def _mark_source(source_path, revision: str) -> None:
    (source_path / ".medquad-revision").write_text(revision, encoding="utf-8")


@pytest.mark.asyncio
async def test_medquad_import_indexes_answered_pairs_and_skips_removed_answers(tmp_path):
    source_path = tmp_path / "medquad"
    _write_medquad_document(
        source_path / "5_NIDDK_QA" / "answered.xml",
        answer="Treatment is individualized according to kidney function.",
    )
    _write_medquad_document(
        source_path / "10_MPlus_ADAM_QA" / "copyright-removed.xml",
        answer="",
        qid="2-1",
    )
    _mark_source(source_path, "test-revision")
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
        medquad_source_path=source_path,
        medquad_revision="test-revision",
        medquad_batch_size=2,
    )
    store = InMemoryVectorStore()
    embedder = FakeEmbedder()
    manager = KnowledgeManager(settings=settings, vector_store=store, embedder=embedder)

    submitted = await manager.submit_medquad_import()
    completed = await manager.process_job(submitted["job_id"])

    assert completed == {
        "document_id": "medquad",
        "job_id": submitted["job_id"],
        "status": "ready",
        "record_count": 1,
        "chunk_count": 1,
        "skipped_empty_answers": 1,
    }
    matches = await store.search([1.0, 1.0], 5, {"document_id": "medquad"})
    assert len(matches) == 1
    chunk = matches[0].chunk
    assert chunk.title == "Chronic kidney disease"
    assert chunk.section == "treatment"
    assert chunk.external_url == "https://www.niddk.nih.gov/example"
    assert "How is CKD treated?" in chunk.text
    assert "Treatment is individualized" in chunk.text
    assert "C1561643" in chunk.text
    assert chunk.status == "ready"
    assert chunk.metadata["umls_cuis"] == ["C1561643"]
    assert chunk.metadata["license"] == "CC BY 4.0"
    job = await manager.get_job(submitted["job_id"])
    assert job["payload"]["progress"] == {
        "processed_chunks": 1,
        "total_chunks": 1,
        "record_count": 1,
        "skipped_empty_answers": 1,
    }

    duplicate = await manager.submit_medquad_import()
    assert duplicate == {
        "document_id": "medquad",
        "job_id": None,
        "status": "ready",
        "duplicate": True,
        "version": "test-revision",
        "chunk_count": 1,
    }

    reindex = await manager.reindex_all()
    rebuilt = await manager.process_job(reindex["job_id"])
    assert rebuilt["status"] == "ready"
    assert rebuilt["chunk_count"] == 1
    assert len(await store.search([1.0, 1.0], 5, {"document_id": "medquad"})) == 1
    assert max(embedder.batch_sizes) <= 2
    document = (await manager.list_documents())[0]
    assert document["chunk_count"] == 1
    assert document["versions"][0]["metadata"]["source_sha256"]
    manifest = (tmp_path / "documents" / "medquad" / "test-revision" / "MedQuAD.md").read_text(
        encoding="utf-8"
    )
    assert "Asma Ben Abacha and Dina Demner-Fushman" in manifest


@pytest.mark.asyncio
async def test_failed_medquad_import_can_retry_the_same_revision(tmp_path):
    source_path = tmp_path / "medquad"
    broken_path = source_path / "5_NIDDK_QA" / "broken.xml"
    broken_path.parent.mkdir(parents=True)
    broken_path.write_text("<Document>", encoding="utf-8")
    _mark_source(source_path, "retry-revision")
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
        medquad_source_path=source_path,
        medquad_revision="retry-revision",
    )
    manager = KnowledgeManager(
        settings=settings,
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
    )

    first = await manager.submit_medquad_import()
    failed = await manager.process_job(first["job_id"])
    assert failed["status"] == "failed"

    _write_medquad_document(broken_path, answer="Use individualized treatment.")
    retry = await manager.submit_medquad_import()
    completed = await manager.process_job(retry["job_id"])

    assert retry["duplicate"] is False
    assert retry["job_id"] != first["job_id"]
    assert completed["status"] == "ready"


@pytest.mark.asyncio
async def test_medquad_import_preserves_reused_question_ids_from_different_files(tmp_path):
    source_path = tmp_path / "medquad"
    _write_medquad_document(
        source_path / "5_NIDDK_QA" / "first.xml",
        answer="First source answer.",
        qid="shared-1",
    )
    _mark_source(source_path, "collision-revision")
    _write_medquad_document(
        source_path / "5_NIDDK_QA" / "second.xml",
        answer="Second source answer.",
        qid="shared-1",
    )
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
        medquad_source_path=source_path,
        medquad_revision="collision-revision",
    )
    store = InMemoryVectorStore()
    manager = KnowledgeManager(settings=settings, vector_store=store, embedder=FakeEmbedder())

    submitted = await manager.submit_medquad_import()
    completed = await manager.process_job(submitted["job_id"])

    assert completed["chunk_count"] == 2
    matches = await store.search([1.0, 1.0], 5, {"document_id": "medquad"})
    assert len(matches) == 2


@pytest.mark.asyncio
async def test_medquad_rejects_changed_content_under_the_same_revision(tmp_path):
    source_path = tmp_path / "medquad"
    document_path = source_path / "5_NIDDK_QA" / "answer.xml"
    _write_medquad_document(document_path, answer="Original answer.")
    _mark_source(source_path, "fixed-revision")
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
        medquad_source_path=source_path,
        medquad_revision="fixed-revision",
    )
    manager = KnowledgeManager(
        settings=settings,
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
    )
    submitted = await manager.submit_medquad_import()
    await manager.process_job(submitted["job_id"])
    _write_medquad_document(document_path, answer="Changed answer.")

    with pytest.raises(ValueError, match="medquad_source_content_changed"):
        await manager.submit_medquad_import()


@pytest.mark.asyncio
async def test_concurrent_medquad_submissions_share_the_active_job(tmp_path):
    import asyncio

    source_path = tmp_path / "medquad"
    _write_medquad_document(
        source_path / "5_NIDDK_QA" / "answer.xml",
        answer="Answer.",
    )
    _mark_source(source_path, "concurrent-revision")
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
        medquad_source_path=source_path,
        medquad_revision="concurrent-revision",
    )
    first_manager = KnowledgeManager(
        settings=settings,
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
    )
    second_manager = KnowledgeManager(
        settings=settings,
        vector_store=InMemoryVectorStore(),
        embedder=FakeEmbedder(),
    )

    first, second = await asyncio.gather(
        first_manager.submit_medquad_import(),
        second_manager.submit_medquad_import(),
    )

    assert first["job_id"] == second["job_id"]
    assert {first["duplicate"], second["duplicate"]} == {False, True}


@pytest.mark.asyncio
async def test_medquad_publish_failure_restores_previous_ready_version(tmp_path):
    from dataclasses import replace

    source_path = tmp_path / "medquad"
    document_path = source_path / "5_NIDDK_QA" / "answer.xml"
    _write_medquad_document(document_path, answer="Version one treatment.")
    _mark_source(source_path, "revision-one")
    settings = KnowledgeSettings(
        enabled=True,
        catalog_path=tmp_path / "catalog.sqlite3",
        documents_path=tmp_path / "documents",
        medquad_source_path=source_path,
        medquad_revision="revision-one",
    )
    store = InMemoryVectorStore()
    first_manager = KnowledgeManager(
        settings=settings,
        vector_store=store,
        embedder=FakeEmbedder(),
    )
    first = await first_manager.submit_medquad_import()
    assert (await first_manager.process_job(first["job_id"]))["status"] == "ready"

    _write_medquad_document(document_path, answer="Version two treatment.")
    _mark_source(source_path, "revision-two")
    second_manager = KnowledgeManager(
        settings=replace(settings, medquad_revision="revision-two"),
        vector_store=store,
        embedder=FakeEmbedder(),
    )
    second = await second_manager.submit_medquad_import()

    def fail_manifest(*_args):
        raise OSError("manifest failed")

    second_manager.medquad.write_manifest = fail_manifest
    failed = await second_manager.process_job(second["job_id"])

    assert failed["status"] == "failed"
    matches = await store.search([1.0, 1.0], 5, {"version": "revision-one"})
    assert len(matches) == 1
    assert (await second_manager.catalog.get_document("medquad"))["version"] == "revision-one"


def test_prefixed_token_split_enforces_total_chunk_budget():
    class WordTokenizer:
        def encode(self, text, **_kwargs):
            return text.split()

        def decode(self, tokens, **_kwargs):
            return " ".join(tokens)

    class LoadedModel:
        tokenizer = WordTokenizer()

    embedder = SentenceTransformerEmbedder("unused")
    embedder._model = LoadedModel()
    chunks = embedder.split_prefixed_tokens(
        " ".join(f"metadata-{index}" for index in range(30)),
        " ".join(f"answer-{index}" for index in range(100)),
        size=40,
        overlap=5,
    )

    assert len(chunks) > 1
    assert max(embedder.count_tokens(chunk) for chunk in chunks) <= 40
