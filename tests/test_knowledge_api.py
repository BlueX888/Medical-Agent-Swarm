from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.server as server
from knowledge import KnowledgeSettings


class FakeKnowledgeBase:
    async def health(self):
        return {"status": "ok", "backend": "fake"}


class FakeKnowledgeManager:
    def __init__(self):
        self.submissions = []
        self.processed = []
        self.deleted = []

    async def initialize(self):
        return None

    async def recover_jobs(self):
        return []

    async def health(self):
        return {"ready_documents": 1, "latest_index_error": None}

    async def submit_document(self, **kwargs):
        self.submissions.append(kwargs)
        return {"document_id": kwargs.get("document_id") or "doc-1", "job_id": "job-1", "status": "queued", "duplicate": False}

    async def process_job(self, job_id):
        self.processed.append(job_id)
        return {"job_id": job_id, "status": "ready"}

    async def list_documents(self):
        return [{"id": "doc-1", "title": "慢性肾病指南", "status": "ready"}]

    async def get_job(self, job_id):
        return {"id": job_id, "document_id": "doc-1", "status": "ready", "payload": {}}

    async def delete_document(self, document_id):
        self.deleted.append(document_id)
        return True

    async def submit_delete(self, document_id):
        self.deleted.append(document_id)
        return {"document_id": document_id, "job_id": "delete-1", "status": "queued"}

    async def reindex_all(self):
        return {"job_id": "reindex-1", "status": "queued"}

    async def submit_medquad_import(self):
        return {
            "document_id": "medquad",
            "job_id": "medquad-import-1",
            "status": "queued",
            "duplicate": False,
            "version": "577bd37",
        }


def test_knowledge_admin_api_requires_token_and_manages_documents(monkeypatch, short_term_memory_factory):
    manager = FakeKnowledgeManager()
    runtime = SimpleNamespace(
        settings=KnowledgeSettings(enabled=True, admin_token="admin-secret"),
        knowledge_base=FakeKnowledgeBase(),
        manager=manager,
    )

    async def create_memory():
        return short_term_memory_factory()

    monkeypatch.setattr(server, "create_short_term_memory", create_memory)
    monkeypatch.setattr(server, "create_knowledge_runtime", lambda: runtime)

    with TestClient(server.app) as client:
        unauthorized = client.get("/api/admin/knowledge/documents")
        assert unauthorized.status_code == 403

        response = client.post(
            "/api/admin/knowledge/documents",
            headers={"X-Knowledge-Admin-Token": "admin-secret"},
            files={"file": ("ckd.md", "# 指南\n内容", "text/markdown")},
            data={"metadata": '{"title":"慢性肾病指南","version":"2026"}'},
        )
        assert response.status_code == 202
        assert response.json() == {
            "document_id": "doc-1",
            "job_id": "job-1",
            "status": "queued",
            "duplicate": False,
        }

        documents = client.get(
            "/api/admin/knowledge/documents",
            headers={"X-Knowledge-Admin-Token": "admin-secret"},
        )
        assert documents.json()[0]["title"] == "慢性肾病指南"

        job = client.get(
            "/api/admin/knowledge/jobs/job-1",
            headers={"X-Knowledge-Admin-Token": "admin-secret"},
        )
        assert job.json()["status"] == "ready"

        medquad = client.post(
            "/api/admin/knowledge/imports/medquad",
            headers={"X-Knowledge-Admin-Token": "admin-secret"},
        )
        assert medquad.status_code == 202
        assert medquad.json()["document_id"] == "medquad"
        assert medquad.json()["job_id"] == "medquad-import-1"

        deleted = client.delete(
            "/api/admin/knowledge/documents/doc-1",
            headers={"X-Knowledge-Admin-Token": "admin-secret"},
        )
        assert deleted.status_code == 202
        assert manager.deleted == ["doc-1"]

        health = client.get("/api/health").json()
        assert health["knowledge"]["status"] == "ok"
