from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger

from .catalog import KnowledgeCatalog
from .documents import (
    DocumentValidationError,
    chunk_sections,
    parse_document,
    safe_identifier,
    validate_file,
    validate_metadata,
)
from .models import KnowledgeChunk
from .settings import KnowledgeSettings


class KnowledgeManager:
    """Administrative interface for the document lifecycle."""

    def __init__(self, *, settings: KnowledgeSettings, vector_store: Any, embedder: Any):
        self.settings = settings
        self.vector_store = vector_store
        self.embedder = embedder
        self.catalog = KnowledgeCatalog(settings.catalog_path)

    async def initialize(self) -> None:
        await self.catalog.initialize()

    async def submit_document(
        self,
        *,
        filename: str,
        content: bytes,
        metadata: Dict[str, Any],
        document_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        metadata = validate_metadata(metadata)
        filename = validate_file(filename, content, self.settings.max_file_bytes)
        checksum = hashlib.sha256(content).hexdigest()
        duplicate = await self.catalog.find_by_checksum(checksum)
        if duplicate:
            return {
                "document_id": duplicate["id"],
                "job_id": duplicate.get("job_id"),
                "status": duplicate["status"],
                "duplicate": True,
            }
        document_id = safe_identifier(document_id or str(uuid.uuid4()), "document_id")
        existing = await self.catalog.get_document(document_id)
        version = safe_identifier(metadata.get("version") or "v1", "version")
        if existing and await self.catalog.get_version(document_id, version):
            raise DocumentValidationError("version_already_exists")
        target_dir = self.settings.documents_path / document_id / version
        target_dir.mkdir(parents=True, exist_ok=True)
        file_path = target_dir / filename
        await asyncio.to_thread(file_path.write_bytes, content)
        job_id = str(uuid.uuid4())
        staged_document = {
            "id": document_id,
            "filename": filename,
            "title": str(metadata.get("title") or Path(filename).stem),
            "source_org": str(metadata.get("source_org") or ""),
            "version": version,
            "published_at": str(metadata.get("published_at") or ""),
            "external_url": str(metadata.get("external_url") or ""),
            "language": str(metadata.get("language") or ""),
            "checksum": checksum,
            "status": "queued",
            "file_path": str(file_path),
            "created_at": existing.get("created_at") if existing else None,
        }
        payload = {
            "file_path": str(file_path),
            "previous_version": existing.get("version") if existing else None,
            "staged_document": staged_document,
        }
        if not existing:
            await self.catalog.upsert_document(staged_document)
        await self.catalog.upsert_version(
            {
                "document_id": document_id,
                "version": version,
                "checksum": checksum,
                "filename": filename,
                "file_path": str(file_path),
                "status": "queued",
                "metadata": metadata,
            }
        )
        await self.catalog.create_job(job_id, document_id, "index", payload)
        return {"document_id": document_id, "job_id": job_id, "status": "queued", "duplicate": False}

    async def process_job(self, job_id: str) -> Dict[str, Any]:
        started_at = time.perf_counter()
        job = await self.catalog.get_job(job_id)
        if not job:
            raise LookupError("knowledge_job_not_found")
        if job["operation"] == "delete":
            result = await self._process_delete_job(job)
            self._log_ingestion_event(job, result, started_at)
            return result
        if job["operation"] == "reindex":
            result = await self._process_reindex_job(job)
            self._log_ingestion_event(job, result, started_at)
            return result
        current_document = await self.catalog.get_document(job["document_id"])
        document = dict(job["payload"].get("staged_document") or current_document or {})
        if not document:
            raise LookupError("knowledge_document_not_found")
        await self.catalog.update_job(job_id, status="indexing", error=None)
        await self.catalog.update_version(document["id"], document["version"], status="indexing", error=None)
        if not job["payload"].get("previous_version"):
            await self.catalog.update_document(document["id"], status="indexing", last_error=None)
        try:
            path = Path(job["payload"]["file_path"])
            previous_version = job["payload"].get("previous_version")
            chunks = await self._build_chunks(
                document,
                path,
                status="staged" if previous_version else "ready",
            )
            vectors = await asyncio.to_thread(self.embedder.embed_documents, [chunk.text for chunk in chunks])
            await self.vector_store.upsert(chunks, vectors)
            if previous_version:
                await self.vector_store.activate_version(
                    document["id"],
                    document["version"],
                    previous_version,
                )
            await self.catalog.upsert_document({**document, "status": "ready", "chunk_count": len(chunks)})
            await self.catalog.update_version(
                document["id"], document["version"], status="ready", chunk_count=len(chunks), error=None
            )
            if previous_version and previous_version != document["version"]:
                await self.catalog.update_version(
                    document["id"], previous_version, status="retired", error=None
                )
                try:
                    await self.vector_store.delete_document(document["id"], previous_version)
                except Exception as cleanup_error:
                    logger.warning(f"Old knowledge version cleanup failed: {type(cleanup_error).__name__}")
            await self.catalog.update_job(job_id, status="ready")
            result = {
                "document_id": document["id"],
                "job_id": job_id,
                "status": "ready",
                "chunk_count": len(chunks),
            }
            self._log_ingestion_event(job, result, started_at)
            return result
        except Exception as exc:
            logger.warning(f"Knowledge ingestion failed ({job_id}): {type(exc).__name__}")
            try:
                await self.vector_store.delete_document(document["id"], document["version"])
            except Exception:
                pass
            await self.catalog.update_version(
                document["id"], document["version"], status="failed", error=type(exc).__name__
            )
            if not job["payload"].get("previous_version"):
                await self.catalog.update_document(document["id"], status="failed", last_error=type(exc).__name__)
            await self.catalog.update_job(job_id, status="failed", error=type(exc).__name__)
            result = {
                "document_id": document["id"],
                "job_id": job_id,
                "status": "failed",
                "error": type(exc).__name__,
            }
            self._log_ingestion_event(job, result, started_at)
            return result

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return await self.catalog.get_job(job_id)

    async def list_documents(self):
        return await self.catalog.list_documents()

    async def delete_document(self, document_id: str) -> bool:
        document = await self.catalog.get_document(document_id)
        if not document:
            return False
        await self.vector_store.delete_document(document_id)
        for version in await self.catalog.list_versions(document_id):
            path = Path(version["file_path"])
            if path.exists():
                await asyncio.to_thread(path.unlink)
        await self.catalog.delete_document(document_id)
        return True

    async def submit_delete(self, document_id: str) -> Dict[str, Any]:
        document = await self.catalog.get_document(document_id)
        if not document:
            raise LookupError("knowledge_document_not_found")
        job_id = str(uuid.uuid4())
        await self.catalog.create_job(job_id, document_id, "delete", {})
        await self.catalog.update_document(document_id, status="deleting")
        return {"document_id": document_id, "job_id": job_id, "status": "queued"}

    async def reindex_all(self) -> Dict[str, Any]:
        job_id = str(uuid.uuid4())
        await self.catalog.create_job(job_id, None, "reindex", {})
        return {"job_id": job_id, "status": "queued"}

    async def _process_delete_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        await self.catalog.update_job(job["id"], status="deleting", error=None)
        document = await self.catalog.get_document(job["document_id"])
        if document:
            await self.vector_store.delete_document(document["id"])
            for version in await self.catalog.list_versions(document["id"]):
                path = Path(version["file_path"])
                if path.exists():
                    await asyncio.to_thread(path.unlink)
            await self.catalog.delete_document(document["id"])
        await self.catalog.update_job(job["id"], status="ready")
        return {"document_id": job["document_id"], "job_id": job["id"], "status": "deleted"}

    async def _process_reindex_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        await self.catalog.update_job(job["id"], status="indexing", error=None)
        try:
            chunks = []
            for document in await self.catalog.list_documents():
                if document["status"] != "ready":
                    continue
                path = Path(document["file_path"])
                chunks.extend(await self._build_chunks(document, path))
            vectors = (
                await asyncio.to_thread(self.embedder.embed_documents, [chunk.text for chunk in chunks])
                if chunks
                else []
            )
            await self.vector_store.replace_all(chunks, vectors)
            await self.catalog.update_job(job["id"], status="ready")
            return {"job_id": job["id"], "status": "ready", "chunk_count": len(chunks)}
        except Exception as exc:
            await self.catalog.update_job(job["id"], status="failed", error=type(exc).__name__)
            return {"job_id": job["id"], "status": "failed", "error": type(exc).__name__}

    async def _build_chunks(
        self,
        document: Dict[str, Any],
        path: Path,
        *,
        status: str = "ready",
    ):
        content = await asyncio.to_thread(path.read_bytes)
        sections = await asyncio.to_thread(parse_document, path.name, content)
        pieces = await asyncio.to_thread(
            chunk_sections,
            sections,
            self.embedder,
            self.settings.chunk_tokens,
            self.settings.chunk_overlap,
        )
        return [
            KnowledgeChunk(
                chunk_id=f"{document['id']}:{document['version']}:{index}",
                document_id=document["id"],
                version=document["version"],
                title=document["title"],
                section=section,
                text=text,
                source_org=document["source_org"],
                published_at=document["published_at"],
                external_url=document["external_url"],
                language=document["language"],
                status=status,
            )
            for index, (section, text) in enumerate(pieces)
        ]

    async def recover_jobs(self):
        return await self.catalog.recoverable_jobs()

    async def health(self) -> Dict[str, Any]:
        return {
            "ready_documents": await self.catalog.ready_count(),
            "latest_index_error": await self.catalog.latest_error(),
        }

    def _log_ingestion_event(
        self,
        job: Dict[str, Any],
        result: Dict[str, Any],
        started_at: float,
    ) -> None:
        logger.bind(
            event="knowledge_ingestion",
            job_id=job["id"],
            operation=job["operation"],
            status=result.get("status"),
            chunk_count=int(result.get("chunk_count") or 0),
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
            embedding_model=getattr(self.embedder, "model_name", ""),
            error=result.get("error"),
        ).info("Knowledge ingestion event")
