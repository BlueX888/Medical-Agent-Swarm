from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from defusedxml import ElementTree
from loguru import logger

from .catalog import KnowledgeCatalog
from .documents import safe_identifier
from .models import KnowledgeChunk
from .settings import KnowledgeSettings


MEDQUAD_DOCUMENT_ID = "medquad"
MEDQUAD_REPOSITORY_URL = "https://github.com/abachaa/MedQuAD"
MEDQUAD_LICENSE = "CC BY 4.0"
MEDQUAD_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
MEDQUAD_AUTHORS = "Asma Ben Abacha and Dina Demner-Fushman"
MEDQUAD_PAPER = "A Question-Entailment Approach to Question Answering (BMC Bioinformatics, 2019)"
MEDQUAD_PAPER_URL = "https://doi.org/10.1186/s12859-019-3119-4"

SOURCE_NAMES = {
    "CancerGov": "National Cancer Institute",
    "CDC": "Centers for Disease Control and Prevention",
    "GARD": "Genetic and Rare Diseases Information Center",
    "GHR": "Genetics Home Reference",
    "MPlus": "MedlinePlus",
    "NHLBI": "National Heart, Lung, and Blood Institute",
    "NIDDK": "National Institute of Diabetes and Digestive and Kidney Diseases",
    "NINDS": "National Institute of Neurological Disorders and Stroke",
    "NIHSeniorHealth": "NIH Senior Health",
    "MPlusHealthTopics": "MedlinePlus",
}


@dataclass(frozen=True)
class MedQuADBuild:
    chunks: List[KnowledgeChunk]
    record_count: int
    skipped_empty_answers: int


class MedQuADImporter:
    """Import answered MedQuAD pairs through the existing knowledge lifecycle."""

    def __init__(
        self,
        *,
        settings: KnowledgeSettings,
        catalog: KnowledgeCatalog,
        vector_store: Any,
        embedder: Any,
    ):
        self.settings = settings
        self.catalog = catalog
        self.vector_store = vector_store
        self.embedder = embedder
        self._submit_lock = asyncio.Lock()

    async def submit(self) -> Dict[str, Any]:
        async with self._submit_lock:
            return await self._submit_locked()

    async def _submit_locked(self) -> Dict[str, Any]:
        source_path = self.settings.medquad_source_path.resolve()
        version = safe_identifier(self.settings.medquad_revision, "medquad_revision")
        source_digest = await asyncio.to_thread(
            self._validate_source,
            source_path,
            version,
        )
        existing_version = await self.catalog.get_version(MEDQUAD_DOCUMENT_ID, version)
        if existing_version and existing_version["status"] != "failed":
            metadata = existing_version.get("metadata") or {}
            recorded_digest = metadata.get("source_sha256")
            if recorded_digest and recorded_digest != source_digest:
                raise ValueError("medquad_source_content_changed")
            if not recorded_digest:
                metadata = {**metadata, **self._provenance(version, source_path, source_digest)}
                await self.catalog.update_version(
                    MEDQUAD_DOCUMENT_ID,
                    version,
                    checksum=source_digest,
                    metadata=json.dumps(metadata, ensure_ascii=False),
                )
                current = await self.catalog.get_document(MEDQUAD_DOCUMENT_ID)
                if current and current.get("version") == version:
                    await self.catalog.update_document(
                        MEDQUAD_DOCUMENT_ID,
                        checksum=source_digest,
                    )
            active_job = await self.catalog.find_active_job(
                MEDQUAD_DOCUMENT_ID,
                "import_medquad",
            )
            return {
                "document_id": MEDQUAD_DOCUMENT_ID,
                "job_id": active_job["id"] if active_job else None,
                "status": existing_version["status"],
                "duplicate": True,
                "version": version,
                "chunk_count": int(existing_version.get("chunk_count") or 0),
            }

        existing = await self.catalog.get_document(MEDQUAD_DOCUMENT_ID)
        target_dir = self.settings.documents_path / MEDQUAD_DOCUMENT_ID / version
        target_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = target_dir / "MedQuAD.md"
        checksum = source_digest
        document = {
            "id": MEDQUAD_DOCUMENT_ID,
            "filename": manifest_path.name,
            "title": "MedQuAD Medical Question Answering Dataset",
            "source_org": "U.S. National Library of Medicine",
            "version": version,
            "published_at": "",
            "external_url": MEDQUAD_REPOSITORY_URL,
            "language": "en",
            "checksum": checksum,
            "status": "queued",
            "file_path": str(manifest_path),
            "created_at": existing.get("created_at") if existing else None,
        }
        metadata = self._provenance(version, source_path, source_digest)
        job_id = str(uuid.uuid4())
        try:
            await self.catalog.create_job(
                job_id,
                MEDQUAD_DOCUMENT_ID,
                "import_medquad",
                {
                    "source_path": str(source_path),
                    "source_sha256": source_digest,
                    "previous_version": existing.get("version") if existing else None,
                    "staged_document": document,
                    "progress": {"processed_chunks": 0},
                },
            )
        except sqlite3.IntegrityError:
            active_job = await self.catalog.find_active_job(
                MEDQUAD_DOCUMENT_ID,
                "import_medquad",
            )
            if active_job is None:
                raise
            return {
                "document_id": MEDQUAD_DOCUMENT_ID,
                "job_id": active_job["id"],
                "status": active_job["status"],
                "duplicate": True,
                "version": version,
                "chunk_count": 0,
            }
        try:
            await self.catalog.upsert_version(
                {
                    "document_id": MEDQUAD_DOCUMENT_ID,
                    "version": version,
                    "checksum": checksum,
                    "filename": manifest_path.name,
                    "file_path": str(manifest_path),
                    "status": "queued",
                    "metadata": metadata,
                }
            )
        except Exception:
            await self.catalog.update_job(
                job_id,
                status="failed",
                error="version_reservation_failed",
            )
            raise
        return {
            "document_id": MEDQUAD_DOCUMENT_ID,
            "job_id": job_id,
            "status": "queued",
            "duplicate": False,
            "version": version,
        }

    async def process(self, job: Dict[str, Any]) -> Dict[str, Any]:
        document = dict(job["payload"]["staged_document"])
        version = document["version"]
        previous_version = job["payload"].get("previous_version")
        await self.catalog.update_job(job["id"], status="indexing", error=None)
        await self.catalog.update_version(MEDQUAD_DOCUMENT_ID, version, status="indexing", error=None)
        try:
            build = await asyncio.to_thread(
                self.build_chunks,
                Path(job["payload"]["source_path"]),
                version,
                "staged",
                job["payload"]["source_sha256"],
            )
            total = len(build.chunks)
            batch_size = max(1, self.settings.medquad_batch_size)
            for start in range(0, total, batch_size):
                batch = build.chunks[start : start + batch_size]
                vectors = await asyncio.to_thread(
                    self.embedder.embed_documents,
                    [chunk.text for chunk in batch],
                )
                await self.vector_store.upsert(batch, vectors)
                payload = dict(job["payload"])
                payload["progress"] = {
                    "processed_chunks": min(start + len(batch), total),
                    "total_chunks": total,
                    "record_count": build.record_count,
                    "skipped_empty_answers": build.skipped_empty_answers,
                }
                await self.catalog.update_job(job["id"], payload=json.dumps(payload, ensure_ascii=False))

            await self.vector_store.activate_version(
                MEDQUAD_DOCUMENT_ID,
                version,
                previous_version,
            )
            await asyncio.to_thread(
                self.write_manifest,
                Path(document["file_path"]),
                version,
                build,
                job["payload"]["source_sha256"],
            )
            await self.catalog.upsert_document(
                {**document, "status": "ready", "chunk_count": total}
            )
            await self.catalog.update_version(
                MEDQUAD_DOCUMENT_ID,
                version,
                status="ready",
                chunk_count=total,
                error=None,
            )
            await self.catalog.update_job(job["id"], status="ready", error=None)
            if previous_version and previous_version != version:
                await self.catalog.update_version(
                    MEDQUAD_DOCUMENT_ID,
                    previous_version,
                    status="retired",
                    error=None,
                )
                try:
                    await self.vector_store.delete_document(
                        MEDQUAD_DOCUMENT_ID,
                        previous_version,
                    )
                except Exception as cleanup_error:
                    logger.warning(
                        "Old MedQuAD version cleanup failed: {}",
                        type(cleanup_error).__name__,
                    )
            return {
                "document_id": MEDQUAD_DOCUMENT_ID,
                "job_id": job["id"],
                "status": "ready",
                "record_count": build.record_count,
                "chunk_count": total,
                "skipped_empty_answers": build.skipped_empty_answers,
            }
        except Exception as exc:
            if previous_version and previous_version != version:
                try:
                    await self.vector_store.activate_version(
                        MEDQUAD_DOCUMENT_ID,
                        previous_version,
                        version,
                    )
                except Exception:
                    pass
            try:
                await self.vector_store.delete_document(MEDQUAD_DOCUMENT_ID, version)
            except Exception:
                pass
            await self.catalog.update_version(
                MEDQUAD_DOCUMENT_ID,
                version,
                status="failed",
                error=type(exc).__name__,
            )
            await self.catalog.update_job(
                job["id"],
                status="failed",
                error=type(exc).__name__,
            )
            return {
                "document_id": MEDQUAD_DOCUMENT_ID,
                "job_id": job["id"],
                "status": "failed",
                "error": type(exc).__name__,
            }

    def build_chunks(
        self,
        source_path: Path,
        version: str,
        status: str = "ready",
        expected_digest: str | None = None,
    ) -> MedQuADBuild:
        digest = self._validate_source(source_path.resolve(), version)
        if expected_digest and digest != expected_digest:
            raise ValueError("medquad_source_content_changed")
        chunks: List[KnowledgeChunk] = []
        records = 0
        skipped = 0
        for path in sorted(source_path.rglob("*.xml")):
            root = ElementTree.parse(path).getroot()
            focus = _element_text(root.find("Focus")) or path.stem
            category = _element_text(root.find("./FocusAnnotations/Category"))
            synonyms = [
                value
                for node in root.findall("./FocusAnnotations/Synonyms/Synonym")
                if (value := _element_text(node))
            ]
            cuis = [
                value
                for node in root.findall("./FocusAnnotations/UMLS/CUIs/CUI")
                if (value := _element_text(node))
            ]
            semantic_types = [
                value
                for node in root.findall("./FocusAnnotations/UMLS/SemanticTypes/SemanticType")
                if (value := _element_text(node))
            ]
            semantic_group = _element_text(
                root.find("./FocusAnnotations/UMLS/SemanticGroup")
            )
            source = str(root.attrib.get("source") or "MedQuAD")
            original_source_org = SOURCE_NAMES.get(source, f"MedQuAD / {source}")
            source_org = (
                f"{original_source_org}; via MedQuAD "
                f"({MEDQUAD_AUTHORS}, {MEDQUAD_LICENSE})"
            )
            external_url = str(root.attrib.get("url") or MEDQUAD_REPOSITORY_URL)
            subset = _safe_slug(path.parent.name)
            source_document = _safe_slug(path.stem)
            for pair in root.findall("./QAPairs/QAPair"):
                question_node = pair.find("Question")
                answer = _element_text(pair.find("Answer"))
                if not answer:
                    skipped += 1
                    continue
                question = _element_text(question_node)
                if not question:
                    continue
                records += 1
                qid = _safe_slug(str((question_node.attrib if question_node is not None else {}).get("qid") or records))
                qtype = str((question_node.attrib if question_node is not None else {}).get("qtype") or category or "medical_qa")
                prefix_lines = [f"Question: {question}", f"Medical focus: {focus}"]
                if category:
                    prefix_lines.append(f"Category: {category}")
                if synonyms:
                    prefix_lines.append(f"Synonyms: {', '.join(synonyms)}")
                if cuis:
                    prefix_lines.append(f"UMLS CUI: {', '.join(cuis)}")
                if semantic_types:
                    prefix_lines.append(f"UMLS semantic type: {', '.join(semantic_types)}")
                if semantic_group:
                    prefix_lines.append(f"UMLS semantic group: {semantic_group}")
                prefix = "\n".join(prefix_lines)
                for piece_index, chunk_text in enumerate(self._split_answer(answer, prefix)):
                    chunks.append(
                        KnowledgeChunk(
                            chunk_id=(
                                f"medquad:{version}:{subset}:{source_document}:"
                                f"{qid}:{piece_index}"
                            ),
                            document_id=MEDQUAD_DOCUMENT_ID,
                            version=version,
                            title=focus,
                            section=qtype,
                            text=chunk_text,
                            source_org=source_org,
                            external_url=external_url,
                            language="en",
                            status=status,
                            metadata={
                                "dataset": "MedQuAD",
                                "source_code": source,
                                "source_org": original_source_org,
                                "question_id": qid,
                                "question_type": qtype,
                                "category": category,
                                "synonyms": synonyms,
                                "umls_cuis": cuis,
                                "umls_semantic_types": semantic_types,
                                "umls_semantic_group": semantic_group,
                                "license": MEDQUAD_LICENSE,
                                "license_url": MEDQUAD_LICENSE_URL,
                            },
                        )
                    )
        if not chunks:
            raise ValueError("medquad_has_no_answered_records")
        return MedQuADBuild(
            chunks=chunks,
            record_count=records,
            skipped_empty_answers=skipped,
        )

    def _split_answer(self, answer: str, prefix: str) -> List[str]:
        if hasattr(self.embedder, "split_prefixed_tokens"):
            return self.embedder.split_prefixed_tokens(
                prefix,
                answer,
                size=self.settings.chunk_tokens,
                overlap=self.settings.chunk_overlap,
            )
        if hasattr(self.embedder, "split_tokens"):
            prefix_tokens = int(self.embedder.count_tokens(prefix))
            size = max(64, self.settings.chunk_tokens - min(prefix_tokens, 160))
            overlap = min(self.settings.chunk_overlap, max(0, size - 1))
            return [
                f"{prefix}\nAnswer: {piece.strip()}"
                for piece in self.embedder.split_tokens(answer, size=size, overlap=overlap)
                if piece.strip()
            ]
        size = max(64, self.settings.chunk_tokens - min(len(prefix), 160))
        step = max(1, size - min(self.settings.chunk_overlap, size - 1))
        return [
            f"{prefix}\nAnswer: {answer[start : start + size].strip()}"
            for start in range(0, len(answer), step)
        ]

    @staticmethod
    def write_manifest(
        path: Path,
        version: str,
        build: MedQuADBuild,
        source_digest: str,
    ) -> None:
        path.write_text(
            "\n".join(
                [
                    "# MedQuAD Medical Question Answering Dataset",
                    "",
                    f"- Repository: {MEDQUAD_REPOSITORY_URL}",
                    f"- Revision: {version}",
                    f"- Source SHA-256: {source_digest}",
                    f"- Authors: {MEDQUAD_AUTHORS}",
                    f"- License: [{MEDQUAD_LICENSE}]({MEDQUAD_LICENSE_URL})",
                    f"- Paper: [{MEDQUAD_PAPER}]({MEDQUAD_PAPER_URL})",
                    "- Transformation: empty answers were omitted; answered records were token-split and embedded for retrieval.",
                    f"- Answered records: {build.record_count}",
                    f"- Indexed chunks: {len(build.chunks)}",
                    f"- Empty/copyright-removed answers skipped: {build.skipped_empty_answers}",
                ]
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _provenance(version: str, source_path: Path, digest: str) -> Dict[str, str]:
        return {
            "dataset": "MedQuAD",
            "authors": MEDQUAD_AUTHORS,
            "license": MEDQUAD_LICENSE,
            "license_url": MEDQUAD_LICENSE_URL,
            "paper": MEDQUAD_PAPER,
            "paper_url": MEDQUAD_PAPER_URL,
            "repository": MEDQUAD_REPOSITORY_URL,
            "revision": version,
            "source_path": str(source_path),
            "source_sha256": digest,
            "transformation": "Empty answers omitted; answered records token-split and embedded.",
        }

    @classmethod
    def _validate_source(cls, source_path: Path, revision: str) -> str:
        if not source_path.is_dir():
            raise FileNotFoundError("medquad_source_not_found")
        actual_revision = cls._read_source_revision(source_path)
        if actual_revision != revision:
            raise ValueError("medquad_revision_mismatch")
        paths = sorted(source_path.rglob("*.xml"))
        if not paths:
            raise ValueError("medquad_source_has_no_xml")
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.relative_to(source_path).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    @staticmethod
    def _read_source_revision(source_path: Path) -> str:
        git_dir = source_path / ".git"
        head_path = git_dir / "HEAD"
        if not head_path.is_file():
            marker = source_path / ".medquad-revision"
            if marker.is_file():
                return marker.read_text(encoding="utf-8").strip()
            raise ValueError("medquad_revision_marker_missing")
        head = head_path.read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        ref_path = git_dir / ref
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        packed_refs = git_dir / "packed-refs"
        if packed_refs.is_file():
            for line in packed_refs.read_text(encoding="utf-8").splitlines():
                if line and not line.startswith(("#", "^")):
                    value, name = line.split(" ", 1)
                    if name == ref:
                        return value
        raise ValueError("medquad_revision_unresolved")


def _element_text(element: Any) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]", "-", value).strip("-")
    return slug[:180] or "unknown"
