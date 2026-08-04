from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class KnowledgeCatalog:
    """Persistent document and ingestion-job catalog."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._lock:
            await asyncio.to_thread(self._initialize_sync)

    async def find_by_checksum(self, checksum: str) -> Optional[Dict[str, Any]]:
        return await self._fetch_one(
            "SELECT d.*, j.id AS job_id FROM document_versions v "
            "JOIN documents d ON d.id=v.document_id "
            "LEFT JOIN jobs j ON j.document_id=d.id "
            "WHERE v.checksum=? AND v.status!='failed' ORDER BY j.created_at DESC LIMIT 1",
            (checksum,),
        )

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        return await self._fetch_one("SELECT * FROM documents WHERE id=?", (document_id,))

    async def list_documents(self) -> List[Dict[str, Any]]:
        documents = await self._fetch_all("SELECT * FROM documents ORDER BY created_at DESC", ())
        for document in documents:
            document["versions"] = await self.list_versions(document["id"])
        return documents

    async def upsert_version(self, value: Dict[str, Any]) -> None:
        await self.initialize()
        now = _now()
        fields = {
            "document_id": value["document_id"],
            "version": value["version"],
            "checksum": value["checksum"],
            "filename": value["filename"],
            "file_path": value["file_path"],
            "status": value.get("status", "queued"),
            "chunk_count": int(value.get("chunk_count", 0)),
            "error": value.get("error"),
            "metadata": json.dumps(value.get("metadata") or {}, ensure_ascii=False),
            "created_at": value.get("created_at") or now,
            "updated_at": now,
        }
        columns = list(fields)
        updates = ",".join(
            f"{column}=excluded.{column}"
            for column in columns
            if column not in {"document_id", "version", "created_at"}
        )
        await self._execute(
            f"INSERT INTO document_versions ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
            f"ON CONFLICT(document_id,version) DO UPDATE SET {updates}",
            tuple(fields[column] for column in columns),
        )

    async def update_version(self, document_id: str, version: str, **changes: Any) -> None:
        changes["updated_at"] = _now()
        await self._execute(
            f"UPDATE document_versions SET {','.join(f'{key}=?' for key in changes)} "
            "WHERE document_id=? AND version=?",
            (*changes.values(), document_id, version),
        )

    async def list_versions(self, document_id: str) -> List[Dict[str, Any]]:
        values = await self._fetch_all(
            "SELECT * FROM document_versions WHERE document_id=? ORDER BY created_at DESC",
            (document_id,),
        )
        for value in values:
            value["metadata"] = json.loads(value.get("metadata") or "{}")
        return values

    async def get_version(self, document_id: str, version: str) -> Optional[Dict[str, Any]]:
        value = await self._fetch_one(
            "SELECT * FROM document_versions WHERE document_id=? AND version=?",
            (document_id, version),
        )
        if value:
            value["metadata"] = json.loads(value.get("metadata") or "{}")
        return value

    async def upsert_document(self, value: Dict[str, Any]) -> None:
        await self.initialize()
        now = _now()
        fields = {
            "id": value["id"],
            "filename": value["filename"],
            "title": value["title"],
            "source_org": value.get("source_org", ""),
            "version": value.get("version", "v1"),
            "published_at": value.get("published_at", ""),
            "external_url": value.get("external_url", ""),
            "language": value.get("language", ""),
            "checksum": value["checksum"],
            "status": value.get("status", "queued"),
            "file_path": value["file_path"],
            "chunk_count": int(value.get("chunk_count", 0)),
            "last_error": value.get("last_error"),
            "created_at": value.get("created_at") or now,
            "updated_at": now,
        }
        columns = list(fields)
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns if column not in {"id", "created_at"})
        await self._execute(
            f"INSERT INTO documents ({','.join(columns)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {updates}",
            tuple(fields[column] for column in columns),
        )

    async def update_document(self, document_id: str, **changes: Any) -> None:
        if not changes:
            return
        changes["updated_at"] = _now()
        await self._execute(
            f"UPDATE documents SET {','.join(f'{key}=?' for key in changes)} WHERE id=?",
            (*changes.values(), document_id),
        )

    async def delete_document(self, document_id: str) -> None:
        await self._execute("DELETE FROM document_versions WHERE document_id=?", (document_id,))
        await self._execute("DELETE FROM documents WHERE id=?", (document_id,))

    async def create_job(self, job_id: str, document_id: Optional[str], operation: str, payload: Dict[str, Any]) -> None:
        now = _now()
        await self._execute(
            "INSERT INTO jobs (id,document_id,operation,status,payload,error,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (job_id, document_id, operation, "queued", json.dumps(payload, ensure_ascii=False), None, now, now),
        )

    async def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        value = await self._fetch_one("SELECT * FROM jobs WHERE id=?", (job_id,))
        if value and value.get("payload"):
            value["payload"] = json.loads(value["payload"])
        return value

    async def update_job(self, job_id: str, **changes: Any) -> None:
        changes["updated_at"] = _now()
        await self._execute(
            f"UPDATE jobs SET {','.join(f'{key}=?' for key in changes)} WHERE id=?",
            (*changes.values(), job_id),
        )

    async def recoverable_jobs(self) -> List[Dict[str, Any]]:
        values = await self._fetch_all(
            "SELECT * FROM jobs WHERE status IN ('queued','indexing','deleting') ORDER BY created_at",
            (),
        )
        for value in values:
            value["payload"] = json.loads(value.get("payload") or "{}")
        return values

    async def ready_count(self) -> int:
        value = await self._fetch_one("SELECT COUNT(*) AS count FROM documents WHERE status='ready'", ())
        return int((value or {}).get("count", 0))

    async def latest_error(self) -> Optional[str]:
        value = await self._fetch_one(
            "SELECT last_error FROM documents WHERE last_error IS NOT NULL ORDER BY updated_at DESC LIMIT 1",
            (),
        )
        return str(value["last_error"]) if value and value.get("last_error") else None

    async def _fetch_one(self, sql: str, params: tuple) -> Optional[Dict[str, Any]]:
        values = await self._fetch_all(sql, params)
        return values[0] if values else None

    async def _fetch_all(self, sql: str, params: tuple) -> List[Dict[str, Any]]:
        await self.initialize()
        async with self._lock:
            return await asyncio.to_thread(self._fetch_all_sync, sql, params)

    async def _execute(self, sql: str, params: tuple) -> None:
        await self.initialize()
        async with self._lock:
            await asyncio.to_thread(self._execute_sync, sql, params)

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize_sync(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, title TEXT NOT NULL,
                    source_org TEXT NOT NULL DEFAULT '', version TEXT NOT NULL,
                    published_at TEXT NOT NULL DEFAULT '', external_url TEXT NOT NULL DEFAULT '',
                    language TEXT NOT NULL DEFAULT '', checksum TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL, file_path TEXT NOT NULL, chunk_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, document_id TEXT, operation TEXT NOT NULL,
                    status TEXT NOT NULL, payload TEXT NOT NULL, error TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL, version TEXT NOT NULL, checksum TEXT NOT NULL,
                    filename TEXT NOT NULL, file_path TEXT NOT NULL, status TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL DEFAULT 0, error TEXT, metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(document_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_document ON jobs(document_id, created_at);
                CREATE INDEX IF NOT EXISTS idx_versions_checksum ON document_versions(checksum);
                """
            )
            jobs_info = connection.execute("PRAGMA table_info(jobs)").fetchall()
            document_column = next((row for row in jobs_info if row[1] == "document_id"), None)
            if document_column and int(document_column[3]) == 1:
                connection.execute("PRAGMA foreign_keys=OFF")
                connection.executescript(
                    """
                    ALTER TABLE jobs RENAME TO jobs_legacy;
                    CREATE TABLE jobs (
                        id TEXT PRIMARY KEY, document_id TEXT, operation TEXT NOT NULL,
                        status TEXT NOT NULL, payload TEXT NOT NULL, error TEXT,
                        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                    );
                    INSERT INTO jobs SELECT * FROM jobs_legacy;
                    DROP TABLE jobs_legacy;
                    CREATE INDEX IF NOT EXISTS idx_jobs_document ON jobs(document_id, created_at);
                    """
                )
                connection.execute("PRAGMA foreign_keys=ON")

    def _fetch_all_sync(self, sql: str, params: tuple) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(sql, params).fetchall()]

    def _execute_sync(self, sql: str, params: tuple) -> None:
        with self._connect() as connection:
            connection.execute(sql, params)
