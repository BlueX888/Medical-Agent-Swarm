"""Checkpoint storage configuration shared by workflow modules.

The application owns the saver lifecycle and injects one saver into each
compiled graph.  LangGraph remains the persistence seam; this module only
selects and initializes concrete adapters.
"""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver


class CheckpointConfigurationError(ValueError):
    """Raised when checkpoint persistence configuration is invalid."""


class CheckpointingDisabledError(RuntimeError):
    """Raised when a persistence-only operation has no configured saver."""


class RunAlreadyActiveError(RuntimeError):
    """Raised when another worker already owns the workflow run."""


class RunAlreadyExistsError(RuntimeError):
    """Raised when a new invocation reuses a durable run identifier."""


_ACTIVE_RUNS: set[str] = set()
_ACTIVE_RUNS_GUARD = threading.Lock()


@dataclass(frozen=True)
class CheckpointSettings:
    """Storage selection for workflow checkpoints."""

    backend: str = "sqlite"
    sqlite_path: Path = Path(".data/checkpoints.sqlite3")
    postgres_dsn: Optional[str] = None
    encryption_key: Optional[str] = None

    def __post_init__(self) -> None:
        backend = self.backend.strip().lower()
        if backend not in {"disabled", "memory", "sqlite", "postgres"}:
            raise CheckpointConfigurationError(
                f"Unsupported CHECKPOINT_BACKEND={self.backend!r}; expected "
                "disabled, memory, sqlite, or postgres"
            )
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "sqlite_path", Path(self.sqlite_path))
        if backend == "postgres" and not self.postgres_dsn:
            raise CheckpointConfigurationError(
                "CHECKPOINT_POSTGRES_DSN is required for the postgres backend"
            )
        if self.encryption_key and len(self.encryption_key.encode("utf-8")) not in {
            16,
            24,
            32,
        }:
            raise CheckpointConfigurationError(
                "CHECKPOINT_AES_KEY must encode to 16, 24, or 32 bytes"
            )

    @classmethod
    def from_env(cls) -> "CheckpointSettings":
        return cls(
            backend=os.getenv("CHECKPOINT_BACKEND", "sqlite"),
            sqlite_path=Path(
                os.getenv("CHECKPOINT_SQLITE_PATH", ".data/checkpoints.sqlite3")
            ),
            postgres_dsn=os.getenv("CHECKPOINT_POSTGRES_DSN"),
            encryption_key=os.getenv("CHECKPOINT_AES_KEY"),
        )


@dataclass(frozen=True)
class CheckpointSnapshot:
    """Stable application view of a LangGraph state snapshot."""

    run_id: str
    checkpoint_id: str
    values: Dict[str, Any]
    next_nodes: tuple[str, ...]
    metadata: Dict[str, Any]
    created_at: Optional[str]
    parent_checkpoint_id: Optional[str]
    status: str

    @classmethod
    def from_langgraph(cls, snapshot: Any) -> "CheckpointSnapshot":
        config = dict(getattr(snapshot, "config", {}) or {})
        configurable = dict(config.get("configurable") or {})
        parent_config = dict(getattr(snapshot, "parent_config", {}) or {})
        parent = dict(parent_config.get("configurable") or {})
        tasks = tuple(getattr(snapshot, "tasks", ()) or ())
        next_nodes = tuple(getattr(snapshot, "next", ()) or ())

        if any(getattr(task, "error", None) for task in tasks):
            status = "failed"
        elif any(getattr(task, "interrupts", None) for task in tasks):
            status = "interrupted"
        elif next_nodes:
            status = "pending"
        else:
            status = "completed"

        return cls(
            run_id=str(configurable.get("thread_id") or ""),
            checkpoint_id=str(configurable.get("checkpoint_id") or ""),
            values=dict(getattr(snapshot, "values", {}) or {}),
            next_nodes=next_nodes,
            metadata=dict(getattr(snapshot, "metadata", {}) or {}),
            created_at=getattr(snapshot, "created_at", None),
            parent_checkpoint_id=(
                str(parent["checkpoint_id"]) if parent.get("checkpoint_id") else None
            ),
            status=status,
        )


def checkpoint_config(
    run_id: str,
    checkpoint_id: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    configurable = {"thread_id": run_id.strip()}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


@asynccontextmanager
async def claim_run(
    checkpointer: Optional[BaseCheckpointSaver[Any]],
    run_id: str,
) -> AsyncIterator[None]:
    """Claim one run locally and, for PostgreSQL, across processes."""
    normalized_run_id = run_id.strip()
    with _ACTIVE_RUNS_GUARD:
        if normalized_run_id in _ACTIVE_RUNS:
            raise RunAlreadyActiveError(
                f"Workflow run is already active: {normalized_run_id}"
            )
        _ACTIVE_RUNS.add(normalized_run_id)

    postgres_claimed = False
    try:
        if _is_postgres_saver(checkpointer):
            postgres_claimed = await _try_postgres_advisory_lock(
                checkpointer,
                normalized_run_id,
            )
            if not postgres_claimed:
                raise RunAlreadyActiveError(
                    f"Workflow run is already active: {normalized_run_id}"
                )
        yield
    finally:
        try:
            if postgres_claimed:
                await _release_postgres_advisory_lock(
                    checkpointer,
                    normalized_run_id,
                )
        finally:
            with _ACTIVE_RUNS_GUARD:
                _ACTIVE_RUNS.discard(normalized_run_id)


def _is_postgres_saver(checkpointer: Optional[BaseCheckpointSaver[Any]]) -> bool:
    if checkpointer is None:
        return False
    return checkpointer.__class__.__module__.startswith(
        "langgraph.checkpoint.postgres"
    )


async def _try_postgres_advisory_lock(
    checkpointer: Any,
    run_id: str,
) -> bool:
    cursor = await checkpointer.conn.execute(
        "SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS acquired",
        (run_id,),
    )
    row = await cursor.fetchone()
    if isinstance(row, dict):
        return bool(row.get("acquired"))
    return bool(row and row[0])


async def _release_postgres_advisory_lock(
    checkpointer: Any,
    run_id: str,
) -> None:
    await checkpointer.conn.execute(
        "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
        (run_id,),
    )


@asynccontextmanager
async def open_checkpointer(
    settings: Optional[CheckpointSettings] = None,
) -> AsyncIterator[Optional[BaseCheckpointSaver[Any]]]:
    """Open and initialize the configured LangGraph saver."""
    settings = settings or CheckpointSettings.from_env()
    serializer = _checkpoint_serializer(settings)

    if settings.backend == "disabled":
        yield None
        return

    if settings.backend == "memory":
        yield InMemorySaver(serde=serializer)
        return

    if settings.backend == "sqlite":
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise CheckpointConfigurationError(
                "SQLite checkpointing requires langgraph-checkpoint-sqlite"
            ) from exc

        database_path = settings.sqlite_path.expanduser().resolve()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(database_path)) as connection:
            saver = AsyncSqliteSaver(connection, serde=serializer)
            await saver.setup()
            yield saver
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:
        raise CheckpointConfigurationError(
            "PostgreSQL checkpointing requires langgraph-checkpoint-postgres"
        ) from exc

    assert settings.postgres_dsn is not None
    async with AsyncPostgresSaver.from_conn_string(
        settings.postgres_dsn,
        serde=serializer,
    ) as saver:
        await saver.setup()
        yield saver


def _checkpoint_serializer(settings: CheckpointSettings):
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    serializer = JsonPlusSerializer(pickle_fallback=False)
    if not settings.encryption_key:
        return serializer

    try:
        from langgraph.checkpoint.serde.encrypted import EncryptedSerializer

        return EncryptedSerializer.from_pycryptodome_aes(
            serializer,
            key=settings.encryption_key.encode("utf-8"),
        )
    except ImportError as exc:
        raise CheckpointConfigurationError(
            "Encrypted checkpointing requires pycryptodome"
        ) from exc
