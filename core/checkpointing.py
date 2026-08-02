"""Checkpoint storage configuration shared by workflow modules.

The application owns the saver lifecycle and injects one saver into each
compiled graph.  LangGraph remains the persistence seam; this module only
selects and initializes concrete adapters.
"""
from __future__ import annotations

import os
import hashlib
import secrets
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncContextManager,
    AsyncIterator,
    BinaryIO,
    Dict,
    Optional,
    Protocol,
)

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


class RunLeaseManager(Protocol):
    """Cross-worker ownership seam for workflow runs."""

    def claim(self, run_id: str) -> AsyncContextManager[None]: ...


class LocalRunLeaseManager:
    """Process-local lease used for memory and disabled backends."""

    def claim(self, run_id: str) -> AsyncContextManager[None]:
        return claim_run(run_id)


class FileRunLeaseManager:
    """OS-backed run lease shared by processes using one SQLite file."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def claim(self, run_id: str) -> AsyncIterator[None]:
        async with claim_run(run_id):
            digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
            path = self.directory / f"{digest}.lock"
            lock_file = path.open("a+b")
            locked = False
            try:
                _lock_file(lock_file, run_id)
                locked = True
                yield
            finally:
                if locked:
                    _unlock_file(lock_file)
                lock_file.close()


class PostgresRunLeaseManager:
    """PostgreSQL advisory lease held on a dedicated owned connection."""

    def __init__(self, connection: Any):
        self.connection = connection

    @asynccontextmanager
    async def claim(self, run_id: str) -> AsyncIterator[None]:
        async with claim_run(run_id):
            cursor = await self.connection.execute(
                "SELECT pg_try_advisory_lock(hashtextextended(%s, 0))",
                (run_id,),
            )
            row = await cursor.fetchone()
            if not row or not bool(row[0]):
                raise RunAlreadyActiveError(
                    f"Workflow run is already active: {run_id}"
                )
            try:
                yield
            finally:
                await self.connection.execute(
                    "SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                    (run_id,),
                )


@dataclass(frozen=True)
class CheckpointSettings:
    """Storage selection for workflow checkpoints."""

    backend: str = "memory"
    sqlite_path: Path = Path(".data/checkpoints.sqlite3")
    postgres_dsn: Optional[str] = None
    encryption_key: Optional[str] = None
    allow_plaintext: bool = False

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
        if (
            backend in {"sqlite", "postgres"}
            and not self.encryption_key
            and not self.allow_plaintext
        ):
            raise CheckpointConfigurationError(
                "Durable checkpoints may contain medical data. Set "
                "CHECKPOINT_AES_KEY, or explicitly opt into plaintext with "
                "CHECKPOINT_ALLOW_PLAINTEXT=true for local development."
            )

    @classmethod
    def from_env(cls) -> "CheckpointSettings":
        backend = os.getenv("CHECKPOINT_BACKEND", "sqlite").strip().lower()
        encryption_key = os.getenv("CHECKPOINT_AES_KEY") or None
        if backend == "sqlite" and not encryption_key:
            encryption_key = _load_or_create_local_key(
                Path(os.getenv("CHECKPOINT_AES_KEY_FILE", ".data/checkpoint.key"))
            )
        return cls(
            backend=backend,
            sqlite_path=Path(
                os.getenv("CHECKPOINT_SQLITE_PATH", ".data/checkpoints.sqlite3")
            ),
            postgres_dsn=os.getenv("CHECKPOINT_POSTGRES_DSN"),
            encryption_key=encryption_key,
            allow_plaintext=_boolean_env("CHECKPOINT_ALLOW_PLAINTEXT", False),
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
    run_id: str,
) -> AsyncIterator[None]:
    """Claim one run within the current process."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id must be a non-empty string")
    normalized_run_id = run_id.strip()
    with _ACTIVE_RUNS_GUARD:
        if normalized_run_id in _ACTIVE_RUNS:
            raise RunAlreadyActiveError(
                f"Workflow run is already active: {normalized_run_id}"
            )
        _ACTIVE_RUNS.add(normalized_run_id)

    try:
        yield
    finally:
        with _ACTIVE_RUNS_GUARD:
            _ACTIVE_RUNS.discard(normalized_run_id)


def _lock_file(lock_file: BinaryIO, run_id: str) -> None:
    lock_file.seek(0, os.SEEK_END)
    if lock_file.tell() == 0:
        lock_file.write(b"\0")
        lock_file.flush()
    lock_file.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        raise RunAlreadyActiveError(
            f"Workflow run is already active: {run_id}"
        ) from exc


def _unlock_file(lock_file: BinaryIO) -> None:
    lock_file.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@asynccontextmanager
async def open_run_lease(
    settings: Optional[CheckpointSettings] = None,
) -> AsyncIterator[RunLeaseManager]:
    settings = settings or CheckpointSettings.from_env()
    if settings.backend == "sqlite":
        database_path = settings.sqlite_path.expanduser().resolve()
        yield FileRunLeaseManager(database_path.parent / ".run-leases")
        return
    if settings.backend == "postgres":
        from psycopg import AsyncConnection

        assert settings.postgres_dsn is not None
        async with await AsyncConnection.connect(
            settings.postgres_dsn,
            autocommit=True,
            prepare_threshold=0,
        ) as connection:
            yield PostgresRunLeaseManager(connection)
        return
    yield LocalRunLeaseManager()


@asynccontextmanager
async def open_checkpointer(
    settings: Optional[CheckpointSettings] = None,
) -> AsyncIterator[Optional[BaseCheckpointSaver[Any]]]:
    """Open and initialize the configured LangGraph saver."""
    settings = settings or CheckpointSettings.from_env()
    serializer = checkpoint_serializer(settings)

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


def checkpoint_serializer(settings: CheckpointSettings):
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


def _boolean_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise CheckpointConfigurationError(
        f"{name} must be true or false, got {value!r}"
    )


def _load_or_create_local_key(path: Path) -> str:
    """Load, or atomically create, a local 32-byte SQLite encryption key."""
    key_path = path.expanduser().resolve()
    key_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        key = key_path.read_text(encoding="ascii").strip()
    except FileNotFoundError:
        key = secrets.token_hex(16)
        temporary_path = key_path.with_name(
            f".{key_path.name}.{secrets.token_hex(8)}.tmp"
        )
        descriptor = os.open(
            temporary_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="ascii") as handle:
                handle.write(key)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_path, key_path)
            except FileExistsError:
                key = key_path.read_text(encoding="ascii").strip()
        finally:
            temporary_path.unlink(missing_ok=True)
    if len(key.encode("utf-8")) not in {16, 24, 32}:
        raise CheckpointConfigurationError(
            f"Invalid SQLite checkpoint key file: {key_path}"
        )
    return key
