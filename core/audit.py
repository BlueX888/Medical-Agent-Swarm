"""Durable, encrypted workflow audit snapshots."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

from .checkpointing import CheckpointSettings, checkpoint_serializer


class AuditStore(Protocol):
    async def save_attempt(
        self,
        run_id: str,
        attempt_id: str,
        payload: Dict[str, Any],
    ) -> None: ...

    async def get_attempts(self, run_id: str) -> List[Dict[str, Any]]: ...

    async def claim_effect(
        self,
        run_id: str,
        effect_name: str,
        *,
        retry_claimed: bool = False,
    ) -> bool: ...

    async def complete_effect(
        self,
        run_id: str,
        effect_name: str,
        status: str,
    ) -> None: ...

    async def get_effects(self, run_id: str) -> List[Dict[str, Any]]: ...

    async def reconcile_effect(
        self, run_id: str, effect_name: str, resolution: str
    ) -> bool: ...


class MemoryAuditStore:
    def __init__(self):
        self._attempts: Dict[tuple[str, str], Dict[str, Any]] = {}
        self._effects: Dict[tuple[str, str], tuple[str, datetime]] = {}

    async def save_attempt(
        self,
        run_id: str,
        attempt_id: str,
        payload: Dict[str, Any],
    ) -> None:
        self._attempts[(run_id, attempt_id)] = payload

    async def get_attempts(self, run_id: str) -> List[Dict[str, Any]]:
        values = [
            {"attempt_id": attempt_id, **payload}
            for (stored_run_id, attempt_id), payload in self._attempts.items()
            if stored_run_id == run_id
        ]
        values.sort(key=lambda item: item.get("run", {}).get("started_at", ""))
        return values

    async def claim_effect(
        self,
        run_id: str,
        effect_name: str,
        *,
        retry_claimed: bool = False,
    ) -> bool:
        key = (run_id, effect_name)
        now = datetime.now(timezone.utc)
        current = self._effects.get(key)
        if current:
            status, updated_at = current
            if status == "completed":
                return False
            if status == "unknown":
                return False
            if status == "claimed" and not retry_claimed:
                self._effects[key] = ("unknown", now)
                return False
        self._effects[key] = ("claimed", now)
        return True

    async def complete_effect(
        self,
        run_id: str,
        effect_name: str,
        status: str,
    ) -> None:
        _validate_effect_status(status)
        self._effects[(run_id, effect_name)] = (
            status,
            datetime.now(timezone.utc),
        )

    async def get_effects(self, run_id: str) -> List[Dict[str, Any]]:
        return [
            {
                "effect_name": effect_name,
                "status": status,
                "updated_at": updated_at.isoformat(),
            }
            for (stored_run_id, effect_name), (status, updated_at) in self._effects.items()
            if stored_run_id == run_id
        ]

    async def reconcile_effect(
        self, run_id: str, effect_name: str, resolution: str
    ) -> bool:
        _validate_reconciliation(resolution)
        key = (run_id, effect_name)
        current = self._effects.get(key)
        if current is None or current[0] != "unknown":
            return False
        self._effects[key] = (resolution, datetime.now(timezone.utc))
        return True


class SqliteAuditStore:
    def __init__(self, connection: Any, serializer: Any):
        self.connection = connection
        self.serializer = serializer

    async def setup(self) -> None:
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_audit (
                run_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                payload_type TEXT NOT NULL,
                payload BLOB NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, attempt_id)
            )
            """
        )
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_effects (
                run_id TEXT NOT NULL,
                effect_name TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (run_id, effect_name)
            )
            """
        )
        await self.connection.commit()

    async def save_attempt(
        self,
        run_id: str,
        attempt_id: str,
        payload: Dict[str, Any],
    ) -> None:
        payload_type, serialized = self.serializer.dumps_typed(payload)
        await self.connection.execute(
            """
            INSERT INTO workflow_audit (
                run_id, attempt_id, payload_type, payload, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(run_id, attempt_id) DO UPDATE SET
                payload_type=excluded.payload_type,
                payload=excluded.payload,
                updated_at=excluded.updated_at
            """,
            (
                run_id,
                attempt_id,
                payload_type,
                serialized,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await self.connection.commit()

    async def get_attempts(self, run_id: str) -> List[Dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT attempt_id, payload_type, payload
            FROM workflow_audit
            WHERE run_id = ?
            ORDER BY updated_at ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "attempt_id": row[0],
                **self.serializer.loads_typed((row[1], bytes(row[2]))),
            }
            for row in rows
        ]

    async def claim_effect(
        self,
        run_id: str,
        effect_name: str,
        *,
        retry_claimed: bool = False,
    ) -> bool:
        now = datetime.now(timezone.utc)
        cursor = await self.connection.execute(
            """
            INSERT INTO workflow_effects (
                run_id, effect_name, status, updated_at
            ) VALUES (?, ?, 'claimed', ?)
            ON CONFLICT(run_id, effect_name) DO UPDATE SET
                status='claimed',
                updated_at=excluded.updated_at
            WHERE workflow_effects.status = 'failed'
               OR (workflow_effects.status = 'claimed' AND ?)
            RETURNING status
            """,
            (run_id, effect_name, now.isoformat(), retry_claimed),
        )
        row = await cursor.fetchone()
        if row is None and not retry_claimed:
            await self.connection.execute(
                """
                UPDATE workflow_effects
                SET status = 'unknown', updated_at = ?
                WHERE run_id = ? AND effect_name = ? AND status = 'claimed'
                """,
                (now.isoformat(), run_id, effect_name),
            )
        await self.connection.commit()
        return row is not None

    async def complete_effect(
        self,
        run_id: str,
        effect_name: str,
        status: str,
    ) -> None:
        _validate_effect_status(status)
        await self.connection.execute(
            """
            UPDATE workflow_effects
            SET status = ?, updated_at = ?
            WHERE run_id = ? AND effect_name = ?
            """,
            (
                status,
                datetime.now(timezone.utc).isoformat(),
                run_id,
                effect_name,
            ),
        )
        await self.connection.commit()

    async def get_effects(self, run_id: str) -> List[Dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT effect_name, status, updated_at
            FROM workflow_effects WHERE run_id = ? ORDER BY effect_name
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        return [
            {"effect_name": row[0], "status": row[1], "updated_at": row[2]}
            for row in rows
        ]

    async def reconcile_effect(
        self, run_id: str, effect_name: str, resolution: str
    ) -> bool:
        _validate_reconciliation(resolution)
        cursor = await self.connection.execute(
            """
            UPDATE workflow_effects SET status = ?, updated_at = ?
            WHERE run_id = ? AND effect_name = ? AND status = 'unknown'
            RETURNING status
            """,
            (
                resolution,
                datetime.now(timezone.utc).isoformat(),
                run_id,
                effect_name,
            ),
        )
        row = await cursor.fetchone()
        await self.connection.commit()
        return row is not None


class PostgresAuditStore:
    def __init__(self, connection: Any, serializer: Any):
        self.connection = connection
        self.serializer = serializer

    async def setup(self) -> None:
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_audit (
                run_id TEXT NOT NULL,
                attempt_id TEXT NOT NULL,
                payload_type TEXT NOT NULL,
                payload BYTEA NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (run_id, attempt_id)
            )
            """
        )
        await self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS workflow_effects (
                run_id TEXT NOT NULL,
                effect_name TEXT NOT NULL,
                status TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (run_id, effect_name)
            )
            """
        )

    async def save_attempt(
        self,
        run_id: str,
        attempt_id: str,
        payload: Dict[str, Any],
    ) -> None:
        payload_type, serialized = self.serializer.dumps_typed(payload)
        await self.connection.execute(
            """
            INSERT INTO workflow_audit (
                run_id, attempt_id, payload_type, payload, updated_at
            ) VALUES (%s, %s, %s, %s, NOW())
            ON CONFLICT(run_id, attempt_id) DO UPDATE SET
                payload_type=EXCLUDED.payload_type,
                payload=EXCLUDED.payload,
                updated_at=EXCLUDED.updated_at
            """,
            (run_id, attempt_id, payload_type, serialized),
        )

    async def get_attempts(self, run_id: str) -> List[Dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT attempt_id, payload_type, payload
            FROM workflow_audit
            WHERE run_id = %s
            ORDER BY updated_at ASC
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "attempt_id": row[0],
                **self.serializer.loads_typed((row[1], bytes(row[2]))),
            }
            for row in rows
        ]

    async def claim_effect(
        self,
        run_id: str,
        effect_name: str,
        *,
        retry_claimed: bool = False,
    ) -> bool:
        cursor = await self.connection.execute(
            """
            INSERT INTO workflow_effects (
                run_id, effect_name, status, updated_at
            ) VALUES (%s, %s, 'claimed', NOW())
            ON CONFLICT(run_id, effect_name) DO UPDATE SET
                status='claimed',
                updated_at=NOW()
            WHERE workflow_effects.status = 'failed'
               OR (workflow_effects.status = 'claimed' AND %s)
            RETURNING run_id
            """,
            (run_id, effect_name, retry_claimed),
        )
        row = await cursor.fetchone()
        if row is None and not retry_claimed:
            await self.connection.execute(
                """
                UPDATE workflow_effects
                SET status = 'unknown', updated_at = NOW()
                WHERE run_id = %s AND effect_name = %s AND status = 'claimed'
                """,
                (run_id, effect_name),
            )
        return row is not None

    async def complete_effect(
        self,
        run_id: str,
        effect_name: str,
        status: str,
    ) -> None:
        _validate_effect_status(status)
        await self.connection.execute(
            """
            UPDATE workflow_effects
            SET status = %s, updated_at = NOW()
            WHERE run_id = %s AND effect_name = %s
            """,
            (status, run_id, effect_name),
        )

    async def get_effects(self, run_id: str) -> List[Dict[str, Any]]:
        cursor = await self.connection.execute(
            """
            SELECT effect_name, status, updated_at
            FROM workflow_effects WHERE run_id = %s ORDER BY effect_name
            """,
            (run_id,),
        )
        rows = await cursor.fetchall()
        return [
            {
                "effect_name": row[0],
                "status": row[1],
                "updated_at": row[2].isoformat(),
            }
            for row in rows
        ]

    async def reconcile_effect(
        self, run_id: str, effect_name: str, resolution: str
    ) -> bool:
        _validate_reconciliation(resolution)
        cursor = await self.connection.execute(
            """
            UPDATE workflow_effects SET status = %s, updated_at = NOW()
            WHERE run_id = %s AND effect_name = %s AND status = 'unknown'
            RETURNING status
            """,
            (resolution, run_id, effect_name),
        )
        return await cursor.fetchone() is not None


def _validate_effect_status(status: str) -> None:
    if status not in {"completed", "failed", "unknown"}:
        raise ValueError("effect status must be completed, failed, or unknown")


def _validate_reconciliation(resolution: str) -> None:
    if resolution not in {"completed", "failed"}:
        raise ValueError("reconciliation must resolve to completed or failed")


@asynccontextmanager
async def open_audit_store(
    settings: Optional[CheckpointSettings] = None,
) -> AsyncIterator[AuditStore]:
    settings = settings or CheckpointSettings.from_env()
    serializer = checkpoint_serializer(settings)
    if settings.backend in {"disabled", "memory"}:
        yield MemoryAuditStore()
        return
    if settings.backend == "sqlite":
        import aiosqlite

        database_path = settings.sqlite_path.expanduser().resolve()
        database_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path = database_path.with_name(f"{database_path.stem}-audit.sqlite3")
        async with aiosqlite.connect(str(audit_path)) as connection:
            store = SqliteAuditStore(connection, serializer)
            await store.setup()
            yield store
        return

    from psycopg import AsyncConnection

    assert settings.postgres_dsn is not None
    async with await AsyncConnection.connect(
        settings.postgres_dsn,
        autocommit=True,
        prepare_threshold=0,
    ) as connection:
        store = PostgresAuditStore(connection, serializer)
        await store.setup()
        yield store
