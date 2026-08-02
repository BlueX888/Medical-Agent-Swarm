"""Session-scoped short-term conversation memory.

Only completed, user-visible turns belong here. Agent scratch messages, tool
results, and debug traces stay in their request-local stores.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional, Protocol

from dotenv import load_dotenv
from loguru import logger


load_dotenv()

DEFAULT_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MAX_MESSAGES = 40
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
KEY_PREFIX = "medical-agent-swarm:stm"
REDIS_CONNECTION_OPTIONS = {
    "decode_responses": True,
    "socket_connect_timeout": 1.0,
    "socket_timeout": 1.0,
}

_IDEMPOTENT_APPEND_SCRIPT = """
if redis.call('EXISTS', KEYS[2]) == 1 then
    return 0
end
local message_count = #ARGV - 2
local max_messages = tonumber(ARGV[#ARGV - 1])
local ttl_seconds = tonumber(ARGV[#ARGV])
redis.call('SET', KEYS[2], '1', 'EX', ttl_seconds)
for index = 1, message_count do
    redis.call('RPUSH', KEYS[1], ARGV[index])
end
redis.call('LTRIM', KEYS[1], -max_messages, -1)
redis.call('EXPIRE', KEYS[1], ttl_seconds)
return 1
"""


class ShortTermMemoryError(RuntimeError):
    """Base error for short-term memory storage failures."""


class ShortTermMemoryUnavailable(ShortTermMemoryError):
    """Raised when the configured short-term memory backend is unavailable."""


class ShortTermMemoryDataError(ShortTermMemoryError):
    """Raised when stored memory cannot be serialized or decoded safely."""


@dataclass(frozen=True)
class MemoryMessage:
    """A single user-visible conversation message."""

    role: str
    content: str
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def create(
        cls,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "MemoryMessage":
        return cls(
            role=role,
            content=content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        message = asdict(self)
        if self.metadata is None:
            message.pop("metadata")
        return message


class ShortTermMemoryAdapter(Protocol):
    """Internal storage seam used by the public short-term memory module."""

    backend_name: str

    async def load_messages(
        self,
        session_id: str,
        message_limit: int,
    ) -> List[Dict[str, Any]]:
        ...

    async def save_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        idempotency_key: Optional[str] = None,
    ) -> bool:
        ...

    async def clear_session(self, session_id: str) -> bool:
        ...

    async def get_session_ttl(self, session_id: str) -> int:
        ...

    async def health(self) -> Dict[str, str]:
        ...

    async def close(self) -> None:
        ...


@dataclass
class _SessionRunLock:
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0


_SESSION_RUN_LOCKS: Dict[
    tuple[asyncio.AbstractEventLoop, str],
    _SessionRunLock,
] = {}
_SESSION_RUN_LOCKS_GUARD = threading.Lock()


class RedisShortTermMemoryAdapter:
    """Redis List adapter with atomic append, trim, and TTL refresh."""

    backend_name = "redis"

    def __init__(
        self,
        *,
        ttl_seconds: int,
        max_messages: int,
        redis_url: Optional[str] = None,
        redis_config: Optional[Dict[str, Any]] = None,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self._redis = self._create_client(
            redis_url=redis_url,
            redis_config=redis_config or {},
        )

    async def load_messages(
        self,
        session_id: str,
        message_limit: int,
    ) -> List[Dict[str, Any]]:
        try:
            values = await self._redis.lrange(
                self._key(session_id),
                -message_limit,
                -1,
            )
        except Exception as exc:
            logger.warning(f"Failed to load Redis short-term memory: {exc}")
            raise ShortTermMemoryUnavailable(
                "Redis short-term memory is unavailable"
            ) from exc

        messages = []
        for value in values:
            try:
                data = json.loads(value)
            except (TypeError, json.JSONDecodeError) as exc:
                logger.error("Redis short-term memory contains invalid JSON")
                raise ShortTermMemoryDataError(
                    "Redis short-term memory contains invalid data"
                ) from exc
            if not isinstance(data, dict):
                raise ShortTermMemoryDataError(
                    "Redis short-term memory contains invalid data"
                )
            if data.get("role") in {"user", "assistant"}:
                message: Dict[str, Any] = {
                    "role": str(data["role"]),
                    "content": str(data.get("content", "")),
                    "timestamp": str(data.get("timestamp", "")),
                }
                if isinstance(data.get("metadata"), dict):
                    message["metadata"] = data["metadata"]
                messages.append(message)
        return messages

    async def save_messages(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        idempotency_key: Optional[str] = None,
    ) -> bool:
        key = self._key(session_id)
        try:
            serialized = [
                json.dumps(message, ensure_ascii=False, separators=(",", ":"))
                for message in messages
            ]
        except (TypeError, ValueError) as exc:
            raise ShortTermMemoryDataError(
                "Short-term memory message is not JSON serializable"
            ) from exc
        try:
            if idempotency_key:
                operation_key = self._operation_key(session_id, idempotency_key)
                result = await self._redis.eval(
                    _IDEMPOTENT_APPEND_SCRIPT,
                    2,
                    key,
                    operation_key,
                    *serialized,
                    self.max_messages,
                    self.ttl_seconds,
                )
                return bool(result)

            async with self._redis.pipeline(transaction=True) as pipeline:
                pipeline.rpush(key, *serialized)
                pipeline.ltrim(key, -self.max_messages, -1)
                pipeline.expire(key, self.ttl_seconds)
                await pipeline.execute()
            return True
        except Exception as exc:
            logger.warning(f"Failed to save Redis short-term memory: {exc}")
            raise ShortTermMemoryUnavailable(
                "Redis short-term memory is unavailable"
            ) from exc

    async def clear_session(self, session_id: str) -> bool:
        try:
            return bool(await self._redis.delete(self._key(session_id)))
        except Exception as exc:
            logger.warning(f"Failed to clear Redis short-term memory: {exc}")
            raise ShortTermMemoryUnavailable(
                "Redis short-term memory is unavailable"
            ) from exc

    async def get_session_ttl(self, session_id: str) -> int:
        try:
            return int(await self._redis.ttl(self._key(session_id)))
        except Exception as exc:
            logger.warning(f"Failed to read Redis short-term memory TTL: {exc}")
            raise ShortTermMemoryUnavailable(
                "Redis short-term memory is unavailable"
            ) from exc

    async def health(self) -> Dict[str, str]:
        try:
            await self._redis.ping()
            return {"backend": self.backend_name, "status": "ok"}
        except Exception as exc:
            logger.warning(f"Redis short-term memory health check failed: {exc}")
            return {"backend": self.backend_name, "status": "degraded"}

    async def close(self) -> None:
        close = getattr(self._redis, "aclose", None) or getattr(self._redis, "close", None)
        if close:
            result = close()
            if hasattr(result, "__await__"):
                await result

    @staticmethod
    def _key(session_id: str) -> str:
        return f"{KEY_PREFIX}:{session_id}"

    @staticmethod
    def _operation_key(session_id: str, idempotency_key: str) -> str:
        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        return f"{KEY_PREFIX}:{session_id}:operations:{digest}"

    @staticmethod
    def _create_client(
        redis_url: Optional[str],
        redis_config: Dict[str, Any],
    ):
        try:
            from redis.asyncio import Redis
        except ImportError as exc:
            raise RuntimeError(
                "Redis backend requires the 'redis' package. "
                "Install project requirements first."
            ) from exc

        if redis_config and not redis_url:
            return Redis(
                host=redis_config.get("host", "localhost"),
                port=int(redis_config.get("port", 6379)),
                db=int(redis_config.get("db", 0)),
                password=redis_config.get("password"),
                **REDIS_CONNECTION_OPTIONS,
            )

        return Redis.from_url(
            redis_url or DEFAULT_REDIS_URL,
            **REDIS_CONNECTION_OPTIONS,
        )


class ShortTermMemory:
    """Session-scoped conversation memory backed exclusively by Redis."""

    def __init__(
        self,
        redis_config: Optional[Dict[str, Any]] = None,
        *,
        redis_url: Optional[str] = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_messages: int = DEFAULT_MAX_MESSAGES,
        adapter: Optional[ShortTermMemoryAdapter] = None,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        if max_messages < 2:
            raise ValueError("max_messages must be at least two")

        # Complete turns always contain two messages.
        self.ttl_seconds = int(ttl_seconds)
        self.max_messages = int(max_messages) - (int(max_messages) % 2)
        self._adapter = adapter or RedisShortTermMemoryAdapter(
            ttl_seconds=self.ttl_seconds,
            max_messages=self.max_messages,
            redis_url=redis_url,
            redis_config=redis_config or {},
        )
        if self._adapter.backend_name != "redis":
            raise ValueError("ShortTermMemory adapters must use the Redis backend")

        logger.info(
            "ShortTermMemory initialized "
            f"(backend={self.backend_name}, ttl={self.ttl_seconds}s, "
            f"max_messages={self.max_messages})"
        )

    @property
    def backend_name(self) -> str:
        return self._adapter.backend_name

    async def load_context(
        self,
        session_id: str,
        max_turns: int = 5,
    ) -> List[Dict[str, Any]]:
        """Load recent complete turns in chronological OpenAI message order."""
        self._validate_session_id(session_id)
        if max_turns <= 0:
            return []
        return await self._adapter.load_messages(session_id, max_turns * 2)

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        assistant_metadata: Optional[Dict[str, Any]] = None,
        idempotency_key: Optional[str] = None,
    ) -> bool:
        """Atomically append one completed user/assistant turn."""
        self._validate_session_id(session_id)
        if not isinstance(user_message, str) or not user_message.strip():
            raise ValueError("user_message must be a non-empty string")
        if not isinstance(assistant_message, str) or not assistant_message.strip():
            raise ValueError("assistant_message must be a non-empty string")

        messages = [
            MemoryMessage.create("user", user_message).to_dict(),
            MemoryMessage.create(
                "assistant",
                assistant_message,
                metadata=assistant_metadata,
            ).to_dict(),
        ]
        if idempotency_key is None:
            result = await self._adapter.save_messages(session_id, messages)
        else:
            result = await self._adapter.save_messages(
                session_id,
                messages,
                idempotency_key=idempotency_key,
            )
        return True if result is None else bool(result)

    async def clear_session(self, session_id: str) -> bool:
        """Delete a session, returning whether it existed."""
        self._validate_session_id(session_id)
        return await self._adapter.clear_session(session_id)

    async def get_session_ttl(self, session_id: str) -> int:
        """Return remaining seconds, using Redis TTL conventions (-2 = absent)."""
        self._validate_session_id(session_id)
        return await self._adapter.get_session_ttl(session_id)

    async def health(self) -> Dict[str, str]:
        """Report whether Redis is available."""
        return await self._adapter.health()

    async def close(self) -> None:
        """Release backend connections when supported."""
        await self._adapter.close()

    @asynccontextmanager
    async def session_scope(self, session_id: str) -> AsyncIterator[None]:
        """Serialize complete runs for one session in the current event loop."""
        self._validate_session_id(session_id)
        lock_key = (asyncio.get_running_loop(), session_id)
        with _SESSION_RUN_LOCKS_GUARD:
            entry = _SESSION_RUN_LOCKS.setdefault(lock_key, _SessionRunLock())
            entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            with _SESSION_RUN_LOCKS_GUARD:
                entry.users -= 1
                if entry.users == 0:
                    _SESSION_RUN_LOCKS.pop(lock_key, None)

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")


async def create_short_term_memory() -> ShortTermMemory:
    """Create Redis short-term memory and fail fast when it is unavailable."""
    storage_type = os.getenv("SHORT_TERM_MEMORY_BACKEND", "redis").strip().lower()
    if storage_type != "redis":
        raise ValueError(
            f"Unsupported SHORT_TERM_MEMORY_BACKEND={storage_type!r}; "
            "only Redis is supported"
        )
    ttl_seconds = _positive_int_env(
        "SHORT_TERM_MEMORY_TTL",
        DEFAULT_TTL_SECONDS,
    )
    max_messages = _positive_int_env(
        "SHORT_TERM_MEMORY_MAX_MESSAGES",
        DEFAULT_MAX_MESSAGES,
    )

    redis_failure: Optional[Exception] = None
    memory: Optional[ShortTermMemory] = None
    try:
        memory = ShortTermMemory(
            redis_url=os.getenv("REDIS_URL", DEFAULT_REDIS_URL),
            ttl_seconds=ttl_seconds,
            max_messages=max_messages,
        )
        if (await memory.health())["status"] == "ok":
            return memory
        redis_failure = ShortTermMemoryUnavailable(
            "Redis health check reported a degraded status"
        )
    except Exception as exc:
        redis_failure = exc
    finally:
        if memory is not None and redis_failure is not None:
            try:
                await memory.close()
            except Exception as close_error:
                logger.warning(
                    f"Failed to close unavailable Redis memory: {close_error}"
                )

    raise ShortTermMemoryUnavailable(
        "Redis short-term memory is required but unavailable"
    ) from redis_failure


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(f"Invalid {name}={raw_value!r}; using {default}")
        return default
    if value <= 0:
        logger.warning(f"{name} must be positive; using {default}")
        return default
    return value
