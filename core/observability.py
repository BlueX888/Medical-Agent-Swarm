"""Optional LangSmith observability helpers.

The project handles medical questions, so tracing is opt-in and text payloads
are redacted by default before they leave the process.
"""
from __future__ import annotations

import hashlib
import os
from typing import Any, Awaitable, Callable, Dict, Optional

from loguru import logger


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "secret_key",
    "access_key",
    "private_key",
    "credentials",
}

MEDICAL_TEXT_KEYS = {
    "answer",
    "arguments",
    "content",
    "context",
    "description",
    "enhanced_context",
    "final_answer",
    "historical_cases",
    "input",
    "messages",
    "output",
    "question",
    "recent_history",
    "result",
    "summary",
}

_warned_missing_langsmith = False
_warned_missing_api_key = False


def env_flag(name: str, default: bool = False) -> bool:
    """Read a permissive boolean environment variable."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def langsmith_enabled() -> bool:
    """Return whether LangSmith tracing should be attempted."""
    global _warned_missing_api_key
    if not env_flag("LANGSMITH_TRACING", default=False):
        return False
    if os.getenv("LANGSMITH_API_KEY"):
        return True
    if not _warned_missing_api_key:
        logger.warning(
            "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is not set; "
            "LangSmith tracing will be skipped."
        )
        _warned_missing_api_key = True
    return False


def redact_medical_text_enabled() -> bool:
    """Default to redacting medical text unless the operator opts out."""
    return env_flag("LANGSMITH_REDACT_MEDICAL_TEXT", default=True)


def sanitize_for_langsmith(value: Any, *, parent_key: str = "", depth: int = 0) -> Any:
    """Return a JSON-like value safe to send to LangSmith."""
    if depth > 6:
        return _redacted_text(repr(value))

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        if _should_redact_text(parent_key):
            return _redacted_text(value)
        return _truncate(value)

    if isinstance(value, dict):
        safe: Dict[str, Any] = {}
        for key, item in list(value.items())[:200]:
            key_text = str(key)
            normalized = _normalize_key(key_text)
            if normalized in SENSITIVE_KEYS or normalized.endswith("_secret"):
                safe[key_text] = "[redacted]"
            else:
                safe[key_text] = sanitize_for_langsmith(
                    item,
                    parent_key=normalized,
                    depth=depth + 1,
                )
        return safe

    if isinstance(value, (list, tuple, set)):
        return [
            sanitize_for_langsmith(item, parent_key=parent_key, depth=depth + 1)
            for item in list(value)[:200]
        ]

    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return sanitize_for_langsmith(
                value.model_dump(),
                parent_key=parent_key,
                depth=depth + 1,
            )
        except Exception:
            pass

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return sanitize_for_langsmith(
                value.to_dict(),
                parent_key=parent_key,
                depth=depth + 1,
            )
        except Exception:
            pass

    return _truncate(repr(value))


async def trace_async(
    *,
    name: str,
    run_type: str,
    func: Callable[[], Awaitable[Any]],
    inputs: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[list[str]] = None,
    output_mapper: Optional[Callable[[Any], Any]] = None,
) -> Any:
    """Run an async callable inside a LangSmith span when tracing is enabled."""
    if not langsmith_enabled():
        return await func()

    traceable = _load_traceable()
    if traceable is None:
        return await func()

    safe_inputs = sanitize_for_langsmith(inputs or {})
    safe_metadata = sanitize_for_langsmith(metadata or {})
    result_box: Dict[str, Any] = {}

    @traceable(
        name=name,
        run_type=run_type,
        metadata=safe_metadata,
        tags=tags or ["medical-agent-swarm"],
    )
    async def _traced(payload: Dict[str, Any]) -> Any:
        result = await func()
        result_box["value"] = result
        mapped = output_mapper(result) if output_mapper else result
        return sanitize_for_langsmith(mapped)

    await _traced(safe_inputs)
    return result_box.get("value")


def _load_traceable() -> Optional[Callable[..., Any]]:
    global _warned_missing_langsmith
    try:
        from langsmith import traceable

        return traceable
    except ImportError:
        if not _warned_missing_langsmith:
            logger.warning(
                "LANGSMITH_TRACING is enabled but langsmith is not installed. "
                "Install dependencies with `pip install -r requirements.txt`."
            )
            _warned_missing_langsmith = True
        return None


def _normalize_key(key: str) -> str:
    return key.lower().replace("-", "_")


def _should_redact_text(parent_key: str) -> bool:
    if not redact_medical_text_enabled():
        return False
    return _normalize_key(parent_key) in MEDICAL_TEXT_KEYS


def _redacted_text(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return f"[redacted text len={len(text)} sha256={digest}]"


def _truncate(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"
