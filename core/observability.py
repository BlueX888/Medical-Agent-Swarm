"""Optional LangSmith observability helpers.

The project handles medical questions, so tracing is opt-in and text payloads
are redacted by default before they leave the process.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from contextvars import ContextVar, Token
from typing import Any, Awaitable, Callable, Dict, Iterable, Mapping, Optional

from loguru import logger


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}
TELEMETRY_SCHEMA_VERSION = "1.0"
APP_NAME = "medical-agent-swarm"

VALID_ENVIRONMENTS = {"local", "test", "staging", "production"}
VALID_ENTRYPOINTS = {"api", "cli", "python", "benchmark"}
VALID_ROUTES = {"single_agent", "swarm", "fallback", "unknown"}
VALID_STATUSES = {"success", "failed", "timeout", "blocked", "degraded"}

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
    "dose",
    "evidence",
    "enhanced_context",
    "error",
    "final_answer",
    "history",
    "historical_cases",
    "knowledge_bundle",
    "knowledge_context",
    "id_card",
    "input",
    "medical_history",
    "medication",
    "messages",
    "name",
    "output",
    "patient_id",
    "phone",
    "prompt",
    "question",
    "recent_history",
    "result",
    "symptoms",
    "summary",
    "text",
}

_warned_missing_langsmith = False
_warned_missing_api_key = False
_run_metrics: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "observability_run_metrics",
    default=None,
)


class _SafeTraceFailure(RuntimeError):
    """Internal sentinel that marks a remote span failed without PHI."""


def log_observability_event(
    event: str,
    *,
    name: str,
    run_type: str,
    run_id: Optional[str] = None,
    status: Optional[str] = None,
    duration_ms: Optional[float] = None,
    retry_count: Optional[int] = None,
    error_type: Optional[str] = None,
    error_code: Optional[str] = None,
    **fields: Any,
) -> None:
    """Emit one structured, correlation-friendly observability log event."""
    payload: Dict[str, Any] = {
        "observability.event": event,
        "span.name": name,
        "run_type": run_type,
    }
    optional = {
        "run_id": run_id,
        "status": status,
        "duration_ms": duration_ms,
        "retry_count": retry_count,
        "error.type": error_type,
        "error.code": error_code,
    }
    payload.update({key: value for key, value in optional.items() if value is not None})
    payload.update({key: value for key, value in fields.items() if value is not None})
    safe_payload = sanitize_for_langsmith(payload)
    level = "warning" if event.endswith("failed") or event.endswith("error") else "info"
    bound = logger.bind(**safe_payload)
    getattr(bound, level)("observability.{}", event)


def session_reference(session_id: Optional[str]) -> Optional[str]:
    """Return an irreversible, keyed reference for a session identifier."""
    key = os.getenv("OBSERVABILITY_HASH_KEY")
    if not session_id or not key:
        return None
    digest = hmac.new(
        key.encode("utf-8"),
        session_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"hmac-sha256:{digest}"


def build_observability_metadata(
    *,
    run_id: Optional[str] = None,
    session_id: Optional[str] = None,
    route: Optional[str] = None,
    status: Optional[str] = None,
    agent_id: Optional[str] = None,
    graph_node: Optional[str] = None,
    entrypoint: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the common, low-cardinality metadata contract for every span."""
    environment = os.getenv("OBSERVABILITY_ENVIRONMENT", "local").strip().lower()
    if environment not in VALID_ENVIRONMENTS:
        environment = "local"
    resolved_entrypoint = (
        entrypoint or os.getenv("OBSERVABILITY_ENTRYPOINT", "python")
    ).strip().lower()
    if resolved_entrypoint not in VALID_ENTRYPOINTS:
        resolved_entrypoint = "python"

    metadata: Dict[str, Any] = {
        "app.name": APP_NAME,
        "app.version": (
            os.getenv("APP_VERSION")
            or os.getenv("GIT_SHA")
            or os.getenv("COMMIT_SHA")
            or "unknown"
        ),
        "deployment.environment": environment,
        "telemetry.schema_version": TELEMETRY_SCHEMA_VERSION,
        "entrypoint": resolved_entrypoint,
    }
    if run_id:
        metadata["run_id"] = str(run_id)
    session_ref = session_reference(session_id)
    if session_ref:
        metadata["session_ref"] = session_ref
    if route:
        metadata["route"] = route if route in VALID_ROUTES else "unknown"
    if status:
        metadata["status"] = status if status in VALID_STATUSES else "failed"
    if agent_id:
        metadata["agent_id"] = str(agent_id)
    if graph_node:
        metadata["graph_node"] = str(graph_node)
    if extra:
        metadata.update(extra)
    return metadata


def normalize_error(error: BaseException) -> Dict[str, str]:
    """Map an exception to safe, stable fields without exporting its message."""
    if isinstance(error, TimeoutError):
        code = "timeout"
    elif isinstance(error, PermissionError):
        code = "permission_denied"
    elif isinstance(error, (ValueError, TypeError)):
        code = "validation_error"
    elif isinstance(error, KeyError):
        code = "not_found"
    else:
        code = "internal_error"
    return {"error.type": type(error).__name__, "error.code": code}


def summarize_tool_input(
    tool_name: str,
    arguments: Optional[Mapping[str, Any]],
    *,
    allowed_keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Describe a tool request without retaining argument values."""
    allowed = set(allowed_keys) if allowed_keys is not None else None
    keys = sorted(
        {
            str(key) if allowed is None or str(key) in allowed else "unknown"
            for key in (arguments or {}).keys()
        }
    )
    return {
        "tool_name": tool_name,
        "argument_keys": keys,
        "argument_count": len(keys),
    }


def summarize_tool_output(result: Any) -> Dict[str, Any]:
    """Describe a tool result without retaining its contents."""
    kind = _result_kind(result)
    success = not (
        isinstance(result, Mapping)
        and (
            result.get("success") is False
            or bool(result.get("error"))
            or bool(result.get("error_code"))
        )
    )
    return {
        "success": success,
        "result_kind": kind,
        "result_size": _serialized_size(result),
    }


def classify_tool_result(result: Any) -> str:
    """Classify a completed tool result using the stable MVP vocabulary."""
    if result is None or result == "" or result == [] or result == {}:
        return "empty"
    if isinstance(result, Mapping) and (
        result.get("success") is False
        or bool(result.get("error"))
        or bool(result.get("error_code"))
    ):
        return "error"
    return "success"


def summarize_tool_trace_output(
    result: Any,
    *,
    outcome: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the complete safe Tool Span output and final status."""
    resolved_outcome = outcome or classify_tool_result(result)
    status = {
        "blocked": "blocked",
        "timeout": "timeout",
        "error": "failed",
    }.get(resolved_outcome, "success")
    return {
        **summarize_tool_output(result),
        "status": status,
        "tool.outcome": resolved_outcome,
    }


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
    span_started = time.perf_counter()
    supplied_metadata = dict(metadata or {})
    metadata_source = metadata if isinstance(metadata, dict) else None

    async def _run_untraced() -> Any:
        log_observability_event(
            "span.started",
            name=name,
            run_type=run_type,
            run_id=supplied_metadata.get("run_id"),
            retry_count=supplied_metadata.get(
                "llm.retry_count", supplied_metadata.get("tool.retry_count")
            ),
        )
        try:
            result = await func()
        except Exception as error:
            normalized = normalize_error(error)
            if name == "medical_swarm_request":
                metrics = _run_metrics.get()
                if metrics is not None:
                    metrics["exception_count"] += 1
                    error_type = normalized["error.type"]
                    metrics["exception_types"][error_type] = (
                        metrics["exception_types"].get(error_type, 0) + 1
                    )
            else:
                duration_ms = round(
                    (time.perf_counter() - span_started) * 1000,
                    3,
                )
                _record_span_metrics(
                    name,
                    run_type,
                    supplied_metadata,
                    {"status": "failed", "duration_ms": duration_ms, **normalized},
                )
            log_observability_event(
                "span.failed",
                name=name,
                run_type=run_type,
                run_id=supplied_metadata.get("run_id"),
                status=(
                    "timeout"
                    if normalized["error.code"] == "timeout"
                    else "failed"
                ),
                duration_ms=round(
                    (time.perf_counter() - span_started) * 1000,
                    3,
                ),
                retry_count=supplied_metadata.get(
                    "llm.retry_count", supplied_metadata.get("tool.retry_count")
                ),
                error_type=normalized["error.type"],
                error_code=normalized["error.code"],
            )
            raise
        try:
            mapped = output_mapper(result) if output_mapper else result
        except Exception as mapper_error:
            mapped = {
                "status": "degraded",
                **normalize_error(mapper_error),
            }
        if isinstance(mapped, Mapping):
            mapped = dict(mapped)
            mapped.setdefault(
                "duration_ms",
                round((time.perf_counter() - span_started) * 1000, 3),
            )
        mapped_values = mapped if isinstance(mapped, Mapping) else {}
        if name == "medical_swarm_request":
            mapped = _augment_root_summary(mapped)
            mapped_values = mapped
        else:
            _record_span_metrics(name, run_type, supplied_metadata, mapped)
        summary_fields = (
            {
                key: mapped_values.get(key)
                for key in (
                    "agent_count",
                    "llm_call_count",
                    "input_tokens",
                    "output_tokens",
                    "total_tokens",
                    "tool_call_count",
                    "tool_duration_ms_total",
                    "tool_duration_ms_avg",
                    "llm_duration_ms_total",
                    "llm_duration_ms_avg",
                    "llm_ttft_ms_count",
                    "llm_ttft_ms_avg",
                    "llm_ttft_ms_min",
                    "llm_ttft_ms_max",
                    "retry_success_count",
                    "retry_exhausted_count",
                    "exception_count",
                    "exception_total_count",
                    "exception_types",
                    "exception_types_all",
                )
                if key in mapped_values
            }
            if name == "medical_swarm_request"
            else {}
        )
        mapped_status = mapped_values.get("status")
        if mapped_status in {"failed", "timeout"}:
            log_observability_event(
                "span.failed",
                name=name,
                run_type=run_type,
                run_id=supplied_metadata.get("run_id"),
                status=mapped_status,
                duration_ms=round((time.perf_counter() - span_started) * 1000, 3),
                retry_count=(
                    mapped_values.get("retry_count")
                    if name == "medical_swarm_request"
                    else supplied_metadata.get(
                        "llm.retry_count", supplied_metadata.get("tool.retry_count")
                    )
                ),
                error_type=mapped_values.get("error.type"),
                error_code=mapped_values.get("error.code"),
                **summary_fields,
            )
            return result
        log_observability_event(
            "span.completed",
            name=name,
            run_type=run_type,
            run_id=supplied_metadata.get("run_id"),
            status=mapped_values.get("status") or "success",
            duration_ms=round((time.perf_counter() - span_started) * 1000, 3),
            retry_count=(
                mapped_values.get("retry_count")
                if name == "medical_swarm_request"
                else supplied_metadata.get(
                    "llm.retry_count", supplied_metadata.get("tool.retry_count")
                )
            ),
            **summary_fields,
        )
        return result

    async def _run_untraced_with_metrics() -> Any:
        metrics_token: Optional[Token] = None
        active_metrics = _run_metrics.get()
        if name == "medical_swarm_request" and _run_metrics.get() is None:
            supplied_metadata.setdefault("run_id", str(uuid.uuid4()))
            metrics_token = _run_metrics.set(
                _new_run_metrics(supplied_metadata)
            )
        elif active_metrics is not None:
            supplied_metadata.setdefault("run_id", active_metrics.get("run_id"))
        try:
            return await _run_untraced()
        finally:
            if metrics_token is not None:
                _run_metrics.reset(metrics_token)

    if not langsmith_enabled():
        return await _run_untraced_with_metrics()

    traceable = _load_traceable()
    if traceable is None:
        return await _run_untraced_with_metrics()

    active_metrics = _run_metrics.get()
    if name == "medical_swarm_request":
        supplied_metadata.setdefault("run_id", str(uuid.uuid4()))
    elif active_metrics is not None:
        supplied_metadata.setdefault("run_id", active_metrics.get("run_id"))
        if (
            not supplied_metadata.get("session_id")
            and active_metrics.get("session_ref")
        ):
            supplied_metadata.setdefault(
                "session_ref",
                active_metrics["session_ref"],
            )
    safe_inputs = sanitize_for_langsmith(inputs or {})
    common_metadata = build_observability_metadata(
        run_id=supplied_metadata.pop("run_id", None),
        session_id=supplied_metadata.pop("session_id", None),
        route=supplied_metadata.pop("route", None),
        status=supplied_metadata.pop("status", None),
        agent_id=supplied_metadata.pop("agent_id", None),
        graph_node=supplied_metadata.pop("graph_node", None),
        entrypoint=supplied_metadata.pop("entrypoint", None),
        extra=supplied_metadata,
    )
    safe_metadata = sanitize_for_langsmith(common_metadata)
    log_observability_event(
        "span.started",
        name=name,
        run_type=run_type,
        run_id=common_metadata.get("run_id"),
        retry_count=safe_metadata.get(
            "llm.retry_count", safe_metadata.get("tool.retry_count")
        ),
        **{
            "deployment.environment": safe_metadata.get(
                "deployment.environment"
            ),
            "app.version": safe_metadata.get("app.version"),
        },
    )
    result_box: Dict[str, Any] = {}
    metrics_token: Optional[Token] = None
    if name == "medical_swarm_request" and _run_metrics.get() is None:
        metrics_token = _run_metrics.set(_new_run_metrics(common_metadata))

    async def _trace_body(payload: Dict[str, Any]) -> Any:
        try:
            result = await func()
            result_box["value"] = result
            duration_ms = round((time.perf_counter() - span_started) * 1000, 3)
            try:
                mapped = output_mapper(result) if output_mapper else result
            except Exception as mapper_error:
                logger.exception("Observability output mapper failed: {}", mapper_error)
                mapped = {
                    "status": "degraded",
                    **normalize_error(mapper_error),
                }
            if isinstance(mapped, Mapping):
                mapped = dict(mapped)
                mapped.setdefault("duration_ms", duration_ms)
            if name == "medical_swarm_request":
                mapped = _augment_root_summary(mapped)
            else:
                _record_span_metrics(name, run_type, safe_metadata, mapped)
            _update_current_span_metadata(mapped)
            mapped_status = (
                mapped.get("status")
                if isinstance(mapped, Mapping)
                else None
            )
            if mapped_status in {"failed", "timeout"}:
                log_observability_event(
                    "span.failed",
                    name=name,
                    run_type=run_type,
                    run_id=safe_metadata.get("run_id"),
                    status=mapped_status,
                    duration_ms=duration_ms,
                    retry_count=safe_metadata.get(
                        "llm.retry_count", safe_metadata.get("tool.retry_count")
                    ),
                    error_type=mapped.get("error.type")
                    if isinstance(mapped, Mapping)
                    else None,
                    error_code=mapped.get("error.code")
                    if isinstance(mapped, Mapping)
                    else None,
                )
                raise _SafeTraceFailure(
                    f"observability outcome: {mapped_status}"
                )
            log_observability_event(
                "span.completed",
                name=name,
                run_type=run_type,
                run_id=safe_metadata.get("run_id"),
                status=(mapped.get("status") if isinstance(mapped, Mapping) else "success"),
                duration_ms=duration_ms,
                retry_count=safe_metadata.get(
                    "llm.retry_count", safe_metadata.get("tool.retry_count")
                ),
            )
            return sanitize_for_langsmith(mapped)
        except Exception as error:
            if isinstance(error, _SafeTraceFailure):
                raise
            result_box["error"] = error
            normalized_error = normalize_error(error)
            failure_status = (
                "timeout"
                if normalized_error["error.code"] == "timeout"
                else "failed"
            )
            mapped_error = {
                "status": failure_status,
                **normalized_error,
                "duration_ms": round(
                    (time.perf_counter() - span_started) * 1000,
                    3,
                ),
            }
            if run_type == "tool":
                mapped_error["tool.outcome"] = (
                    "timeout" if failure_status == "timeout" else "error"
                )
                if metadata_source:
                    if "tool.retry_count" in metadata_source:
                        mapped_error["tool.retry_count"] = _safe_integer(
                            metadata_source["tool.retry_count"]
                        )
                    if metadata_source.get("tool.retry_exhausted"):
                        mapped_error["tool.retry_exhausted"] = True
            elif run_type == "llm":
                mapped_error["llm.outcome"] = (
                    "timeout" if failure_status == "timeout" else "error"
                )
            elif name.startswith("safety."):
                mapped_error.update(
                    {
                        "safety.executed": False,
                        "safety.passed": False,
                        "safety.outcome": "error",
                    }
                )
            if name == "medical_swarm_request":
                metrics = _run_metrics.get()
                if metrics is not None:
                    metrics["exception_count"] += 1
                    error_type = normalized_error["error.type"]
                    metrics["exception_types"][error_type] = (
                        metrics["exception_types"].get(error_type, 0) + 1
                    )
                mapped_error = _augment_root_summary(mapped_error)
            else:
                _record_span_metrics(name, run_type, safe_metadata, mapped_error)
            log_observability_event(
                "span.failed",
                name=name,
                run_type=run_type,
                run_id=safe_metadata.get("run_id"),
                status=failure_status,
                duration_ms=mapped_error["duration_ms"],
                retry_count=safe_metadata.get(
                    "llm.retry_count", safe_metadata.get("tool.retry_count")
                ),
                error_type=normalized_error["error.type"],
                error_code=normalized_error["error.code"],
            )
            _update_current_span_metadata(mapped_error)
            raise _SafeTraceFailure(
                f"observability outcome: {failure_status}"
            )

    try:
        decorator = traceable(
            name=name,
            run_type=run_type,
            metadata=safe_metadata,
            tags=tags or ["medical-agent-swarm"],
        )
        traced_call = decorator(_trace_body)
    except Exception as setup_error:
        logger.warning("Observability span setup failed: {}", setup_error)
        if metrics_token is not None:
            _run_metrics.reset(metrics_token)
        return await _run_untraced_with_metrics()

    try:
        await traced_call(safe_inputs)
    except Exception as exporter_error:
        if "error" in result_box:
            if metrics_token is not None:
                _run_metrics.reset(metrics_token)
            raise result_box["error"]
        if "value" in result_box:
            if not isinstance(exporter_error, _SafeTraceFailure):
                logger.warning(
                    "Observability exporter failed after request: {}",
                    exporter_error,
                )
            if metrics_token is not None:
                _run_metrics.reset(metrics_token)
            return result_box["value"]
        logger.warning("Observability exporter failed before request: {}", exporter_error)
        if metrics_token is not None:
            _run_metrics.reset(metrics_token)
        return await func()

    if "error" in result_box:
        if metrics_token is not None:
            _run_metrics.reset(metrics_token)
        raise result_box["error"]
    result = result_box.get("value")
    if metrics_token is not None:
        _run_metrics.reset(metrics_token)
    return result


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
    return f"[redacted text len={len(text)}]"


def _truncate(text: str, limit: int = 20000) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... [truncated {len(text) - limit} chars]"


def _result_kind(result: Any) -> str:
    if result is None or result == "" or result == [] or result == {}:
        return "empty"
    if isinstance(result, Mapping):
        return "object"
    if isinstance(result, (list, tuple, set)):
        return "list"
    if isinstance(result, str):
        return "text"
    return "unknown"


def _serialized_size(value: Any) -> int:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            default=lambda item: repr(item),
            separators=(",", ":"),
        )
    except Exception:
        serialized = repr(value)
    return len(serialized.encode("utf-8", errors="ignore"))


def _new_run_metrics(
    common_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "run_id": (common_metadata or {}).get("run_id"),
        "session_ref": (common_metadata or {}).get("session_ref"),
        "agent_ids": set(),
        "llm_call_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "tool_call_count": 0,
        "tool_success_count": 0,
        "tool_blocked": 0,
        "tool_failed": 0,
        "tool_duration_ms_total": 0.0,
        "tool_duration_ms_count": 0,
        "llm_duration_ms_total": 0.0,
        "llm_duration_ms_count": 0,
        "llm_ttft_ms_total": 0.0,
        "llm_ttft_ms_count": 0,
        "llm_ttft_ms_min": None,
        "llm_ttft_ms_max": None,
        "retry_count": 0,
        "retry_success_count": 0,
        "retry_exhausted_count": 0,
        "agent_retry_count": 0,
        "agent_exception_count": 0,
        "agent_exception_types": {},
        "exception_count": 0,
        "exception_types": {},
        "safety_checked": False,
        "safety_passed": False,
        "safety_error": False,
    }


def _record_span_metrics(
    name: str,
    run_type: str,
    metadata: Mapping[str, Any],
    output: Any,
) -> None:
    metrics = _run_metrics.get()
    if metrics is None:
        return
    values = output if isinstance(output, Mapping) else {}
    retry_count = _safe_integer(
        values.get(
            "retry_count",
            values.get(
                "llm.retry_count",
                values.get(
                    "tool.retry_count",
                    metadata.get("llm.retry_count", metadata.get("tool.retry_count", 0)),
                ),
            ),
        )
    )
    metric_retry_count = retry_count if run_type in {"llm", "tool"} else 0
    if metric_retry_count > 0:
        metrics["retry_count"] += 1
        if values.get("tool.retry_exhausted") or values.get("status") in {"failed", "timeout"} or values.get(
            "llm.outcome"
        ) in {"error", "timeout"} or values.get("tool.outcome") in {
            "error",
            "timeout",
        }:
            metrics["retry_exhausted_count"] += 1
        else:
            metrics["retry_success_count"] += 1

    if values.get("error.type"):
        metrics["exception_count"] += 1
        error_type = str(values["error.type"])
        metrics["exception_types"][error_type] = (
            metrics["exception_types"].get(error_type, 0) + 1
        )
    if values.get("safety.outcome") == "error" and not values.get("error.type"):
        metrics["exception_count"] += 1
        metrics["exception_types"]["SafetyCheckError"] = (
            metrics["exception_types"].get("SafetyCheckError", 0) + 1
        )

    duration_ms = values.get("duration_ms")
    if isinstance(duration_ms, (int, float)):
        if run_type == "llm":
            metrics["llm_duration_ms_total"] += float(duration_ms)
            metrics["llm_duration_ms_count"] += 1
        elif run_type == "tool":
            metrics["tool_duration_ms_total"] += float(duration_ms)
            metrics["tool_duration_ms_count"] += 1

    if name.startswith("agent."):
        agent_id = metadata.get("agent_id") or name.removeprefix("agent.")
        if agent_id:
            metrics["agent_ids"].add(str(agent_id))
        agent_retry_count = _safe_integer(values.get("agent.retry_count"))
        metrics["agent_retry_count"] += agent_retry_count
        agent_exception_count = _safe_integer(values.get("agent.exception_count"))
        metrics["agent_exception_count"] += agent_exception_count
        for error_type, count in (values.get("agent.exception_types") or {}).items():
            error_type_text = str(error_type)
            metrics["agent_exception_types"][error_type_text] = (
                metrics["agent_exception_types"].get(error_type_text, 0)
                + _safe_integer(count)
            )
    if run_type == "llm":
        metrics["llm_call_count"] += 1
        metrics["input_tokens"] += _safe_integer(values.get("llm.input_tokens"))
        metrics["output_tokens"] += _safe_integer(values.get("llm.output_tokens"))
        metrics["total_tokens"] += _safe_integer(values.get("llm.total_tokens"))
        ttft_ms = values.get("llm.ttft_ms")
        if isinstance(ttft_ms, (int, float)):
            ttft = float(ttft_ms)
            metrics["llm_ttft_ms_total"] += ttft
            metrics["llm_ttft_ms_count"] += 1
            previous_min = metrics["llm_ttft_ms_min"]
            previous_max = metrics["llm_ttft_ms_max"]
            metrics["llm_ttft_ms_min"] = (
                ttft if previous_min is None else min(previous_min, ttft)
            )
            metrics["llm_ttft_ms_max"] = (
                ttft if previous_max is None else max(previous_max, ttft)
            )
    if run_type == "tool":
        metrics["tool_call_count"] += 1
        outcome = values.get("tool.outcome", "error")
        if outcome == "success":
            metrics["tool_success_count"] += 1
        elif outcome == "blocked":
            metrics["tool_blocked"] += 1
        else:
            metrics["tool_failed"] += 1
    if name.startswith("safety."):
        metrics["safety_checked"] = bool(values.get("safety.executed"))
        metrics["safety_passed"] = bool(values.get("safety.passed"))
        metrics["safety_error"] = values.get("safety.outcome") == "error"


def _augment_root_summary(mapped: Any) -> Dict[str, Any]:
    summary = dict(mapped) if isinstance(mapped, Mapping) else {}
    summary.setdefault("route", "unknown")
    metrics = _run_metrics.get() or _new_run_metrics()
    exception_types = dict(metrics["exception_types"])
    for error_type, count in metrics["agent_exception_types"].items():
        exception_types[error_type] = exception_types.get(error_type, 0) + count
    summary.update(
        {
            "agent_count": len(metrics["agent_ids"]),
            "llm_call_count": metrics["llm_call_count"],
            "input_tokens": metrics["input_tokens"],
            "output_tokens": metrics["output_tokens"],
            "total_tokens": metrics["total_tokens"],
            "tool_call_count": metrics["tool_call_count"],
            "tool_success_count": metrics["tool_success_count"],
            "tool_blocked": metrics["tool_blocked"],
            "tool_failed": metrics["tool_failed"],
            "tool_duration_ms_total": round(metrics["tool_duration_ms_total"], 3),
            "tool_duration_ms_avg": round(
                metrics["tool_duration_ms_total"]
                / metrics["tool_duration_ms_count"],
                3,
            )
            if metrics["tool_duration_ms_count"]
            else 0.0,
            "llm_duration_ms_total": round(metrics["llm_duration_ms_total"], 3),
            "llm_duration_ms_avg": round(
                metrics["llm_duration_ms_total"]
                / metrics["llm_duration_ms_count"],
                3,
            )
            if metrics["llm_duration_ms_count"]
            else 0.0,
            "llm_ttft_ms_count": metrics["llm_ttft_ms_count"],
            "llm_ttft_ms_avg": round(
                metrics["llm_ttft_ms_total"] / metrics["llm_ttft_ms_count"],
                3,
            )
            if metrics["llm_ttft_ms_count"]
            else None,
            "llm_ttft_ms_min": metrics["llm_ttft_ms_min"],
            "llm_ttft_ms_max": metrics["llm_ttft_ms_max"],
            "retry_count": metrics["retry_count"],
            "retry_success_count": metrics["retry_success_count"],
            "retry_exhausted_count": metrics["retry_exhausted_count"],
            "agent_retry_count": metrics["agent_retry_count"],
            "agent_exception_count": metrics["agent_exception_count"],
            "agent_exception_types": dict(metrics["agent_exception_types"]),
            "exception_count": metrics["exception_count"],
            "exception_total_count": (
                metrics["exception_count"] + metrics["agent_exception_count"]
            ),
            "exception_types": dict(metrics["exception_types"]),
            "exception_types_all": exception_types,
            "safety_checked": metrics["safety_checked"],
            "safety_passed": metrics["safety_passed"],
            "safety_error": metrics["safety_error"],
        }
    )
    return summary


def _safe_integer(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _update_current_span_metadata(mapped: Any) -> None:
    """Attach result-dependent safe fields to the active LangSmith run."""
    if not isinstance(mapped, Mapping):
        return
    root_summary_keys = {
        "route",
        "agent_count",
        "llm_call_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "tool_call_count",
        "tool_success_count",
        "tool_blocked",
        "tool_failed",
        "duration_ms",
        "tool_duration_ms_total",
        "tool_duration_ms_avg",
        "llm_duration_ms_total",
        "llm_duration_ms_avg",
        "llm_ttft_ms_count",
        "llm_ttft_ms_avg",
        "llm_ttft_ms_min",
        "llm_ttft_ms_max",
        "retry_count",
        "retry_success_count",
        "retry_exhausted_count",
        "agent_retry_count",
        "agent_exception_count",
        "agent_exception_types",
        "exception_count",
        "exception_total_count",
        "exception_types",
        "exception_types_all",
        "safety_checked",
        "safety_passed",
        "safety_error",
        "answer_length",
    }
    dynamic = {
        key: value
        for key, value in mapped.items()
        if key == "status"
        or key in root_summary_keys
        or key.startswith("tool.")
        or key.startswith("llm.")
        or key.startswith("safety.")
        or key.startswith("error.")
    }
    if "tool.outcome" in mapped:
        dynamic.update(
            {
                "tool.result_kind": mapped.get("result_kind", "unknown"),
                "tool.result_size": _safe_integer(mapped.get("result_size")),
            }
        )
    if not dynamic:
        return
    try:
        from langsmith.run_helpers import get_current_run_tree

        run_tree = get_current_run_tree()
        if run_tree is not None:
            run_tree.add_metadata(sanitize_for_langsmith(dynamic))
    except Exception as metadata_error:
        logger.debug("Could not update active trace metadata: {}", metadata_error)
