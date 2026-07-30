"""Optional LangSmith observability helpers.

The project handles medical questions, so tracing is opt-in and text payloads
are redacted by default before they leave the process.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
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
    if not langsmith_enabled():
        return await func()

    traceable = _load_traceable()
    if traceable is None:
        return await func()

    supplied_metadata = dict(metadata or {})
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
    result_box: Dict[str, Any] = {}
    metrics_token: Optional[Token] = None
    if name == "medical_swarm_request" and _run_metrics.get() is None:
        metrics_token = _run_metrics.set(_new_run_metrics(common_metadata))

    async def _trace_body(payload: Dict[str, Any]) -> Any:
        try:
            result = await func()
            result_box["value"] = result
            try:
                mapped = output_mapper(result) if output_mapper else result
            except Exception as mapper_error:
                logger.exception("Observability output mapper failed: {}", mapper_error)
                mapped = {
                    "status": "degraded",
                    **normalize_error(mapper_error),
                }
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
                raise _SafeTraceFailure(
                    f"observability outcome: {mapped_status}"
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
            }
            if run_type == "tool":
                mapped_error["tool.outcome"] = (
                    "timeout" if failure_status == "timeout" else "error"
                )
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
            if name != "medical_swarm_request":
                _record_span_metrics(name, run_type, safe_metadata, mapped_error)
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
        return await func()

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
    if name.startswith("agent."):
        agent_id = metadata.get("agent_id") or name.removeprefix("agent.")
        if agent_id:
            metrics["agent_ids"].add(str(agent_id))
    if run_type == "llm":
        metrics["llm_call_count"] += 1
        metrics["input_tokens"] += _safe_integer(values.get("llm.input_tokens"))
        metrics["output_tokens"] += _safe_integer(values.get("llm.output_tokens"))
        metrics["total_tokens"] += _safe_integer(values.get("llm.total_tokens"))
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
