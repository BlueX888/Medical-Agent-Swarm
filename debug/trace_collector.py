"""Lightweight in-memory collector for structured agent debug traces."""
from __future__ import annotations

import copy
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from threading import RLock
from typing import Any, Dict, List, Optional

from .models import DebugEvent, DebugRun


class DebugTraceCollector:
    """Collect structured events for a single request run."""

    def __init__(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.run = DebugRun(
            run_id=run_id or str(uuid.uuid4()),
            session_id=session_id,
            question=question,
            context=self._safe_value(context or {}),
            metadata=self._safe_value(metadata or {}),
        )
        self._events: List[DebugEvent] = []
        self._sequence = 0
        self._lock = RLock()

    @property
    def run_id(self) -> str:
        return self.run.run_id

    def update_run(self, **updates: Any) -> None:
        with self._lock:
            for key, value in updates.items():
                if hasattr(self.run, key):
                    setattr(self.run, key, self._safe_value(value))

    def finish_success(
        self,
        result_json: Dict[str, Any],
        route: Optional[str] = None,
        final_answer: Optional[str] = None,
        timeout: bool = False,
    ) -> None:
        with self._lock:
            if self.run.ended_at and self.run.status in {"failed", "success", "timeout"}:
                return
            self.run.ended_at = datetime.now()
            self.run.status = "timeout" if timeout else "success"
            self.run.route = route or self.run.route
            self.run.result_json = self._safe_value(result_json)
            self.run.final_answer = final_answer if final_answer is not None else (
                result_json.get("answer", "") if isinstance(result_json, dict) else ""
            )

    def finish_failed(self, error: Exception | str) -> None:
        with self._lock:
            if self.run.ended_at and self.run.status in {"failed", "success", "timeout"}:
                return
            message = str(error)
            self.run.ended_at = datetime.now()
            self.run.status = "failed"
            self.run.result_json = {"error": message}
            self.record_event(
                stage="agent_loop",
                status="failed",
                error=message,
                output={"error": message},
                name="run_failed",
            )

    def record_event(
        self,
        stage: str,
        *,
        agent_id: Optional[str] = None,
        skill_name: Optional[str] = None,
        input: Any = None,
        output: Any = None,
        duration_ms: Optional[float] = None,
        status: str = "success",
        error: Optional[str] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DebugEvent:
        with self._lock:
            self._sequence += 1
            event = DebugEvent(
                event_id=str(uuid.uuid4()),
                sequence=self._sequence,
                timestamp=datetime.now(),
                stage=stage,
                name=name,
                agent_id=agent_id,
                skill_name=skill_name,
                input=self._safe_value(input),
                output=self._safe_value(output),
                metadata=self._safe_value(metadata or {}),
                duration_ms=round(duration_ms, 2) if duration_ms is not None else None,
                status=status,
                error=error,
            )
            self._events.append(event)
        return event

    def time_event(
        self,
        stage: str,
        *,
        agent_id: Optional[str] = None,
        skill_name: Optional[str] = None,
        input: Any = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "DebugEventTimer":
        return DebugEventTimer(
            collector=self,
            stage=stage,
            agent_id=agent_id,
            skill_name=skill_name,
            input=input,
            name=name,
            metadata=metadata,
        )

    def get_run(self) -> DebugRun:
        with self._lock:
            return copy.deepcopy(self.run)

    def get_events(self) -> List[DebugEvent]:
        with self._lock:
            return copy.deepcopy(self._events)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run": self.get_run().to_dict(),
            "events": [event.to_dict() for event in self.get_events()],
        }

    def _safe_value(self, value: Any, *, depth: int = 0) -> Any:
        if depth > 5:
            return self._truncate_text(repr(value))

        if value is None or isinstance(value, (bool, int, float, str)):
            return self._truncate_text(value) if isinstance(value, str) else value

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Enum):
            return value.value

        if is_dataclass(value):
            return self._safe_value(asdict(value), depth=depth + 1)

        if isinstance(value, dict):
            safe: Dict[str, Any] = {}
            for key, item in list(value.items())[:200]:
                key_text = str(key)
                if key_text == "debug_collector":
                    continue
                if self._is_sensitive_key(key_text):
                    safe[key_text] = "[redacted]"
                    continue
                safe[key_text] = self._safe_value(item, depth=depth + 1)
            return safe

        if isinstance(value, (list, tuple, set)):
            return [self._safe_value(item, depth=depth + 1) for item in list(value)[:200]]

        if hasattr(value, "to_dict") and callable(value.to_dict):
            try:
                return self._safe_value(value.to_dict(), depth=depth + 1)
            except Exception:
                pass

        if hasattr(value, "get_summary") and callable(value.get_summary):
            try:
                return self._safe_value(value.get_summary(), depth=depth + 1)
            except Exception:
                pass

        if hasattr(value, "__dict__"):
            public_values = {
                key: item
                for key, item in vars(value).items()
                if not key.startswith("_")
                and key != "debug_collector"
                and not self._is_sensitive_key(key)
            }
            if public_values:
                return self._safe_value(public_values, depth=depth + 1)

        return self._truncate_text(repr(value))

    def _truncate_text(self, text: str, limit: int = 20000) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}... [truncated {len(text) - limit} chars]"

    def _is_sensitive_key(self, key: str) -> bool:
        normalized = key.lower().replace("-", "_")
        sensitive_keys = {
            "api_key",
            "apikey",
            "authorization",
            "x_api_key",
            "password",
            "passwd",
            "secret",
            "secret_key",
            "access_key",
            "private_key",
            "credentials",
        }
        return normalized in sensitive_keys or normalized.endswith("_secret")


class DebugEventTimer:
    """Context manager helper for timed DebugEvent emission."""

    def __init__(
        self,
        collector: DebugTraceCollector,
        stage: str,
        *,
        agent_id: Optional[str] = None,
        skill_name: Optional[str] = None,
        input: Any = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.collector = collector
        self.stage = stage
        self.agent_id = agent_id
        self.skill_name = skill_name
        self.input = input
        self.name = name
        self.metadata = metadata or {}
        self.started = time.perf_counter()

    def finish(
        self,
        *,
        output: Any = None,
        status: str = "success",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DebugEvent:
        duration_ms = (time.perf_counter() - self.started) * 1000
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(metadata)
        return self.collector.record_event(
            self.stage,
            agent_id=self.agent_id,
            skill_name=self.skill_name,
            input=self.input,
            output=output,
            duration_ms=duration_ms,
            status=status,
            error=error,
            name=self.name,
            metadata=merged_metadata,
        )


class InMemoryTraceStore:
    """Thread-safe process-local debug run store for the local API service."""

    def __init__(self):
        self._runs: Dict[str, DebugTraceCollector] = {}
        self._lock = RLock()

    def add(self, collector: DebugTraceCollector) -> None:
        with self._lock:
            self._runs[collector.run_id] = collector

    def get(self, run_id: str) -> Optional[DebugTraceCollector]:
        with self._lock:
            return self._runs.get(run_id)

    def list(self, limit: int = 50) -> List[DebugRun]:
        with self._lock:
            runs = [collector.get_run() for collector in self._runs.values()]
        runs.sort(key=lambda run: run.started_at, reverse=True)
        return runs[:limit]
