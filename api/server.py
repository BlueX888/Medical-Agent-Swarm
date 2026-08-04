"""Local FastAPI server for the Agent debug console."""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from core.skill_loader import discover_skills, is_active_skill
from core.audit import open_audit_store
from core.checkpointing import (
    CheckpointSettings,
    CheckpointingDisabledError,
    open_checkpointer,
    open_run_lease,
)
from debug import DebugTraceCollector, InMemoryTraceStore
from memory import (
    LongTermMemory,
    ShortTermMemory,
    ShortTermMemoryError,
    create_short_term_memory,
)
from knowledge import DocumentValidationError, KnowledgeRuntime, create_knowledge_runtime
from swarm import SwarmCoordinator

from .schemas import (
    AgentInfo,
    ConsultationCreateRequest,
    ConsultationCreateResponse,
    ConsultationSnapshot,
    CitationSource,
    DebugEventsResponse,
    DebugRunResponse,
    EffectReconcileRequest,
    MemoryClearResponse,
    MemoryResponse,
    RunCreateRequest,
    RunCreateResponse,
    RunResumeRequest,
    SkillInfo,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_STORE = InMemoryTraceStore()
LONG_TERM_MEMORY = LongTermMemory()

PUBLIC_PHASES = [
    "understanding",
    "planning",
    "consulting",
    "safety_review",
    "finalizing",
]
PUBLIC_AGENT_ROLES = {
    "consultation_agent": ("health_consultation", "健康咨询"),
    "diagnostic_agent": ("symptom_analysis", "风险与症状分析"),
    "research_agent": ("evidence_research", "医学证据检索"),
}


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.short_term_memory = await create_short_term_memory()
    application.state.workflow_tasks = set()
    application.state.knowledge_runtime = create_knowledge_runtime()
    await application.state.knowledge_runtime.manager.initialize()
    try:
        settings = CheckpointSettings.from_env()
        async with open_checkpointer(settings) as checkpointer:
            async with open_run_lease(settings) as run_lease:
                async with open_audit_store(settings) as audit_store:
                    application.state.checkpoint_settings = settings
                    application.state.checkpointer = checkpointer
                    application.state.run_lease = run_lease
                    application.state.audit_store = audit_store
                    for job in await application.state.knowledge_runtime.manager.recover_jobs():
                        _spawn_workflow_task(
                            application,
                            application.state.knowledge_runtime.manager.process_job(job["id"]),
                        )
                    try:
                        yield
                    finally:
                        await _drain_workflow_tasks(application)
    finally:
        await application.state.short_term_memory.close()


app = FastAPI(
    title="Medical-Agent-Swarm Debug API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health(request: Request) -> Dict[str, Any]:
    memory_health = await _short_term_memory(request).health()
    knowledge_health = await _knowledge_runtime(request).knowledge_base.health()
    knowledge_health.update(await _knowledge_runtime(request).manager.health())
    return {
        "status": (
            "ok"
            if memory_health["status"] == "ok"
            and knowledge_health.get("status") in {"ok", "disabled"}
            else "degraded"
        ),
        "memory": memory_health,
        "knowledge": knowledge_health,
    }


@app.post("/api/admin/knowledge/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_knowledge_document(
    request: Request,
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
) -> Dict[str, Any]:
    _require_knowledge_admin(request)
    metadata_value = _parse_knowledge_metadata(metadata)
    content = await file.read(_knowledge_runtime(request).settings.max_file_bytes + 1)
    try:
        result = await _knowledge_runtime(request).manager.submit_document(
            filename=file.filename or "",
            content=content,
            metadata=metadata_value,
        )
    except DocumentValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not result.get("duplicate"):
        _spawn_workflow_task(
            request,
            _knowledge_runtime(request).manager.process_job(result["job_id"]),
        )
    return result


@app.put("/api/admin/knowledge/documents/{document_id}", status_code=status.HTTP_202_ACCEPTED)
async def replace_knowledge_document(
    request: Request,
    document_id: str,
    file: UploadFile = File(...),
    metadata: str = Form("{}"),
) -> Dict[str, Any]:
    _require_knowledge_admin(request)
    metadata_value = _parse_knowledge_metadata(metadata)
    content = await file.read(_knowledge_runtime(request).settings.max_file_bytes + 1)
    try:
        result = await _knowledge_runtime(request).manager.submit_document(
            filename=file.filename or "",
            content=content,
            metadata=metadata_value,
            document_id=document_id,
        )
    except DocumentValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not result.get("duplicate"):
        _spawn_workflow_task(request, _knowledge_runtime(request).manager.process_job(result["job_id"]))
    return result


@app.get("/api/admin/knowledge/documents")
async def list_knowledge_documents(request: Request) -> List[Dict[str, Any]]:
    _require_knowledge_admin(request)
    return await _knowledge_runtime(request).manager.list_documents()


@app.delete("/api/admin/knowledge/documents/{document_id}", status_code=status.HTTP_202_ACCEPTED)
async def delete_knowledge_document(request: Request, document_id: str) -> Dict[str, Any]:
    _require_knowledge_admin(request)
    manager = _knowledge_runtime(request).manager
    try:
        result = await manager.submit_delete(document_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail="Knowledge document not found") from exc
    _spawn_workflow_task(request, manager.process_job(result["job_id"]))
    return result


@app.post("/api/admin/knowledge/reindex", status_code=status.HTTP_202_ACCEPTED)
async def reindex_knowledge(request: Request) -> Dict[str, Any]:
    _require_knowledge_admin(request)
    manager = _knowledge_runtime(request).manager
    result = await manager.reindex_all()
    _spawn_workflow_task(request, manager.process_job(result["job_id"]))
    return result


@app.get("/api/admin/knowledge/jobs/{job_id}")
async def get_knowledge_job(request: Request, job_id: str) -> Dict[str, Any]:
    _require_knowledge_admin(request)
    job = await _knowledge_runtime(request).manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Knowledge job not found")
    return job


def _start_workflow(
    request: Request,
    payload: RunCreateRequest,
    *,
    source: str,
) -> DebugTraceCollector:
    collector = DebugTraceCollector(
        question=payload.question,
        context=payload.context,
        session_id=payload.session_id,
        metadata={
            "source": source,
            "enable_swarm": payload.enable_swarm,
            "enable_memory": payload.enable_memory,
            "enable_short_term_memory": payload.enable_short_term_memory,
            "enable_long_term_memory": payload.enable_long_term_memory,
            "enable_rag": payload.enable_rag,
            "context_keys": sorted(payload.context.keys()),
        },
    )
    RUN_STORE.add(collector)
    _spawn_workflow_task(
        request,
        _execute_run(
            payload,
            collector,
            _short_term_memory(request),
            _checkpointer(request),
            _run_lease(request),
            _audit_store(request),
            _knowledge_runtime(request),
        ),
    )
    return collector


@app.post("/api/runs", response_model=RunCreateResponse)
async def create_run(payload: RunCreateRequest, request: Request) -> RunCreateResponse:
    collector = _start_workflow(request, payload, source="api")

    run = collector.get_run().to_dict()
    return RunCreateResponse(
        run_id=collector.run_id,
        status=run["status"],
        run=run,
    )


@app.post(
    "/api/consultations",
    response_model=ConsultationCreateResponse,
)
async def create_consultation(
    payload: ConsultationCreateRequest,
    request: Request,
) -> ConsultationCreateResponse:
    run_payload = RunCreateRequest(
        question=payload.question,
        context=payload.context,
        session_id=payload.session_id,
        enable_swarm=True,
        enable_memory=True,
    )
    collector = _start_workflow(request, run_payload, source="consultation_api")
    return ConsultationCreateResponse(
        consultation_id=collector.run_id,
        status="queued",
    )


@app.get(
    "/api/consultations/{consultation_id}",
    response_model=ConsultationSnapshot,
)
async def get_consultation(
    request: Request,
    consultation_id: str,
) -> ConsultationSnapshot:
    run, events = await _get_public_consultation_data(request, consultation_id)
    supplied_session_id = request.headers.get("X-Session-ID", "")
    expected_session_id = str(run.get("session_id") or "")
    if not supplied_session_id or not secrets.compare_digest(
        supplied_session_id,
        expected_session_id,
    ):
        raise HTTPException(status_code=404, detail="Consultation not found")
    return ConsultationSnapshot(**_build_public_consultation_snapshot(run, events))


@app.get("/api/runs/{run_id}/checkpoints")
async def list_run_checkpoints(
    request: Request,
    run_id: str,
    limit: int = Query(default=100, ge=1, le=500),
    include_values: bool = Query(default=False),
) -> Dict[str, Any]:
    _require_checkpoint_admin(request)
    coordinator = _checkpoint_coordinator(request)
    try:
        checkpoints = await coordinator.list_checkpoints(run_id, limit=limit)
    except CheckpointingDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    serialized = []
    for item in checkpoints:
        value = asdict(item)
        if not include_values:
            value["state_keys"] = sorted(value.pop("values").keys())
        serialized.append(value)
    return {"run_id": run_id, "checkpoints": serialized}


@app.get("/api/runs/{run_id}/effects")
async def list_run_effects(request: Request, run_id: str) -> Dict[str, Any]:
    """List outbox states, including writes requiring reconciliation."""
    _require_checkpoint_admin(request)
    return {
        "run_id": run_id,
        "effects": await _audit_store(request).get_effects(run_id),
    }


@app.patch("/api/runs/{run_id}/effects/{effect_name}")
async def reconcile_run_effect(
    request: Request,
    run_id: str,
    effect_name: str,
    payload: EffectReconcileRequest,
) -> Dict[str, Any]:
    """Record an administrator's reconciliation of an uncertain write."""
    _require_checkpoint_admin(request)
    reconciled = await _audit_store(request).reconcile_effect(
        run_id,
        effect_name,
        payload.resolution,
    )
    if not reconciled:
        raise HTTPException(
            status_code=409,
            detail="Effect is missing or is not awaiting reconciliation",
        )
    return {
        "run_id": run_id,
        "effect_name": effect_name,
        "status": payload.resolution,
    }


@app.post("/api/runs/{run_id}/resume", response_model=RunCreateResponse)
async def resume_run(
    request: Request,
    run_id: str,
    payload: RunResumeRequest,
) -> RunCreateResponse:
    _require_checkpoint_admin(request)
    coordinator = _checkpoint_coordinator(request)
    try:
        checkpoint = await coordinator.get_checkpoint(run_id, payload.checkpoint_id)
    except CheckpointingDisabledError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if checkpoint is None:
        raise HTTPException(status_code=404, detail=f"Checkpoint run not found: {run_id}")
    if checkpoint.status == "completed" and payload.checkpoint_id is None:
        raise HTTPException(
            status_code=409,
            detail="Run is already completed; select an earlier checkpoint to replay",
        )

    coordinator = _checkpoint_coordinator(request, checkpoint.values)

    collector = DebugTraceCollector(
        question=str(checkpoint.values.get("question") or ""),
        context=dict(checkpoint.values.get("context") or {}),
        session_id=checkpoint.values.get("session_id"),
        run_id=run_id,
        metadata={
            "source": "checkpoint_resume",
            "checkpoint_id": payload.checkpoint_id or checkpoint.checkpoint_id,
        },
    )
    RUN_STORE.add(collector)
    _spawn_workflow_task(
        request,
        _execute_resume(
            coordinator,
            collector,
            run_id,
            payload.checkpoint_id,
        )
    )
    run = collector.get_run().to_dict()
    return RunCreateResponse(run_id=run_id, status=run["status"], run=run)


@app.get("/api/runs")
async def list_runs(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
) -> Dict[str, List[Dict[str, Any]]]:
    _require_checkpoint_admin(request)
    return {"runs": [run.to_dict() for run in RUN_STORE.list(limit=limit)]}


@app.get("/api/runs/{run_id}", response_model=DebugRunResponse)
async def get_run(request: Request, run_id: str) -> DebugRunResponse:
    _require_checkpoint_admin(request)
    collector = RUN_STORE.get(run_id)
    if collector:
        return DebugRunResponse(run=collector.get_run().to_dict())
    attempts = await _audit_store(request).get_attempts(run_id)
    if not attempts:
        raise HTTPException(status_code=404, detail=f"Debug run not found: {run_id}")
    return DebugRunResponse(run=attempts[-1]["run"])


@app.get("/api/runs/{run_id}/events", response_model=DebugEventsResponse)
async def get_run_events(request: Request, run_id: str) -> DebugEventsResponse:
    _require_checkpoint_admin(request)
    collector = RUN_STORE.get(run_id)
    if collector:
        return DebugEventsResponse(
            events=[event.to_dict() for event in collector.get_events()]
        )
    attempts = await _audit_store(request).get_attempts(run_id)
    if not attempts:
        raise HTTPException(status_code=404, detail=f"Debug run not found: {run_id}")
    events = []
    for attempt in attempts:
        for event in attempt.get("events", []):
            value = dict(event)
            metadata = dict(value.get("metadata") or {})
            metadata["audit_attempt_id"] = attempt["attempt_id"]
            value["metadata"] = metadata
            events.append(value)
    events.sort(key=lambda event: event.get("timestamp", ""))
    return DebugEventsResponse(events=events)


@app.get("/api/skills", response_model=List[SkillInfo])
async def get_skills() -> List[SkillInfo]:
    discovered = discover_skills(PROJECT_ROOT)
    skills: List[SkillInfo] = []

    for skill in discovered:
        metadata = skill.get("metadata") or {}
        skills.append(
            SkillInfo(
                name=skill.get("name", ""),
                function_name=skill.get("function_name", ""),
                script_name=skill.get("script_name", ""),
                description=metadata.get("description", ""),
                active=is_active_skill(skill),
                metadata=_json_safe_metadata(metadata),
            )
        )

    return skills


@app.get("/api/agents", response_model=List[AgentInfo])
async def get_agents(enable_swarm: bool = True) -> List[AgentInfo]:
    coordinator = SwarmCoordinator(enable_swarm=enable_swarm)
    agents = []
    for agent in coordinator.worker_pool:
        agents.append(
            AgentInfo(
                agent_id=agent.agent_id,
                class_name=agent.__class__.__name__,
                capabilities=agent.get_capabilities(),
                config=_json_safe_metadata(agent.config),
                skills=_agent_skills(agent),
            )
        )
    return agents


@app.get("/api/sessions/{session_id}/memory", response_model=MemoryResponse)
async def get_session_memory(
    request: Request,
    session_id: str,
    query: Optional[str] = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
) -> MemoryResponse:
    short_term_memory = _short_term_memory(request)
    try:
        recent_history = await short_term_memory.load_context(
            session_id=session_id,
            max_turns=max(1, (limit + 1) // 2),
        )
        ttl_seconds = await short_term_memory.get_session_ttl(session_id)
    except ShortTermMemoryError as exc:
        raise HTTPException(
            status_code=503,
            detail="Short-term memory backend unavailable",
        ) from exc
    recent_history = recent_history[-limit:]
    historical_cases = (
        LONG_TERM_MEMORY.search_similar_sessions(query=query, limit=min(limit, 10))
        if query
        else []
    )
    return MemoryResponse(
        session_id=session_id,
        backend=short_term_memory.backend_name,
        ttl_seconds=ttl_seconds,
        recent_history=recent_history,
        historical_cases=historical_cases,
        long_term_enabled=bool(getattr(LONG_TERM_MEMORY, "enabled", False)),
    )


@app.delete(
    "/api/sessions/{session_id}/memory",
    response_model=MemoryClearResponse,
)
async def clear_session_memory(
    request: Request,
    session_id: str,
) -> MemoryClearResponse:
    try:
        cleared = await _short_term_memory(request).clear_session(session_id)
    except ShortTermMemoryError as exc:
        raise HTTPException(
            status_code=503,
            detail="Short-term memory backend unavailable",
        ) from exc
    return MemoryClearResponse(session_id=session_id, cleared=cleared)


async def _execute_run(
    payload: RunCreateRequest,
    collector: DebugTraceCollector,
    short_term_memory: ShortTermMemory,
    checkpointer: Any,
    run_lease: Any,
    audit_store: Any,
    knowledge_runtime: KnowledgeRuntime,
) -> None:
    try:
        coordinator = SwarmCoordinator(
            enable_swarm=payload.enable_swarm,
            enable_memory=payload.enable_memory,
            enable_short_term_memory=payload.enable_short_term_memory,
            enable_long_term_memory=payload.enable_long_term_memory,
            short_term_memory=short_term_memory,
            checkpointer=checkpointer,
            run_lease=run_lease,
            audit_store=audit_store,
            enable_rag=(
                knowledge_runtime.settings.enabled
                if payload.enable_rag is None
                else payload.enable_rag
            ),
            knowledge_base=knowledge_runtime.knowledge_base,
        )
        await coordinator.process(
            question=payload.question,
            context=payload.context,
            session_id=payload.session_id,
            debug_collector=collector,
            run_id=collector.run_id,
        )
    except Exception as exc:
        collector.finish_failed(exc)


async def _execute_resume(
    coordinator: SwarmCoordinator,
    collector: DebugTraceCollector,
    run_id: str,
    checkpoint_id: Optional[str],
) -> None:
    try:
        result = await coordinator.resume(
            run_id,
            checkpoint_id,
            debug_collector=collector,
        )
        collector.finish_success(
            result_json=result,
            route=result.get("route"),
            final_answer=result.get("answer", ""),
            timeout=bool(result.get("timeout_occurred", False)),
        )
    except Exception as exc:
        collector.finish_failed(exc)


async def _get_public_consultation_data(
    request: Request,
    consultation_id: str,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    collector = RUN_STORE.get(consultation_id)
    if collector:
        run = collector.get_run().to_dict()
        if not _is_public_consultation_run(run):
            raise HTTPException(status_code=404, detail="Consultation not found")
        return run, [event.to_dict() for event in collector.get_events()]
    attempts = await _audit_store(request).get_attempts(consultation_id)
    if not attempts:
        raise HTTPException(status_code=404, detail="Consultation not found")
    latest = attempts[-1]
    run = dict(latest.get("run") or {})
    if not _is_public_consultation_run(run):
        raise HTTPException(status_code=404, detail="Consultation not found")
    return run, [dict(event) for event in latest.get("events") or []]


def _is_public_consultation_run(run: Dict[str, Any]) -> bool:
    metadata = run.get("metadata")
    return isinstance(metadata, dict) and metadata.get("source") == "consultation_api"


def _build_public_consultation_snapshot(
    run: Dict[str, Any],
    events: List[Dict[str, Any]],
) -> Dict[str, Any]:
    raw_status = str(run.get("status") or "running").lower()
    terminal = raw_status in {"success", "failed", "timeout"}
    status = raw_status if terminal else ("running" if events else "queued")
    current_phase = "finalizing" if raw_status == "success" else _current_public_phase(events)
    current_index = PUBLIC_PHASES.index(current_phase)
    completed_phases = PUBLIC_PHASES[:current_index]
    if raw_status == "success":
        completed_phases = list(PUBLIC_PHASES)

    completed_agent_ids = list(
        dict.fromkeys(
            str(event.get("agent_id"))
            for event in events
            if event.get("agent_id") in PUBLIC_AGENT_ROLES
            and event.get("stage") == "agent_loop"
            and event.get("name") == "agent_loop"
            and event.get("status") == "success"
        )
    )
    failed_agent_ids = {
        str(event.get("agent_id"))
        for event in events
        if event.get("agent_id") in PUBLIC_AGENT_ROLES
        and event.get("stage") == "agent_loop"
        and event.get("name") == "agent_loop"
        and event.get("status") == "failed"
    }
    planned_agent_ids = _planned_public_agent_ids(events)
    agent_ids = list(dict.fromkeys([*planned_agent_ids, *completed_agent_ids]))
    participants = []
    for agent_id in agent_ids:
        public_id, label = PUBLIC_AGENT_ROLES[agent_id]
        if agent_id in failed_agent_ids:
            state = "failed"
        elif raw_status == "success" or current_index > PUBLIC_PHASES.index("consulting"):
            state = "done"
        elif agent_id in completed_agent_ids:
            state = "done"
        elif raw_status in {"failed", "timeout"}:
            state = "failed"
        elif current_phase == "consulting":
            state = "active"
        else:
            state = "waiting"
        participants.append({"id": public_id, "label": label, "state": state})

    result_json = run.get("result_json")
    result_data = result_json if isinstance(result_json, dict) else {}
    result = None
    failure = None
    if raw_status == "success":
        risk_level = _find_public_risk_level(result_data)
        result_agent_ids = [
            agent_id
            for agent_id in _string_list(result_data.get("agents_involved"))
            if agent_id in PUBLIC_AGENT_ROLES
        ]
        visible_agent_ids = result_agent_ids or agent_ids
        result = {
            "answer": str(run.get("final_answer") or result_data.get("answer") or ""),
            "risk_level": risk_level,
            "suggestions": _string_list(result_data.get("suggestions")),
            "disclaimer": str(result_data.get("disclaimer") or ""),
            "participants": [PUBLIC_AGENT_ROLES[agent_id][1] for agent_id in visible_agent_ids],
            "sources": [
                _public_knowledge_source(source)
                for source in (result_data.get("sources") or [])
                if isinstance(source, dict)
            ],
        }
    elif raw_status in {"failed", "timeout"}:
        timed_out = raw_status == "timeout"
        failure = {
            "code": "analysis_timeout" if timed_out else "analysis_failed",
            "message": (
                "分析时间较长，请重新尝试；如症状正在加重，请及时线下就医。"
                if timed_out
                else "本次分析未能完成，请重新尝试；如症状严重或正在加重，请及时线下就医。"
            ),
            "retryable": True,
        }

    return {
        "consultation_id": str(run.get("run_id") or ""),
        "status": status,
        "progress": {
            "current_phase": current_phase,
            "completed_phases": completed_phases,
            "participants": participants,
            "safety_checked": _public_safety_checked(events),
        },
        "result": result,
        "failure": failure,
    }


def _public_phase_for_event(event: Dict[str, Any]) -> Optional[str]:
    stage = str(event.get("stage") or "")
    name = str(event.get("name") or "")
    if stage == "memory" and name == "save_memory":
        return "finalizing"
    if stage in {"memory", "load_memory"}:
        return "understanding"
    if stage in {"planning", "routing", "constraint_check"}:
        return "planning"
    if stage in {"agent_loop", "skill_call", "swarm_context"}:
        return "consulting"
    if stage == "safety_check":
        return "safety_review"
    return None


def _current_public_phase(events: List[Dict[str, Any]]) -> str:
    if any(
        (
            str(event.get("stage") or "") == "save_memory"
            or (
                str(event.get("stage") or "") == "memory"
                and str(event.get("name") or "") == "save_memory"
            )
        )
        and str(event.get("status") or "success") == "success"
        for event in events
    ):
        return "finalizing"
    if any(str(event.get("stage") or "") == "safety_check" for event in events):
        return "finalizing" if _public_safety_checked(events) else "safety_review"
    if any(
        str(event.get("stage") or "") == "agent_loop"
        and str(event.get("name") or "") in {"run_single_agent", "run_swarm", "run_fallback"}
        and str(event.get("status") or "success") == "success"
        for event in events
    ):
        return "safety_review"
    if any(
        str(event.get("stage") or "") == "routing"
        and str(event.get("status") or "success") == "success"
        for event in events
    ):
        return "consulting"
    if any(
        str(event.get("stage") or "") == "agent_loop"
        and str(event.get("name") or "") in {"run_single_agent", "run_swarm", "run_fallback"}
        for event in events
    ):
        return "consulting"
    if any(
        str(event.get("stage") or "") in {"planning", "constraint_check"}
        for event in events
    ):
        return "planning"
    if any(_public_phase_for_event(event) == "understanding" for event in events):
        return "planning"
    return "understanding"


def _planned_public_agent_ids(events: List[Dict[str, Any]]) -> List[str]:
    planned: List[str] = []
    for event in events:
        name = str(event.get("name") or "")
        output = event.get("output")
        tasks: Any = None
        if name == "subtasks_created" and isinstance(output, list):
            tasks = output
        elif name == "route_plan" and isinstance(output, dict):
            tasks = output.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            if not isinstance(task, dict):
                continue
            agent_id = task.get("assigned_agent")
            if isinstance(agent_id, str) and agent_id in PUBLIC_AGENT_ROLES:
                planned.append(agent_id)
    return list(dict.fromkeys(planned))


def _public_safety_checked(events: List[Dict[str, Any]]) -> bool:
    for event in reversed(events):
        if str(event.get("stage") or "") != "safety_check":
            continue
        output = event.get("output")
        if isinstance(output, dict) and output.get("safety_checked") is True:
            return True
    return False


def _find_public_risk_level(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        raw = value.get("risk_level")
        if isinstance(raw, str) and raw.lower() in {
            "low",
            "medium",
            "high",
            "emergency",
        }:
            return raw.lower()
        for child in value.values():
            found = _find_public_risk_level(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_public_risk_level(child)
            if found:
                return found
    return None


def _string_list(value: Any) -> List[str]:
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


def _public_knowledge_source(value: Dict[str, Any]) -> Dict[str, str]:
    return CitationSource.model_validate(value).model_dump()


def _short_term_memory(request: Request) -> ShortTermMemory:
    return request.app.state.short_term_memory


def _checkpointer(request: Request) -> Any:
    return request.app.state.checkpointer


def _run_lease(request: Request) -> Any:
    return request.app.state.run_lease


def _audit_store(request: Request) -> Any:
    return request.app.state.audit_store


def _knowledge_runtime(request: Request) -> KnowledgeRuntime:
    return request.app.state.knowledge_runtime


def _require_knowledge_admin(request: Request) -> None:
    expected = _knowledge_runtime(request).settings.admin_token
    supplied = request.headers.get("X-Knowledge-Admin-Token", "")
    if not expected or not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="Knowledge administrator required")


def _parse_knowledge_metadata(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="metadata must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=422, detail="metadata must be a JSON object")
    return parsed


def _checkpoint_coordinator(
    request: Request,
    checkpoint_values: Optional[Dict[str, Any]] = None,
) -> SwarmCoordinator:
    values = checkpoint_values or {}
    return SwarmCoordinator(
        short_term_memory=_short_term_memory(request),
        checkpointer=_checkpointer(request),
        run_lease=_run_lease(request),
        audit_store=_audit_store(request),
        enable_swarm=bool(values.get("enable_swarm", True)),
        enable_short_term_memory=bool(
            values.get("enable_short_term_memory", True)
        ),
        enable_long_term_memory=bool(values.get("enable_long_term_memory", True)),
        enable_rag=bool(values.get("enable_rag", False)),
        knowledge_base=_knowledge_runtime(request).knowledge_base,
        swarm_timeout_s=float(values.get("swarm_timeout_s") or 120.0),
    )


def _require_checkpoint_admin(request: Request) -> None:
    expected = os.getenv("CHECKPOINT_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Checkpoint administration is not configured",
        )
    supplied = request.headers.get("X-Checkpoint-Admin-Token", "")
    if not secrets.compare_digest(supplied, expected):
        raise HTTPException(status_code=403, detail="Checkpoint administrator required")


def _spawn_workflow_task(request: Any, coroutine: Any) -> asyncio.Task[Any]:
    task = asyncio.create_task(coroutine)
    application = request if isinstance(request, FastAPI) else request.app
    tasks = application.state.workflow_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return task


async def _drain_workflow_tasks(application: FastAPI) -> None:
    tasks = set(application.state.workflow_tasks)
    if not tasks:
        return
    _, pending = await asyncio.wait(tasks, timeout=30.0)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)


def _get_collector_or_404(run_id: str) -> DebugTraceCollector:
    collector = RUN_STORE.get(run_id)
    if not collector:
        raise HTTPException(status_code=404, detail=f"Debug run not found: {run_id}")
    return collector


def _agent_skills(agent: Any) -> List[Dict[str, Any]]:
    output = []
    for name, skill in agent.skill_registry.get_all().items():
        output.append(
            {
                "name": name,
                "description": skill.get("description", ""),
                "is_async": bool(skill.get("is_async")),
                "parameters": [
                    {
                        "name": param.name,
                        "type": param.type,
                        "description": param.description,
                        "required": param.required,
                        "enum": param.enum,
                    }
                    for param in skill.get("parameters", [])
                ],
            }
        )
    return output


def _json_safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    safe: Dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if inspect.isfunction(value) or inspect.ismethod(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        elif isinstance(value, (list, tuple)):
            safe[key] = [
                item
                for item in value
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
        elif isinstance(value, dict):
            safe[key] = _json_safe_metadata(value)
        else:
            safe[key] = str(value)
    return safe
