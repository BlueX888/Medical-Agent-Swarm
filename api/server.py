"""Local FastAPI server for the Agent debug console."""
from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from core.skill_loader import discover_skills, is_active_skill
from debug import DebugTraceCollector, InMemoryTraceStore
from memory import LongTermMemory, ShortTermMemory, create_short_term_memory
from swarm import SwarmCoordinator

from .schemas import (
    AgentInfo,
    DebugEventsResponse,
    DebugRunResponse,
    MemoryClearResponse,
    MemoryResponse,
    RunCreateRequest,
    RunCreateResponse,
    SkillInfo,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_STORE = InMemoryTraceStore()
LONG_TERM_MEMORY = LongTermMemory()


@asynccontextmanager
async def lifespan(application: FastAPI):
    application.state.short_term_memory = await create_short_term_memory()
    try:
        yield
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
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health(request: Request) -> Dict[str, Any]:
    memory_health = await _short_term_memory(request).health()
    return {
        "status": "ok" if memory_health["status"] == "ok" else "degraded",
        "memory": memory_health,
    }


@app.post("/api/runs", response_model=RunCreateResponse)
async def create_run(payload: RunCreateRequest, request: Request) -> RunCreateResponse:
    collector = DebugTraceCollector(
        question=payload.question,
        context=payload.context,
        session_id=payload.session_id,
        metadata={
            "source": "api",
            "enable_swarm": payload.enable_swarm,
            "enable_memory": payload.enable_memory,
            "enable_short_term_memory": payload.enable_short_term_memory,
            "enable_long_term_memory": payload.enable_long_term_memory,
            "context_keys": sorted(payload.context.keys()),
        },
    )
    RUN_STORE.add(collector)

    asyncio.create_task(
        _execute_run(payload, collector, _short_term_memory(request))
    )

    run = collector.get_run().to_dict()
    return RunCreateResponse(
        run_id=collector.run_id,
        status=run["status"],
        run=run,
    )


@app.get("/api/runs")
async def list_runs(limit: int = Query(default=50, ge=1, le=200)) -> Dict[str, List[Dict[str, Any]]]:
    return {"runs": [run.to_dict() for run in RUN_STORE.list(limit=limit)]}


@app.get("/api/runs/{run_id}", response_model=DebugRunResponse)
async def get_run(run_id: str) -> DebugRunResponse:
    collector = _get_collector_or_404(run_id)
    return DebugRunResponse(run=collector.get_run().to_dict())


@app.get("/api/runs/{run_id}/events", response_model=DebugEventsResponse)
async def get_run_events(run_id: str) -> DebugEventsResponse:
    collector = _get_collector_or_404(run_id)
    return DebugEventsResponse(
        events=[event.to_dict() for event in collector.get_events()]
    )


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
    recent_history = await short_term_memory.load_context(
        session_id=session_id,
        max_turns=max(1, (limit + 1) // 2),
    )
    recent_history = recent_history[-limit:]
    historical_cases = (
        LONG_TERM_MEMORY.search_similar_sessions(query=query, limit=min(limit, 10))
        if query
        else []
    )
    return MemoryResponse(
        session_id=session_id,
        backend=short_term_memory.backend_name,
        ttl_seconds=await short_term_memory.get_session_ttl(session_id),
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
    cleared = await _short_term_memory(request).clear_session(session_id)
    return MemoryClearResponse(session_id=session_id, cleared=cleared)


async def _execute_run(
    payload: RunCreateRequest,
    collector: DebugTraceCollector,
    short_term_memory: ShortTermMemory,
) -> None:
    try:
        coordinator = SwarmCoordinator(
            enable_swarm=payload.enable_swarm,
            enable_memory=payload.enable_memory,
            enable_short_term_memory=payload.enable_short_term_memory,
            enable_long_term_memory=payload.enable_long_term_memory,
            short_term_memory=short_term_memory,
        )
        await coordinator.process(
            question=payload.question,
            context=payload.context,
            session_id=payload.session_id,
            debug_collector=collector,
        )
    except Exception as exc:
        collector.finish_failed(exc)


def _short_term_memory(request: Request) -> ShortTermMemory:
    return request.app.state.short_term_memory


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
