"""Runtime Worker capability catalog used by the Orchestrator and executor."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class AgentDescriptor:
    agent_id: str
    description: str
    capabilities: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "description": self.description,
            "capabilities": list(self.capabilities),
        }


class AgentCatalog:
    """Stable read interface over the live Worker pool."""

    def __init__(self, workers: Iterable[Any]):
        self._workers = {
            str(worker.agent_id): worker
            for worker in workers
        }
        if not self._workers:
            raise ValueError("AgentCatalog requires at least one Worker")

    def list_agents(self) -> List[Dict[str, Any]]:
        descriptors = []
        for agent_id, worker in self._workers.items():
            config = getattr(worker, "config", {}) or {}
            capabilities = list(dict.fromkeys(self._capabilities(worker)))
            descriptors.append(
                AgentDescriptor(
                    agent_id=agent_id,
                    description=str(config.get("description") or worker.__class__.__doc__ or "").strip(),
                    capabilities=capabilities,
                ).to_dict()
            )
        return descriptors

    def has_agent(self, agent_id: str) -> bool:
        return agent_id in self._workers

    def supports(self, agent_id: str, required_capabilities: Iterable[str]) -> bool:
        worker = self._workers.get(agent_id)
        if worker is None:
            return False
        available = set(self._capabilities(worker))
        return set(required_capabilities).issubset(available)

    def find_supporting(self, required_capabilities: Iterable[str]) -> Optional[str]:
        required = list(required_capabilities)
        for agent_id in self._workers:
            if self.supports(agent_id, required):
                return agent_id
        return None

    def get_worker(self, agent_id: str) -> Optional[Any]:
        return self._workers.get(agent_id)

    def capabilities_for(self, agent_id: str) -> List[str]:
        worker = self._workers.get(agent_id)
        return self._capabilities(worker) if worker is not None else []

    @staticmethod
    def _capabilities(worker: Any) -> List[str]:
        getter = getattr(worker, "get_capabilities", None)
        if callable(getter):
            return list(getter())
        return list(getattr(worker, "capabilities", []) or [])
