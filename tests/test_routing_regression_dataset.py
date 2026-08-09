import json
from pathlib import Path

import pytest

from swarm.agent_catalog import AgentCatalog
from swarm.orchestrator import Orchestrator


DATASET = json.loads(
    (Path(__file__).parent / "data" / "routing_regression.json").read_text(
        encoding="utf-8"
    )
)


class Worker:
    def __init__(self, agent_id, capabilities):
        self.agent_id = agent_id
        self.capabilities = capabilities
        self.config = {"description": agent_id}

    def get_capabilities(self):
        return self.capabilities


class DatasetLLM:
    def __init__(self, item):
        self.item = item

    async def chat(self, messages, **kwargs):
        return json.dumps(
            {
                "intent_summary": self.item["name"],
                "intents": self.item["intents"],
                "risk_level": self.item["expected_risk"],
                "confidence": 0.9,
                "tasks": self.item["tasks"],
                "execution_mode": self.item["expected_mode"],
                "source": "llm",
                "reasons": ["regression fixture"],
                "needs_clarification": self.item.get("needs_clarification", False),
            },
            ensure_ascii=False,
        )


@pytest.fixture
def catalog():
    return AgentCatalog(
        [
            Worker(
                "consultation_agent",
                ["general_health_advice", "risk_assessment", "symptom_triage"],
            ),
            Worker(
                "diagnostic_agent",
                [
                    "risk_assessment",
                    "symptom_triage",
                    "symptom_analysis",
                    "clinical_reasoning",
                ],
            ),
            Worker(
                "research_agent",
                ["guideline_lookup", "evidence_synthesis", "literature_search"],
            ),
        ]
    )


@pytest.mark.parametrize("item", DATASET, ids=lambda item: item["name"])
@pytest.mark.asyncio
async def test_routing_regression_case_is_stable_and_valid(item, catalog):
    orchestrator = Orchestrator(DatasetLLM(item), catalog)

    plan = await orchestrator.plan(
        item["question"],
        item.get("context", {}),
    )

    selected = [task.assigned_agent for task in plan.tasks]
    assert plan.risk_level.value == item["expected_risk"]
    assert plan.execution_mode.value == item["expected_mode"]
    assert plan.tasks
    if item.get("must_include"):
        assert item["must_include"] in selected
    else:
        assert selected == item["expected_agents"]

    # A single requested Worker must not turn into an unnecessary swarm.
    if len(item["expected_agents"]) == 1:
        assert len(plan.tasks) == 1
