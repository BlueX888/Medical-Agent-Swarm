from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from research import deep_research_workflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATHS = [
    PROJECT_ROOT / ".agents/skills/deep-research/script/research.py",
    PROJECT_ROOT / ".claude/skills/deep-research/script/research.py",
]


def _load_script(path: Path):
    module_name = f"test_deep_research_{path.parts[-4]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script_path", SCRIPT_PATHS, ids=["agents", "claude"])
@pytest.mark.asyncio
async def test_deep_research_runs_web_workflow_without_local_evidence_cache(
    monkeypatch,
    script_path,
):
    calls = []

    class FakeWorkflow:
        async def run(self, *, question, max_web_results):
            calls.append((question, max_web_results))
            return SimpleNamespace(
                key_findings=["公开资料结论"],
                confidence=0.8,
                sources=[{"title": "公开来源"}],
                evidence_level="A",
                summary="综合结果",
                conflicts=[],
                recommendations=["继续核对"],
            )

    monkeypatch.setattr(
        deep_research_workflow,
        "DeepResearchWorkflow",
        FakeWorkflow,
    )
    module = _load_script(script_path)

    result = await module.deep_research("公开指南问题", max_iterations=1)

    assert calls == [("公开指南问题", 5)]
    assert result["status"] == "completed"
    assert result["data_sources"] == "Web Search + Evidence Synthesis"
    assert "cache_hit" not in result
    assert "memory_source" not in result
