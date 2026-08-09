"""Web-only deep research implementation shared by tool entrypoints."""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict

from loguru import logger


async def deep_research(query: str, max_iterations: int = 2) -> Dict[str, Any]:
    """Conduct evidence-oriented medical research using public web sources."""
    logger.info(
        f"Starting deep research: query={query}, max_iterations={max_iterations}"
    )
    from research.deep_research_workflow import DeepResearchWorkflow

    try:
        max_iterations = int(max_iterations or 2)
    except (TypeError, ValueError):
        max_iterations = 2
    max_iterations = max(1, max_iterations)
    max_iterations_cap = int(os.getenv("DEEP_RESEARCH_MAX_ITERATIONS", "2") or 2)
    max_iterations = min(max_iterations, max_iterations_cap)

    workflow = DeepResearchWorkflow()
    try:
        report = await workflow.run(
            question=query,
            max_web_results=max_iterations * 5,
        )
        return {
            "answer": format_research_report(query, report),
            "findings": report.key_findings,
            "confidence": (
                "high"
                if report.confidence > 0.7
                else "medium"
                if report.confidence > 0.4
                else "low"
            ),
            "sources": len(report.sources),
            "evidence_level": report.evidence_level,
            "status": "completed",
            "data_sources": "Web Search + Evidence Synthesis",
        }
    except Exception as exc:
        logger.error(f"Deep research failed: {exc}")
        return {
            "answer": f"深度研究失败：{exc}",
            "findings": [],
            "confidence": "low",
            "sources": 0,
            "status": "error",
        }


def format_research_report(query: str, report: Any) -> str:
    """Format a ResearchReport into a compact Chinese report."""
    output = ["【深度研究报告】", "", f"研究问题：{query}", ""]
    if report.key_findings:
        output.append("关键发现：")
        for index, finding in enumerate(report.key_findings, 1):
            output.append(f"{index}. {finding}")
        output.append("")
    if report.summary:
        output.extend(["综合分析：", str(report.summary), ""])

    output.append(f"证据等级：{report.evidence_level} 级")
    output.append(f"置信度：{report.confidence:.0%}")
    if report.conflicts:
        output.extend(["", "信息冲突："])
        output.extend(f"- {conflict}" for conflict in report.conflicts)
    if report.recommendations:
        output.extend(["", "建议："])
        for index, recommendation in enumerate(report.recommendations, 1):
            output.append(f"{index}. {recommendation}")
    if report.sources:
        output.extend(["", f"参考来源数量：{len(report.sources)}"])
    output.extend(["", "💡 数据来源：网络搜索 + 证据综合"])
    return "\n".join(output)


def deep_research_sync(query: str, max_iterations: int = 2) -> Dict[str, Any]:
    return asyncio.run(deep_research(query, max_iterations))
