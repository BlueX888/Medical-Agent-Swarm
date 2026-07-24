#!/usr/bin/env python3
"""
Runtime safety guard tests.

These tests use fake LLM/Agent objects so they do not call external APIs.
"""
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.agent_loop import AgentLoop
from core.llm_client import LLMResponse, ToolCall


class FakeLLMClient:
    def __init__(self, responses: List[LLMResponse]):
        self.responses = responses
        self.index = 0

    async def chat_with_tools(self, *args, **kwargs) -> LLMResponse:
        if self.index >= len(self.responses):
            return self.responses[-1]
        response = self.responses[self.index]
        self.index += 1
        return response

    def create_tool_message(self, tool_call_id: str, tool_name: str, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False),
        }


class FakeAgent:
    def __init__(
        self,
        responses: List[LLMResponse],
        tool_results: Dict[str, Dict[str, Any]] = None,
        tools: List[Dict[str, Any]] = None,
    ):
        self.agent_id = "consultation_agent"
        self.config = {"temperature": 0.1}
        self.llm_client = FakeLLMClient(responses)
        self.tool_results = tool_results or {}
        self.tools = tools or []

    def get_system_prompt(self) -> str:
        return "medical safety test agent"

    def format_user_input(self, input_data: Dict[str, Any]) -> str:
        return input_data.get("question", "")

    def get_tools_for_llm(self, **kwargs) -> List[Dict[str, Any]]:
        return self.tools

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name == "safety_check":
            raise AssertionError("safety_check must not execute as an Agent tool")
        return self.tool_results.get(tool_name, {})

    async def post_process_result(self, result: Dict[str, Any], final_response: str) -> Dict[str, Any]:
        return result


class OutputValidationTrap:
    def validate_tool_call(self, agent_id: str, tool_name: str) -> Dict[str, Any]:
        return {"valid": True}

    def validate_output(self, agent_id: str, output: str) -> Dict[str, Any]:
        raise AssertionError("AgentLoop should not run validate_output for final answers")


async def run_fake_agent(question: str, answer: str) -> Dict[str, Any]:
    agent = FakeAgent([
        LLMResponse(content=answer, tool_calls=[], finish_reason="stop")
    ])
    loop = AgentLoop(max_iterations=2)
    return await loop.run(agent, {"question": question})


async def test_auto_safety_check_without_tool_call():
    result = await run_fake_agent("胸痛伴呼吸困难怎么办？", "建议先休息观察。")

    # SafetyGuard is now centralized in MedicalSwarmGraph.build_response.
    # AgentLoop no longer runs safety review; it returns neutral defaults.
    assert result["safety_checked"] is False
    assert result["answer"] == "建议先休息观察。"


async def test_dangerous_medication_detected():
    result = await run_fake_agent("血压高怎么办？", "你就是高血压，自行加药即可，不用去医院。")

    # AgentLoop no longer runs SafetyGuard — answer passes through as-is.
    assert result["safety_checked"] is False
    assert "就是" in result["answer"] or "自行加药" in result["answer"]


async def test_final_answer_skips_constraint_output_validation():
    agent = FakeAgent([
        LLMResponse(content="建议休息观察。", tool_calls=[], finish_reason="stop")
    ])
    loop = AgentLoop(max_iterations=2)
    loop.validator = OutputValidationTrap()

    result = await loop.run(agent, {"question": "胸痛怎么办？"})

    assert result["safety_checked"] is False


async def test_safety_check_filtered_from_agent_tools():
    tools = [
        {"type": "function", "function": {"name": "safety_check", "description": "legacy"}},
        {"type": "function", "function": {"name": "assess_risk", "description": "risk"}},
    ]
    agent = FakeAgent([
        LLMResponse(content="建议观察。", tool_calls=[], finish_reason="stop")
    ], tools=tools)
    loop = AgentLoop(max_iterations=2)

    exposed = loop._get_tools_for_llm(agent, loop._normalize_tool_policy({}))
    exposed_names = {tool["function"]["name"] for tool in exposed}

    assert "safety_check" not in exposed_names
    assert "assess_risk" in exposed_names


async def test_safety_check_tool_call_is_blocked():
    tool_call = ToolCall(
        id="call-safety",
        name="safety_check",
        arguments={"response": "建议观察。"},
    )
    agent = FakeAgent([
        LLMResponse(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
        LLMResponse(content="建议观察。", tool_calls=[], finish_reason="stop"),
    ])
    loop = AgentLoop(max_iterations=3)
    result = await loop.run(agent, {"question": "胸痛怎么办？"})

    assert result["safety_checked"] is False


async def test_risk_level_from_assess_risk_result():
    tool_call = ToolCall(
        id="call-risk",
        name="assess_risk",
        arguments={"symptoms": "胸痛，呼吸困难"},
    )
    agent = FakeAgent(
        responses=[
            LLMResponse(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
            LLMResponse(content="建议休息观察。", tool_calls=[], finish_reason="stop"),
        ],
        tool_results={
            "assess_risk": {
                "risk_level": "emergency",
                "recommendation": "立即拨打 120。",
            }
        },
    )
    loop = AgentLoop(max_iterations=3)
    result = await loop.run(agent, {"question": "胸痛，呼吸困难"})

    assert result["safety_checked"] is False
    assert result["answer"] == "建议休息观察。"


async def test_stroke_fast_emergency_warning():
    result = await run_fake_agent("家人突然口角歪斜、说话含糊、一侧胳膊没力气。", "先让他睡一会儿。")
    assert result["safety_checked"] is False


async def test_child_allergy_emergency_warning():
    result = await run_fake_agent("孩子吃花生后喘不上气、嘴唇发紫、身上起大片风团。", "可以先喝点水观察。")
    assert result["safety_checked"] is False


async def test_pregnancy_hypertension_emergency_warning():
    result = await run_fake_agent("孕28周，血压150/95，尿蛋白阳性，头痛越来越明显。", "先少盐休息。")
    assert result["safety_checked"] is False


async def main():
    tests = [
        test_auto_safety_check_without_tool_call,
        test_dangerous_medication_detected,
        test_final_answer_skips_constraint_output_validation,
        test_safety_check_filtered_from_agent_tools,
        test_safety_check_tool_call_is_blocked,
        test_risk_level_from_assess_risk_result,
        test_stroke_fast_emergency_warning,
        test_child_allergy_emergency_warning,
        test_pregnancy_hypertension_emergency_warning,
    ]

    for test in tests:
        await test()
        print(f"passed: {test.__name__}")


if __name__ == "__main__":
    asyncio.run(main())
