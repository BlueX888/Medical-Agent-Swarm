"""
Agent循环引擎
实现 LLM 驱动的 Skill 调用循环
消费由工作流注入的会话历史
支持约束验证（Harness Engineering）
"""
import uuid
import json
import inspect
from typing import Dict, Any, List, Optional
from loguru import logger

from .state_manager import TaskStatus, AgentState
from .llm_client import LLMResponse

# Harness Engineering: runtime behavior constraints
try:
    from constraints import ConstraintValidator
    CONSTRAINTS_ENABLED = True
except ImportError:
    logger.warning("Constraints module not found, running without constraint validation")
    CONSTRAINTS_ENABLED = False


RUNTIME_ONLY_TOOLS = {"safety_check"}


class AgentLoop:
    """
    Agent循环引擎
    LLM 自主决策 Skill 调用，循环直到任务完成

    功能：
    - 消费请求中已经加载好的会话历史
    - 不直接读写持久化记忆
    """

    def __init__(self, max_iterations: int = 10, max_tool_calls: int = 4):
        """
        初始化Agent循环引擎

        Args:
            max_iterations: 最大迭代次数（防止无限循环）
            max_tool_calls: 最大 Skill 调用次数（硬性限制，默认4次；可通过 Agent 类型覆盖）
        """
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.tool_call_count = 0

        # Harness Engineering: runtime behavior constraint validator
        self.validator = ConstraintValidator() if CONSTRAINTS_ENABLED else None
        if CONSTRAINTS_ENABLED:
            logger.debug("✅ Constraint validation enabled")

    def _normalize_tool_policy(self, policy: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize per-request tool policy into allow/deny lists."""
        policy = policy or {}
        allow_tools = policy.get("allow_tools") or policy.get("allowed_tools") or []
        deny_tools = policy.get("deny_tools") or policy.get("denied_tools") or []
        return {
            "allow_tools": list(dict.fromkeys(allow_tools)),
            "deny_tools": list(dict.fromkeys(deny_tools)),
            "reason": policy.get("reason", ""),
        }

    def _is_tool_allowed(self, tool_name: str, tool_policy: Dict[str, Any]) -> bool:
        if tool_name in RUNTIME_ONLY_TOOLS:
            return False
        allow_tools = set(tool_policy.get("allow_tools") or [])
        deny_tools = set(tool_policy.get("deny_tools") or [])
        if allow_tools and tool_name not in allow_tools:
            return False
        return tool_name not in deny_tools

    def _get_tools_for_llm(self, agent, tool_policy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get tools from new or legacy agent implementations and apply policy."""
        allow_tools = tool_policy["allow_tools"]
        deny_tools = tool_policy["deny_tools"]
        method = agent.get_tools_for_llm
        parameters = inspect.signature(method).parameters
        supports_policy = (
            "allow_tools" in parameters
            or "deny_tools" in parameters
            or any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        )

        if supports_policy:
            tools = method(allow_tools=allow_tools, deny_tools=deny_tools)
        else:
            tools = method()

        allow_set = set(allow_tools or [])
        deny_set = set(deny_tools or [])
        filtered_tools = []
        for tool in tools or []:
            name = ((tool or {}).get("function") or {}).get("name")
            if name in RUNTIME_ONLY_TOOLS:
                continue
            if allow_set and name not in allow_set:
                continue
            if name in deny_set:
                continue
            filtered_tools.append(tool)
        return filtered_tools

    async def run(self, agent, input_data: Dict[str, Any], session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        执行Agent循环

        Args:
            agent: Agent实例
            input_data: 输入数据

        Returns:
            最终结果
        """
        task_id = str(uuid.uuid4())
        debug_collector = input_data.get("debug_collector")
        loop_input_data = {
            key: value
            for key, value in input_data.items()
            if key != "debug_collector"
        }
        agent_timer = None
        if debug_collector:
            agent_timer = debug_collector.time_event(
                "agent_loop",
                agent_id=agent.agent_id,
                input=loop_input_data,
                name="agent_loop",
            )
        state = AgentState(
            task_id=task_id,
            agent_id=agent.agent_id,
            input_data=loop_input_data,
            max_iterations=self.max_iterations
        )

        # 重置计数
        self.tool_call_count = 0

        logger.info(f"Starting Agent Loop for {agent.agent_id}, task_id={task_id}")

        try:
            state.status = TaskStatus.IN_PROGRESS

            # 初始化消息历史（包含历史对话）
            messages = self._initialize_messages(agent, loop_input_data, session_id)
            latest_risk_level = ""
            if debug_collector:
                role_counts: Dict[str, int] = {}
                for message in messages:
                    role = str(message.get("role", "unknown"))
                    role_counts[role] = role_counts.get(role, 0) + 1
                debug_collector.record_event(
                    "memory",
                    agent_id=agent.agent_id,
                    name="initialize_messages",
                    input={
                        "session_id": session_id,
                        "conversation_history_provided": bool(
                            loop_input_data.get("conversation_history")
                        ),
                    },
                    output={
                        "message_count": len(messages),
                        "role_counts": role_counts,
                        "messages": messages,
                    },
                    metadata={
                        "history_message_count": max(0, len(messages) - 2),
                    },
                )

            # 获取 Agent 的 Skills (OpenAI format)
            tool_policy = self._normalize_tool_policy(loop_input_data.get("tool_policy"))
            tools_openai_format = self._get_tools_for_llm(agent, tool_policy)

            logger.debug(
                f"Agent has {len(tools_openai_format) if tools_openai_format else 0} skills available "
                f"(deny={tool_policy['deny_tools']})"
            )

            # 主循环：LLM → Skill Calls → Results → LLM
            while state.should_continue():
                state.iteration += 1
                logger.debug(f"=== Iteration {state.iteration}/{state.max_iterations} ===")

                try:
                    # 调用 LLM（可能返回 tool_calls）
                    llm_response: LLMResponse = await agent.llm_client.chat_with_tools(
                        messages=messages,
                        tools=tools_openai_format,
                        tool_choice="auto",
                        temperature=agent.config.get('temperature', 0.7),
                        debug_collector=debug_collector,
                        trace_name=f"{agent.agent_id}_iteration_{state.iteration}",
                        agent_id=agent.agent_id,
                    )

                    # 记录中间结果
                    state.add_intermediate_result({
                        'iteration': state.iteration,
                        'llm_response': {
                            'content': llm_response.content,
                            'tool_calls': [
                                {'name': tc.name, 'arguments': tc.arguments}
                                for tc in llm_response.tool_calls
                            ],
                            'finish_reason': llm_response.finish_reason
                        }
                    })

                    # 情况1: LLM 返回 tool_calls，执行 Skills
                    if llm_response.has_tool_calls():
                        # 硬性限制：检查是否已达到最大调用次数
                        if self.tool_call_count >= self.max_tool_calls:
                            logger.warning(f"⚠️ 已达到最大 Skill 调用次数限制 ({self.max_tool_calls})，强制生成最终答案")
                            # 强制要求 LLM 提供最终答案
                            messages.append({
                                'role': 'user',
                                'content': f'已完成 {self.max_tool_calls} 次信息检索。请基于已获取的信息提供最终答复。'
                            })
                            continue

                        logger.info(f"LLM requested {len(llm_response.tool_calls)} tool calls (当前已调用 {self.tool_call_count}/{self.max_tool_calls})")

                        # 添加 assistant 消息（包含 tool_calls）
                        messages.append(self._create_assistant_message_with_tools(llm_response))

                        # 执行每个 Skill 调用
                        for tool_call in llm_response.tool_calls:
                            # 增加计数
                            self.tool_call_count += 1
                            logger.debug(
                                f"Requested tool: {tool_call.name}({tool_call.arguments}) "
                                f"- 第 {self.tool_call_count} 次调用"
                            )

                            if not self._is_tool_allowed(tool_call.name, tool_policy):
                                reason = (
                                    tool_policy.get("reason")
                                    or f"Tool {tool_call.name} is not allowed for this request."
                                )
                                logger.warning(
                                    f"Blocked tool call by policy: {tool_call.name} "
                                    f"(agent={agent.agent_id}, reason={reason})"
                                )
                                tool_result = {
                                    "success": False,
                                    "error": f"Tool call blocked by policy: {tool_call.name}",
                                    "skill": tool_call.name,
                                    "policy_reason": reason,
                                }
                                if debug_collector:
                                    debug_collector.record_event(
                                        "constraint_check",
                                        agent_id=agent.agent_id,
                                        skill_name=tool_call.name,
                                        name="tool_policy",
                                        input={
                                            "agent_id": agent.agent_id,
                                            "tool_name": tool_call.name,
                                            "tool_policy": tool_policy,
                                        },
                                        output=tool_result,
                                        status="failed",
                                        error=tool_result["error"],
                                        metadata={
                                            "iteration": state.iteration,
                                            "tool_call_index": self.tool_call_count,
                                        },
                                    )
                                messages.append(
                                    agent.llm_client.create_tool_message(
                                        tool_call_id=tool_call.id,
                                        tool_name=tool_call.name,
                                        result=tool_result
                                    )
                                )
                                continue

                            logger.debug(
                                f"Executing: {tool_call.name}({tool_call.arguments}) "
                                f"- 第 {self.tool_call_count} 次调用"
                            )

                            # Harness Engineering: 验证调用
                            if self.validator:
                                validation_result = self.validator.validate_tool_call(
                                    agent.agent_id,
                                    tool_call.name
                                )
                                if debug_collector:
                                    debug_collector.record_event(
                                        "constraint_check",
                                        agent_id=agent.agent_id,
                                        skill_name=tool_call.name,
                                        name="validate_tool_call",
                                        input={
                                            "agent_id": agent.agent_id,
                                            "tool_name": tool_call.name,
                                        },
                                        output=validation_result,
                                        status="success" if validation_result.get("valid") else "failed",
                                        error=None if validation_result.get("valid") else validation_result.get("reason"),
                                        metadata={
                                            "iteration": state.iteration,
                                            "tool_call_index": self.tool_call_count,
                                        },
                                    )
                                if not validation_result.get("valid"):
                                    logger.warning(
                                        f"⚠️ 约束警告: {validation_result.get('reason')}"
                                    )

                            skill_timer = None
                            if debug_collector:
                                skill_timer = debug_collector.time_event(
                                    "skill_call",
                                    agent_id=agent.agent_id,
                                    skill_name=tool_call.name,
                                    input=tool_call.arguments,
                                    name=tool_call.name,
                                    metadata={
                                        "iteration": state.iteration,
                                        "tool_call_id": tool_call.id,
                                        "tool_call_index": self.tool_call_count,
                                    },
                                )

                            try:
                                tool_result = await agent.execute_tool(
                                    tool_name=tool_call.name,
                                    arguments=tool_call.arguments
                                )
                            except Exception as tool_exc:
                                if skill_timer:
                                    skill_timer.finish(
                                        output={"error": str(tool_exc)},
                                        status="failed",
                                        error=str(tool_exc),
                                    )
                                raise

                            if skill_timer:
                                tool_error = (
                                    tool_result.get("error")
                                    if isinstance(tool_result, dict)
                                    else None
                                )
                                tool_failed = (
                                    isinstance(tool_result, dict)
                                    and tool_result.get("success") is False
                                ) or bool(tool_error)
                                skill_timer.finish(
                                    output=tool_result,
                                    status="failed" if tool_failed else "success",
                                    error=str(tool_error) if tool_error else None,
                                )
                            extracted_risk_level = self._extract_risk_level(tool_call.name, tool_result)
                            if extracted_risk_level:
                                latest_risk_level = extracted_risk_level

                            # 添加结果消息
                            messages.append(
                                agent.llm_client.create_tool_message(
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_call.name,
                                    result=tool_result
                                )
                            )

                        # 继续下一轮循环
                        continue

                    # 情况2: LLM 返回文本响应，任务完成
                    else:
                        logger.info(f"LLM provided final response (no tool calls)")

                        # Final-answer content safety is centralized in SafetyGuard.
                        final_answer = llm_response.content

                        result = {
                            'answer': final_answer,
                            'iterations': state.iteration,
                            'agent_id': agent.agent_id,
                            'safety_checked': False,
                            'safety_passed': False,
                            'safety_issues': [],
                        }

                        # 让 Agent 进行结果后处理（如提取建议等）
                        if hasattr(agent, 'post_process_result'):
                            result = await agent.post_process_result(result, final_answer)

                        state.mark_completed(result)
                        break

                except Exception as e:
                    logger.error(f"Error in iteration {state.iteration}: {e}")
                    if state.iteration >= state.max_iterations:
                        state.mark_failed(str(e))
                        break
                    # 否则继续尝试

            # 如果达到最大迭代次数但没有完成
            if not state.is_completed():
                logger.warning(f"Max iterations reached without completion")

                # 强制调用 LLM 生成最终总结
                try:
                    logger.info("Forcing LLM to provide final answer")

                    # 添加强制总结的提示
                    messages.append({
                        'role': 'user',
                        'content': '请基于以上信息，提供最终的答复。'
                    })

                    # 调用 LLM（禁用 function calling）
                    final_response = await agent.llm_client.chat_with_tools(
                        messages=messages,
                        tools=None,
                        temperature=0.7,
                        debug_collector=debug_collector,
                        trace_name=f"{agent.agent_id}_forced_final",
                        agent_id=agent.agent_id,
                    )

                    final_answer = final_response.content or '抱歉，未能完成任务'

                    result = {
                        'answer': final_answer,
                        'iterations': state.iteration,
                        'warning': 'max_iterations_reached',
                        'safety_checked': False,
                        'safety_passed': False,
                        'safety_issues': [],
                    }

                    state.mark_completed(result)
                    logger.info("Generated fallback answer after max iterations")

                except Exception as e:
                    logger.error(f"Failed to generate fallback answer: {e}")
                    # 降级到简单提取
                    result = {
                        'answer': '抱歉，系统在处理您的问题时遇到了问题。建议您简化问题或稍后重试。',
                        'iterations': state.iteration,
                        'warning': 'max_iterations_reached',
                        'error': str(e),
                        'safety_checked': False,
                        'safety_passed': False,
                        'safety_issues': [{
                            "type": "fallback_generation_error",
                            "severity": "high",
                            "message": str(e),
                        }],
                    }
                    state.mark_completed(result)

            logger.info(f"Agent Loop finished: status={state.status.value}, iterations={state.iteration}")
            final_result = state.final_result or {}
            if agent_timer:
                agent_timer.finish(
                    output=final_result,
                    status="success" if state.status == TaskStatus.COMPLETED else "failed",
                    error=state.error,
                )
            return final_result

        except Exception as e:
            logger.error(f"Agent Loop failed: {e}")
            state.mark_failed(str(e))
            if agent_timer:
                agent_timer.finish(status="failed", error=str(e), output={"error": str(e)})
            raise

    def _initialize_messages(self, agent, input_data: Dict[str, Any], session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """初始化消息列表，包含历史对话上下文"""
        messages = []

        # 系统提示词
        system_prompt = agent.get_system_prompt()

        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })

        # 会话历史由工作流统一加载，AgentLoop 只消费，不接触存储。
        history = input_data.get("conversation_history") or []
        normalized_history = [
            {
                "role": message["role"],
                "content": message["content"],
            }
            for message in history
            if isinstance(message, dict)
            and message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        ]
        if normalized_history:
            logger.info(f"Loaded {len(normalized_history)} conversation history messages")
            messages.extend(normalized_history)

        # 用户输入
        user_message = agent.format_user_input(input_data)
        messages.append({
            'role': 'user',
            'content': user_message
        })

        return messages

    def _create_assistant_message_with_tools(self, llm_response: LLMResponse) -> Dict[str, Any]:
        """创建包含 tool_calls 的 assistant 消息"""
        message = {
            'role': 'assistant',
            'content': llm_response.content or None
        }

        # 添加 tool_calls（OpenAI 格式）
        if llm_response.tool_calls:
            message['tool_calls'] = [
                {
                    'id': tc.id,
                    'type': 'function',
                    'function': {
                        'name': tc.name,
                        'arguments': json.dumps(tc.arguments, ensure_ascii=False)
                    }
                }
                for tc in llm_response.tool_calls
            ]

        return message

    def _extract_risk_level(self, tool_name: str, tool_result: Dict[str, Any]) -> str:
        """Extract risk level from assess_risk or compatible tool results."""
        if not isinstance(tool_result, dict):
            return ""

        risk_level = tool_result.get("risk_level", "")
        if risk_level:
            return str(risk_level)

        # Keep this permissive in case a future triage Skill nests the result.
        if tool_name == "assess_risk":
            data = tool_result.get("data") or tool_result.get("result") or {}
            if isinstance(data, dict) and data.get("risk_level"):
                return str(data["risk_level"])

        return ""
