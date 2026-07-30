"""
LLM客户端
支持调用 OpenAI 兼容的 API（如字节跳动豆包、OpenAI、Deepseek 等）
支持 function calling
"""
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import AsyncOpenAI
from dotenv import dotenv_values
from loguru import logger

PROJECT_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

try:
    from config import LLM_CONFIG
except ImportError:
    LLM_CONFIG = {}
from core.observability import trace_async


def resolve_llm_config() -> Dict[str, Any]:
    """Resolve OpenAI-compatible settings, preferring environment variables."""
    config = dict(LLM_CONFIG)
    dotenv_config = dotenv_values(PROJECT_ENV_FILE)

    def get_setting(name: str) -> Optional[str]:
        return os.getenv(name) or dotenv_config.get(name)

    environment_overrides = {
        "api_key": get_setting("OPENAI_API_KEY"),
        "model_name": get_setting("OPENAI_MODEL"),
        "base_url": get_setting("OPENAI_BASE_URL"),
    }
    config.update({
        key: value
        for key, value in environment_overrides.items()
        if value
    })

    if temperature := get_setting("OPENAI_TEMPERATURE"):
        try:
            config["temperature"] = float(temperature)
        except ValueError as exc:
            raise ValueError("OPENAI_TEMPERATURE must be a number.") from exc

    if max_tokens := get_setting("OPENAI_MAX_TOKENS"):
        try:
            config["max_tokens"] = int(max_tokens)
        except ValueError as exc:
            raise ValueError("OPENAI_MAX_TOKENS must be an integer.") from exc

    missing = [
        key
        for key in ("api_key", "model_name")
        if not config.get(key)
    ]
    if missing:
        variable_names = {
            "api_key": "OPENAI_API_KEY",
            "model_name": "OPENAI_MODEL",
        }
        expected = ", ".join(variable_names[key] for key in missing)
        raise ValueError(
            f"Missing LLM configuration: set {expected} in .env "
            "or provide the corresponding values in config.py."
        )

    config.setdefault("base_url", "https://api.openai.com/v1")
    config.setdefault("temperature", 0.7)
    config.setdefault("max_tokens", 8192)
    return config


@dataclass
class ToolCall:
    """Function call 数据结构"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    """LLM 响应数据结构（支持 function calling）"""
    content: Optional[str]
    tool_calls: List[ToolCall]
    finish_reason: str  # "stop", "tool_calls", "length", "content_filter"
    usage: Dict[str, Any] = None
    model: Optional[str] = None
    response_id: Optional[str] = None

    def has_tool_calls(self) -> bool:
        """是否包含 function calls"""
        return len(self.tool_calls) > 0


class LLMClient:
    """统一的LLM客户端，支持多种模型"""

    def __init__(self, model_type: str = "openai_compatible"):
        """
        初始化LLM客户端

        Args:
            model_type: 模型类型，默认 "openai_compatible"（支持 OpenAI 兼容的 API）
        """
        self.model_type = model_type

        if model_type == "openai_compatible":
            # 环境变量优先，config.py 作为向后兼容的回退配置。
            self.config = resolve_llm_config()
            self.client = AsyncOpenAI(
                api_key=self.config["api_key"],
                base_url=self.config["base_url"]
            )
            self.model_name = self.config["model_name"]
            self.temperature = self.config.get("temperature", 0.7)
            self.max_tokens = self.config.get("max_tokens", 8192)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        debug_collector: Optional[Any] = None,
        trace_name: str = "llm_chat",
        agent_id: Optional[str] = None,
        iteration: int = 0,
        retry_count: int = 0,
        **kwargs
    ) -> str:
        """
        异步聊天接口

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数（可选）
            max_tokens: 最大token数（可选）

        Returns:
            模型返回的文本
        """
        try:
            temperature = self.temperature if temperature is None else temperature
            max_tokens = self.max_tokens if max_tokens is None else max_tokens

            logger.debug(f"Calling LLM ({self.model_type}) with {len(messages)} messages")
            timer = None
            if debug_collector:
                timer = debug_collector.time_event(
                    "llm_call",
                    agent_id=agent_id,
                    input={
                        "messages": messages,
                        "tools": None,
                        "tool_choice": None,
                    },
                    name=trace_name,
                    metadata={
                        "model": self.model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "message_count": len(messages),
                        "tools_count": 0,
                    },
                )

            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }

            response = await trace_async(
                name=self._llm_span_name(trace_name),
                run_type="llm",
                func=lambda: self.client.chat.completions.create(**request_params),
                inputs={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                metadata=self._langsmith_metadata(
                    trace_name=trace_name,
                    agent_id=agent_id,
                    debug_collector=debug_collector,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools_count=0,
                    iteration=iteration,
                    retry_count=retry_count,
                ),
                tags=["medical-agent-swarm", "llm"],
                output_mapper=self._response_trace_output,
            )

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            usage = self._usage_to_dict(getattr(response, "usage", None))
            if timer:
                timer.finish(
                    output={
                        "content": content,
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "model": getattr(response, "model", self.model_name),
                        "response_id": getattr(response, "id", None),
                    },
                    metadata={
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "model": getattr(response, "model", self.model_name),
                        "response_id": getattr(response, "id", None),
                    },
                )
            logger.debug(f"LLM response length: {len(content or '')} chars")
            return content or ""

        except Exception as e:
            if "timer" in locals() and timer:
                timer.finish(status="failed", error=str(e), output={"error": str(e)})
            logger.error(f"LLM call failed: {e}")
            raise

    async def chat_with_retry(
        self,
        messages: List[Dict[str, str]],
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        带重试的聊天接口

        Args:
            messages: 消息列表
            max_retries: 最大重试次数

        Returns:
            模型返回的文本
        """
        for attempt in range(max_retries):
            try:
                return await self.chat(
                    messages,
                    retry_count=attempt,
                    **kwargs,
                )
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                logger.warning(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                await asyncio.sleep(2 ** attempt)  # 指数退避

    def create_message(self, role: str, content: str) -> Dict[str, str]:
        """
        创建消息对象

        Args:
            role: 角色，"user" 或 "assistant" 或 "system"
            content: 消息内容

        Returns:
            消息字典
        """
        return {"role": role, "content": content}

    async def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        debug_collector: Optional[Any] = None,
        trace_name: str = "llm_chat_with_tools",
        agent_id: Optional[str] = None,
        iteration: int = 0,
        retry_count: int = 0,
        **kwargs
    ) -> LLMResponse:
        """
        带工具支持的聊天接口

        Args:
            messages: 消息列表
            tools: 工具定义列表（OpenAI format）
            tool_choice: 工具选择策略 ("auto"/"required"/"none")
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLMResponse 对象
        """
        try:
            temperature = self.temperature if temperature is None else temperature
            max_tokens = self.max_tokens if max_tokens is None else max_tokens

            logger.debug(f"Calling LLM with {len(tools) if tools else 0} tools")

            # 准备请求参数
            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs
            }

            # 添加工具参数（如果提供）
            if tools:
                request_params["tools"] = tools
                if tool_choice != "auto":
                    request_params["tool_choice"] = tool_choice

            timer = None
            if debug_collector:
                timer = debug_collector.time_event(
                    "llm_call",
                    agent_id=agent_id,
                    input={
                        "messages": messages,
                        "tools": tools or [],
                        "tool_choice": tool_choice,
                    },
                    name=trace_name,
                    metadata={
                        "model": self.model_name,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "message_count": len(messages),
                        "tools_count": len(tools or []),
                        "tool_choice": tool_choice,
                    },
                )

            response = await trace_async(
                name=self._llm_span_name(trace_name),
                run_type="llm",
                func=lambda: self.client.chat.completions.create(**request_params),
                inputs={
                    "messages": messages,
                    "tools": tools or [],
                    "tool_choice": tool_choice,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                metadata=self._langsmith_metadata(
                    trace_name=trace_name,
                    agent_id=agent_id,
                    debug_collector=debug_collector,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools_count=len(tools or []),
                    tool_choice=tool_choice,
                    iteration=iteration,
                    retry_count=retry_count,
                ),
                tags=["medical-agent-swarm", "llm", "tools"],
                output_mapper=self._response_trace_output,
            )

            # 解析响应
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # 提取工具调用
            tool_calls = []
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))
                logger.debug(f"LLM requested {len(tool_calls)} tool calls")

            usage = self._usage_to_dict(getattr(response, "usage", None))
            response_model = getattr(response, "model", self.model_name)
            response_id = getattr(response, "id", None)
            if timer:
                timer.finish(
                    output={
                        "content": message.content,
                        "tool_calls": [
                            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                            for tc in tool_calls
                        ],
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "model": response_model,
                        "response_id": response_id,
                    },
                    metadata={
                        "finish_reason": finish_reason,
                        "tool_calls_count": len(tool_calls),
                        "usage": usage,
                        "model": response_model,
                        "response_id": response_id,
                    },
                )

            return LLMResponse(
                content=message.content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
                model=response_model,
                response_id=response_id,
            )

        except Exception as e:
            if "timer" in locals() and timer:
                timer.finish(status="failed", error=str(e), output={"error": str(e)})
            logger.error(f"LLM call with tools failed: {e}")
            raise

    def create_tool_message(
        self,
        tool_call_id: str,
        tool_name: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建工具执行结果消息

        Args:
            tool_call_id: 工具调用ID
            tool_name: 工具名称
            result: 工具执行结果

        Returns:
            工具消息字典
        """
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": json.dumps(result, ensure_ascii=False)
        }

    def _usage_to_dict(self, usage: Any) -> Dict[str, Any]:
        """Convert provider token usage objects to a JSON-safe dictionary."""
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return usage
        if hasattr(usage, "model_dump") and callable(usage.model_dump):
            return usage.model_dump()
        if hasattr(usage, "dict") and callable(usage.dict):
            return usage.dict()
        if hasattr(usage, "__dict__"):
            return {
                key: value
                for key, value in vars(usage).items()
                if not key.startswith("_")
            }
        return {}

    def _langsmith_metadata(
        self,
        *,
        trace_name: str,
        agent_id: Optional[str],
        debug_collector: Optional[Any],
        temperature: float,
        max_tokens: int,
        tools_count: int,
        tool_choice: Optional[str] = None,
        iteration: int = 0,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "llm.model": self.model_name,
            "llm.provider": "openai-compatible",
            "llm.purpose": trace_name,
            "llm.agent_id": agent_id,
            "llm.iteration": iteration,
            "llm.retry_count": retry_count,
            "llm.outcome": "success",
            "status": "success",
            "agent_id": agent_id,
            "llm.available_tools": tools_count,
            "llm.tool_choice": tool_choice,
            "ls_provider": "openai-compatible",
            "ls_model_name": self.model_name,
            "ls_model_type": "chat",
            "ls_temperature": temperature,
            "ls_max_tokens": max_tokens,
        }
        if debug_collector:
            run = debug_collector.get_run()
            metadata["session_id"] = run.session_id
            metadata["run_id"] = run.run_id
        return metadata

    def _response_trace_output(self, response: Any) -> Dict[str, Any]:
        choices = getattr(response, "choices", []) or []
        first_choice = choices[0] if choices else None
        message = getattr(first_choice, "message", None) if first_choice else None
        tool_calls_count = 0
        if message and getattr(message, "tool_calls", None):
            tool_calls_count = len(message.tool_calls)
        usage = self._usage_to_dict(getattr(response, "usage", None))
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        output_tokens = usage.get(
            "output_tokens",
            usage.get("completion_tokens", 0),
        ) or 0
        return {
            "status": "success",
            "llm.input_tokens": input_tokens,
            "llm.output_tokens": output_tokens,
            "llm.total_tokens": usage.get("total_tokens", input_tokens + output_tokens) or 0,
            "llm.finish_reason": getattr(first_choice, "finish_reason", None),
            "llm.tool_calls_requested": tool_calls_count,
            "llm.outcome": "success",
        }

    def _llm_span_name(self, purpose: str) -> str:
        normalized = purpose.removeprefix("llm.").replace("_", ".")
        return f"llm.{normalized}"
