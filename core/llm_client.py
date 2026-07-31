"""
LLM客户端
支持调用 OpenAI 兼容的 API（如字节跳动豆包、OpenAI、Deepseek 等）
支持 function calling
"""
import asyncio
import json
import os
import time
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

    if streaming := get_setting("LLM_STREAMING"):
        config["streaming"] = streaming.strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

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
    ttft_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    streamed: bool = False
    stream_fallback: bool = False

    def has_tool_calls(self) -> bool:
        """是否包含 function calls"""
        return len(self.tool_calls) > 0


@dataclass
class _CompletionSnapshot:
    """Provider-neutral completion collected from a streaming or full response."""

    content: Optional[str]
    tool_calls: List[ToolCall]
    finish_reason: Optional[str]
    usage: Dict[str, Any]
    model: Optional[str]
    response_id: Optional[str]
    ttft_ms: Optional[float]
    duration_ms: float
    streamed: bool
    stream_fallback: bool = False


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
            self.streaming = bool(self.config.get("streaming", True))
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
        retry_reason: Optional[str] = None,
        stream: Optional[bool] = None,
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
                        "retry_count": retry_count,
                        "retry_reason": retry_reason,
                    },
                )

            request_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
            response = await self._invoke_completion(
                request_params,
                trace_name=trace_name,
                trace_inputs={
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                trace_metadata=self._langsmith_metadata(
                    trace_name=trace_name,
                    agent_id=agent_id,
                    debug_collector=debug_collector,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools_count=0,
                    iteration=iteration,
                    retry_count=retry_count,
                    retry_reason=retry_reason,
                    stream=getattr(self, "streaming", True) if stream is None else stream,
                ),
                tags=["medical-agent-swarm", "llm"],
                stream=getattr(self, "streaming", True) if stream is None else stream,
            )

            content = response.content
            finish_reason = response.finish_reason
            usage = response.usage
            if timer:
                timer.finish(
                    output={
                        "content": content,
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "model": response.model or self.model_name,
                        "response_id": response.response_id,
                    },
                    metadata={
                        "finish_reason": finish_reason,
                        "usage": usage,
                        "model": response.model or self.model_name,
                        "response_id": response.response_id,
                        "ttft_ms": response.ttft_ms,
                        "duration_ms": response.duration_ms,
                        "streamed": response.streamed,
                    },
                )
            logger.debug(f"LLM response length: {len(content or '')} chars")
            return content or ""

        except Exception as e:
            if "timer" in locals() and timer:
                timer.finish(
                    status="failed",
                    error=str(e),
                    output={"error": str(e)},
                    metadata={"error_type": type(e).__name__},
                )
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
        last_error: Optional[str] = None
        for attempt in range(max_retries + 1):
            try:
                return await self.chat(
                    messages,
                    retry_count=attempt,
                    retry_reason=last_error,
                    **kwargs,
                )
            except Exception as e:
                last_error = self._retry_reason(e)
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "Retry {}/{} after error type={} code={}",
                    attempt + 1,
                    max_retries,
                    type(e).__name__,
                    last_error,
                )
                await asyncio.sleep(2 ** attempt)  # 指数退避

    async def chat_with_tools_retry(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 2,
        **kwargs,
    ) -> LLMResponse:
        """Retry tool-capable LLM calls while tracing every attempt."""
        last_error: Optional[str] = None
        for attempt in range(max_retries + 1):
            try:
                return await self.chat_with_tools(
                    messages=messages,
                    retry_count=attempt,
                    retry_reason=last_error,
                    **kwargs,
                )
            except Exception as error:
                last_error = self._retry_reason(error)
                if attempt >= max_retries:
                    raise
                logger.warning(
                    "Retry {}/{} after error type={} code={}",
                    attempt + 1,
                    max_retries,
                    type(error).__name__,
                    last_error,
                )
                await asyncio.sleep(2 ** attempt)

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
        retry_reason: Optional[str] = None,
        stream: Optional[bool] = None,
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
                        "retry_count": retry_count,
                        "retry_reason": retry_reason,
                    },
                )

            response = await self._invoke_completion(
                request_params,
                trace_name=trace_name,
                trace_inputs={
                    "messages": messages,
                    "tools": tools or [],
                    "tool_choice": tool_choice,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                trace_metadata=self._langsmith_metadata(
                    trace_name=trace_name,
                    agent_id=agent_id,
                    debug_collector=debug_collector,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools_count=len(tools or []),
                    tool_choice=tool_choice,
                    iteration=iteration,
                    retry_count=retry_count,
                    retry_reason=retry_reason,
                    stream=getattr(self, "streaming", True) if stream is None else stream,
                ),
                tags=["medical-agent-swarm", "llm", "tools"],
                stream=getattr(self, "streaming", True) if stream is None else stream,
            )

            # 解析响应
            finish_reason = response.finish_reason
            tool_calls = response.tool_calls
            if tool_calls:
                logger.debug(f"LLM requested {len(tool_calls)} tool calls")

            usage = response.usage
            response_model = response.model or self.model_name
            response_id = response.response_id
            if timer:
                timer.finish(
                    output={
                        "content": response.content,
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
                        "ttft_ms": response.ttft_ms,
                        "duration_ms": response.duration_ms,
                        "streamed": response.streamed,
                    },
                )

            return LLMResponse(
                content=response.content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
                model=response_model,
                response_id=response_id,
                ttft_ms=response.ttft_ms,
                duration_ms=response.duration_ms,
                streamed=response.streamed,
                stream_fallback=response.stream_fallback,
            )

        except Exception as e:
            if "timer" in locals() and timer:
                timer.finish(
                    status="failed",
                    error=str(e),
                    output={"error": str(e)},
                    metadata={"error_type": type(e).__name__},
                )
            logger.error(f"LLM call with tools failed: {e}")
            raise

    async def _invoke_completion(
        self,
        request_params: Dict[str, Any],
        *,
        trace_name: str,
        trace_inputs: Dict[str, Any],
        trace_metadata: Dict[str, Any],
        tags: List[str],
        stream: bool,
    ) -> _CompletionSnapshot:
        retry_count = int(trace_metadata.get("llm.retry_count", 0) or 0)
        return await trace_async(
            name=self._llm_span_name(trace_name),
            run_type="llm",
            func=lambda: self._create_completion(
                request_params,
                stream=stream,
            ),
            inputs=trace_inputs,
            metadata=trace_metadata,
            tags=tags,
            output_mapper=lambda response: self._response_trace_output(
                response,
                retry_count=retry_count,
            ),
        )

    async def _create_completion(
        self,
        request_params: Dict[str, Any],
        *,
        stream: bool,
    ) -> _CompletionSnapshot:
        params = dict(request_params)
        params.pop("stream", None)
        if not stream:
            params.pop("stream_options", None)
            started = time.perf_counter()
            response = await self.client.chat.completions.create(**params)
            return self._snapshot_from_response(
                response,
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        params["stream"] = True
        params.setdefault("stream_options", {"include_usage": True})
        started = time.perf_counter()
        try:
            response_stream = await self.client.chat.completions.create(**params)
            if hasattr(response_stream, "choices") and not hasattr(
                response_stream, "__aiter__"
            ):
                snapshot = self._snapshot_from_response(
                    response_stream,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                snapshot.stream_fallback = True
                return snapshot
            return await self._collect_stream(
                response_stream,
                started=started,
            )
        except Exception as error:
            if not self._is_stream_unsupported(error):
                raise
            fallback_params = dict(request_params)
            fallback_params.pop("stream", None)
            fallback_params.pop("stream_options", None)
            fallback_started = time.perf_counter()
            response = await self.client.chat.completions.create(**fallback_params)
            snapshot = self._snapshot_from_response(
                response,
                duration_ms=(time.perf_counter() - fallback_started) * 1000,
            )
            snapshot.stream_fallback = True
            return snapshot

    async def _collect_stream(
        self,
        response_stream: Any,
        *,
        started: float,
    ) -> _CompletionSnapshot:
        content_parts: List[str] = []
        tool_states: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        usage: Dict[str, Any] = {}
        model: Optional[str] = None
        response_id: Optional[str] = None
        first_token_at: Optional[float] = None

        async for chunk in response_stream:
            model = self._field(chunk, "model") or model
            response_id = self._field(chunk, "id") or response_id
            chunk_usage = self._usage_to_dict(self._field(chunk, "usage"))
            if chunk_usage:
                usage = chunk_usage
            for choice in self._field(chunk, "choices", []) or []:
                finish_reason = self._field(choice, "finish_reason") or finish_reason
                delta = self._field(choice, "delta")
                if delta is None:
                    continue
                delta_content = self._field(delta, "content")
                if delta_content:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    content_parts.append(str(delta_content))
                for tool_delta in self._field(delta, "tool_calls", []) or []:
                    index = int(self._field(tool_delta, "index", 0) or 0)
                    state = tool_states.setdefault(
                        index,
                        {"id": "", "name": "", "arguments": ""},
                    )
                    delta_id = self._field(tool_delta, "id")
                    function = self._field(tool_delta, "function")
                    delta_name = self._field(function, "name") if function else None
                    delta_arguments = (
                        self._field(function, "arguments") if function else None
                    )
                    if delta_id or delta_name or delta_arguments:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                    if delta_id:
                        state["id"] += str(delta_id)
                    if delta_name:
                        state["name"] += str(delta_name)
                    if delta_arguments:
                        state["arguments"] += str(delta_arguments)

        tool_calls = [
            ToolCall(
                id=state["id"] or f"stream-tool-{index}",
                name=state["name"],
                arguments=self._parse_tool_arguments(state["arguments"]),
            )
            for index, state in sorted(tool_states.items())
        ]
        return _CompletionSnapshot(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            model=model or self.model_name,
            response_id=response_id,
            ttft_ms=(
                (first_token_at - started) * 1000
                if first_token_at is not None
                else None
            ),
            duration_ms=(time.perf_counter() - started) * 1000,
            streamed=True,
        )

    def _snapshot_from_response(
        self,
        response: Any,
        *,
        duration_ms: float,
    ) -> _CompletionSnapshot:
        choices = self._field(response, "choices", []) or []
        first_choice = choices[0] if choices else None
        message = self._field(first_choice, "message") if first_choice else None
        content = self._field(message, "content") if message else None
        tool_calls: List[ToolCall] = []
        for tool_call in self._field(message, "tool_calls", []) or []:
            function = self._field(tool_call, "function")
            tool_calls.append(
                ToolCall(
                    id=str(self._field(tool_call, "id", "")),
                    name=str(self._field(function, "name", "")),
                    arguments=self._parse_tool_arguments(
                        self._field(function, "arguments", "{}")
                    ),
                )
            )
        return _CompletionSnapshot(
            content=content,
            tool_calls=tool_calls,
            finish_reason=self._field(first_choice, "finish_reason")
            if first_choice
            else None,
            usage=self._usage_to_dict(self._field(response, "usage")),
            model=self._field(response, "model") or self.model_name,
            response_id=self._field(response, "id"),
            ttft_ms=None,
            duration_ms=duration_ms,
            streamed=False,
        )

    @staticmethod
    def _field(value: Any, name: str, default: Any = None) -> Any:
        if value is None:
            return default
        if isinstance(value, dict):
            return value.get(name, default)
        return getattr(value, name, default)

    @staticmethod
    def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if not arguments:
            return {}
        parsed = json.loads(arguments)
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _is_stream_unsupported(error: BaseException) -> bool:
        message = str(error).lower()
        return isinstance(error, TypeError) or any(
            marker in message
            for marker in (
                "stream not supported",
                "unknown parameter: stream",
                "unexpected keyword argument 'stream'",
                "stream_options",
            )
        )

    @staticmethod
    def _retry_reason(error: BaseException) -> str:
        if isinstance(error, TimeoutError):
            return "timeout"
        if isinstance(error, PermissionError):
            return "permission_denied"
        if isinstance(error, (ValueError, TypeError)):
            return "validation_error"
        return type(error).__name__.lower()

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
        retry_reason: Optional[str] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        metadata: Dict[str, Any] = {
            "llm.model": self.model_name,
            "llm.provider": "openai-compatible",
            "llm.purpose": trace_name,
            "llm.agent_id": agent_id,
            "llm.iteration": iteration,
            "llm.retry_count": retry_count,
            "llm.attempt_index": retry_count + 1,
            "llm.retry_reason": retry_reason,
            "llm.streaming": stream,
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

    def _response_trace_output(
        self,
        response: Any,
        *,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        if isinstance(response, _CompletionSnapshot):
            tool_calls_count = len(response.tool_calls)
            usage = response.usage
            finish_reason = response.finish_reason
            ttft_ms = response.ttft_ms
            duration_ms = response.duration_ms
            streamed = response.streamed
            stream_fallback = response.stream_fallback
        else:
            choices = getattr(response, "choices", []) or []
            first_choice = choices[0] if choices else None
            message = getattr(first_choice, "message", None) if first_choice else None
            tool_calls_count = 0
            if message and getattr(message, "tool_calls", None):
                tool_calls_count = len(message.tool_calls)
            usage = self._usage_to_dict(getattr(response, "usage", None))
            finish_reason = getattr(first_choice, "finish_reason", None)
            ttft_ms = None
            duration_ms = None
            streamed = False
            stream_fallback = False
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
        output_tokens = usage.get(
            "output_tokens",
            usage.get("completion_tokens", 0),
        ) or 0
        mapped = {
            "status": "success",
            "llm.input_tokens": input_tokens,
            "llm.output_tokens": output_tokens,
            "llm.total_tokens": usage.get("total_tokens", input_tokens + output_tokens) or 0,
            "llm.finish_reason": finish_reason,
            "llm.tool_calls_requested": tool_calls_count,
            "llm.retry_count": retry_count,
            "llm.streaming": streamed,
            "llm.stream_fallback": stream_fallback,
            "llm.outcome": "success",
        }
        if ttft_ms is not None:
            mapped["llm.ttft_ms"] = round(float(ttft_ms), 3)
        if duration_ms is not None:
            mapped["llm.duration_ms"] = round(float(duration_ms), 3)
            mapped["duration_ms"] = round(float(duration_ms), 3)
        return mapped

    def _llm_span_name(self, purpose: str) -> str:
        normalized = purpose.removeprefix("llm.").replace("_", ".")
        return f"llm.{normalized}"
