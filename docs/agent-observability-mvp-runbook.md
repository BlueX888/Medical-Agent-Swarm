# Agent Observability MVP 运行手册

## 安全默认值

外部追踪默认关闭。启用后，Span 只发送低基数 metadata、参数键名、计数、类型、大小和状态，不发送问题、病史、回答、Prompt、工具参数/结果正文或原始 `session_id`。

只有在完全使用合成或已验证脱敏的数据时，才允许临时设置 `LANGSMITH_REDACT_MEDICAL_TEXT=false`。真实患者数据不得关闭正文脱敏。

## 启用

复制 `.env.example` 并设置：

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=medical-agent-swarm
LANGSMITH_REDACT_MEDICAL_TEXT=true
OBSERVABILITY_ENVIRONMENT=staging
OBSERVABILITY_ENTRYPOINT=api
OBSERVABILITY_HASH_KEY=<random-secret>
APP_VERSION=<release-or-git-sha>
```

`OBSERVABILITY_HASH_KEY` 未配置时不会发送 `session_ref`。密钥应通过部署平台的 Secret 管理，不应提交到仓库。

## Trace 结构

一次请求只应产生一个 `medical_swarm_request` 根 Trace，其下包含：

- `graph.<node>`：固定 Graph 节点；
- `agent.<agent_id>`：参与执行的 Agent；
- `llm.<purpose>`：模型调用、Token、finish reason 和请求工具数；
- `tool.<tool_name>`：允许、验证、结果状态和耗时；
- `safety.runtime_guard`：安全检查是否执行、通过和修改回答。

并行 Agent 必须显示为同一父 Span 下的并列子 Span。

## MVP 视图

在 LangSmith 项目中保存以下可筛选视图；若套餐不支持直接计算比率，则分别保存分子和分母查询：

| 视图 | 数据 |
|---|---|
| 请求量与状态 | Root 数量，按 environment/entrypoint/route/status |
| E2E 延迟 | Root duration P50/P95/P99 |
| Tool 触发 | 至少一个 Tool Span 的 Root / 全部 Root |
| Tool 调用 | Tool Span / Root；按 tool.name/agent_id/outcome |
| Tool 延迟 | Tool duration P50/P95 |
| LLM 使用 | LLM calls、input/output/total tokens / Root |
| 路由 | Root 按 route 分布 |
| Safety | executed/passed/error |
| 排障 | 最慢 Trace、Tool 最多 Trace |

统一过滤器：`deployment.environment`、`route`、`agent_id`、`tool.name`、`llm.model`、`status`、`app.version`。

## 本地核对

`debug.observability_summary.summarize_debug_run(run, events)` 是本地事实源。它从 `DebugRun` 和 `DebugEvent[]` 计算 Root 同构摘要，用于测试 Tool/LLM/Token/Safety 计数与 LangSmith 输出是否一致，不承担生产聚合。

运行验证：

```powershell
python -m pytest tests/test_observability_schema.py tests/test_agent_loop_tool_observability.py tests/test_observability_summary.py -q
python -m pytest -m "not integration" -q
```

## 排障

1. 没有 Trace：检查 `LANGSMITH_TRACING=true`、API Key 和项目名。
2. 没有 `session_ref`：这是缺少 `OBSERVABILITY_HASH_KEY` 时的预期安全行为。
3. Tool 数量不一致：对照本地 `skill_call` 和 `tool_policy` blocked 事件。
4. Token 为零：确认模型供应商响应包含 usage。
5. Safety 缺失：展开 `graph.build_response`，确认 `safety.runtime_guard` 存在。
6. exporter 或网络异常：业务请求应继续返回；查看本地 Loguru 警告。
7. 发现疑似正文：立即设置 `LANGSMITH_TRACING=false`，撤销访问、按组织流程清理远端数据并轮换密钥。

## 关闭

```env
LANGSMITH_TRACING=false
```

关闭后 `trace_async()` 是 no-op，业务函数的调用次数、返回值和异常传播保持不变。
