import { DebugEvent, DebugRun, DebugStats, MemoryMessage } from "../types/debug";

// 提交 context 前移除空字符串，避免后端收到大量无意义字段。
export function compactObject(values: Record<string, string>) {
  return Object.fromEntries(
    Object.entries(values)
      .map(([key, value]) => [key, value.trim()])
      .filter(([, value]) => value)
  );
}

// 终态判断集中维护，轮询和 UI 状态都复用这一个规则。
export function isTerminalStatus(status?: string | null) {
  return status === "success" || status === "failed" || status === "timeout";
}

// 统一 JSON 格式化口径，保证 Raw、事件详情和导出内容一致。
export function safeStringify(value: unknown) {
  return JSON.stringify(value, null, 2);
}

// 耗时展示尽量短：毫秒级保留整数，超过一秒后用秒显示。
export function formatMs(value: number) {
  if (value >= 1000) {
    return `${(value / 1000).toFixed(1)} s`;
  }
  return `${Math.round(value)} ms`;
}

// run 还在执行时用当前时间估算耗时；结束后使用 ended_at 固定结果。
export function computeDurationMs(run: DebugRun | null) {
  if (!run) return 0;
  const start = new Date(run.started_at).getTime();
  const end = run.ended_at ? new Date(run.ended_at).getTime() : Date.now();
  return Math.max(0, end - start);
}

// 后端返回的 JSON 是弱类型，读取前先收窄成普通对象。
export function getRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

// 读取 event.metadata.usage 这类嵌套对象时使用，避免到处写类型判断。
export function getNestedRecord(value: unknown, key: string): Record<string, unknown> {
  return getRecord(getRecord(value)[key]);
}

// token usage 等字段可能缺失或不是数字，统一转成安全的 number。
export function getNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

// 从 run/events 派生顶部指标和 Overview 统计；不把这些统计写回后端。
export function buildStats(run: DebugRun | null, events: DebugEvent[]): DebugStats {
  const llmEvents = events.filter((event) => event.stage === "llm_call");
  const skillEvents = events.filter((event) => event.stage === "skill_call");
  const safetyEvents = events.filter((event) => event.stage === "safety_check");
  const constraintEvents = events.filter((event) => event.stage === "constraint_check");
  const memoryEvents = events.filter((event) => event.stage === "memory");

  let promptTokens = 0;
  let completionTokens = 0;
  let tokenTotal = 0;

  // token 使用量来自每个 llm_call 事件的 metadata.usage。
  for (const event of llmEvents) {
    const usage = getNestedRecord(event.metadata, "usage");
    promptTokens += getNumber(usage.prompt_tokens);
    completionTokens += getNumber(usage.completion_tokens);
    tokenTotal += getNumber(usage.total_tokens);
  }

  if (!tokenTotal) {
    tokenTotal = promptTokens + completionTokens;
  }

  const latestSafety = safetyEvents[safetyEvents.length - 1];
  const result = getRecord(run?.result_json);
  // 安全状态优先看最新 safety_check 事件，兼容旧 run 时再看 result_json。
  const safetyStatus =
    latestSafety?.status === "failed" || result.safety_passed === false
      ? "failed"
      : latestSafety?.status === "success" || result.safety_passed === true
        ? "passed"
        : "pending";

  return {
    eventCount: events.length,
    failedCount: events.filter((event) => event.status === "failed").length,
    llmCallCount: llmEvents.length,
    skillCallCount: skillEvents.length,
    constraintCheckCount: constraintEvents.length,
    memoryEventCount: memoryEvents.length,
    tokenTotal,
    promptTokens,
    completionTokens,
    safetyStatus,
    agentsInvolved: Array.from(
      new Set(events.map((event) => event.agent_id).filter(Boolean) as string[])
    ),
    durationMs: computeDurationMs(run)
  };
}

// 纯前端下载：把当前观测台收集到的数据打包成 JSON 文件。
export function downloadJson(filename: string, payload: unknown) {
  const blob = new Blob([safeStringify(payload)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

// 复制单个事件 JSON，便于把某次 LLM/Skill 调用单独发给其他人排查。
export async function copyJson(value: unknown) {
  await navigator.clipboard.writeText(safeStringify(value));
}

// 搜索用完整事件 JSON 做匹配，牺牲一点性能换取调试时“搜任意字段”的便利。
export function eventMatchesSearch(event: DebugEvent, search: string) {
  if (!search.trim()) return true;
  const needle = search.trim().toLowerCase();
  return safeStringify(event).toLowerCase().includes(needle);
}

// 事件标题按 name > skill_name > stage 降级，保证旧事件也有可读标题。
export function eventTitle(event: DebugEvent) {
  return event.name || event.skill_name || event.stage;
}

// Memory 消息角色用于会话气泡展示；保留原始 role 兼容未来新增的后端角色。
export function formatMemoryRole(role: unknown) {
  const value = String(role || "unknown").toLowerCase();
  if (value === "user") return "用户";
  if (value === "assistant") return "助手";
  if (value === "tool") return "工具";
  if (value === "system") return "系统";
  return value;
}

// role 同时作为 CSS class 使用，因此需要收敛到页面已定义的安全枚举。
export function memoryRoleClass(role: unknown) {
  const value = String(role || "unknown").toLowerCase();
  return ["user", "assistant", "tool", "system"].includes(value) ? value : "unknown";
}

// 对话内容优先展示 content；如果后端返回扩展结构，则回退到完整 JSON。
export function getMemoryMessageContent(message: MemoryMessage) {
  return typeof message.content === "string" ? message.content : safeStringify(message);
}

// 后端 timestamp 是 ISO 字符串；无法解析时保持空字符串，避免 UI 出现 Invalid Date。
export function formatMemoryTimestamp(message: MemoryMessage) {
  if (!message.timestamp || typeof message.timestamp !== "string") return "";
  const time = new Date(message.timestamp);
  return Number.isNaN(time.getTime()) ? "" : time.toLocaleTimeString();
}
