import {
  AgentInfo,
  DebugEvent,
  DebugRun,
  MemoryResponse,
  RunCreatePayload,
  RunCreateResponse,
  SkillInfo
} from "../types/debug";

// 前端默认连接本地 FastAPI；部署或联调其他端口时用 VITE_API_BASE_URL 覆盖。
export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

// 所有 API 请求统一走这里，保证错误处理和 JSON 解析口径一致。
async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<T>;
}

// 健康检查只用于前端显示后端可用性，不参与业务流程判断。
export async function getHealth(): Promise<{ status: string }> {
  return requestJson<{ status: string }>("/api/health");
}

// 创建 run 后后端会立刻返回 run_id，真正的 Agent 执行在后台异步进行。
export async function createRun(payload: RunCreatePayload): Promise<RunCreateResponse> {
  return requestJson<RunCreateResponse>("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

// 运行历史来自后端内存中的 RUN_STORE，刷新服务后历史会丢失。
export async function listRuns(limit = 50): Promise<DebugRun[]> {
  const payload = await requestJson<{ runs: DebugRun[] }>(`/api/runs?limit=${limit}`);
  return payload.runs;
}

// 获取 run 当前快照，主要用于轮询 status、route、final_answer 和 result_json。
export async function getRun(runId: string): Promise<DebugRun> {
  const payload = await requestJson<{ run: DebugRun }>(`/api/runs/${runId}`);
  return payload.run;
}

// 获取 run 的完整事件流；观测台的 Timeline/LLM/Skill 等视图都基于它派生。
export async function getRunEvents(runId: string): Promise<DebugEvent[]> {
  const payload = await requestJson<{ events: DebugEvent[] }>(`/api/runs/${runId}/events`);
  return payload.events;
}

// Agent 元数据包括配置、能力标签和已注册的 Skill 列表。
export async function getAgents(): Promise<AgentInfo[]> {
  return requestJson<AgentInfo[]>("/api/agents");
}

// Skill 元数据来自后端磁盘扫描，用于展示可发现/可调用工具。
export async function getSkills(): Promise<SkillInfo[]> {
  return requestJson<SkillInfo[]>("/api/skills");
}

// Memory 查询支持可选 query，用于同时取短期历史和长期相似案例。
export async function getSessionMemory(
  sessionId: string,
  query?: string,
  limit = 10
): Promise<MemoryResponse> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (query?.trim()) {
    params.set("query", query.trim());
  }
  return requestJson<MemoryResponse>(`/api/sessions/${encodeURIComponent(sessionId)}/memory?${params}`);
}
