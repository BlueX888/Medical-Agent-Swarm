import { ApiHealth, MemoryResponse, RunEvent, RunSnapshot } from "../types";

// 默认连接本地 FastAPI；部署时用 VITE_API_BASE_URL 覆盖。
export const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<T>;
}

export async function getHealth(): Promise<ApiHealth> {
  return requestJson("/api/health");
}

// 创建 run 后端立即返回 run_id，Agent 分析在后台异步执行，前端轮询取结果。
export async function createRun(payload: {
  question: string;
  context: Record<string, string>;
  session_id: string;
}): Promise<{ run_id: string; status: string }> {
  return requestJson("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, enable_swarm: true, enable_memory: true })
  });
}

export async function getRun(runId: string): Promise<RunSnapshot> {
  const payload = await requestJson<{ run: RunSnapshot }>(`/api/runs/${runId}`);
  return payload.run;
}

export async function getRunEvents(runId: string): Promise<RunEvent[]> {
  const payload = await requestJson<{ events: RunEvent[] }>(`/api/runs/${runId}/events`);
  return payload.events;
}

export async function getSessionMemory(sessionId: string, limit = 40): Promise<MemoryResponse> {
  return requestJson(`/api/sessions/${encodeURIComponent(sessionId)}/memory?limit=${limit}`);
}

export async function clearSessionMemory(sessionId: string): Promise<{ cleared: boolean }> {
  return requestJson(`/api/sessions/${encodeURIComponent(sessionId)}/memory`, {
    method: "DELETE"
  });
}
