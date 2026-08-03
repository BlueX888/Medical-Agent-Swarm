import { ApiHealth, ConsultationSnapshot, MemoryResponse } from "../types";

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

export async function createConsultation(payload: {
  question: string;
  context: Record<string, string>;
  session_id: string;
}): Promise<{ consultation_id: string; status: string }> {
  return requestJson("/api/consultations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
}

export async function getConsultation(
  consultationId: string,
  sessionId: string
): Promise<ConsultationSnapshot> {
  return requestJson(`/api/consultations/${encodeURIComponent(consultationId)}`, {
    headers: { "X-Session-ID": sessionId }
  });
}

export async function getSessionMemory(sessionId: string, limit = 40): Promise<MemoryResponse> {
  return requestJson(`/api/sessions/${encodeURIComponent(sessionId)}/memory?limit=${limit}`);
}

export async function clearSessionMemory(sessionId: string): Promise<{ cleared: boolean }> {
  return requestJson(`/api/sessions/${encodeURIComponent(sessionId)}/memory`, {
    method: "DELETE"
  });
}
