export type RiskLevel = "low" | "medium" | "high" | "emergency";
export type ServiceStatus = "checking" | "healthy" | "degraded" | "offline";
export type Sex = "" | "女" | "男" | "其他";

export interface Profile {
  age: string;
  sex: Sex;
  medicalHistory: string;
  medications: string;
}

export interface RunSnapshot {
  run_id: string;
  session_id: string | null;
  status: string;
  final_answer: string;
  result_json: Record<string, unknown> | null;
}

export interface RunEvent {
  sequence: number;
  stage: string;
  name: string | null;
  agent_id: string | null;
  skill_name: string | null;
  output: unknown;
  status: string;
}

export interface ApiHealth {
  status: string;
  memory: { backend: string; status: string };
}

export interface MemoryResponse {
  session_id: string;
  backend: string;
  recent_history: Array<{
    role: string;
    content: string;
    metadata?: {
      risk_level?: unknown;
      suggestions?: unknown;
      disclaimer?: unknown;
      agents_involved?: unknown;
    };
  }>;
}

// 助手回答里除正文外还带一组从 result_json / events 提炼出来的展示字段。
export interface AssistantPayload {
  riskLevel: RiskLevel | null;
  suggestions: string[];
  disclaimer: string;
  agentsInvolved: string[];
  failed: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  payload?: AssistantPayload;
}

// 等待回答期间进度卡的状态：已出现过的阶段 + 参与协作的 Agent 数。
export interface RunProgress {
  runId: string;
  stagesSeen: string[];
  agentCount: number;
}
