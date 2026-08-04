export type RiskLevel = "low" | "medium" | "high" | "emergency";
export type ServiceStatus = "checking" | "healthy" | "degraded" | "offline";
export type Sex = "" | "女" | "男" | "其他";
export type ConsultationStatus = "queued" | "running" | "success" | "failed" | "timeout";
export type ConsultationPhase =
  | "understanding"
  | "planning"
  | "consulting"
  | "safety_review"
  | "finalizing";

export interface Profile {
  age: string;
  sex: Sex;
  medicalHistory: string;
  medications: string;
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
      safety_checked?: unknown;
    };
  }>;
}

export interface ConsultationParticipant {
  id: string;
  label: string;
  state: "waiting" | "active" | "done" | "failed";
}

export interface ConsultationProgress {
  current_phase: ConsultationPhase;
  completed_phases: ConsultationPhase[];
  participants: ConsultationParticipant[];
  safety_checked: boolean;
}

export interface ConsultationResult {
  answer: string;
  risk_level: RiskLevel | null;
  suggestions: string[];
  disclaimer: string;
  participants: string[];
  sources: CitationSource[];
}

export interface CitationSource {
  citation_id: string;
  title: string;
  source_org: string;
  version: string;
  published_at: string;
  section: string;
  external_url: string;
}

export interface ConsultationFailure {
  code: "analysis_failed" | "analysis_timeout";
  message: string;
  retryable: boolean;
}

export interface ConsultationSnapshot {
  consultation_id: string;
  status: ConsultationStatus;
  progress: ConsultationProgress;
  result: ConsultationResult | null;
  failure: ConsultationFailure | null;
}

export interface AssistantPayload {
  riskLevel: RiskLevel | null;
  suggestions: string[];
  disclaimer: string;
  participants: string[];
  sources?: CitationSource[];
  safetyChecked: boolean;
  failed: boolean;
  timedOut?: boolean;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  payload?: AssistantPayload;
  retryQuestion?: string;
}
