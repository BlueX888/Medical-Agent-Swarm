// 后端 run 的执行状态；timeout 也是终态，不再继续轮询。
export type RunStatus = "running" | "success" | "failed" | "timeout";

// 前端健康检查状态，不直接等同于后端业务 run 状态。
export type ApiHealth = "checking" | "ok" | "failed";

// DebugRun 对应后端一次完整请求执行，承载最终答案、路由和运行级 metadata。
export type DebugRun = {
  run_id: string;
  session_id: string | null;
  question: string;
  context: Record<string, unknown>;
  started_at: string;
  ended_at: string | null;
  route: string | null;
  status: RunStatus;
  final_answer: string;
  result_json: Record<string, unknown> | null;
  metadata: Record<string, unknown>;
};

// DebugEvent 是观测台的核心数据单位；不同 stage 会被前端拆分到不同 Tab。
export type DebugEvent = {
  event_id: string | null;
  sequence: number;
  timestamp: string;
  stage: string;
  name: string | null;
  agent_id: string | null;
  skill_name: string | null;
  input: unknown;
  output: unknown;
  metadata: Record<string, unknown>;
  duration_ms: number | null;
  status: string;
  error: string | null;
};

// Skill 参数描述来自后端 SkillRegistry 的函数签名推断。
export type SkillParameter = {
  name: string;
  type: string;
  description: string;
  required: boolean;
  enum?: string[] | null;
};

// AgentInfo.skills 中的单个 Skill，包含给 LLM function calling 使用的参数信息。
export type AgentSkill = {
  name: string;
  description: string;
  is_async?: boolean;
  parameters?: SkillParameter[];
};

// Agent 静态元数据：能力、配置和已注册工具，用于 Agents Tab。
export type AgentInfo = {
  agent_id: string;
  class_name: string;
  capabilities: string[];
  config: Record<string, unknown>;
  skills: AgentSkill[];
};

// SkillInfo 来自磁盘扫描结果，用于 Skills Tab 展示可发现和 active 状态。
export type SkillInfo = {
  name: string;
  function_name: string;
  script_name: string;
  description: string;
  active: boolean;
  metadata: Record<string, unknown>;
};

// MemoryMessage 对应 ShortTermMemory 中保存的单条对话消息，允许后端后续附加更多调试字段。
export type MemoryMessage = Record<string, unknown> & {
  role?: string;
  content?: string;
  timestamp?: string;
};

// Memory API 返回值：短期历史、长期相似案例和长期记忆开关状态。
export type MemoryResponse = {
  session_id: string;
  backend: "memory" | "redis";
  ttl_seconds: number;
  recent_history: MemoryMessage[];
  historical_cases: Array<Record<string, unknown>>;
  long_term_enabled: boolean;
};

export type MemoryClearResponse = {
  session_id: string;
  cleared: boolean;
};

// 创建 run 的请求体，字段名保持与 FastAPI schema 一致。
export type RunCreatePayload = {
  question: string;
  context: Record<string, string>;
  session_id?: string;
  enable_swarm: boolean;
  enable_memory: boolean;
  enable_short_term_memory?: boolean;
  enable_long_term_memory?: boolean;
};

// POST /api/runs 的响应：run_id 用于后续轮询，run 是初始快照。
export type RunCreateResponse = {
  run_id: string;
  status: string;
  run: DebugRun;
};

// DebugStats 是前端派生统计，不是后端接口字段。
export type DebugStats = {
  eventCount: number;
  failedCount: number;
  llmCallCount: number;
  skillCallCount: number;
  constraintCheckCount: number;
  memoryEventCount: number;
  tokenTotal: number;
  promptTokens: number;
  completionTokens: number;
  safetyStatus: "passed" | "failed" | "pending";
  agentsInvolved: string[];
  durationMs: number;
};

// 右侧 Inspector 的固定视角列表；新增 Tab 时同步更新组件中的 tabs 常量。
export type InspectorTab =
  | "overview"
  | "timeline"
  | "llm"
  | "agents"
  | "skills"
  | "memory"
  | "safety"
  | "constraints"
  | "raw";

// 左侧快捷病例模板，只影响表单填充，不会自动触发请求。
export type CaseTemplate = {
  label: string;
  question: string;
  context: Record<string, string>;
};
