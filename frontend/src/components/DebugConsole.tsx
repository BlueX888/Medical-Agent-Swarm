import {
  Activity,
  AlertTriangle,
  BrainCircuit,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CirclePlus,
  Clipboard,
  Clock3,
  Database,
  Download,
  FileJson,
  History,
  ListFilter,
  MessageSquarePlus,
  MessageSquareText,
  Network,
  Play,
  RefreshCcw,
  Search,
  Server,
  ShieldCheck,
  Wrench
} from "lucide-react";
import { ReactNode, useMemo, useState } from "react";
import {
  AgentInfo,
  ApiHealth,
  CaseTemplate,
  DebugEvent,
  DebugRun,
  DebugStats,
  InspectorTab,
  MemoryMessage,
  MemoryResponse,
  SkillInfo
} from "../types/debug";
import {
  copyJson,
  eventMatchesSearch,
  eventTitle,
  formatMemoryRole,
  formatMemoryTimestamp,
  formatMs,
  getMemoryMessageContent,
  getNestedRecord,
  getRecord,
  memoryRoleClass,
  safeStringify
} from "../utils/debug";

// 表单值与 setter 分开建模，是为了让 InputPanel 保持“受控组件”但不拥有页面状态。
type FormValues = {
  question: string;
  age: string;
  sex: string;
  medicalHistory: string;
  medications: string;
  background: string;
  sessionId: string;
  enableSwarm: boolean;
  enableMemory: boolean;
  showRawData: boolean;
};

type FormSetters = {
  setQuestion: (value: string) => void;
  setAge: (value: string) => void;
  setSex: (value: string) => void;
  setMedicalHistory: (value: string) => void;
  setMedications: (value: string) => void;
  setBackground: (value: string) => void;
  setSessionId: (value: string) => void;
  setEnableSwarm: (value: boolean) => void;
  setEnableMemory: (value: boolean) => void;
  setShowRawData: (value: boolean) => void;
};

// 顶部状态栏：把 run 状态、后端健康、耗时和关键统计压缩成一眼可扫的运行摘要。
export function Topbar({
  run,
  stats,
  apiHealth,
  onRerun,
  onExport,
  canRerun,
  canExport
}: {
  run: DebugRun | null;
  stats: DebugStats;
  apiHealth: ApiHealth;
  onRerun: () => void;
  onExport: () => void;
  canRerun: boolean;
  canExport: boolean;
}) {
  return (
    <header className="topbar">
      <div className="brand-block">
        <div className="eyebrow">Medical-Agent-Swarm</div>
        <h1>全量参数日志观测台</h1>
      </div>
      <div className="status-strip">
        <StatusPill status={run?.status ?? "running"} idle={!run} />
        <Metric icon={<Server size={16} />} label="API" value={apiHealth} />
        <Metric icon={<Network size={16} />} label="Route" value={run?.route ?? "pending"} />
        <Metric icon={<Clock3 size={16} />} label="Elapsed" value={formatMs(stats.durationMs)} />
        <Metric icon={<Activity size={16} />} label="Events" value={String(stats.eventCount)} />
        <Metric icon={<BrainCircuit size={16} />} label="LLM" value={String(stats.llmCallCount)} />
        <Metric icon={<Wrench size={16} />} label="Skill" value={String(stats.skillCallCount)} />
        <Metric icon={<FileJson size={16} />} label="Tokens" value={String(stats.tokenTotal)} />
        <Metric icon={<ShieldCheck size={16} />} label="Safety" value={stats.safetyStatus} />
        <button className="icon-button" onClick={onRerun} disabled={!canRerun} title="重新运行">
          <RefreshCcw size={18} />
        </button>
        <button className="icon-button" onClick={onExport} disabled={!canExport} title="导出当前 run JSON">
          <Download size={18} />
        </button>
      </div>
    </header>
  );
}

// 左侧输入面板：负责构造请求、展示模板和运行历史，不直接知道后端调用细节。
export function InputPanel({
  values,
  setters,
  templates,
  runHistory,
  error,
  memoryError,
  isSubmitting,
  apiHealth,
  conversationMessages,
  onSubmit,
  onApplyTemplate,
  onSelectHistory,
  onRefreshHistory,
  onNewSession
}: {
  values: FormValues;
  setters: FormSetters;
  templates: CaseTemplate[];
  runHistory: DebugRun[];
  error: string;
  memoryError: string;
  isSubmitting: boolean;
  apiHealth: ApiHealth;
  conversationMessages: MemoryMessage[];
  onSubmit: (event: React.FormEvent) => void;
  onApplyTemplate: (index: number) => void;
  onSelectHistory: (runId: string) => void;
  onRefreshHistory: () => void;
  onNewSession: () => void;
}) {
  const isFollowUp = Boolean(values.sessionId.trim());
  const submitLabel = isSubmitting ? "启动中" : isFollowUp ? "发送追问" : "运行";

  return (
    <aside className="panel input-panel">
      <form id="run-form" onSubmit={onSubmit}>
        <PanelHeading icon={<BrainCircuit size={18} />} title="病例输入" />

        <label className="field">
          <span>{isFollowUp ? "追问" : "问题"}</span>
          <textarea
            value={values.question}
            onChange={(event) => setters.setQuestion(event.target.value)}
            rows={7}
            required
          />
        </label>

        <div className="template-row">
          {templates.map((template, index) => (
            <button key={template.label} type="button" onClick={() => onApplyTemplate(index)}>
              {template.label}
            </button>
          ))}
        </div>

        <div className="field-grid">
          <label className="field">
            <span>年龄</span>
            <input value={values.age} onChange={(event) => setters.setAge(event.target.value)} />
          </label>
          <label className="field">
            <span>性别</span>
            <input value={values.sex} onChange={(event) => setters.setSex(event.target.value)} />
          </label>
        </div>

        <label className="field">
          <span>病史</span>
          <textarea
            value={values.medicalHistory}
            onChange={(event) => setters.setMedicalHistory(event.target.value)}
            rows={3}
          />
        </label>

        <label className="field">
          <span>用药</span>
          <textarea
            value={values.medications}
            onChange={(event) => setters.setMedications(event.target.value)}
            rows={3}
          />
        </label>

        <label className="field">
          <span>背景</span>
          <textarea
            value={values.background}
            onChange={(event) => setters.setBackground(event.target.value)}
            rows={3}
          />
        </label>

        <div className="session-row">
          <label className="field session-field">
            <span>Session ID</span>
            <input value={values.sessionId} onChange={(event) => setters.setSessionId(event.target.value)} />
          </label>
          <button className="secondary-button" type="button" onClick={onNewSession} disabled={isSubmitting} title="开始新会话">
            <CirclePlus size={15} />
            <span>新会话</span>
          </button>
        </div>

        <div className="toggle-stack">
          <Toggle label="启用 Swarm" checked={values.enableSwarm} onChange={setters.setEnableSwarm} />
          <Toggle label="启用 Memory" checked={values.enableMemory} onChange={setters.setEnableMemory} />
          <Toggle label="显示 Raw Data" checked={values.showRawData} onChange={setters.setShowRawData} />
        </div>

        <div className={`api-health ${apiHealth}`}>
          <Server size={15} />
          <span>后端状态：{apiHealth}</span>
        </div>

        {error && <ErrorLine>{error}</ErrorLine>}
        {memoryError && <ErrorLine>{memoryError}</ErrorLine>}

        <button className="primary-button" type="submit" disabled={isSubmitting || !values.question.trim()}>
          {isFollowUp ? <MessageSquarePlus size={18} /> : <Play size={18} />}
          <span>{submitLabel}</span>
        </button>
      </form>

      <ConversationPanel sessionId={values.sessionId} messages={conversationMessages} />

      <section className="history-section">
        {/* 运行历史来自 /api/runs，只用于恢复本地内存中的 run，不做持久化假设。 */}
        <div className="subheading">
          <span>
            <History size={16} />
            运行历史
          </span>
          <button className="small-icon-button" type="button" onClick={onRefreshHistory} title="刷新历史">
            <RefreshCcw size={15} />
          </button>
        </div>
        <div className="history-list">
          {runHistory.length === 0 && <div className="empty-row">暂无历史</div>}
          {runHistory.map((historyRun) => (
            <button
              className="history-item"
              key={historyRun.run_id}
              type="button"
              onClick={() => onSelectHistory(historyRun.run_id)}
            >
              <strong>{historyRun.status}</strong>
              <span>{historyRun.route ?? "pending"}</span>
              <small>{historyRun.question || historyRun.run_id}</small>
            </button>
          ))}
        </div>
      </section>
    </aside>
  );
}

// 会话对话区把 Memory API 的 recent_history 转成可读消息流，让连续追问是否命中同一 session 一眼可见。
function ConversationPanel({ sessionId, messages }: { sessionId: string; messages: MemoryMessage[] }) {
  const visibleMessages = messages.slice(-12);

  return (
    <section className="conversation-section">
      <div className="subheading">
        <span>
          <MessageSquareText size={16} />
          会话对话
        </span>
        <small>{messages.length} 条</small>
      </div>
      <div className="session-summary">
        <span>session_id</span>
        <strong title={sessionId || undefined}>{sessionId || "未创建"}</strong>
      </div>
      <div className="conversation-list">
        {visibleMessages.length === 0 && <div className="empty-row">暂无会话消息</div>}
        {visibleMessages.map((message, index) => {
          const timestamp = formatMemoryTimestamp(message);
          return (
            <article className={`chat-message ${memoryRoleClass(message.role)}`} key={`${message.timestamp ?? "message"}-${index}`}>
              <div className="chat-message-head">
                <strong>{formatMemoryRole(message.role)}</strong>
                {timestamp && <time>{timestamp}</time>}
              </div>
              <pre>{getMemoryMessageContent(message)}</pre>
            </article>
          );
        })}
      </div>
    </section>
  );
}

// 中间上方面板：只展示最终答案，保留 pre 格式以避免模型输出的分段丢失。
export function AnswerPanel({ finalAnswer }: { finalAnswer: string }) {
  return (
    <section className="panel answer-panel">
      <PanelHeading icon={<Activity size={18} />} title="最终回答" />
      <div className="answer-body">
        {finalAnswer ? <pre>{finalAnswer}</pre> : <span className="muted">暂无完成回答</span>}
      </div>
    </section>
  );
}

// 中间下方面板：展示完整事件流的紧凑版本，详情仍可按事件展开。
export function TimelinePanel({ events }: { events: DebugEvent[] }) {
  const sortedEvents = useMemo(() => sortEvents(events), [events]);
  return (
    <section className="panel timeline-panel">
      <PanelHeading icon={<Clock3 size={18} />} title="事件时间线" />
      <div className="timeline">
        {sortedEvents.length === 0 && <div className="empty-row">暂无事件</div>}
        {sortedEvents.map((event) => (
          <EventRow event={event} key={event.event_id ?? `${event.stage}-${event.timestamp}`} />
        ))}
      </div>
    </section>
  );
}

// 右侧 Inspector 是观测台主体：同一批 run/events 按不同调试视角重新组织。
export function InspectorPanel({
  run,
  events,
  agents,
  skills,
  memory,
  stats,
  showRawData
}: {
  run: DebugRun | null;
  events: DebugEvent[];
  agents: AgentInfo[];
  skills: SkillInfo[];
  memory: MemoryResponse | null;
  stats: DebugStats;
  showRawData: boolean;
}) {
  const [activeTab, setActiveTab] = useState<InspectorTab>("overview");
  const sortedEvents = useMemo(() => sortEvents(events), [events]);

  return (
    <aside className="panel inspector-panel">
      <div className="tab-row wide">
        {tabs.map((tab) => (
          <button
            className={`tab-button ${activeTab === tab.id ? "active" : ""}`}
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            type="button"
          >
            {tab.icon}
            <span>{tab.label}</span>
          </button>
        ))}
      </div>

      <div className="inspector-section">
        {/* 每个 Tab 都只做前端聚合，不改变后端原始 trace。 */}
        {activeTab === "overview" && <OverviewTab run={run} events={sortedEvents} stats={stats} />}
        {activeTab === "timeline" && <TimelineTab events={sortedEvents} agents={agents} skills={skills} />}
        {activeTab === "llm" && <LLMTab events={sortedEvents.filter((event) => event.stage === "llm_call")} />}
        {activeTab === "agents" && <AgentsTab agents={agents} events={sortedEvents} />}
        {activeTab === "skills" && <SkillsTab skills={skills} events={sortedEvents} />}
        {activeTab === "memory" && <MemoryTab memory={memory} events={sortedEvents} />}
        {activeTab === "safety" && <SafetyTab events={sortedEvents.filter((event) => event.stage === "safety_check")} />}
        {activeTab === "constraints" && (
          <ConstraintsTab events={sortedEvents.filter((event) => event.stage === "constraint_check")} />
        )}
        {activeTab === "raw" && (
          <RawTab
            showRawData={showRawData}
            payload={{ run, events: sortedEvents, agents, skills, memory, stats }}
          />
        )}
      </div>
    </aside>
  );
}

// Overview 用于回答“这次运行整体发生了什么”：参数、结果、统计和事件阶段分布。
function OverviewTab({
  run,
  events,
  stats
}: {
  run: DebugRun | null;
  events: DebugEvent[];
  stats: DebugStats;
}) {
  return (
    <>
      <SummaryGrid
        items={[
          ["Events", stats.eventCount],
          ["Failed", stats.failedCount],
          ["LLM Calls", stats.llmCallCount],
          ["Skill Calls", stats.skillCallCount],
          ["Constraints", stats.constraintCheckCount],
          ["Memory", stats.memoryEventCount],
          ["Tokens", stats.tokenTotal],
          ["Safety", stats.safetyStatus]
        ]}
      />
      <JsonSection title="run metadata" value={run?.metadata ?? {}} />
      <JsonSection title="请求参数" value={run ? { question: run.question, context: run.context } : null} />
      <JsonSection title="结果字段" value={run?.result_json ?? null} />
      <JsonSection title="事件阶段统计" value={countBy(events, "stage")} />
    </>
  );
}

// Timeline Tab 提供多维筛选：stage、agent、skill、status 和全文 JSON 搜索。
function TimelineTab({
  events,
  agents,
  skills
}: {
  events: DebugEvent[];
  agents: AgentInfo[];
  skills: SkillInfo[];
}) {
  const [stage, setStage] = useState("");
  const [agentId, setAgentId] = useState("");
  const [skillName, setSkillName] = useState("");
  const [status, setStatus] = useState("");
  const [search, setSearch] = useState("");

  const stages = unique(events.map((event) => event.stage));
  const statuses = unique(events.map((event) => event.status));
  const agentOptions = unique([
    ...agents.map((agent) => agent.agent_id),
    ...events.map((event) => event.agent_id).filter(Boolean)
  ] as string[]);
  const skillOptions = unique([
    ...skills.map((skill) => skill.name),
    ...events.map((event) => event.skill_name).filter(Boolean)
  ] as string[]);

  const filtered = events.filter((event) => {
    // 筛选条件全部是前端本地计算，避免为了调试视图反复请求后端。
    if (stage && event.stage !== stage) return false;
    if (agentId && event.agent_id !== agentId) return false;
    if (skillName && event.skill_name !== skillName) return false;
    if (status && event.status !== status) return false;
    return eventMatchesSearch(event, search);
  });

  return (
    <>
      <div className="filter-bar">
        <FilterSelect icon={<ListFilter size={14} />} value={stage} onChange={setStage} options={stages} label="stage" />
        <FilterSelect value={agentId} onChange={setAgentId} options={agentOptions} label="agent" />
        <FilterSelect value={skillName} onChange={setSkillName} options={skillOptions} label="skill" />
        <FilterSelect value={status} onChange={setStatus} options={statuses} label="status" />
        <label className="search-box">
          <Search size={14} />
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索完整事件 JSON" />
        </label>
      </div>
      <div className="timeline inspector-timeline">
        {filtered.length === 0 && <div className="empty-row">没有匹配事件</div>}
        {filtered.map((event) => (
          <EventRow event={event} key={event.event_id ?? `${event.stage}-${event.timestamp}`} defaultOpen={false} />
        ))}
      </div>
    </>
  );
}

// LLM Tab 展示每次模型调用的请求、响应和 token 使用情况。
function LLMTab({ events }: { events: DebugEvent[] }) {
  return (
    <div className="card-list">
      {events.length === 0 && <div className="empty-row">暂无 LLM 调用</div>}
      {events.map((event) => {
        // usage 被放在 event.metadata.usage 中；没有 token 数据时用 0 占位。
        const usage = getNestedRecord(event.metadata, "usage");
        return (
          <div className="data-card" key={event.event_id ?? event.timestamp}>
            <CardHeader event={event} />
            <SummaryGrid
              items={[
                ["model", String(event.metadata?.model ?? "-")],
                ["finish_reason", String(event.metadata?.finish_reason ?? "-")],
                ["messages", String(event.metadata?.message_count ?? "-")],
                ["tools", String(event.metadata?.tools_count ?? "-")],
                ["prompt", String(usage.prompt_tokens ?? 0)],
                ["completion", String(usage.completion_tokens ?? 0)]
              ]}
            />
            <JsonSection title="请求 messages / tools" value={event.input} />
            <JsonSection title="响应" value={event.output} />
            <JsonSection title="metadata" value={event.metadata} />
          </div>
        );
      })}
    </div>
  );
}

// Agents Tab 合并静态 Agent 元数据和动态事件，用于定位哪个 Agent 参与、耗时或失败。
function AgentsTab({ agents, events }: { agents: AgentInfo[]; events: DebugEvent[] }) {
  const swarmEvents = events.filter((event) => event.stage === "swarm_context");
  return (
    <>
      <div className="agent-list">
        {agents.map((agent) => {
          // relatedEvents 是该 Agent 在当前 run 中留下的所有 trace 事件。
          const relatedEvents = events.filter((event) => event.agent_id === agent.agent_id);
          const failed = relatedEvents.some((event) => event.status === "failed");
          const totalMs = relatedEvents.reduce((sum, event) => sum + (event.duration_ms ?? 0), 0);
          return (
            <div className="agent-card" key={agent.agent_id}>
              <div className="agent-card-main">
                <div>
                  <strong>{agent.agent_id}</strong>
                  <span>{agent.class_name}</span>
                </div>
                <span className={failed ? "bad" : "good"}>{failed ? "failed" : "ok"}</span>
              </div>
              <SummaryGrid
                items={[
                  ["events", relatedEvents.length],
                  ["duration", formatMs(totalMs)],
                  ["skills", agent.skills.length],
                  ["capabilities", agent.capabilities.length]
                ]}
              />
              <TagRow values={agent.capabilities} />
              <JsonSection title="config" value={agent.config} />
              <JsonSection title="registered skills" value={agent.skills} />
            </div>
          );
        })}
      </div>
      <JsonSection title="swarm_context events" value={swarmEvents.map((event) => event.output)} />
    </>
  );
}

// Skills Tab 同时展示已注册 Skill 和本次实际调用记录，便于对比“可用”和“已用”。
function SkillsTab({ skills, events }: { skills: SkillInfo[]; events: DebugEvent[] }) {
  const skillEvents = events.filter((event) => event.stage === "skill_call");
  return (
    <>
      <SummaryGrid
        items={[
          ["registered", skills.filter((skill) => skill.active).length],
          ["called", skillEvents.length],
          ["failed", skillEvents.filter((event) => event.status === "failed").length],
          ["inactive", skills.filter((skill) => !skill.active).length]
        ]}
      />
      <div className="card-list">
        {skills.map((skill) => {
          // 后端事件中 skill_name 可能是展示名或函数名，这里两者都兼容。
          const calls = skillEvents.filter(
            (event) => event.skill_name === skill.name || event.skill_name === skill.function_name
          );
          return (
            <div className="data-card" key={`${skill.name}-${skill.function_name}`}>
              <div className="card-title-row">
                <strong>{skill.name}</strong>
                <span className={skill.active ? "good" : "bad"}>{skill.active ? "active" : "inactive"}</span>
              </div>
              <p className="card-description">{skill.description || "无描述"}</p>
              <SummaryGrid
                items={[
                  ["function", skill.function_name],
                  ["script", skill.script_name || "-"],
                  ["calls", calls.length],
                  ["failed", calls.filter((event) => event.status === "failed").length]
                ]}
              />
              <JsonSection title="metadata" value={skill.metadata} />
              <JsonSection title="call events" value={calls} />
            </div>
          );
        })}
      </div>
    </>
  );
}

// Memory Tab 展示显式 Memory API 返回值，并附带运行过程中记录的 memory 事件。
function MemoryTab({
  memory,
  events
}: {
  memory: MemoryResponse | null;
  events: DebugEvent[];
}) {
  const memoryEvents = events.filter((event) => event.stage === "memory" || event.stage === "load_memory" || event.stage === "save_memory");
  return (
    <>
      <SummaryGrid
        items={[
          ["session", memory?.session_id ?? "-"],
          ["recent", memory?.recent_history.length ?? 0],
          ["historical", memory?.historical_cases.length ?? 0],
          ["long_term", memory?.long_term_enabled ? "enabled" : "disabled"]
        ]}
      />
      <JsonSection title="recent_history" value={memory?.recent_history ?? []} />
      <JsonSection title="historical_cases" value={memory?.historical_cases ?? []} />
      <JsonSection title="memory events" value={memoryEvents} />
    </>
  );
}

// Safety Tab 专注最终安全审查与运行时兜底检查，失败项直接保留原始 output。
function SafetyTab({ events }: { events: DebugEvent[] }) {
  return (
    <div className="card-list">
      {events.length === 0 && <div className="empty-row">暂无安全审查</div>}
      {events.map((event) => (
        <div className="data-card" key={event.event_id ?? event.timestamp}>
          <CardHeader event={event} />
          <JsonSection title="input" value={event.input} />
          <JsonSection title="output" value={event.output} />
          <JsonSection title="metadata" value={event.metadata} />
        </div>
      ))}
    </div>
  );
}

// Constraints Tab 展示工具调用和输出约束校验，帮助判断系统是否越界或被自动修复。
function ConstraintsTab({ events }: { events: DebugEvent[] }) {
  return (
    <div className="card-list">
      {events.length === 0 && <div className="empty-row">暂无约束校验</div>}
      {events.map((event) => (
        <div className="data-card" key={event.event_id ?? event.timestamp}>
          <CardHeader event={event} />
          <JsonSection title="input" value={event.input} />
          <JsonSection title="output" value={event.output} />
          <JsonSection title="metadata" value={event.metadata} />
        </div>
      ))}
    </div>
  );
}

// Raw Tab 是最后兜底视图：只在用户显式打开 Raw Data 开关后展示完整 JSON。
function RawTab({ showRawData, payload }: { showRawData: boolean; payload: unknown }) {
  if (!showRawData) {
    return <div className="empty-row">Raw data hidden</div>;
  }
  return <pre className="json-block">{safeStringify(payload)}</pre>;
}

// 单条事件行：默认紧凑展示，展开后可查看 Input / Output / Metadata 三段原始 JSON。
function EventRow({ event, defaultOpen = false }: { event: DebugEvent; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const [copied, setCopied] = useState(false);
  const isFailed = event.status === "failed";

  async function handleCopy() {
    // 复制单条事件，便于把具体异常或 LLM 调用上下文单独贴给别人分析。
    await copyJson(event);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1100);
  }

  return (
    <div className={`timeline-event ${isFailed ? "failed" : ""}`}>
      <button className="event-toggle" type="button" onClick={() => setOpen((value) => !value)} title="展开事件详情">
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      <div className="event-marker">{isFailed ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}</div>
      <div className="event-content">
        <div className="event-title">
          <strong>#{event.sequence || "-"} {eventTitle(event)}</strong>
          <span>{event.stage}</span>
        </div>
        <div className="event-meta">
          {event.agent_id && <span>{event.agent_id}</span>}
          {event.skill_name && <span>{event.skill_name}</span>}
          {event.duration_ms !== null && <span>{formatMs(event.duration_ms)}</span>}
          <span>{new Date(event.timestamp).toLocaleTimeString()}</span>
          <span className={isFailed ? "bad" : "good"}>{event.status}</span>
        </div>
        {event.error && <div className="event-error">{event.error}</div>}
        {open && (
          <div className="event-details">
            <div className="detail-toolbar">
              <button className="text-button" type="button" onClick={handleCopy}>
                <Clipboard size={14} />
                <span>{copied ? "已复制" : "复制事件 JSON"}</span>
              </button>
            </div>
            <JsonSection title="Input" value={event.input} />
            <JsonSection title="Output" value={event.output} />
            <JsonSection title="Metadata" value={event.metadata} />
          </div>
        )}
      </div>
    </div>
  );
}

// 各类卡片复用的标题行，统一展示事件序号和状态颜色。
function CardHeader({ event }: { event: DebugEvent }) {
  return (
    <div className="card-title-row">
      <strong>#{event.sequence || "-"} {eventTitle(event)}</strong>
      <span className={event.status === "failed" ? "bad" : "good"}>{event.status}</span>
    </div>
  );
}

// 小型指标网格，专门用于高密度展示运行数字和短文本。
function SummaryGrid({ items }: { items: Array<[string, ReactNode]> }) {
  return (
    <div className="summary-grid">
      {items.map(([label, value]) => (
        <div key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

// JSON 折叠区：保持原始结构完整展示，避免调试时丢失嵌套字段。
function JsonSection({ title, value }: { title: string; value: unknown }) {
  const [open, setOpen] = useState(true);
  return (
    <section className="json-section">
      <button type="button" onClick={() => setOpen((value) => !value)}>
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span>{title}</span>
      </button>
      {open && <pre className="json-block">{safeStringify(value)}</pre>}
    </section>
  );
}

// 标题、状态徽标和基础控件都是轻量展示组件，保持业务组件更聚焦。
function PanelHeading({ icon, title }: { icon: ReactNode; title: string }) {
  return (
    <div className="panel-heading">
      {icon}
      <span>{title}</span>
    </div>
  );
}

function StatusPill({ status, idle }: { status: string; idle: boolean }) {
  const label = idle ? "idle" : status;
  return <span className={`status-pill ${label}`}>{label}</span>;
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="metric">
      {icon}
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  );
}

function ErrorLine({ children }: { children: ReactNode }) {
  return (
    <div className="error-line">
      <AlertTriangle size={16} />
      <span>{children}</span>
    </div>
  );
}

// select 筛选器统一封装，保证 Timeline Tab 的四类筛选在视觉上保持一致。
function FilterSelect({
  value,
  onChange,
  options,
  label,
  icon
}: {
  value: string;
  onChange: (value: string) => void;
  options: string[];
  label: string;
  icon?: ReactNode;
}) {
  return (
    <label className="filter-select">
      {icon}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        <option value="">{label}: all</option>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );
}

// Agent 能力标签展示；空数组时给出明确空状态，避免误以为加载失败。
function TagRow({ values }: { values: string[] }) {
  if (values.length === 0) {
    return <div className="empty-row">无能力标签</div>;
  }
  return (
    <div className="capability-row">
      {values.map((value) => (
        <span key={value}>{value}</span>
      ))}
    </div>
  );
}

// 按 stage/status 做简单计数，用于 Overview 的事件分布。
function countBy(events: DebugEvent[], key: "stage" | "status") {
  return events.reduce<Record<string, number>>((acc, event) => {
    const value = event[key] || "unknown";
    acc[value] = (acc[value] ?? 0) + 1;
    return acc;
  }, {});
}

// 事件排序优先使用后端 sequence；旧事件没有 sequence 时退回 timestamp。
function sortEvents(events: DebugEvent[]) {
  return [...events].sort((a, b) => {
    const sequenceDelta = (a.sequence || 0) - (b.sequence || 0);
    if (sequenceDelta !== 0) return sequenceDelta;
    return new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime();
  });
}

// 生成筛选下拉选项时去重并排序，保证列表稳定不跳动。
function unique(values: string[]) {
  return Array.from(new Set(values.filter(Boolean))).sort();
}

// Inspector 固定九个视角；新增 Tab 时要同步补充对应的渲染分支。
const tabs: Array<{ id: InspectorTab; label: string; icon: ReactNode }> = [
  { id: "overview", label: "Overview", icon: <Activity size={15} /> },
  { id: "timeline", label: "Timeline", icon: <Clock3 size={15} /> },
  { id: "llm", label: "LLM", icon: <BrainCircuit size={15} /> },
  { id: "agents", label: "Agents", icon: <Network size={15} /> },
  { id: "skills", label: "Skills", icon: <Wrench size={15} /> },
  { id: "memory", label: "Memory", icon: <Database size={15} /> },
  { id: "safety", label: "Safety", icon: <ShieldCheck size={15} /> },
  { id: "constraints", label: "Constraints", icon: <ListFilter size={15} /> },
  { id: "raw", label: "Raw", icon: <FileJson size={15} /> }
];
