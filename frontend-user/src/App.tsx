import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, Clock3, HeartPulse, MessageSquarePlus } from "lucide-react";
import {
  createRun,
  getHealth,
  getRun,
  getRunEvents,
  getSessionMemory
} from "./api/client";
import { Composer } from "./components/Composer";
import { ChatMessageView } from "./components/ChatMessage";
import { ProfileCard } from "./components/ProfileCard";
import { ProgressCard } from "./components/ProgressCard";
import {
  ApiHealth,
  AssistantPayload,
  ChatMessage,
  Profile,
  RiskLevel,
  RunEvent,
  RunProgress,
  RunSnapshot,
  ServiceStatus
} from "./types";

const SESSION_KEY = "mas-user-session";
const PROFILE_KEY = "mas-user-profile";
const POLL_INTERVAL_MS = 1500;
const TERMINAL_STATUSES = new Set(["success", "failed", "timeout"]);
const RISK_VALUES: RiskLevel[] = ["low", "medium", "high", "emergency"];

const EXAMPLE_QUESTIONS = [
  "最近两天头痛并伴有恶心，需要马上就医吗？",
  "发烧 38.5 度还有点咳嗽，在家怎么护理？",
  "长期熬夜之后心悸，有什么生活上的建议？"
];

const EMPTY_PROFILE: Profile = { age: "", sex: "", medicalHistory: "", medications: "" };
const SERVICE_LABELS: Record<ServiceStatus, string> = {
  checking: "正在连接",
  healthy: "服务已连接",
  degraded: "部分功能受限",
  offline: "服务未连接"
};

function newSessionId(): string {
  return `user-${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`;
}

function loadSessionId(): string {
  const stored = localStorage.getItem(SESSION_KEY);
  if (stored) {
    return stored;
  }
  const created = newSessionId();
  localStorage.setItem(SESSION_KEY, created);
  return created;
}

function loadProfile(): Profile {
  try {
    const stored = JSON.parse(localStorage.getItem(PROFILE_KEY) ?? "{}") as Record<string, unknown>;
    const sex = stored.sex;
    return {
      age: typeof stored.age === "string" ? stored.age : "",
      sex: sex === "女" || sex === "男" || sex === "其他" ? sex : "",
      medicalHistory: typeof stored.medicalHistory === "string" ? stored.medicalHistory : "",
      medications: typeof stored.medications === "string" ? stored.medications : ""
    };
  } catch {
    return { ...EMPTY_PROFILE };
  }
}

function hasStoredProfile(): boolean {
  const loaded = loadProfile();
  return [loaded.age, loaded.sex, loaded.medicalHistory, loaded.medications].some((value) =>
    value.trim()
  );
}

// 在任意嵌套结构里递归找某个键，用于从 result_json / 事件输出中提取 risk_level。
function findKeyDeep(value: unknown, key: string): unknown {
  if (Array.isArray(value)) {
    for (const item of value) {
      const found = findKeyDeep(item, key);
      if (found !== undefined) {
        return found;
      }
    }
    return undefined;
  }
  if (value && typeof value === "object") {
    const record = value as Record<string, unknown>;
    if (record[key] !== undefined) {
      return record[key];
    }
    for (const child of Object.values(record)) {
      const found = findKeyDeep(child, key);
      if (found !== undefined) {
        return found;
      }
    }
  }
  return undefined;
}

function extractRiskLevel(run: RunSnapshot, events: RunEvent[]): RiskLevel | null {
  const candidates = [run.result_json, ...events.slice().reverse().map((event) => event.output)];
  for (const candidate of candidates) {
    const raw = findKeyDeep(candidate, "risk_level");
    if (typeof raw === "string" && RISK_VALUES.includes(raw.toLowerCase() as RiskLevel)) {
      return raw.toLowerCase() as RiskLevel;
    }
  }
  return null;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function toServiceStatus(health: ApiHealth): ServiceStatus {
  return health.status === "ok" && health.memory.status === "ok" ? "healthy" : "degraded";
}

function buildFailureMessage(id: string, content: string): ChatMessage {
  return {
    id,
    role: "assistant",
    content,
    payload: {
      riskLevel: null,
      suggestions: [],
      disclaimer: "",
      agentsInvolved: [],
      failed: true
    }
  };
}

function buildPayloadFromMetadata(
  metadata: Record<string, unknown> | undefined
): AssistantPayload | undefined {
  if (!metadata) {
    return undefined;
  }
  const rawRisk = metadata.risk_level;
  const riskLevel =
    typeof rawRisk === "string" && RISK_VALUES.includes(rawRisk.toLowerCase() as RiskLevel)
      ? (rawRisk.toLowerCase() as RiskLevel)
      : null;
  return {
    riskLevel,
    suggestions: asStringArray(metadata.suggestions),
    disclaimer: typeof metadata.disclaimer === "string" ? metadata.disclaimer : "",
    agentsInvolved: asStringArray(metadata.agents_involved),
    failed: false
  };
}

function buildAssistantMessage(run: RunSnapshot, events: RunEvent[]): ChatMessage {
  const result = run.result_json ?? {};
  const failed = run.status === "failed";
  const answer = failed
    ? "抱歉，这次分析没有完成。请稍后重试；如果症状严重或在加重，请直接就医。"
    : run.final_answer ||
      (typeof result["answer"] === "string" ? (result["answer"] as string) : "") ||
      "抱歉，这次没有生成有效的回答，请换个说法再试一次。";
  const payload: AssistantPayload = {
    riskLevel: failed ? null : extractRiskLevel(run, events),
    suggestions: failed ? [] : asStringArray(result["suggestions"]),
    disclaimer: typeof result["disclaimer"] === "string" ? (result["disclaimer"] as string) : "",
    agentsInvolved: asStringArray(result["agents_involved"]),
    failed
  };
  return { id: `assistant-${run.run_id}`, role: "assistant", content: answer, payload };
}

function App() {
  const [sessionId, setSessionId] = useState(loadSessionId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [profile, setProfile] = useState<Profile>(loadProfile);
  const [rememberProfile, setRememberProfile] = useState(hasStoredProfile);
  const [profileOpen, setProfileOpen] = useState(false);
  const [progress, setProgress] = useState<RunProgress | null>(null);
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus>("checking");
  const [confirmNewChat, setConfirmNewChat] = useState(false);
  const [previousSessionId, setPreviousSessionId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const isBusy = progress !== null;

  // 持续探测服务状态，区分正常、降级与离线，避免把 HTTP 200 一律显示为正常。
  useEffect(() => {
    let active = true;
    const refreshHealth = () => {
      getHealth()
        .then((health) => {
          if (!active) return;
          setServiceStatus(toServiceStatus(health));
        })
        .catch(() => {
          if (active) setServiceStatus("offline");
        });
    };
    refreshHealth();
    const timer = window.setInterval(refreshHealth, 30_000);
    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  // 会话切换时恢复历史，并取消旧请求，避免新对话被旧会话结果覆盖。
  useEffect(() => {
    let cancelled = false;
    setMessages([]);
    getSessionMemory(sessionId)
      .then((memory) => {
        if (cancelled) return;
        const restored = memory.recent_history
          .filter((item) => item.role === "user" || item.role === "assistant")
          .map((item, index): ChatMessage => ({
            id: `history-${index}`,
            role: item.role as "user" | "assistant",
            content: item.content,
            payload:
              item.role === "assistant"
                ? buildPayloadFromMetadata(item.metadata as Record<string, unknown> | undefined)
                : undefined
          }));
        setMessages(restored);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  useEffect(() => {
    if (rememberProfile) {
      localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    } else {
      localStorage.removeItem(PROFILE_KEY);
    }
  }, [profile, rememberProfile]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, progress]);

  // 轮询当前 run：更新进度卡阶段，进入终态后转成助手消息。
  useEffect(() => {
    if (!progress) {
      return;
    }
    const runId = progress.runId;
    let cancelled = false;

    const poll = async () => {
      try {
        const [run, events] = await Promise.all([getRun(runId), getRunEvents(runId)]);
        if (cancelled) {
          return;
        }
        const stagesSeen = [...new Set(events.map((event) => event.stage))];
        const agentCount = new Set(
          events.filter((event) => event.agent_id).map((event) => event.agent_id)
        ).size;
        if (TERMINAL_STATUSES.has(run.status)) {
          setProgress(null);
          // 以 run_id 去重，避免终态时并发的两次轮询重复追加同一条回答。
          setMessages((current) =>
            current.some((message) => message.id === `assistant-${run.run_id}`)
              ? current
              : [...current, buildAssistantMessage(run, events)]
          );
        } else {
          setProgress({ runId, stagesSeen, agentCount });
        }
      } catch {
        if (!cancelled) {
          setServiceStatus("offline");
          setProgress(null);
          setMessages((current) => [
            ...current,
            buildFailureMessage(
              `assistant-error-${runId}`,
              "与服务器的连接中断了，请确认服务恢复后重试。"
            )
          ]);
        }
      }
    };

    const timer = window.setInterval(poll, POLL_INTERVAL_MS);
    void poll();
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [progress?.runId]);

  const context = useMemo(() => {
    const entries: Record<string, string> = {};
    const age = Number(profile.age);
    if (
      profile.age.trim() &&
      Number.isInteger(age) &&
      age >= 0 &&
      age <= 120
    ) {
      entries.age = String(age);
    }
    if (profile.sex.trim()) entries.sex = profile.sex.trim();
    if (profile.medicalHistory.trim()) entries.medical_history = profile.medicalHistory.trim();
    if (profile.medications.trim()) entries.medications = profile.medications.trim();
    return entries;
  }, [profile]);

  const sendQuestion = async () => {
    const question = input.trim();
    if (!question || isBusy) {
      return;
    }
    setInput("");
    setPreviousSessionId(null);
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", content: question }
    ]);
    try {
      const created = await createRun({ question, context, session_id: sessionId });
      getHealth()
        .then((health) => setServiceStatus(toServiceStatus(health)))
        .catch(() => setServiceStatus("offline"));
      setProgress({ runId: created.run_id, stagesSeen: [], agentCount: 0 });
    } catch {
      setServiceStatus("offline");
      setMessages((current) => [
        ...current,
        buildFailureMessage(
          `assistant-error-${Date.now()}`,
          "暂时联系不上分析服务，请确认服务恢复后重试。"
        )
      ]);
    }
  };

  const startNewConversation = () => {
    if (isBusy) {
      return;
    }
    setConfirmNewChat(true);
  };

  const confirmStartNewConversation = () => {
    const fresh = newSessionId();
    setPreviousSessionId(sessionId);
    localStorage.setItem(SESSION_KEY, fresh);
    setSessionId(fresh);
    setInput("");
    setProfileOpen(false);
    setConfirmNewChat(false);
  };

  const restorePreviousConversation = () => {
    if (!previousSessionId) return;
    localStorage.setItem(SESSION_KEY, previousSessionId);
    setSessionId(previousSessionId);
    setPreviousSessionId(null);
    setInput("");
    setProfileOpen(false);
  };

  const clearProfile = () => {
    localStorage.removeItem(PROFILE_KEY);
    setRememberProfile(false);
    setProfile({ ...EMPTY_PROFILE });
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="topbar-title">
          <h1>健康助手</h1>
          <div
            className={`service-status service-status-${serviceStatus}`}
            role="status"
            aria-live="polite"
          >
            <span className="status-dot" aria-hidden="true" />
            <span className="status-text">{SERVICE_LABELS[serviceStatus]}</span>
          </div>
        </div>
        <div className="new-chat-control">
          <button
            type="button"
            className="new-chat-button"
            onClick={startNewConversation}
            disabled={isBusy}
            aria-expanded={confirmNewChat}
          >
            <MessageSquarePlus size={15} />
            新对话
          </button>
          {confirmNewChat && (
            <div className="new-chat-confirmation" role="dialog" aria-label="确认开始新对话">
              <p>当前对话仍会保留约 24 小时。开始后可立即返回。</p>
              <div>
                <button type="button" onClick={() => setConfirmNewChat(false)}>
                  取消
                </button>
                <button type="button" className="confirm-button" onClick={confirmStartNewConversation}>
                  开始新对话
                </button>
              </div>
            </div>
          )}
        </div>
      </header>
      <div className="notice-banner" role="note">
        <span>健康建议不能替代医生诊断。</span>
        <strong>如有胸痛、呼吸困难、意识异常或严重出血，请立即拨打 120。</strong>
      </div>

      <main className="chat-area">
        {messages.length === 0 && !progress && (
          <div className="empty-state">
            {previousSessionId && (
              <button
                type="button"
                className="restore-conversation-button"
                onClick={restorePreviousConversation}
              >
                返回上一对话
              </button>
            )}
            <p className="empty-kicker">先判断风险，再整理下一步</p>
            <p className="empty-greeting">你好，说说哪里不舒服？</p>
            <p className="empty-sub">描述得越具体，越容易得到清晰的就医与护理建议。</p>
            <div className="question-guide" aria-label="描述症状时建议包含">
              <div className="guide-item">
                <Activity size={17} aria-hidden="true" />
                <span>
                  <strong>哪里不舒服</strong>
                  <small>部位与感觉</small>
                </span>
              </div>
              <div className="guide-item">
                <Clock3 size={17} aria-hidden="true" />
                <span>
                  <strong>从何时开始</strong>
                  <small>持续、反复或突然</small>
                </span>
              </div>
              <div className="guide-item">
                <HeartPulse size={17} aria-hidden="true" />
                <span>
                  <strong>还有什么异常</strong>
                  <small>伴随症状与用药</small>
                </span>
              </div>
            </div>
            <p className="example-heading">也可以从这些描述开始</p>
            <div className="example-list">
              {EXAMPLE_QUESTIONS.map((question) => (
                <button
                  key={question}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => setInput(question)}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((message) => (
          <ChatMessageView key={message.id} message={message} />
        ))}
        {progress && <ProgressCard progress={progress} />}
        <div ref={bottomRef} />
      </main>

      <footer className="input-area">
        <ProfileCard
          profile={profile}
          onChange={setProfile}
          open={profileOpen}
          onToggle={() => setProfileOpen((open) => !open)}
          rememberProfile={rememberProfile}
          onRememberChange={setRememberProfile}
          onClear={clearProfile}
        />
        <Composer value={input} onChange={setInput} onSend={sendQuestion} disabled={isBusy} />
      </footer>
    </div>
  );
}

export default App;
