import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  Clock3,
  HeartPulse,
  MessageSquarePlus,
  RefreshCcw,
  ShieldAlert
} from "lucide-react";
import {
  createConsultation,
  getConsultation,
  getHealth,
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
  ConsultationSnapshot,
  Profile,
  RiskLevel,
  ServiceStatus
} from "./types";

const SESSION_KEY = "mas-user-session";
const PROFILE_KEY = "mas-user-profile";
type ConversationState =
  | "initializing"
  | "restoring_history"
  | "ready"
  | "creating_consultation"
  | "collaborating"
  | "success"
  | "failed"
  | "offline";
const POLL_INTERVAL_MS = import.meta.env.MODE === "test" ? 10 : 1500;
const TERMINAL_STATUSES = new Set(["success", "failed", "timeout"]);
const RISK_VALUES: RiskLevel[] = ["low", "medium", "high", "emergency"];
const AGENT_LABELS: Record<string, string> = {
  consultation_agent: "健康咨询",
  diagnostic_agent: "风险与症状分析",
  research_agent: "医学证据检索"
};

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
  if (stored) return stored;
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
  return [loaded.age, loaded.sex, loaded.medicalHistory, loaded.medications].some((value) => value.trim());
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function asSources(value: unknown): AssistantPayload["sources"] {
  if (!Array.isArray(value)) return [];
  return value.filter(
    (item): item is NonNullable<AssistantPayload["sources"]>[number] =>
      Boolean(item) && typeof item === "object" && typeof (item as { title?: unknown }).title === "string"
  );
}

function toServiceStatus(health: ApiHealth): ServiceStatus {
  return health.status === "ok" && health.memory.status === "ok" ? "healthy" : "degraded";
}

function buildPayloadFromMetadata(metadata: Record<string, unknown> | undefined): AssistantPayload | undefined {
  if (!metadata) return undefined;
  const rawRisk = metadata.risk_level;
  const riskLevel =
    typeof rawRisk === "string" && RISK_VALUES.includes(rawRisk.toLowerCase() as RiskLevel)
      ? (rawRisk.toLowerCase() as RiskLevel)
      : null;
  return {
    riskLevel,
    suggestions: asStringArray(metadata.suggestions),
    disclaimer: typeof metadata.disclaimer === "string" ? metadata.disclaimer : "",
    participants: asStringArray(metadata.agents_involved)
      .map((id) => AGENT_LABELS[id])
      .filter((label): label is string => Boolean(label)),
    sources: asSources(metadata.sources),
    safetyChecked: metadata.safety_checked === true,
    failed: false
  };
}

function buildAssistantMessage(snapshot: ConsultationSnapshot): ChatMessage {
  if (snapshot.status === "success" && snapshot.result) {
    return {
      id: `assistant-${snapshot.consultation_id}`,
      role: "assistant",
      content: snapshot.result.answer || "本次没有生成有效回答，请换个说法再试一次。",
      payload: {
        riskLevel: snapshot.result.risk_level,
        suggestions: snapshot.result.suggestions,
        disclaimer: snapshot.result.disclaimer,
        participants: snapshot.result.participants,
        sources: snapshot.result.sources,
        safetyChecked: snapshot.progress.safety_checked,
        failed: false
      }
    };
  }
  return {
    id: `assistant-${snapshot.consultation_id}`,
    role: "assistant",
    content: snapshot.failure?.message ?? "本轮会诊未能完成，请重新尝试。",
    payload: {
      riskLevel: null,
      suggestions: [],
      disclaimer: "",
      participants: [],
      safetyChecked: snapshot.progress.safety_checked,
      failed: true,
      timedOut: snapshot.status === "timeout"
    }
  };
}

function App() {
  const [sessionId, setSessionId] = useState(loadSessionId);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [historyStatus, setHistoryStatus] = useState<"loading" | "ready">("loading");
  const [input, setInput] = useState("");
  const [profile, setProfile] = useState<Profile>(loadProfile);
  const [rememberProfile, setRememberProfile] = useState(hasStoredProfile);
  const [profileOpen, setProfileOpen] = useState(false);
  const [snapshot, setSnapshot] = useState<ConsultationSnapshot | null>(null);
  const [activeConsultationId, setActiveConsultationId] = useState<string | null>(null);
  const [activeQuestion, setActiveQuestion] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [serviceStatus, setServiceStatus] = useState<ServiceStatus>("checking");
  const [confirmNewChat, setConfirmNewChat] = useState(false);
  const [previousSessionId, setPreviousSessionId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  const isBusy = submitting || activeConsultationId !== null;
  const conversationState: ConversationState =
    serviceStatus === "offline"
      ? "offline"
      : serviceStatus === "checking"
        ? "initializing"
        : historyStatus === "loading"
          ? "restoring_history"
          : submitting
            ? "creating_consultation"
            : activeConsultationId
              ? "collaborating"
              : snapshot?.status === "success"
                ? "success"
                : snapshot?.status === "failed" || snapshot?.status === "timeout"
                  ? "failed"
                  : "ready";
  const profileCount = [profile.age, profile.sex, profile.medicalHistory, profile.medications].filter(
    (value) => value.trim()
  ).length;

  const refreshHealth = useCallback(async () => {
    try {
      setServiceStatus(toServiceStatus(await getHealth()));
    } catch {
      setServiceStatus("offline");
    }
  }, []);

  useEffect(() => {
    void refreshHealth();
    const timer = window.setInterval(refreshHealth, 30_000);
    return () => window.clearInterval(timer);
  }, [refreshHealth]);

  useEffect(() => {
    let cancelled = false;
    setHistoryStatus("loading");
    setMessages([]);
    setSnapshot(null);
    getSessionMemory(sessionId)
      .then((memory) => {
        if (cancelled) return;
        setMessages(
          memory.recent_history
            .filter((item) => item.role === "user" || item.role === "assistant")
            .map((item, index): ChatMessage => ({
              id: `history-${sessionId}-${index}`,
              role: item.role as "user" | "assistant",
              content: item.content,
              payload:
                item.role === "assistant"
                  ? buildPayloadFromMetadata(item.metadata as Record<string, unknown> | undefined)
                  : undefined
            }))
        );
      })
      .catch(() => undefined)
      .finally(() => {
        if (!cancelled) setHistoryStatus("ready");
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  useEffect(() => {
    if (rememberProfile) localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
    else localStorage.removeItem(PROFILE_KEY);
  }, [profile, rememberProfile]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, snapshot]);

  useEffect(() => {
    if (!activeConsultationId) return;
    const consultationId = activeConsultationId;
    let cancelled = false;
    let timer: number | undefined;
    let failures = 0;

    const poll = async () => {
      try {
        const next = await getConsultation(consultationId, sessionId);
        if (cancelled) return;
        failures = 0;
        setSnapshot(next);
        if (TERMINAL_STATUSES.has(next.status)) {
          const completedMessage = buildAssistantMessage(next);
          if (next.status !== "success") completedMessage.retryQuestion = activeQuestion;
          setMessages((current) =>
            current.some((message) => message.id === completedMessage.id)
              ? current
              : [...current, completedMessage]
          );
          setActiveConsultationId(null);
          return;
        }
        timer = window.setTimeout(poll, POLL_INTERVAL_MS);
      } catch {
        failures += 1;
        if (cancelled) return;
        if (failures < 3) {
          timer = window.setTimeout(poll, POLL_INTERVAL_MS);
          return;
        }
        setServiceStatus("offline");
        setActiveConsultationId(null);
        setSnapshot(null);
        setMessages((current) => [
          ...current,
          {
            id: `assistant-network-${consultationId}`,
            role: "assistant",
            content: "连接中断了。请重新连接后再次分析；如症状严重或正在加重，请及时线下就医。",
            retryQuestion: activeQuestion,
            payload: {
              riskLevel: null,
              suggestions: [],
              disclaimer: "",
              participants: [],
              safetyChecked: false,
              failed: true
            }
          }
        ]);
      }
    };

    void poll();
    return () => {
      cancelled = true;
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [activeConsultationId, activeQuestion, sessionId]);

  const context = useMemo(() => {
    const entries: Record<string, string> = {};
    const age = Number(profile.age);
    if (profile.age.trim() && Number.isInteger(age) && age >= 0 && age <= 120) entries.age = String(age);
    if (profile.sex.trim()) entries.sex = profile.sex.trim();
    if (profile.medicalHistory.trim()) entries.medical_history = profile.medicalHistory.trim();
    if (profile.medications.trim()) entries.medications = profile.medications.trim();
    return entries;
  }, [profile]);

  const submitQuestion = async (question: string, appendUserMessage: boolean) => {
    const normalized = question.trim();
    if (!normalized || isBusy || serviceStatus === "offline") return;
    setSubmitting(true);
    setActiveQuestion(normalized);
    setSnapshot(null);
    if (appendUserMessage) {
      setMessages((current) => [
        ...current,
        { id: `user-${Date.now()}`, role: "user", content: normalized }
      ]);
    } else {
      setMessages((current) => current.filter((message) => message.retryQuestion !== normalized));
    }

    try {
      const created = await createConsultation({ question: normalized, context, session_id: sessionId });
      setInput("");
      setPreviousSessionId(null);
      setActiveConsultationId(created.consultation_id);
      void refreshHealth();
    } catch {
      setInput(normalized);
      setMessages((current) => [
        ...current,
        {
          id: `assistant-create-error-${Date.now()}`,
          role: "assistant",
          content: "暂时无法开始会诊。请检查连接后重试；如症状严重或正在加重，请及时线下就医。",
          retryQuestion: normalized,
          payload: {
            riskLevel: null,
            suggestions: [],
            disclaimer: "",
            participants: [],
            safetyChecked: false,
            failed: true
          }
        }
      ]);
      void refreshHealth();
    } finally {
      setSubmitting(false);
    }
  };

  const sendQuestion = () => void submitQuestion(input, true);
  const retryQuestion = (question: string) => void submitQuestion(question, false);

  const chooseExample = (question: string) => {
    setInput(question);
    composerRef.current?.focus();
  };

  const startNewConversation = () => {
    if (!isBusy) setConfirmNewChat(true);
  };

  const confirmStartNewConversation = () => {
    const fresh = newSessionId();
    setPreviousSessionId(sessionId);
    localStorage.setItem(SESSION_KEY, fresh);
    setSessionId(fresh);
    setInput("");
    setProfileOpen(false);
    setConfirmNewChat(false);
    setActiveConsultationId(null);
  };

  const restorePreviousConversation = () => {
    if (!previousSessionId) return;
    localStorage.setItem(SESSION_KEY, previousSessionId);
    setSessionId(previousSessionId);
    setPreviousSessionId(null);
    setInput("");
  };

  const clearProfile = () => {
    localStorage.removeItem(PROFILE_KEY);
    setRememberProfile(false);
    setProfile({ ...EMPTY_PROFILE });
  };

  const hasMessages = messages.length > 0;
  const composerDisabled =
    isBusy || historyStatus === "loading" || serviceStatus === "checking" || serviceStatus === "offline";

  return (
    <div className="app-shell" data-conversation-state={conversationState}>
      <header className="topbar">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true"><HeartPulse size={20} /></span>
          <div>
            <div className="brand-line">
              <h1>健康导航</h1>
              <span className="prototype-badge">研究原型</span>
            </div>
            <p>多角色协作的健康信息参考</p>
          </div>
        </div>
        <div className="topbar-actions">
          <div className={`service-status service-status-${serviceStatus}`} role="status" aria-live="polite">
            <span className="status-dot" aria-hidden="true" />
            <span>{SERVICE_LABELS[serviceStatus]}</span>
          </div>
          <div className="new-chat-control">
            <button type="button" className="new-chat-button" onClick={startNewConversation} disabled={isBusy} aria-expanded={confirmNewChat}>
              <MessageSquarePlus size={16} />新对话
            </button>
            {confirmNewChat && (
              <div className="new-chat-confirmation" role="dialog" aria-label="确认开始新对话">
                <p>当前对话仍会保留约 24 小时。开始后可立即返回。</p>
                <div>
                  <button type="button" onClick={() => setConfirmNewChat(false)}>取消</button>
                  <button type="button" className="confirm-button" onClick={confirmStartNewConversation}>开始新对话</button>
                </div>
              </div>
            )}
          </div>
        </div>
      </header>

      <div className="notice-banner" role="note">
        <span><ShieldAlert size={16} aria-hidden="true" />健康建议不能替代医生诊断</span>
        <strong>胸痛、呼吸困难、意识异常或严重出血，请立即拨打 120</strong>
        {serviceStatus === "offline" && (
          <button type="button" onClick={() => void refreshHealth()}><RefreshCcw size={14} />重新连接</button>
        )}
      </div>

      <div className="workspace">
        <section className="conversation-panel" aria-label="健康咨询对话">
          <main className="chat-area">
            {historyStatus === "loading" && (
              <div className="history-loading" role="status">正在恢复对话…</div>
            )}
            {historyStatus === "ready" && !hasMessages && !isBusy && (
              <div className="empty-state">
                {previousSessionId && (
                  <button type="button" className="restore-conversation-button" onClick={restorePreviousConversation}>返回上一对话</button>
                )}
                <p className="empty-kicker">先判断风险，再找到下一步</p>
                <h2>说说最担心的症状</h2>
                <p className="empty-sub">从最影响你的问题开始。信息越具体，建议越容易落实。</p>
                <div className="question-guide" aria-label="描述症状时建议包含">
                  <div className="guide-item"><Activity size={18} /><span><strong>不舒服的部位</strong><small>感觉与严重程度</small></span></div>
                  <div className="guide-item"><Clock3 size={18} /><span><strong>持续了多久</strong><small>突然、反复或持续</small></span></div>
                  <div className="guide-item"><HeartPulse size={18} /><span><strong>伴随的情况</strong><small>其他症状与用药</small></span></div>
                </div>
                <div className="example-list" aria-label="示例问题">
                  {EXAMPLE_QUESTIONS.map((question) => (
                    <button key={question} type="button" className="suggestion-chip" onClick={() => chooseExample(question)}>{question}</button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((message) => (
              <ChatMessageView key={message.id} message={message} onRetry={retryQuestion} />
            ))}
            {(snapshot || isBusy) && <div className="mobile-progress"><ProgressCard snapshot={snapshot} /></div>}
            <div ref={bottomRef} />
          </main>
          <footer className="input-area">
            <Composer
              ref={composerRef}
              value={input}
              onChange={setInput}
              onSend={sendQuestion}
              onOpenProfile={() => setProfileOpen(true)}
              disabled={composerDisabled}
              busy={isBusy}
              hasMessages={hasMessages}
              profileCount={profileCount}
            />
          </footer>
        </section>

        <aside className="consultation-sidebar"><ProgressCard snapshot={snapshot} /></aside>
      </div>

      <ProfileCard
        profile={profile}
        onChange={setProfile}
        open={profileOpen}
        onClose={() => setProfileOpen(false)}
        rememberProfile={rememberProfile}
        onRememberChange={setRememberProfile}
        onClear={clearProfile}
      />
    </div>
  );
}

export default App;
