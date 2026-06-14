import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  createRun,
  getAgents,
  getHealth,
  getRun,
  getRunEvents,
  getSessionMemory,
  getSkills,
  listRuns
} from "./api/debugApi";
import {
  AnswerPanel,
  InputPanel,
  InspectorPanel,
  TimelinePanel,
  Topbar
} from "./components/DebugConsole";
import { AgentInfo, ApiHealth, CaseTemplate, DebugEvent, DebugRun, MemoryResponse, SkillInfo } from "./types/debug";
import { buildStats, compactObject, downloadJson, getRecord, isTerminalStatus } from "./utils/debug";

// 常用调试病例模板：用于快速验证急症、感染和用药咨询三类典型链路。
const templates: CaseTemplate[] = [
  {
    label: "胸痛",
    question: "Chest tightness and shortness of breath for 30 minutes. Is this urgent?",
    context: { age: "58", sex: "male", medical_history: "Hypertension", medications: "Amlodipine" }
  },
  {
    label: "发热",
    question: "Fever, sore throat, and cough for two days. What should I do?",
    context: { age: "31", sex: "female", medical_history: "No major history", medications: "None" }
  },
  {
    label: "用药",
    question: "Can I stop my blood pressure medication if my home readings improved?",
    context: { age: "46", sex: "female", medical_history: "Hypertension", medications: "Losartan" }
  }
];

function App() {
  // 左侧病例输入区的表单状态。字段名尽量贴近后端 context，提交时再统一压缩空值。
  const [question, setQuestion] = useState("");
  const [age, setAge] = useState("");
  const [sex, setSex] = useState("");
  const [medicalHistory, setMedicalHistory] = useState("");
  const [medications, setMedications] = useState("");
  const [background, setBackground] = useState("");
  const [sessionId, setSessionId] = useState("");
  const [enableSwarm, setEnableSwarm] = useState(true);
  const [enableMemory, setEnableMemory] = useState(true);
  const [showRawData, setShowRawData] = useState(false);

  // 当前 run 及其周边调试数据：events 是观测台最核心的数据源。
  const [run, setRun] = useState<DebugRun | null>(null);
  const [events, setEvents] = useState<DebugEvent[]>([]);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [memory, setMemory] = useState<MemoryResponse | null>(null);
  const [runHistory, setRunHistory] = useState<DebugRun[]>([]);

  // UI 状态：错误、提交中、API 健康状态和 Memory 独立错误分开记录，避免互相覆盖。
  const [apiHealth, setApiHealth] = useState<ApiHealth>("checking");
  const [isSubmitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [memoryError, setMemoryError] = useState("");

  // 首屏启动时并行加载健康检查、Agent/Skill 元数据和最近 run 历史。
  useEffect(() => {
    void bootstrap();
  }, []);

  // 后端会为新会话生成 session_id；拿到后回填输入框，便于后续连续追问。
  useEffect(() => {
    if (run?.session_id) {
      setSessionId(run.session_id);
    }
  }, [run?.session_id]);

  // run 处于 running 时每秒轮询 run 和 events；终态后停止，避免无意义请求。
  useEffect(() => {
    if (!run?.run_id || isTerminalStatus(run.status)) {
      return;
    }

    const poll = async () => {
      try {
        const [nextRun, nextEvents] = await Promise.all([
          getRun(run.run_id),
          getRunEvents(run.run_id)
        ]);
        setRun(nextRun);
        setEvents(nextEvents);
        if (isTerminalStatus(nextRun.status)) {
          void refreshRunHistory();
        }
      } catch (pollError) {
        setError(pollError instanceof Error ? pollError.message : "Polling failed");
      }
    };

    void poll();
    const timer = window.setInterval(poll, 1000);
    return () => window.clearInterval(timer);
  }, [run?.run_id, run?.status]);

  // Memory 面板依赖 session_id。这里单独加载，失败时只影响 Memory 区，不阻塞主流程。
  useEffect(() => {
    const activeSession = run?.session_id || sessionId;
    if (!activeSession) {
      setMemory(null);
      setMemoryError("");
      return;
    }

    let cancelled = false;
    const loadMemory = async () => {
      try {
        const payload = await getSessionMemory(activeSession, question || run?.question || "");
        if (!cancelled) {
          setMemory(payload);
          setMemoryError("");
        }
      } catch (memoryLoadError) {
        if (!cancelled) {
          setMemory(null);
          setMemoryError(memoryLoadError instanceof Error ? memoryLoadError.message : "Memory load failed");
        }
      }
    };

    void loadMemory();
    return () => {
      cancelled = true;
    };
  }, [run?.session_id, sessionId, run?.status]);

  // stats 是从 events/run 派生出的顶部指标和 Overview 指标，避免在多个组件里重复计算。
  const stats = useMemo(() => buildStats(run, events), [run, events]);
  const finalAnswer = run?.final_answer || String(run?.result_json?.answer ?? "");

  // 启动加载被拆成三个函数，方便页面内局部刷新历史或健康状态。
  async function bootstrap() {
    await Promise.all([
      refreshHealth(),
      refreshMetadata(),
      refreshRunHistory()
    ]);
  }

  async function refreshHealth() {
    setApiHealth("checking");
    try {
      await getHealth();
      setApiHealth("ok");
    } catch {
      setApiHealth("failed");
    }
  }

  async function refreshMetadata() {
    try {
      const [nextAgents, nextSkills] = await Promise.all([getAgents(), getSkills()]);
      setAgents(nextAgents);
      setSkills(nextSkills);
    } catch (metadataError) {
      setError(metadataError instanceof Error ? metadataError.message : "Metadata load failed");
    }
  }

  async function refreshRunHistory() {
    try {
      setRunHistory(await listRuns(50));
    } catch {
      setRunHistory([]);
    }
  }

  async function submitRun(event: FormEvent) {
    event.preventDefault();
    // 新请求开始时清空旧 run/events，避免短时间内旧结果和新提交混在一起。
    setError("");
    setMemoryError("");
    setSubmitting(true);
    setEvents([]);
    setRun(null);

    const context = compactObject({
      age,
      sex,
      medical_history: medicalHistory,
      medications,
      background
    });

    try {
      // POST /api/runs 会立即返回 run_id，真正的 Agent 执行由后端后台任务完成。
      const payload = await createRun({
        question,
        context,
        session_id: sessionId || undefined,
        enable_swarm: enableSwarm,
        enable_memory: enableMemory
      });
      setRun(payload.run);
      if (payload.run.session_id) {
        setSessionId(payload.run.session_id);
      }
      void refreshRunHistory();
    } catch (submitError) {
      setError(submitError instanceof Error ? submitError.message : "Run failed");
    } finally {
      setSubmitting(false);
    }
  }

  function applyTemplate(index: number) {
    // 模板只填充输入，不自动运行，方便用户在运行前微调病例。
    const template = templates[index];
    setQuestion(template.question);
    setAge(String(template.context.age ?? ""));
    setSex(String(template.context.sex ?? ""));
    setMedicalHistory(String(template.context.medical_history ?? ""));
    setMedications(String(template.context.medications ?? ""));
    setBackground("");
  }

  async function selectHistoryRun(runId: string) {
    setError("");
    setMemoryError("");
    try {
      // 运行历史恢复需要同时取 run 和 events，才能完整还原右侧观测台。
      const [selectedRun, selectedEvents] = await Promise.all([
        getRun(runId),
        getRunEvents(runId)
      ]);
      setRun(selectedRun);
      setEvents(selectedEvents);
      setQuestion(selectedRun.question ?? "");
      setSessionId(selectedRun.session_id ?? "");

      const context = getRecord(selectedRun.context);
      // 历史 run 的 context 是弱类型 JSON，这里按已知表单字段做安全回填。
      setAge(String(context.age ?? ""));
      setSex(String(context.sex ?? ""));
      setMedicalHistory(String(context.medical_history ?? ""));
      setMedications(String(context.medications ?? ""));
      setBackground(String(context.background ?? ""));
    } catch (historyError) {
      setError(historyError instanceof Error ? historyError.message : "Run history load failed");
    }
  }

  function rerun() {
    // 复用浏览器原生表单提交能力，确保与点击“运行”走同一条提交链路。
    const form = document.getElementById("run-form") as HTMLFormElement | null;
    form?.requestSubmit();
  }

  function startNewSession() {
    // 新会话必须同时清空前端当前 run 与 Memory 快照，否则 UI 会混入上一轮对话。
    setQuestion("");
    setAge("");
    setSex("");
    setMedicalHistory("");
    setMedications("");
    setBackground("");
    setSessionId("");
    setRun(null);
    setEvents([]);
    setMemory(null);
    setError("");
    setMemoryError("");
  }

  function exportRun() {
    if (!run) return;
    // 导出包包含页面当前可见的所有调试数据，便于离线复盘或提交问题。
    downloadJson(`medical-agent-debug-${run.run_id}.json`, {
      run,
      events,
      agents,
      skills,
      memory,
      stats,
      exported_at: new Date().toISOString()
    });
  }

  return (
    <div className="app-shell">
      <Topbar
        run={run}
        stats={stats}
        apiHealth={apiHealth}
        onRerun={rerun}
        onExport={exportRun}
        canRerun={Boolean(question.trim()) && !isSubmitting}
        canExport={Boolean(run)}
      />

      <main className="workspace">
        <InputPanel
          values={{
            question,
            age,
            sex,
            medicalHistory,
            medications,
            background,
            sessionId,
            enableSwarm,
            enableMemory,
            showRawData
          }}
          setters={{
            setQuestion,
            setAge,
            setSex,
            setMedicalHistory,
            setMedications,
            setBackground,
            setSessionId,
            setEnableSwarm,
            setEnableMemory,
            setShowRawData
          }}
          templates={templates}
          runHistory={runHistory}
          error={error}
          memoryError={memoryError}
          isSubmitting={isSubmitting}
          apiHealth={apiHealth}
          conversationMessages={memory?.recent_history ?? []}
          onSubmit={submitRun}
          onApplyTemplate={applyTemplate}
          onSelectHistory={selectHistoryRun}
          onRefreshHistory={refreshRunHistory}
          onNewSession={startNewSession}
        />

        <section className="main-column">
          <AnswerPanel finalAnswer={finalAnswer} />
          <TimelinePanel events={events} />
        </section>

        <InspectorPanel
          run={run}
          events={events}
          agents={agents}
          skills={skills}
          memory={memory}
          stats={stats}
          showRawData={showRawData}
        />
      </main>
    </div>
  );
}

export default App;
