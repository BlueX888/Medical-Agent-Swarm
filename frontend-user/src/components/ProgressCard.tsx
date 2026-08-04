import {
  BookOpenCheck,
  Check,
  CircleAlert,
  Circle,
  HeartHandshake,
  ListChecks,
  Microscope,
  Minus,
  ShieldCheck,
  Stethoscope,
  UsersRound
} from "lucide-react";
import {
  ConsultationAnalysisStep,
  ConsultationParticipant,
  ConsultationPhase,
  ConsultationSnapshot
} from "../types";

const PHASES: Array<{ id: ConsultationPhase; label: string }> = [
  { id: "understanding", label: "理解你的描述" },
  { id: "planning", label: "判断风险与资料需求" },
  { id: "consulting", label: "多角色分析" },
  { id: "safety_review", label: "安全复核" },
  { id: "finalizing", label: "整理行动建议" }
];

const ROLE_SLOTS = [
  { id: "health_consultation", label: "健康咨询", description: "整理问题与日常建议", icon: HeartHandshake },
  { id: "symptom_analysis", label: "风险与症状分析", description: "识别风险与就医时机", icon: Stethoscope },
  { id: "evidence_research", label: "医学证据检索", description: "按需核对医学资料", icon: Microscope }
] as const;

const ANALYSIS_ICONS = {
  risk: CircleAlert,
  focus: ListChecks,
  evidence: BookOpenCheck,
  collaboration: UsersRound,
  safety: ShieldCheck
} as const;

const ANALYSIS_STATE_LABELS: Record<ConsultationAnalysisStep["state"], string> = {
  pending: "待进行",
  active: "进行中",
  done: "已确认",
  skipped: "本次跳过",
  attention: "需留意"
};

function participantState(
  participants: ConsultationParticipant[],
  id: string
): ConsultationParticipant["state"] {
  return participants.find((participant) => participant.id === id)?.state ?? "waiting";
}

export function ProgressCard({ snapshot }: { snapshot: ConsultationSnapshot | null }) {
  const progress = snapshot?.progress;
  const isRunning = snapshot?.status === "queued" || snapshot?.status === "running";
  const statusCopy = isRunning
    ? "分析进行中"
    : snapshot?.status === "success"
      ? "本轮分析已完成"
      : snapshot?.status === "failed" || snapshot?.status === "timeout"
        ? "本轮未完成"
        : "提交后逐步更新";
  const analysisSteps = progress?.analysis_steps ?? [];

  return (
    <section className="consultation-rail" aria-label="分析进度" aria-live="polite">
      <div className="rail-heading">
        <div>
          <p className="section-eyebrow">Analysis trace</p>
          <h2>分析进度</h2>
        </div>
        <span className={`rail-status${isRunning ? " is-active" : ""}`}>{statusCopy}</span>
      </div>

      <ol className="phase-list">
        {PHASES.map((phase) => {
          const done = progress?.completed_phases.includes(phase.id) ?? false;
          const active = isRunning && progress?.current_phase === phase.id;
          return (
            <li key={phase.id} className={`phase-item${done ? " is-done" : ""}${active ? " is-active" : ""}`}>
              <span className="phase-marker" aria-hidden="true">
                {done ? <Check size={12} /> : <Circle size={9} />}
              </span>
              <span>{phase.label}</span>
            </li>
          );
        })}
      </ol>

      <section className="analysis-receipt" aria-labelledby="analysis-receipt-title">
        <div className="analysis-receipt-heading">
          <span className="analysis-receipt-icon" aria-hidden="true"><ListChecks size={16} /></span>
          <div>
            <h3 id="analysis-receipt-title">本次分析摘要</h3>
            <p>这里展示可核验的步骤与结论，不包含模型内部思维记录。</p>
          </div>
        </div>

        {analysisSteps.length > 0 ? (
          <ol className="analysis-step-list">
            {analysisSteps.map((step) => {
              const Icon = ANALYSIS_ICONS[step.id];
              return (
                <li key={step.id} className={`analysis-step analysis-step-${step.state}`}>
                  <span className="analysis-step-marker" aria-hidden="true">
                    {step.state === "done" ? (
                      <Check size={12} />
                    ) : step.state === "skipped" ? (
                      <Minus size={12} />
                    ) : (
                      <Icon size={13} />
                    )}
                  </span>
                  <div className="analysis-step-copy">
                    <div>
                      <strong>{step.label}</strong>
                      <span>{ANALYSIS_STATE_LABELS[step.state]}</span>
                    </div>
                    <p>{step.summary}</p>
                  </div>
                </li>
              );
            })}
          </ol>
        ) : (
          <p className="analysis-receipt-empty">提交问题后，将在这里说明风险判断、资料使用和安全复核状态。</p>
        )}
      </section>

      <div className="role-section">
        <p className="rail-subtitle">参与分析</p>
        <div className="role-list">
          {ROLE_SLOTS.map((role) => {
            const state = participantState(progress?.participants ?? [], role.id);
            const Icon = role.icon;
            return (
              <div key={role.id} className={`role-card role-${state}`}>
                <span className="role-icon"><Icon size={17} aria-hidden="true" /></span>
                <span className="role-copy">
                  <strong>{role.label}</strong>
                  <small>{role.description}</small>
                </span>
                <span className="role-state">
                  {state === "active" ? "分析中" : state === "done" ? "已完成" : state === "failed" ? "未完成" : "按需加入"}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      <div className={`safety-seal${progress?.safety_checked ? " is-checked" : ""}`}>
        <ShieldCheck size={18} aria-hidden="true" />
        <span>
          <strong>{progress?.safety_checked ? "已完成安全复核" : "最终回答将经过安全复核"}</strong>
          <small>检查急症提醒、过度诊断与用药风险</small>
        </span>
      </div>
    </section>
  );
}
