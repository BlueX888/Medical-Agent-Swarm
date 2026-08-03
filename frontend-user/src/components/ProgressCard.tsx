import {
  Check,
  Circle,
  HeartHandshake,
  Microscope,
  ShieldCheck,
  Stethoscope
} from "lucide-react";
import {
  ConsultationParticipant,
  ConsultationPhase,
  ConsultationSnapshot
} from "../types";

const PHASES: Array<{ id: ConsultationPhase; label: string }> = [
  { id: "understanding", label: "理解你的描述" },
  { id: "planning", label: "制定会诊路径" },
  { id: "consulting", label: "协作分析" },
  { id: "safety_review", label: "安全复核" },
  { id: "finalizing", label: "整理行动建议" }
];

const ROLE_SLOTS = [
  { id: "health_consultation", label: "健康咨询", description: "整理问题与日常建议", icon: HeartHandshake },
  { id: "symptom_analysis", label: "风险与症状分析", description: "识别风险与就医时机", icon: Stethoscope },
  { id: "evidence_research", label: "医学证据检索", description: "按需核对医学资料", icon: Microscope }
] as const;

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
    ? "会诊进行中"
    : snapshot?.status === "success"
      ? "本轮会诊已完成"
      : snapshot?.status === "failed" || snapshot?.status === "timeout"
        ? "本轮未完成"
        : "提交问题后开始协作";

  return (
    <section className="consultation-rail" aria-label="会诊路径" aria-live="polite">
      <div className="rail-heading">
        <div>
          <p className="section-eyebrow">Consultation path</p>
          <h2>会诊路径</h2>
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

      <div className="role-section">
        <p className="rail-subtitle">协作角色</p>
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
