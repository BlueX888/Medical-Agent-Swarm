import { Check, LoaderCircle, LockKeyhole } from "lucide-react";
import { ConsultationAnalysisStep, ConsultationSnapshot } from "../types";

const PHASE_COPY = {
  understanding: "正在整理症状、持续时间和相关背景",
  planning: "正在判断风险等级、回答重点与资料需求",
  consulting: "相关分析角色正在分别核对建议",
  safety_review: "正在检查急症提示、过度诊断和用药风险",
  finalizing: "正在整理行动建议和可引用资料"
} as const;

function currentAnalysisStep(
  steps: ConsultationAnalysisStep[]
): ConsultationAnalysisStep | undefined {
  return (
    steps.find((step) => step.state === "active") ??
    steps.find((step) => step.state === "attention") ??
    steps.find((step) => step.state === "pending")
  );
}

export function LiveAnalysisCard({ snapshot }: { snapshot: ConsultationSnapshot | null }) {
  const progress = snapshot?.progress;
  const steps = progress?.analysis_steps ?? [];
  const current = currentAnalysisStep(steps);
  const confirmed = steps.filter((step) => step.state === "done").length;
  const phaseCopy = progress
    ? PHASE_COPY[progress.current_phase]
    : "正在建立安全分析任务";

  return (
    <section className="live-analysis-card" aria-label="实时分析摘要" aria-live="polite">
      <div className="live-analysis-heading">
        <span className="live-analysis-pulse" aria-hidden="true"><LoaderCircle size={17} /></span>
        <div>
          <p className="section-eyebrow">Live analysis</p>
          <h2>分析正在进行</h2>
        </div>
        <span className="live-analysis-count">{confirmed} 项已确认</span>
      </div>

      <div className="live-analysis-current">
        <span className="live-analysis-index" aria-hidden="true">{Math.min(steps.indexOf(current as ConsultationAnalysisStep) + 1 || 1, 5)}</span>
        <div>
          <strong>{current?.label ?? "准备分析"}</strong>
          <p>{current?.summary ?? phaseCopy}</p>
        </div>
      </div>

      <div className="live-analysis-confirmed" aria-label="当前阶段">
        <Check size={13} aria-hidden="true" />
        <span>{phaseCopy}</span>
      </div>
      <p className="live-analysis-privacy"><LockKeyhole size={12} aria-hidden="true" />展示结构化进度，不展示模型内部思维记录</p>
    </section>
  );
}
