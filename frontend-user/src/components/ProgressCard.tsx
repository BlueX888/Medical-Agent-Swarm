import { RunProgress } from "../types";

// 后端 LangGraph 各节点的 stage 顺序，见 swarm/medical_swarm_graph.py 的 _trace_node 注册。
const STAGE_ORDER = [
  "load_memory",
  "planning",
  "routing",
  "agent_loop",
  "safety_check",
  "save_memory"
] as const;

function stageLabel(stage: string, agentCount: number): string {
  switch (stage) {
    case "load_memory":
      return "正在回顾你之前的对话";
    case "planning":
      return "正在理解你的描述";
    case "routing":
      return "正在安排合适的分析角色";
    case "agent_loop":
      return agentCount > 1 ? `${agentCount} 位智能体协作分析中` : "智能体分析中";
    case "save_memory":
      return "正在记录本次咨询";
    case "safety_check":
      return "正在做最后的安全检查";
    default:
      return "处理中";
  }
}

export function ProgressCard({ progress }: { progress: RunProgress }) {
  const seen = STAGE_ORDER.filter((stage) => progress.stagesSeen.includes(stage));
  const activeStage = seen[seen.length - 1] ?? STAGE_ORDER[0];

  return (
    <div className="msg-row msg-row-assistant">
      <div
        className="bubble bubble-assistant progress-card"
        aria-live="polite"
        aria-busy="true"
      >
        <p className="progress-title">正在为你会诊</p>
        <ol className="progress-timeline">
          {STAGE_ORDER.map((stage) => {
            const isSeen = seen.includes(stage);
            const isActive = stage === activeStage;
            const state = isActive ? "active" : isSeen ? "done" : "todo";
            return (
              <li key={stage} className={`progress-step progress-${state}`}>
                <span className="progress-dot" />
                <span className="progress-label">{stageLabel(stage, progress.agentCount)}</span>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}
