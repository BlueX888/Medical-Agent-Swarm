import ReactMarkdown from "react-markdown";
import { AlertTriangle, Users } from "lucide-react";
import { ChatMessage, RiskLevel } from "../types";

const RISK_LABELS: Record<RiskLevel, string> = {
  low: "风险较低",
  medium: "需要留意",
  high: "风险较高",
  emergency: "急症警示"
};

interface Props {
  message: ChatMessage;
}

export function ChatMessageView({ message }: Props) {
  if (message.role === "user") {
    return (
      <div className="msg-row msg-row-user">
        <div className="bubble bubble-user" aria-label="你的问题">
          {message.content}
        </div>
      </div>
    );
  }

  const payload = message.payload;
  return (
    <div className="msg-row msg-row-assistant">
      <article
        className={`bubble bubble-assistant${payload?.failed ? " bubble-failed" : ""}`}
        aria-label={payload?.failed ? "分析服务提示" : "健康助手回复"}
      >
        {payload?.riskLevel === "emergency" && (
          <div className="emergency-banner" role="alert">
            <AlertTriangle size={16} />
            <span>可能属于急症情况，请立即拨打 120 或前往急诊，不要等待线上回复。</span>
          </div>
        )}
        {payload?.riskLevel && (
          <span className={`risk-badge risk-${payload.riskLevel}`}>
            {RISK_LABELS[payload.riskLevel]}
          </span>
        )}
        <div className="bubble-body">
          <ReactMarkdown>{message.content}</ReactMarkdown>
        </div>
        {payload && payload.suggestions.length > 0 && (
          <div className="follow-up-block">
            <p className="follow-up-title">重点建议</p>
            <ul className="advice-list">
              {payload.suggestions.map((text) => (
                <li key={text}>{text}</li>
              ))}
            </ul>
          </div>
        )}
        {payload && payload.agentsInvolved.length > 0 && (
          <div className="agents-note">
            <Users size={13} />
            <span>已综合 {payload.agentsInvolved.length} 个分析角色的意见</span>
          </div>
        )}
        {payload?.disclaimer && <p className="disclaimer-note">{payload.disclaimer}</p>}
      </article>
    </div>
  );
}
