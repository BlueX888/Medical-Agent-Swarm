import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { AlertTriangle, Check, Clipboard, RotateCcw, ShieldCheck, Users } from "lucide-react";
import { ChatMessage, RiskLevel } from "../types";

const RISK_CONTENT: Record<RiskLevel, { label: string; summary: string }> = {
  low: { label: "风险较低", summary: "目前可先观察与护理，留意后续变化" },
  medium: { label: "需要留意", summary: "建议持续观察，必要时安排线下咨询" },
  high: { label: "风险较高", summary: "建议尽快接受线下医疗评估" },
  emergency: { label: "急症警示", summary: "请立即联系急救服务或前往急诊" }
};

const STRUCTURED_METADATA_KEYS = new Set([
  "suggestions",
  "disclaimer",
  "risk_level",
  "key_findings"
]);

function stripTrailingStructuredMetadata(content: string): string {
  const match = content.match(/\n*```json\s*(\{[\s\S]*?\})\s*```\s*$/i);
  if (!match || match.index === undefined) return content;
  try {
    const payload = JSON.parse(match[1]) as unknown;
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) return content;
    const keys = Object.keys(payload);
    if (keys.length === 0 || !keys.every((key) => STRUCTURED_METADATA_KEYS.has(key))) return content;
    return content.slice(0, match.index).trimEnd();
  } catch {
    return content;
  }
}

interface Props {
  message: ChatMessage;
  onRetry?: (question: string) => void;
}

export function ChatMessageView({ message, onRetry }: Props) {
  const [copied, setCopied] = useState(false);

  if (message.role === "user") {
    return (
      <div className="msg-row msg-row-user">
        <div className="message-label">你</div>
        <div className="bubble bubble-user" aria-label="你的问题">{message.content}</div>
      </div>
    );
  }

  const payload = message.payload;
  const risk = payload?.riskLevel ? RISK_CONTENT[payload.riskLevel] : null;
  const visibleContent = stripTrailingStructuredMetadata(message.content);

  const copyAnswer = async () => {
    if (!navigator.clipboard) return;
    await navigator.clipboard.writeText(visibleContent);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  };

  if (payload?.failed) {
    return (
      <div className="msg-row msg-row-assistant">
        <article className="bubble bubble-assistant failure-card" aria-label="分析服务提示">
          <div className="failure-heading">
            <AlertTriangle size={18} aria-hidden="true" />
            <strong>{payload.timedOut ? "分析时间较长" : "本轮分析未完成"}</strong>
          </div>
          <p>{message.content}</p>
          {message.retryQuestion && onRetry && (
            <button type="button" className="retry-button" onClick={() => onRetry(message.retryQuestion!)}>
              <RotateCcw size={15} aria-hidden="true" />重新分析
            </button>
          )}
        </article>
      </div>
    );
  }

  return (
    <div className="msg-row msg-row-assistant">
      <article className="bubble bubble-assistant result-card" aria-label="健康助手回复">
        <header className="result-heading">
          <div>
            <p className="section-eyebrow">Analysis note</p>
            <h2>分析建议</h2>
          </div>
          <button type="button" className="copy-button" onClick={copyAnswer}>
            {copied ? <Check size={15} /> : <Clipboard size={15} />}
            {copied ? "已复制" : "复制建议"}
          </button>
        </header>

        {payload?.riskLevel === "emergency" && (
          <div className="emergency-banner" role="alert">
            <AlertTriangle size={20} />
            <span><strong>不要等待线上回复。</strong>立即拨打 120 或前往急诊，并尽量请他人陪同。</span>
          </div>
        )}

        {risk && (
          <div className={`risk-summary risk-${payload?.riskLevel}`}>
            <span className="risk-badge">{risk.label}</span>
            <strong>{risk.summary}</strong>
          </div>
        )}

        <div className="bubble-body"><ReactMarkdown>{visibleContent}</ReactMarkdown></div>

        {payload?.sources && payload.sources.length > 0 && (
          <section className="source-block" aria-labelledby={`sources-${message.id}`}>
            <h3 id={`sources-${message.id}`}>参考资料</h3>
            <ol>
              {payload.sources.map((source) => (
                <li key={`${source.citation_id}-${source.title}`}>
                  <span className="source-id">[{source.citation_id}]</span>{" "}
                  {source.external_url ? (
                    <a href={source.external_url} target="_blank" rel="noreferrer noopener">
                      {source.title}
                    </a>
                  ) : (
                    <strong>{source.title}</strong>
                  )}
                  <small>
                    {[source.source_org, source.version, source.published_at, source.section]
                      .filter(Boolean)
                      .join(" · ")}
                  </small>
                </li>
              ))}
            </ol>
          </section>
        )}

        {payload && payload.suggestions.length > 0 && (
          <section className="follow-up-block" aria-labelledby={`advice-${message.id}`}>
            <h3 id={`advice-${message.id}`}>现在可以做什么</h3>
            <ul className="advice-list">
              {payload.suggestions.map((text) => <li key={text}>{text}</li>)}
            </ul>
          </section>
        )}

        <footer className="result-meta">
          {payload && payload.participants.length > 0 && (
            <span><Users size={14} />{payload.participants.join("、")}协作完成</span>
          )}
          {payload?.safetyChecked && <span><ShieldCheck size={14} />已完成安全复核</span>}
        </footer>
        {payload?.disclaimer && <p className="disclaimer-note">{payload.disclaimer}</p>}
      </article>
    </div>
  );
}
