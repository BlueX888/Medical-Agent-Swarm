import { forwardRef, KeyboardEvent, useEffect, useRef } from "react";
import { ArrowUp, UserRound } from "lucide-react";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  onOpenProfile: () => void;
  disabled: boolean;
  busy: boolean;
  hasMessages: boolean;
  profileCount: number;
}

export const Composer = forwardRef<HTMLTextAreaElement, Props>(function Composer(
  {
    value,
    onChange,
    onSend,
    onOpenProfile,
    disabled,
    busy,
    hasMessages,
    profileCount
  },
  forwardedRef
) {
  const isComposing = useRef(false);
  const localRef = useRef<HTMLTextAreaElement | null>(null);

  const setTextareaRef = (node: HTMLTextAreaElement | null) => {
    localRef.current = node;
    if (typeof forwardedRef === "function") {
      forwardedRef(node);
    } else if (forwardedRef) {
      forwardedRef.current = node;
    }
  };

  useEffect(() => {
    const textarea = localRef.current;
    if (!textarea) return;
    textarea.style.height = "auto";
    textarea.style.height = `${Math.min(textarea.scrollHeight, 144)}px`;
  }, [value]);

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (isComposing.current || event.nativeEvent.isComposing || event.keyCode === 229) return;
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };

  return (
    <div className="composer-shell">
      <div className="composer-tools">
        <button type="button" className="profile-tool" onClick={onOpenProfile} disabled={busy}>
          <UserRound size={16} aria-hidden="true" />
          <span>健康资料</span>
          <small>{profileCount > 0 ? `已填 ${profileCount} 项` : "选填"}</small>
        </button>
        <span className="privacy-hint">资料仅用于本次健康分析</span>
      </div>
      <div className="composer">
        <label className="sr-only" htmlFor="health-question">
          描述你的症状或健康问题
        </label>
        <textarea
          ref={setTextareaRef}
          id="health-question"
          rows={1}
          maxLength={2000}
          placeholder={busy ? "会诊进行中，请稍候……" : "描述症状、持续时间和伴随情况……"}
          value={value}
          disabled={disabled}
          aria-describedby="composer-help"
          onChange={(event) => onChange(event.target.value)}
          onCompositionStart={() => {
            isComposing.current = true;
          }}
          onCompositionEnd={() => {
            isComposing.current = false;
          }}
          onKeyDown={handleKeyDown}
        />
        <button
          type="button"
          className="send-button"
          onClick={onSend}
          disabled={disabled || !value.trim()}
          aria-label={hasMessages ? "继续提问" : "开始会诊"}
        >
          <span>{hasMessages ? "继续提问" : "开始会诊"}</span>
          <ArrowUp size={18} aria-hidden="true" />
        </button>
      </div>
      <div className="composer-meta" id="composer-help">
        <span>Enter 发送 · Shift + Enter 换行</span>
        {value.length >= 1600 && <span>{value.length}/2000</span>}
      </div>
    </div>
  );
});
