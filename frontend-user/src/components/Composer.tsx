import { KeyboardEvent, useRef } from "react";
import { SendHorizonal } from "lucide-react";

interface Props {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled: boolean;
}

export function Composer({ value, onChange, onSend, disabled }: Props) {
  const isComposing = useRef(false);

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (isComposing.current || event.nativeEvent.isComposing || event.keyCode === 229) {
      return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSend();
    }
  };

  return (
    <div className="composer-shell">
      <div className="composer">
        <label className="sr-only" htmlFor="health-question">
          描述你的症状或健康问题
        </label>
        <textarea
          id="health-question"
          rows={2}
          maxLength={2000}
          placeholder="描述症状、持续时间和伴随情况……"
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
          aria-label="发送问题"
        >
          <SendHorizonal size={18} />
        </button>
      </div>
      <div className="composer-meta" id="composer-help">
        <span>Enter 发送 · Shift + Enter 换行</span>
        <span>{value.length}/2000</span>
      </div>
    </div>
  );
}
