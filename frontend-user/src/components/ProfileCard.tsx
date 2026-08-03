import { useEffect, useRef } from "react";
import { ShieldCheck, X } from "lucide-react";
import { Profile } from "../types";

interface Props {
  profile: Profile;
  onChange: (profile: Profile) => void;
  open: boolean;
  onClose: () => void;
  rememberProfile: boolean;
  onRememberChange: (remember: boolean) => void;
  onClear: () => void;
}

export function ProfileCard({
  profile,
  onChange,
  open,
  onClose,
  rememberProfile,
  onRememberChange,
  onClear
}: Props) {
  const drawerRef = useRef<HTMLElement>(null);
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const previouslyFocused = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const backdrop = drawerRef.current?.parentElement;
    const backgroundElements = backdrop?.parentElement
      ? Array.from(backdrop.parentElement.children).filter((element) => element !== backdrop)
      : [];
    const backgroundState = backgroundElements.map((element) => ({
      element,
      ariaHidden: element.getAttribute("aria-hidden"),
      inert: element.hasAttribute("inert")
    }));
    for (const element of backgroundElements) {
      element.setAttribute("aria-hidden", "true");
      element.setAttribute("inert", "");
    }
    closeButtonRef.current?.focus();

    const handleKeyDown = (event: globalThis.KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !drawerRef.current) return;
      const focusable = Array.from(
        drawerRef.current.querySelectorAll<HTMLElement>(
          'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
        )
      );
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      for (const { element, ariaHidden, inert } of backgroundState) {
        if (ariaHidden === null) element.removeAttribute("aria-hidden");
        else element.setAttribute("aria-hidden", ariaHidden);
        if (!inert) element.removeAttribute("inert");
      }
      previouslyFocused?.focus();
    };
  }, [open]);

  if (!open) return null;

  const invalidAge =
    profile.age.trim() !== "" &&
    (!Number.isInteger(Number(profile.age)) || Number(profile.age) < 0 || Number(profile.age) > 120);
  const filledCount = [profile.age, profile.sex, profile.medicalHistory, profile.medications].filter(
    (value) => value.trim()
  ).length;

  return (
    <div className="drawer-backdrop" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      <section
        ref={drawerRef}
        className="profile-drawer"
        role="dialog"
        aria-modal="true"
        aria-labelledby="profile-title"
      >
        <header className="drawer-header">
          <div>
            <p className="section-eyebrow">咨询背景</p>
            <h2 id="profile-title">补充健康资料</h2>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className="icon-button"
            onClick={onClose}
            aria-label="关闭健康资料"
          >
            <X size={20} />
          </button>
        </header>
        <div className="profile-privacy-copy">
          <ShieldCheck size={18} aria-hidden="true" />
          <p>资料会随问题发送给本服务。不开启本设备保存时，关闭页面后不会保留。</p>
        </div>
        <div className="profile-fields">
          <label>
            年龄
            <input
              type="number"
              inputMode="numeric"
              min="0"
              max="120"
              placeholder="如 32"
              value={profile.age}
              aria-invalid={invalidAge}
              aria-describedby={invalidAge ? "age-error" : undefined}
              onChange={(event) => onChange({ ...profile, age: event.target.value })}
            />
            {invalidAge && <small id="age-error" className="field-error">请输入 0–120 的整数</small>}
          </label>
          <label>
            性别
            <select
              value={profile.sex}
              onChange={(event) => onChange({ ...profile, sex: event.target.value as Profile["sex"] })}
            >
              <option value="">不填</option>
              <option value="女">女</option>
              <option value="男">男</option>
              <option value="其他">其他</option>
            </select>
          </label>
          <label className="profile-wide">
            既往病史
            <input
              type="text"
              placeholder="如高血压、糖尿病；没有可不填"
              value={profile.medicalHistory}
              onChange={(event) => onChange({ ...profile, medicalHistory: event.target.value })}
            />
          </label>
          <label className="profile-wide">
            正在使用的药物
            <input
              type="text"
              placeholder="如氨氯地平；没有可不填"
              value={profile.medications}
              onChange={(event) => onChange({ ...profile, medications: event.target.value })}
            />
          </label>
        </div>
        <div className="profile-controls">
          <label className="remember-profile">
            <input
              type="checkbox"
              checked={rememberProfile}
              onChange={(event) => onRememberChange(event.target.checked)}
            />
            <span>
              <strong>保存在本设备</strong>
              <small>下次打开时自动带入，可随时清除</small>
            </span>
          </label>
          <button
            type="button"
            className="clear-profile-button"
            onClick={onClear}
            disabled={filledCount === 0 && !rememberProfile}
          >
            清除资料
          </button>
        </div>
        <button type="button" className="drawer-done-button" onClick={onClose} disabled={invalidAge}>
          完成
        </button>
      </section>
    </div>
  );
}
