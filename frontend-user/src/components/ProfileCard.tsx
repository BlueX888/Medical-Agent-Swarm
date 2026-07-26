import { ChevronDown, ChevronUp, UserRound } from "lucide-react";
import { Profile } from "../types";

interface Props {
  profile: Profile;
  onChange: (profile: Profile) => void;
  open: boolean;
  onToggle: () => void;
  rememberProfile: boolean;
  onRememberChange: (remember: boolean) => void;
  onClear: () => void;
}

export function ProfileCard({
  profile,
  onChange,
  open,
  onToggle,
  rememberProfile,
  onRememberChange,
  onClear
}: Props) {
  const filledCount = [profile.age, profile.sex, profile.medicalHistory, profile.medications].filter(
    (value) => value.trim()
  ).length;

  return (
    <section className="profile-card">
      <button
        type="button"
        className="profile-toggle"
        onClick={onToggle}
        aria-expanded={open}
        aria-controls="health-profile-fields"
      >
        <UserRound size={15} />
        <span>我的资料</span>
        <span className="profile-hint">
          {filledCount > 0 ? `已填 ${filledCount} 项，会随提问一起提供` : "选填，帮助分析更准确"}
        </span>
        {open ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
      </button>
      {open && (
        <div className="profile-panel" id="health-profile-fields">
          <p className="profile-privacy-copy">
            资料会随问题发送给本服务。不开启保存时，仅在当前页面使用。
          </p>
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
                aria-invalid={
                  profile.age.trim() !== "" &&
                  (!Number.isInteger(Number(profile.age)) ||
                    Number(profile.age) < 0 ||
                    Number(profile.age) > 120)
                }
                onChange={(event) => onChange({ ...profile, age: event.target.value })}
              />
              {profile.age.trim() !== "" &&
                (!Number.isInteger(Number(profile.age)) ||
                  Number(profile.age) < 0 ||
                  Number(profile.age) > 120) && (
                  <small className="field-error">请输入 0–120 的整数</small>
                )}
            </label>
            <label>
              性别
              <select
                value={profile.sex}
                onChange={(event) =>
                  onChange({ ...profile, sex: event.target.value as Profile["sex"] })
                }
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
                placeholder="如 高血压、糖尿病；没有可不填"
                value={profile.medicalHistory}
                onChange={(event) => onChange({ ...profile, medicalHistory: event.target.value })}
              />
            </label>
            <label className="profile-wide">
              正在使用的药物
              <input
                type="text"
                placeholder="如 氨氯地平；没有可不填"
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
        </div>
      )}
    </section>
  );
}
