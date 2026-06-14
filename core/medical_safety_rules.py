"""
Deterministic medical safety review rules for final answers.

This module is runtime-only. It is intentionally not exposed as an Agent Skill
or LLM-callable tool; SafetyGuard invokes it before final output.
"""
from typing import Any, Dict, List

from loguru import logger


DISCLAIMER_MARKERS = ["仅供参考", "不能替代", "不替代", "免责声明", "请咨询医生", "及时就医"]

OVER_DIAGNOSIS_PATTERNS = [
    "你患有", "您患有", "确诊为", "就是", "一定是", "肯定是", "明确诊断为",
    "无需检查即可判断", "不用去医院就能确定",
]

EMERGENCY_KEYWORDS = [
    "胸痛", "呼吸困难", "意识不清", "意识模糊", "昏迷", "晕厥", "抽搐",
    "偏瘫", "口角歪斜", "说话不清", "剧烈头痛", "严重出血", "咯血",
    "呕血", "黑便", "剧烈腹痛", "高热不退", "孕期出血", "喘不上气",
    "嘴唇发紫", "口唇发紫", "一侧无力", "一侧胳膊没力气", "肢体无力",
    "说话含糊", "面部下垂",
]

EMERGENCY_ACTION_MARKERS = [
    "立即就医", "急诊", "急救", "120", "尽快就医", "马上就医",
    "产科评估", "产科急诊", "卒中绿色通道",
]

DANGEROUS_MEDICATION_PATTERNS = [
    "自行停药", "自己停药", "直接停药", "自行加药", "自己加药", "加大剂量",
    "减少剂量", "不用咨询医生", "不用复诊", "不用就医", "不要去医院",
]


def review_medical_safety(
    response: str,
    original_question: str = "",
    risk_level: str = "",
) -> Dict[str, Any]:
    """
    Review a draft medical answer using deterministic runtime rules.

    Returns the same shape previously produced by the safety_check Skill so
    SafetyGuard and callers keep a stable contract.
    """
    answer = response or ""
    logger.info(
        f"Running runtime medical safety review: "
        f"risk_level={risk_level}, response_len={len(answer)}"
    )

    issues: List[Dict[str, str]] = []
    fixed_suggestions: List[str] = []

    _check_disclaimer(answer, issues, fixed_suggestions)
    _check_over_diagnosis(answer, issues, fixed_suggestions)
    _check_emergency_warning(
        original_question or "",
        answer,
        risk_level or "",
        issues,
        fixed_suggestions,
    )
    _check_dangerous_medication(answer, issues, fixed_suggestions)

    passed = not issues

    return {
        "answer": _format_safety_report(passed, issues, fixed_suggestions),
        "passed": passed,
        "issues": issues,
        "fixed_suggestions": fixed_suggestions,
        "requires_human_review": any(issue["severity"] == "high" for issue in issues),
    }


def _check_disclaimer(response: str, issues: List[Dict[str, str]], suggestions: List[str]) -> None:
    if not any(marker in response for marker in DISCLAIMER_MARKERS):
        issues.append({
            "type": "missing_disclaimer",
            "severity": "medium",
            "message": "回答缺少医疗免责声明或就医提醒。",
        })
        suggestions.append("补充：以上信息仅供参考，不能替代专业医生诊断和治疗；如症状持续、加重或存在疑虑，请及时就医。")


def _check_over_diagnosis(response: str, issues: List[Dict[str, str]], suggestions: List[str]) -> None:
    matched = [pattern for pattern in OVER_DIAGNOSIS_PATTERNS if pattern in response]
    if matched:
        issues.append({
            "type": "over_diagnosis",
            "severity": "high",
            "message": f"回答可能存在过度诊断表达：{', '.join(matched)}。",
        })
        suggestions.append("将确诊性表述改为“可能与...有关”“需要医生结合检查判断”，避免直接诊断。")


def _check_emergency_warning(
    original_question: str,
    response: str,
    risk_level: str,
    issues: List[Dict[str, str]],
    suggestions: List[str],
) -> None:
    combined = f"{original_question}\n{response}"
    has_emergency_signal = _has_emergency_signal(combined, risk_level)
    has_emergency_action = any(marker in response for marker in EMERGENCY_ACTION_MARKERS)

    if has_emergency_signal and not has_emergency_action:
        issues.append({
            "type": "missing_emergency_warning",
            "severity": "high",
            "message": "问题或回答涉及高危症状，但未明确提醒立即就医/急诊/120。",
        })
        suggestions.append("在回答开头补充：如出现胸痛、呼吸困难、意识异常、严重出血等高危情况，请立即就医或拨打 120。")


def _has_emergency_signal(text: str, risk_level: str) -> bool:
    if risk_level in {"high", "emergency"}:
        return True

    if any(keyword in text for keyword in EMERGENCY_KEYWORDS):
        return True

    if "胸痛" in text and any(term in text for term in ["呼吸困难", "胸闷", "大汗", "出汗", "左臂", "放射痛"]):
        return True

    stroke_terms = ["口角歪斜", "说话含糊", "说话不清", "一侧胳膊没力气", "一侧无力", "肢体无力", "偏瘫", "面部下垂"]
    if any(term in text for term in stroke_terms):
        return True

    allergy_terms = ["花生", "过敏", "风团", "荨麻疹"]
    breathing_terms = ["喘不上气", "呼吸困难", "嘴唇发紫", "口唇发紫", "发紫"]
    if any(term in text for term in allergy_terms) and any(term in text for term in breathing_terms):
        return True

    pregnancy_terms = ["孕", "妊娠", "怀孕"]
    hypertension_terms = ["血压", "高血压", "尿蛋白"]
    preeclampsia_terms = ["头痛", "视物", "眼花", "上腹痛", "抽搐", "胎动"]
    if (
        any(term in text for term in pregnancy_terms)
        and any(term in text for term in hypertension_terms)
        and any(term in text for term in preeclampsia_terms)
    ):
        return True

    return False


def _check_dangerous_medication(response: str, issues: List[Dict[str, str]], suggestions: List[str]) -> None:
    matched = [pattern for pattern in DANGEROUS_MEDICATION_PATTERNS if pattern in response]
    if matched:
        issues.append({
            "type": "dangerous_medication_advice",
            "severity": "high",
            "message": f"回答可能包含危险用药建议：{', '.join(matched)}。",
        })
        suggestions.append("删除自行停药、加药、减药等建议，改为“请在医生指导下调整药物”。")


def _format_safety_report(
    passed: bool,
    issues: List[Dict[str, str]],
    suggestions: List[str],
) -> str:
    if passed:
        return "【安全审查】通过：未发现明显医疗安全问题。"

    lines = ["【安全审查】未通过"]
    lines.append("\n发现的问题：")
    for index, issue in enumerate(issues, 1):
        lines.append(f"{index}. [{issue['severity']}] {issue['message']}")

    if suggestions:
        lines.append("\n修正建议：")
        for index, suggestion in enumerate(suggestions, 1):
            lines.append(f"{index}. {suggestion}")

    return "\n".join(lines)
