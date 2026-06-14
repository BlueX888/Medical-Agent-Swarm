"""
Runtime safety guard for final medical answers.

This guard runs outside the LLM tool loop. Safety review is a runtime system
module, not an Agent Skill or model-selectable tool.
"""
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

from .medical_safety_rules import review_medical_safety


class SafetyGuard:
    """Run runtime safety rules and apply deterministic safety supplements."""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent

    async def review(
        self,
        response: str,
        original_question: str = "",
        risk_level: str = "",
    ) -> Dict[str, Any]:
        """Run runtime safety review and return rewritten answer plus metadata."""
        answer = response or ""

        try:
            review_result = review_medical_safety(
                response=answer,
                original_question=original_question or "",
                risk_level=risk_level or "",
            )
        except Exception as exc:
            logger.error(f"Runtime medical safety review failed: {exc}")
            return {
                "answer": answer,
                "safety_checked": False,
                "safety_passed": False,
                "safety_issues": [{
                    "type": "safety_review_error",
                    "severity": "high",
                    "message": str(exc),
                }],
                "safety_review": {"error": str(exc)},
            }

        issues = review_result.get("issues", [])
        passed = bool(review_result.get("passed"))
        rewritten = answer if passed else self._apply_safety_fixes(
            answer=answer,
            issues=issues,
            suggestions=review_result.get("fixed_suggestions", []),
            risk_level=risk_level or "",
        )

        return {
            "answer": rewritten,
            "safety_checked": True,
            "safety_passed": passed,
            "safety_issues": issues,
            "safety_review": review_result,
        }

    def _apply_safety_fixes(
        self,
        answer: str,
        issues: List[Dict[str, str]],
        suggestions: List[str],
        risk_level: str,
    ) -> str:
        issue_types = {issue.get("type") for issue in issues}
        has_high_issue = any(issue.get("severity") == "high" for issue in issues)

        rewritten = answer
        if "dangerous_medication_advice" in issue_types:
            rewritten = self._neutralize_dangerous_medication(rewritten)
        if "over_diagnosis" in issue_types:
            rewritten = self._soften_overdiagnosis(rewritten)

        prefix_lines = []
        if has_high_issue:
            if (
                "missing_emergency_warning" in issue_types
                or risk_level in {"high", "emergency"}
            ):
                prefix_lines.append(
                    "【紧急提醒】如正在出现胸痛、呼吸困难、意识异常、严重过敏、卒中表现、孕期高血压伴头痛等高危情况，请立即拨打 120 或前往急诊/产科急诊评估。"
                )
            if "dangerous_medication_advice" in issue_types:
                prefix_lines.append(
                    "【用药安全】不要自行停药、加药、减药或替换处方药；药物调整必须由医生结合病情决定。"
                )
            if "over_diagnosis" in issue_types:
                prefix_lines.append(
                    "【诊断边界】以下内容不能作为确诊结论，只能作为就医沟通和风险识别参考。"
                )

        suffix_lines = []
        if "missing_disclaimer" in issue_types:
            suffix_lines.append(
                "【免责声明】以上信息仅供参考，不能替代专业医生的诊断和治疗；如症状持续、加重或存在疑虑，请及时就医。"
            )

        # Keep non-emergency suggestions visible without exposing raw stack-like data.
        if suggestions and not has_high_issue:
            suffix_lines.append("【安全补充】" + "；".join(suggestions[:2]))

        parts = []
        if prefix_lines:
            parts.append("\n".join(prefix_lines))
        parts.append(rewritten.strip())
        if suffix_lines:
            parts.append("\n".join(suffix_lines))

        return "\n\n".join(part for part in parts if part)

    def _neutralize_dangerous_medication(self, text: str) -> str:
        replacements = {
            "自行停药": "不要自行停药",
            "自己停药": "不要自己停药",
            "直接停药": "不要直接停药",
            "自行加药": "不要自行加药",
            "自己加药": "不要自己加药",
            "加大剂量": "不要自行加大剂量",
            "减少剂量": "不要自行减少剂量",
            "不用咨询医生": "需要咨询医生",
            "不用复诊": "需要按医嘱复诊",
            "不用就医": "需要根据症状及时就医",
            "不要去医院": "需要根据症状及时就医",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _soften_overdiagnosis(self, text: str) -> str:
        replacements = {
            "你患有": "不能仅凭描述确定，可能存在",
            "您患有": "不能仅凭描述确定，可能存在",
            "确诊为": "需要医生评估是否为",
            "明确诊断为": "需要医生评估是否为",
            "一定是": "可能与",
            "肯定是": "可能与",
            "你就是": "不能仅凭描述确定为",
            "您就是": "不能仅凭描述确定为",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
