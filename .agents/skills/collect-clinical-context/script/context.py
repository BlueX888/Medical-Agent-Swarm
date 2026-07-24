"""
Collect Clinical Context Skill
问诊信息补全 Skill（规则实现）
"""
import re
from typing import Dict, Any, List
from loguru import logger


REQUIRED_FIELDS = {
    "age": "年龄",
    "sex": "性别",
    "symptoms": "主要症状",
    "duration": "持续时间",
    "severity": "严重程度",
    "accompanying_symptoms": "伴随症状",
    "medical_history": "既往史/慢病史",
    "medication_history": "用药史",
}

RED_FLAG_KEYWORDS = [
    "胸痛", "胸闷", "呼吸困难", "气短", "喘不上气", "意识不清", "意识模糊",
    "昏迷", "晕厥", "抽搐", "偏瘫", "口角歪斜", "说话不清", "剧烈头痛",
    "严重出血", "咯血", "呕血", "黑便", "剧烈腹痛", "高热不退",
    "胎动减少", "孕期出血", "过敏性休克",
]

SYMPTOM_KEYWORDS = [
    "发热", "发烧", "咳嗽", "咳痰", "头痛", "头晕", "胸痛", "胸闷",
    "心悸", "腹痛", "腹泻", "恶心", "呕吐", "乏力", "皮疹", "水肿",
    "失眠", "疼痛", "呼吸困难", "气短", "鼻塞", "咽痛", "血压高",
    "血糖高", "尿频", "尿急", "尿痛",
]


async def collect_clinical_context(
    text: str,
    age: str = "",
    sex: str = "",
    symptoms: str = "",
    duration: str = "",
    severity: str = "",
    accompanying_symptoms: str = "",
    medical_history: str = "",
    medication_history: str = "",
) -> Dict[str, Any]:
    """
    抽取并补全问诊上下文。

    Args:
        text: 用户原始描述
        age: 已知年龄
        sex: 已知性别
        symptoms: 已知主要症状
        duration: 已知持续时间
        severity: 已知严重程度
        accompanying_symptoms: 已知伴随症状
        medical_history: 已知既往史/慢病史
        medication_history: 已知用药史
    """
    logger.info(f"Collecting clinical context: text={text[:80]}")

    extracted = {
        "age": age or _extract_age(text),
        "sex": sex or _extract_sex(text),
        "symptoms": symptoms or _extract_symptoms(text),
        "duration": duration or _extract_duration(text),
        "severity": severity or _extract_severity(text),
        "accompanying_symptoms": accompanying_symptoms or _extract_accompanying(text),
        "medical_history": medical_history or _extract_medical_history(text),
        "medication_history": medication_history or _extract_medication_history(text),
    }

    missing_fields = [
        {"field": field, "label": label}
        for field, label in REQUIRED_FIELDS.items()
        if not extracted.get(field)
    ]

    red_flags = _detect_red_flags(text)
    special_populations = _detect_special_populations(text, extracted.get("age", ""))
    follow_up_questions = _build_follow_up_questions(missing_fields, red_flags)
    completeness_score = round((len(REQUIRED_FIELDS) - len(missing_fields)) / len(REQUIRED_FIELDS), 2)

    return {
        "answer": _format_context_summary(
            extracted,
            missing_fields,
            follow_up_questions,
            red_flags,
            special_populations,
            completeness_score,
        ),
        "extracted_context": extracted,
        "missing_fields": missing_fields,
        "follow_up_questions": follow_up_questions,
        "high_risk_flags": red_flags,
        "special_populations": special_populations,
        "needs_urgent_attention": bool(red_flags),
        "completeness_score": completeness_score,
    }


def _extract_age(text: str) -> str:
    patterns = [
        r"(\d{1,3})\s*岁",
        r"年龄[:：]?\s*(\d{1,3})",
        r"(\d{1,3})\s*y(?:ears?)?\s*old",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"{match.group(1)}岁"
    return ""


def _extract_sex(text: str) -> str:
    if any(word in text for word in ["男性", "男士", "男孩", "男童", "男"]):
        return "男"
    if any(word in text for word in ["女性", "女士", "女孩", "女童", "孕妇", "妊娠", "怀孕", "女"]):
        return "女"
    return ""


def _extract_symptoms(text: str) -> str:
    found = []
    for keyword in SYMPTOM_KEYWORDS:
        if keyword in text and keyword not in found:
            found.append(keyword)
    return "、".join(found)


def _extract_duration(text: str) -> str:
    patterns = [
        r"(?:持续|已经|有|出现|反复)?\s*(\d+\s*(?:分钟|小时|天|日|周|个月|年))",
        r"(半小时|一小时|两小时|一天|两天|三天|一周|两周|一个月|半年)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(" ", "")
    return ""


def _extract_severity(text: str) -> str:
    if any(word in text for word in ["剧烈", "严重", "无法忍受", "很痛", "特别痛", "最痛"]):
        return "重度"
    if any(word in text for word in ["明显", "加重", "比较痛", "中等"]):
        return "中度"
    if any(word in text for word in ["轻微", "一点", "轻度"]):
        return "轻度"
    score_match = re.search(r"(\d{1,2})\s*/\s*10", text)
    if score_match:
        score = int(score_match.group(1))
        if score >= 7:
            return "重度"
        if score >= 4:
            return "中度"
        return "轻度"
    return ""


def _extract_accompanying(text: str) -> str:
    match = re.search(r"(?:伴|伴有|同时有|并有)([^。；;，,]+)", text)
    if match:
        return match.group(1).strip()
    return ""


def _extract_medical_history(text: str) -> str:
    chronic = []
    for keyword in ["高血压", "糖尿病", "冠心病", "哮喘", "慢阻肺", "肾病", "肝病", "肿瘤", "过敏史"]:
        if keyword in text and keyword not in chronic:
            chronic.append(keyword)
    if any(word in text for word in ["既往", "病史", "慢病"]):
        chronic.append("用户提到既往史/病史")
    return "、".join(chronic)


def _extract_medication_history(text: str) -> str:
    match = re.search(r"(?:服用|吃|使用|正在用|用药)([^。；;，,]+)", text)
    if match:
        return match.group(1).strip()
    if any(word in text for word in ["药", "降压药", "降糖药", "抗凝", "胰岛素"]):
        return "用户提到用药，但药名/剂量不完整"
    return ""


def _detect_red_flags(text: str) -> List[str]:
    return [keyword for keyword in RED_FLAG_KEYWORDS if keyword in text]


def _detect_special_populations(text: str, age_text: str) -> List[str]:
    populations = []
    age_match = re.search(r"\d{1,3}", age_text)
    if age_match:
        age = int(age_match.group(0))
        if age < 6:
            populations.append("儿童")
        if age >= 65:
            populations.append("老人")
    if any(word in text for word in ["孕妇", "妊娠", "怀孕", "产后"]):
        populations.append("孕产妇")
    if any(word in text for word in ["高血压", "糖尿病", "冠心病", "慢阻肺", "肾病"]):
        populations.append("慢病患者")
    return populations


def _build_follow_up_questions(missing_fields: List[Dict[str, str]], red_flags: List[str]) -> List[str]:
    question_map = {
        "age": "请补充患者年龄。",
        "sex": "请补充患者性别。",
        "symptoms": "请描述最主要的不适症状。",
        "duration": "症状持续了多久，是突然出现还是逐渐加重？",
        "severity": "症状严重程度如何？如果是疼痛，可用 0-10 分描述。",
        "accompanying_symptoms": "是否伴有发热、呼吸困难、胸痛、呕吐、意识改变等其他症状？",
        "medical_history": "是否有高血压、糖尿病、冠心病、哮喘、肾病等既往病史？",
        "medication_history": "近期是否正在用药、停药、加药或对药物过敏？",
    }
    questions = [question_map[item["field"]] for item in missing_fields[:4]]
    if red_flags:
        questions.insert(0, "已出现潜在高危信号，请优先确认是否需要立即就医或拨打 120。")
    return questions


def _format_context_summary(
    extracted: Dict[str, str],
    missing_fields: List[Dict[str, str]],
    follow_up_questions: List[str],
    red_flags: List[str],
    special_populations: List[str],
    completeness_score: float,
) -> str:
    lines = ["【问诊信息整理】"]
    for field, label in REQUIRED_FIELDS.items():
        value = extracted.get(field) or "未提供"
        lines.append(f"- {label}: {value}")

    lines.append(f"\n完整度: {completeness_score:.0%}")

    if special_populations:
        lines.append(f"特殊人群: {'、'.join(special_populations)}")

    if red_flags:
        lines.append(f"潜在高危信息: {'、'.join(red_flags)}")
    else:
        lines.append("潜在高危信息: 暂未识别到明确红旗症状")

    if missing_fields:
        labels = "、".join(item["label"] for item in missing_fields)
        lines.append(f"\n缺失字段: {labels}")

    if follow_up_questions:
        lines.append("\n建议追问:")
        for index, question in enumerate(follow_up_questions, 1):
            lines.append(f"{index}. {question}")

    return "\n".join(lines)


def collect_clinical_context_sync(*args, **kwargs) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(collect_clinical_context(*args, **kwargs))
