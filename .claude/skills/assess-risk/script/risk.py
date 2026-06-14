"""
Assess Risk Skill
风险评估 Skill（规则实现）
"""
import re
from typing import Dict, Any, List, Tuple
from loguru import logger


EMERGENCY_SYMPTOMS = {
    "意识不清": "意识异常可能提示严重神经系统或全身性问题",
    "昏迷": "昏迷属于急症",
    "抽搐": "抽搐需要紧急评估",
    "严重出血": "严重出血可能导致休克",
    "咯血": "咯血需要尽快排查呼吸道或循环系统风险",
    "呕血": "呕血提示消化道出血风险",
    "黑便": "黑便可能提示消化道出血",
    "偏瘫": "偏瘫可能提示卒中",
    "口角歪斜": "口角歪斜可能提示卒中",
    "说话不清": "言语不清可能提示卒中",
    "过敏性休克": "过敏性休克属于急症",
}

HIGH_RISK_SYMPTOMS = {
    "胸痛": "胸痛需要警惕心血管急症",
    "呼吸困难": "呼吸困难需要尽快评估氧合和心肺情况",
    "剧烈头痛": "突发或剧烈头痛需要排查神经系统急症",
    "剧烈腹痛": "剧烈腹痛需要排查急腹症",
    "高热不退": "持续高热可能提示严重感染",
    "晕厥": "晕厥需要评估心脑血管和代谢风险",
    "胎动减少": "胎动减少需要产科紧急评估",
    "孕期出血": "孕期出血需要产科紧急评估",
}

MEDIUM_RISK_KEYWORDS = ["持续", "加重", "反复", "明显", "严重", "夜间", "影响睡眠"]

SPECIAL_POPULATION_KEYWORDS = {
    "pregnant": ["孕妇", "妊娠", "怀孕", "产后"],
    "chronic": ["高血压", "糖尿病", "冠心病", "慢阻肺", "哮喘", "肾病", "肿瘤", "免疫抑制"],
}


async def assess_risk(
    symptoms: str,
    age: str = "",
    medical_history: str = "",
    pregnancy_status: str = "",
) -> Dict[str, Any]:
    """
    评估症状风险等级。

    Args:
        symptoms: 症状描述
        age: 年龄，可为空
        medical_history: 既往史/慢病史，可为空
        pregnancy_status: 妊娠/产后状态，可为空
    """
    logger.info(f"Assessing risk: symptoms={symptoms}")

    text = " ".join(part for part in [symptoms, age, medical_history, pregnancy_status] if part)
    symptom_list = _split_symptoms(symptoms)
    special_populations = _detect_special_populations(text, age)

    risk_level, reasons = _evaluate_base_risk(text, symptom_list)
    combo_level, combo_reasons = _evaluate_combinations(text)
    risk_level = _max_risk(risk_level, combo_level)
    reasons.extend(combo_reasons)

    weighted_level, weighted_reasons = _apply_special_population_weighting(
        risk_level,
        text,
        special_populations,
    )
    risk_level = weighted_level
    reasons.extend(weighted_reasons)

    recommendation = _build_recommendation(risk_level)
    action_steps = _build_action_steps(risk_level)

    return {
        "answer": format_assessment(
            symptoms=symptoms,
            level=risk_level,
            reasons=reasons,
            recommendation=recommendation,
            action_steps=action_steps,
            special_populations=special_populations,
        ),
        "risk_level": risk_level,
        "recommendation": recommendation,
        "reasons": reasons,
        "action_steps": action_steps,
        "special_populations": special_populations,
        "source": "rule_engine",
    }


def _split_symptoms(symptoms: str) -> List[str]:
    parts = re.split(r"[,，、;；\s]+", symptoms)
    return [part.strip() for part in parts if part.strip()] or [symptoms]


def _evaluate_base_risk(text: str, symptom_list: List[str]) -> Tuple[str, List[str]]:
    risk_level = "low"
    reasons: List[str] = []

    for keyword, reason in EMERGENCY_SYMPTOMS.items():
        if keyword in text:
            risk_level = _max_risk(risk_level, "emergency")
            reasons.append(f"检测到急症信号：{keyword}。{reason}")

    for keyword, reason in HIGH_RISK_SYMPTOMS.items():
        if keyword in text:
            risk_level = _max_risk(risk_level, "high")
            reasons.append(f"检测到高风险症状：{keyword}。{reason}")

    if risk_level == "low":
        for symptom in symptom_list:
            if any(keyword in symptom for keyword in MEDIUM_RISK_KEYWORDS):
                risk_level = "medium"
                reasons.append(f"症状描述提示需要关注：{symptom}")

    if not reasons:
        reasons.append("暂未识别到明确红旗症状，但仍需结合完整问诊判断。")

    return risk_level, reasons


def _evaluate_combinations(text: str) -> Tuple[str, List[str]]:
    checks = [
        ("emergency", ["胸痛", "呼吸困难"], "胸痛伴呼吸困难需要按心肺急症优先处理"),
        ("emergency", ["胸痛", "出汗"], "胸痛伴出汗需要警惕急性冠脉综合征"),
        ("emergency", ["胸痛", "晕厥"], "胸痛伴晕厥属于高危组合"),
        ("emergency", ["头痛", "偏瘫"], "头痛伴偏瘫需要排查卒中或颅内急症"),
        ("emergency", ["发热", "意识"], "发热伴意识改变需要紧急评估感染或神经系统风险"),
        ("high", ["腹痛", "呕吐", "发热"], "腹痛伴呕吐和发热需要排查急腹症或严重感染"),
        ("high", ["咳嗽", "呼吸困难"], "咳嗽伴呼吸困难需要评估肺部感染或哮喘/慢阻肺急性发作"),
    ]

    risk_level = "low"
    reasons: List[str] = []
    for level, keywords, reason in checks:
        if all(keyword in text for keyword in keywords):
            risk_level = _max_risk(risk_level, level)
            reasons.append(f"高危症状组合：{reason}")
    return risk_level, reasons


def _detect_special_populations(text: str, age_text: str) -> List[str]:
    populations: List[str] = []
    age_match = re.search(r"\d{1,3}", age_text or text)
    if age_match:
        age = int(age_match.group(0))
        if age < 6:
            populations.append("儿童")
        elif age >= 65:
            populations.append("老人")

    if any(keyword in text for keyword in SPECIAL_POPULATION_KEYWORDS["pregnant"]):
        populations.append("孕产妇")
    if any(keyword in text for keyword in SPECIAL_POPULATION_KEYWORDS["chronic"]):
        populations.append("慢病患者")
    return populations


def _apply_special_population_weighting(
    current_level: str,
    text: str,
    special_populations: List[str],
) -> Tuple[str, List[str]]:
    reasons: List[str] = []
    level = current_level

    if not special_populations:
        return level, reasons

    concerning_terms = ["发热", "胸痛", "呼吸困难", "腹痛", "呕吐", "头痛", "出血", "乏力"]
    has_concerning_symptom = any(term in text for term in concerning_terms)

    if has_concerning_symptom and level == "low":
        level = "medium"
        reasons.append(f"特殊人群（{'、'.join(special_populations)}）出现症状，风险上调至中危。")
    elif has_concerning_symptom and level == "medium":
        level = "high"
        reasons.append(f"特殊人群（{'、'.join(special_populations)}）症状持续或加重，风险上调至高危。")

    if "孕产妇" in special_populations and any(term in text for term in ["出血", "腹痛", "头痛", "胸痛", "呼吸困难"]):
        level = _max_risk(level, "emergency")
        reasons.append("孕产妇出现出血、腹痛、剧烈头痛、胸痛或呼吸困难，应按急症处理。")

    return level, reasons


def _max_risk(left: str, right: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2, "emergency": 3}
    return left if order[left] >= order[right] else right


def _build_recommendation(level: str) -> str:
    if level == "emergency":
        return "请立即拨打 120 或前往急诊，不要自行开车或等待观察。"
    if level == "high":
        return "建议尽快就医，优先选择急诊或当天门诊评估。"
    if level == "medium":
        return "建议在 24-48 小时内就医或线上咨询医生，若加重则立即就医。"
    return "可先观察和记录症状变化；如持续不缓解、加重或出现红旗症状，请及时就医。"


def _build_action_steps(level: str) -> List[str]:
    if level == "emergency":
        return [
            "立即停止活动并保持安全体位。",
            "拨打 120 或请身边人协助送急诊。",
            "不要自行服用或调整处方药，除非医生曾明确交代。",
        ]
    if level == "high":
        return [
            "尽快安排医疗评估。",
            "记录症状开始时间、诱因、持续时间和伴随症状。",
            "避免自行停药、加药或延误就医。",
        ]
    if level == "medium":
        return [
            "补充体温、血压、血糖等可测指标。",
            "观察症状是否持续、反复或加重。",
            "准备既往史和用药清单，便于医生判断。",
        ]
    return [
        "注意休息和补充水分。",
        "记录症状变化。",
        "如出现胸痛、呼吸困难、意识异常、严重出血等情况，立即就医。",
    ]


def format_assessment(
    symptoms: str,
    level: str,
    reasons: List[str],
    recommendation: str,
    action_steps: List[str],
    special_populations: List[str],
) -> str:
    """格式化风险评估结果。"""
    level_map = {
        "low": "低危",
        "medium": "中危",
        "high": "高危",
        "emergency": "紧急",
    }

    output = [
        "【症状风险评估】",
        f"\n症状描述：{symptoms}",
        f"\n风险等级：{level_map.get(level, level)}",
    ]

    if special_populations:
        output.append(f"\n特殊人群：{'、'.join(special_populations)}")

    output.append("\n判断依据：")
    for reason in reasons:
        output.append(f"- {reason}")

    output.append(f"\n行动建议：{recommendation}")
    output.append("\n下一步：")
    for step in action_steps:
        output.append(f"- {step}")

    output.append("\n数据来源：内置风险分诊规则。")
    return "\n".join(output)


def assess_risk_sync(*args, **kwargs) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(assess_risk(*args, **kwargs))
