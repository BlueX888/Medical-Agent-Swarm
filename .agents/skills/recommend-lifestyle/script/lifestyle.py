"""
Recommend Lifestyle Skill
生活方式建议 Skill（模板实现）
"""
from typing import Dict, Any, List
from loguru import logger


LIFESTYLE_TEMPLATES = {
    "hypertension": {
        "aliases": ["高血压", "血压高", "降压"],
        "title": "高血压",
        "diet": [
            "减少钠盐摄入，少吃腌制、加工和重口味食物。",
            "增加蔬菜、水果、全谷物和低脂奶制品摄入。",
            "限制酒精摄入，避免大量饮酒。",
        ],
        "exercise": [
            "在医生允许的前提下，每周进行至少 150 分钟中等强度有氧运动。",
            "避免突然高强度运动，运动中如胸痛、气短或头晕应停止并就医。",
        ],
        "sleep": [
            "保持规律作息，减少熬夜。",
            "如打鼾明显或白天嗜睡，建议评估睡眠呼吸暂停。",
        ],
        "medication": [
            "按医嘱规律服药，不要自行停药、加药或换药。",
            "记录家庭血压，复诊时带给医生参考。",
        ],
    },
    "diabetes": {
        "aliases": ["糖尿病", "血糖高", "控糖", "胰岛素"],
        "title": "糖尿病/血糖管理",
        "diet": [
            "控制总能量和精制糖摄入，主食注意定量。",
            "优先选择全谷物、豆类、蔬菜和优质蛋白。",
            "避免含糖饮料和频繁高糖零食。",
        ],
        "exercise": [
            "规律运动有助于改善胰岛素敏感性，运动前后注意低血糖风险。",
            "如存在足部溃疡、严重视网膜病变或心血管症状，应先咨询医生。",
        ],
        "sleep": [
            "保持规律睡眠，避免长期睡眠不足影响血糖控制。",
        ],
        "medication": [
            "降糖药或胰岛素需按医嘱使用，不要自行调整剂量。",
            "出现明显低血糖症状时需及时处理并联系医生。",
        ],
    },
    "cold": {
        "aliases": ["感冒", "发热", "咳嗽", "咽痛", "鼻塞", "流感"],
        "title": "感冒/呼吸道症状",
        "diet": [
            "保证饮水，饮食清淡，避免烟酒刺激。",
            "发热或出汗多时注意补充水分和电解质。",
        ],
        "exercise": [
            "发热、乏力明显时暂停剧烈运动。",
            "症状缓解后逐步恢复活动。",
        ],
        "sleep": [
            "保证休息，避免熬夜和过度劳累。",
        ],
        "medication": [
            "不要重复叠加含相同成分的感冒药。",
            "儿童、孕妇、老人或慢病患者用药前应咨询医生或药师。",
        ],
    },
    "sleep": {
        "aliases": ["失眠", "睡眠", "睡不着", "早醒"],
        "title": "睡眠改善",
        "diet": [
            "下午或晚上减少咖啡因、浓茶和酒精摄入。",
            "睡前避免大量进食。",
        ],
        "exercise": [
            "白天规律活动有助于睡眠，但睡前避免剧烈运动。",
        ],
        "sleep": [
            "固定起床时间，睡前减少屏幕刺激。",
            "卧室保持安静、黑暗和舒适温度。",
            "若失眠持续超过 2-4 周或伴明显焦虑抑郁，建议就医评估。",
        ],
        "medication": [
            "不要自行长期使用安眠药或镇静药。",
        ],
    },
    "weight": {
        "aliases": ["减重", "减肥", "肥胖", "体重管理", "超重"],
        "title": "体重管理",
        "diet": [
            "建立可持续的热量缺口，避免极端节食。",
            "增加蛋白质、蔬菜和全谷物比例，减少含糖饮料。",
        ],
        "exercise": [
            "结合有氧运动和抗阻训练，循序渐进增加运动量。",
            "有关节疼痛、胸闷或基础病时先咨询医生。",
        ],
        "sleep": [
            "保证睡眠，长期睡眠不足会影响食欲和代谢。",
        ],
        "medication": [
            "减重药物和代谢手术需医生评估，不建议自行购买使用。",
        ],
    },
    "general": {
        "aliases": ["一般健康", "健康", "生活方式", "保健"],
        "title": "一般健康",
        "diet": [
            "饮食多样化，保证蔬菜、水果、全谷物和优质蛋白。",
            "减少高盐、高糖、高油和过度加工食品。",
        ],
        "exercise": [
            "保持规律活动，久坐人群每小时起身活动。",
            "运动强度应结合年龄、基础病和当前症状调整。",
        ],
        "sleep": [
            "保持规律作息，成人通常需要 7-9 小时睡眠。",
        ],
        "medication": [
            "任何处方药调整都应先咨询医生。",
        ],
    },
}


async def recommend_lifestyle(
    diagnosis: str,
    risk_level: str = "",
    age: str = "",
    medical_history: str = "",
) -> Dict[str, Any]:
    """
    提供生活方式建议。

    Args:
        diagnosis: 疾病名称、症状或健康目标
        risk_level: 已知风险等级（low/medium/high/emergency）
        age: 年龄，可为空
        medical_history: 既往史/慢病史，可为空
    """
    logger.info(f"Recommending lifestyle for: {diagnosis}, risk_level={risk_level}")

    if risk_level in {"high", "emergency"} or _contains_emergency_signal(diagnosis):
        recommendation = "当前描述可能存在高风险信号，生活方式建议不能替代医疗评估。请优先就医或按急诊处理。"
        return {
            "answer": format_refusal(diagnosis, recommendation),
            "diagnosis": diagnosis,
            "categories": [],
            "source": "built_in_templates",
            "refused": True,
            "recommendation": recommendation,
        }

    template_key = _match_template(diagnosis)
    template = LIFESTYLE_TEMPLATES[template_key]
    cautions = _build_cautions(age, medical_history, diagnosis)

    return {
        "answer": format_advice(diagnosis, template, cautions),
        "diagnosis": diagnosis,
        "categories": ["diet", "exercise", "sleep", "medication"],
        "source": "built_in_templates",
        "template": template_key,
        "refused": False,
        "cautions": cautions,
    }


def _match_template(text: str) -> str:
    for key, template in LIFESTYLE_TEMPLATES.items():
        if key == "general":
            continue
        if any(alias in text for alias in template["aliases"]):
            return key
    return "general"


def _contains_emergency_signal(text: str) -> bool:
    emergency_terms = [
        "胸痛", "呼吸困难", "意识不清", "昏迷", "晕厥", "严重出血",
        "剧烈腹痛", "剧烈头痛", "偏瘫", "口角歪斜", "孕期出血",
    ]
    return any(term in text for term in emergency_terms)


def _build_cautions(age: str, medical_history: str, diagnosis: str) -> List[str]:
    cautions: List[str] = []
    text = f"{age} {medical_history} {diagnosis}"
    if any(term in text for term in ["老人", "65岁", "70岁", "80岁"]):
        cautions.append("老人调整运动和饮食应循序渐进，注意跌倒、低血压和低血糖风险。")
    if any(term in text for term in ["孕妇", "妊娠", "怀孕"]):
        cautions.append("孕产妇饮食、运动和用药应优先咨询产科医生。")
    if any(term in text for term in ["冠心病", "胸痛", "心衰", "慢阻肺"]):
        cautions.append("存在心肺基础病时，运动处方应由医生评估后制定。")
    if any(term in text for term in ["肾病", "肾功能"]):
        cautions.append("肾病患者的蛋白、盐和水分摄入需结合肾功能由医生或营养师制定。")
    return cautions


def format_refusal(diagnosis: str, recommendation: str) -> str:
    return "\n".join([
        f"【{diagnosis}生活方式建议】",
        recommendation,
        "",
        "出现胸痛、呼吸困难、意识异常、严重出血、剧烈头痛或剧烈腹痛等情况时，请立即就医或拨打 120。",
        "",
        "【免责声明】",
        "以上信息仅供参考，不能替代专业医生的诊断和治疗。",
    ])


def format_advice(diagnosis: str, template: Dict[str, List[str]], cautions: List[str]) -> str:
    output = [f"【{template['title']}生活方式建议】", f"\n适用问题：{diagnosis}"]

    sections = [
        ("饮食", template["diet"]),
        ("运动", template["exercise"]),
        ("睡眠/作息", template["sleep"]),
        ("用药安全", template["medication"]),
    ]

    for title, items in sections:
        output.append(f"\n{title}：")
        for item in items:
            output.append(f"- {item}")

    if cautions:
        output.append("\n特殊注意：")
        for caution in cautions:
            output.append(f"- {caution}")

    output.extend([
        "\n【需要就医的情况】",
        "- 症状持续不缓解、明显加重，或出现胸痛、呼吸困难、意识异常等高危信号。",
        "- 需要调整处方药、出现药物副作用，或基础病控制不佳。",
        "\n【免责声明】",
        "以上建议仅供参考，不能替代专业医生或营养师的个体化诊疗建议。",
    ])

    return "\n".join(output)


def recommend_lifestyle_sync(*args, **kwargs) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(recommend_lifestyle(*args, **kwargs))
