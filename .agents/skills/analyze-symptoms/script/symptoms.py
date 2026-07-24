"""
Analyze Symptoms Skill
症状分析 Skill（规则实现）
"""
import re
from typing import Dict, Any, List
from loguru import logger


SYMPTOM_CATEGORIES = {
    "respiratory": {
        "keywords": ["咳嗽", "呼吸困难", "气短", "鼻塞", "咽痛", "喉咙痛", "咳痰", "胸闷", "喘"],
        "name": "呼吸系统",
    },
    "digestive": {
        "keywords": ["腹痛", "腹泻", "恶心", "呕吐", "胃痛", "便秘", "反酸", "黑便"],
        "name": "消化系统",
    },
    "neurological": {
        "keywords": ["头痛", "头晕", "眩晕", "失眠", "麻木", "乏力", "抽搐", "意识", "偏瘫"],
        "name": "神经系统",
    },
    "cardiovascular": {
        "keywords": ["胸痛", "心悸", "心慌", "血压", "胸闷", "晕厥"],
        "name": "心血管系统",
    },
    "metabolic": {
        "keywords": ["血糖", "口渴", "多尿", "体重下降", "肥胖", "尿频"],
        "name": "代谢/内分泌系统",
    },
    "urinary": {
        "keywords": ["尿频", "尿急", "尿痛", "血尿", "腰痛"],
        "name": "泌尿系统",
    },
    "musculoskeletal": {
        "keywords": ["关节", "肌肉", "骨骼", "腰痛", "肿胀", "僵硬"],
        "name": "骨骼肌肉系统",
    },
    "skin": {
        "keywords": ["皮疹", "瘙痒", "红肿", "水疱", "过敏"],
        "name": "皮肤/过敏相关",
    },
}

COMBINATION_RULES = [
    {
        "keywords": ["发热", "咳嗽", "咽痛"],
        "direction": "上呼吸道感染、流感样疾病等方向",
        "reason": "发热合并咳嗽/咽痛常见于呼吸道感染类问题",
        "urgency": "medium",
    },
    {
        "keywords": ["咳嗽", "呼吸困难"],
        "direction": "肺部感染、哮喘/慢阻肺急性发作等方向",
        "reason": "呼吸道症状合并呼吸困难需要优先评估氧合和肺部情况",
        "urgency": "high",
    },
    {
        "keywords": ["胸痛", "呼吸困难"],
        "direction": "心肺急症方向",
        "reason": "胸痛伴呼吸困难属于高危组合，需尽快就医评估",
        "urgency": "emergency",
    },
    {
        "keywords": ["胸痛", "出汗"],
        "direction": "急性冠脉综合征等心血管急症方向",
        "reason": "胸痛伴大汗需警惕急性心血管事件",
        "urgency": "emergency",
    },
    {
        "keywords": ["腹痛", "呕吐", "发热"],
        "direction": "急性胃肠炎、胆囊/阑尾等急腹症方向",
        "reason": "腹痛伴呕吐和发热需要区分感染性胃肠炎与外科急腹症",
        "urgency": "high",
    },
    {
        "keywords": ["头痛", "偏瘫"],
        "direction": "卒中、颅内病变等神经系统急症方向",
        "reason": "头痛伴局灶神经功能异常需要急诊评估",
        "urgency": "emergency",
    },
    {
        "keywords": ["尿频", "尿急", "尿痛"],
        "direction": "泌尿系统感染方向",
        "reason": "尿路刺激症状组合常见于泌尿系统感染",
        "urgency": "medium",
    },
    {
        "keywords": ["口渴", "多尿", "体重下降"],
        "direction": "血糖异常、糖尿病相关方向",
        "reason": "口渴、多尿和体重下降提示代谢异常可能",
        "urgency": "medium",
    },
]


async def analyze_symptoms(symptoms: str) -> Dict[str, Any]:
    """
    分析症状模式。

    Args:
        symptoms: 症状描述
    """
    logger.info(f"Analyzing symptoms: {symptoms}")

    symptom_list = _split_symptoms(symptoms)
    categories = _detect_categories(symptoms, symptom_list)
    patterns = _build_patterns(categories, symptom_list)
    directions = _match_combination_rules(symptoms)

    if not directions:
        directions = _directions_from_categories(categories)

    red_flags = [
        item for item in directions
        if item.get("urgency") in {"high", "emergency"}
    ]

    # Backward-compatible field name. Values are deliberately phrased as directions, not diagnoses.
    possible_diseases = [item["direction"] for item in directions][:6]

    return {
        "answer": format_analysis(symptoms, patterns, directions, categories, red_flags),
        "patterns": patterns,
        "categories": categories,
        "possible_directions": directions,
        "possible_diseases": possible_diseases,
        "red_flags": red_flags,
        "source": "rule_engine",
    }


def _split_symptoms(symptoms: str) -> List[str]:
    parts = re.split(r"[,，、;；\s]+", symptoms)
    return [part.strip() for part in parts if part.strip()] or [symptoms]


def _detect_categories(text: str, symptom_list: List[str]) -> List[Dict[str, str]]:
    detected = []
    for category_id, data in SYMPTOM_CATEGORIES.items():
        if any(keyword in text for keyword in data["keywords"]):
            detected.append({"id": category_id, "name": data["name"]})
            continue
        if any(any(keyword in symptom for keyword in data["keywords"]) for symptom in symptom_list):
            detected.append({"id": category_id, "name": data["name"]})
    return detected


def _build_patterns(categories: List[Dict[str, str]], symptom_list: List[str]) -> List[str]:
    patterns: List[str] = []
    if categories:
        patterns.append(f"症状涉及：{'、'.join(item['name'] for item in categories)}")
    if len(categories) > 1:
        patterns.append("症状涉及多个系统，需要优先排查高危组合并补全问诊信息。")
    if len(symptom_list) >= 3:
        patterns.append("用户描述了多个症状，建议按起病时间、主次症状和伴随症状排序。")
    if not patterns:
        patterns.append("现有描述较少，暂不能形成稳定症状模式。")
    return patterns


def _match_combination_rules(text: str) -> List[Dict[str, str]]:
    directions: List[Dict[str, str]] = []
    for rule in COMBINATION_RULES:
        if all(keyword in text for keyword in rule["keywords"]):
            directions.append({
                "direction": rule["direction"],
                "reason": rule["reason"],
                "urgency": rule["urgency"],
            })
    return directions


def _directions_from_categories(categories: List[Dict[str, str]]) -> List[Dict[str, str]]:
    mapping = {
        "respiratory": ("呼吸道感染、过敏或气道疾病等方向", "由呼吸系统相关症状提示", "low"),
        "digestive": ("胃肠功能紊乱、感染或急腹症等方向", "由消化系统相关症状提示", "medium"),
        "neurological": ("紧张疲劳、偏头痛、前庭或神经系统问题等方向", "由神经系统相关症状提示", "medium"),
        "cardiovascular": ("血压波动、心律问题或心血管风险方向", "由心血管相关症状提示", "high"),
        "metabolic": ("血糖、甲状腺或体重代谢相关方向", "由代谢/内分泌相关症状提示", "medium"),
        "urinary": ("泌尿系统感染或结石等方向", "由泌尿系统相关症状提示", "medium"),
        "musculoskeletal": ("肌肉骨骼劳损、炎症或外伤方向", "由骨骼肌肉系统症状提示", "low"),
        "skin": ("皮肤感染、过敏或炎症方向", "由皮肤/过敏相关症状提示", "low"),
    }
    directions = []
    for category in categories:
        direction, reason, urgency = mapping.get(
            category["id"],
            ("非特异性症状方向", "当前信息有限", "low"),
        )
        directions.append({"direction": direction, "reason": reason, "urgency": urgency})
    return directions[:6]


def format_analysis(
    symptoms: str,
    patterns: List[str],
    directions: List[Dict[str, str]],
    categories: List[Dict[str, str]],
    red_flags: List[Dict[str, str]],
) -> str:
    """格式化症状分析结果。"""
    output = [
        "【症状模式分析】",
        f"\n症状描述：{symptoms}",
    ]

    if categories:
        output.append(f"\n涉及系统：{'、'.join(item['name'] for item in categories)}")

    output.append("\n识别到的症状模式：")
    for pattern in patterns:
        output.append(f"- {pattern}")

    if directions:
        output.append("\n可能方向（非诊断）：")
        for item in directions:
            output.append(f"- {item['direction']}：{item['reason']}（紧急度：{item['urgency']}）")

    if red_flags:
        output.append("\n高危提示：")
        for item in red_flags:
            output.append(f"- {item['reason']}")

    output.append("\n注意：以上仅为症状模式分析，不能作为确诊依据。需要结合体征、检查和医生评估。")
    output.append("数据来源：内置症状分类和症状组合规则。")
    return "\n".join(output)


def analyze_symptoms_sync(*args, **kwargs) -> Dict[str, Any]:
    import asyncio
    return asyncio.run(analyze_symptoms(*args, **kwargs))
