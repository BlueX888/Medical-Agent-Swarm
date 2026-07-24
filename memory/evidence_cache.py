"""
Local evidence memory for expensive medical research queries.

DeepResearch can use this module as a deterministic local memory layer before
falling back to live web search. It contains a small curated seed memory for
common guideline topics and can persist successful research outputs for future
similar queries.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from loguru import logger

try:
    import filelock  # noqa: F401
except ImportError:
    pass


DEFAULT_CACHE_PATH = Path("memory/evidence/evidence_cache.json")


SEED_EVIDENCE: List[Dict[str, Any]] = [
    {
        "id": "seed_hypertension_bp_targets",
        "preferred_agent": "research_agent",
        "query": "高血压患者血压控制目标 老年人 放宽 指南",
        "title": "成人高血压血压控制目标与老年人个体化管理",
        "triggers": [
            ["高血压", "控制目标"],
            ["高血压", "血压目标"],
            ["高血压", "目标值"],
            ["高血压", "老年"],
            ["血压目标值", "老年"],
            ["ACC", "AHA", "高血压"],
            ["ESC", "高血压"],
            ["血压", "老年", "放宽"],
            ["hypertension", "blood pressure", "target"],
            ["bp", "target", "older"],
        ],
        "keywords": ["高血压", "血压", "控制目标", "血压目标", "目标值", "老年", "个体化", "140/90", "130/80", "ACC", "AHA", "ESC", "hypertension", "older"],
        "answer": (
            "【本地循证记忆】成人高血压的常见诊室血压控制目标通常至少应低于 140/90 mmHg；"
            "若患者耐受良好且心血管风险较高，可在医生评估下考虑更严格目标，例如接近或低于 130/80 mmHg。"
            "老年人不宜机械套用单一阈值，应结合年龄、衰弱程度、体位性低血压、合并症、用药耐受性和跌倒风险进行个体化；"
            "高龄或衰弱患者可适当放宽，但仍需避免长期明显高血压。家庭血压通常参考 135/85 mmHg 作为需要关注或进一步评估的界值。"
        ),
        "findings": [
            "成人高血压常见控制目标至少为诊室血压 <140/90 mmHg。",
            "耐受良好或高危患者可在医生评估下考虑更严格目标，如接近 <130/80 mmHg。",
            "老年人目标应个体化，关注衰弱、体位性低血压、跌倒风险与合并症。",
            "家庭血压监测常以 135/85 mmHg 作为警戒或进一步评估界值。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 4,
    },
    {
        "id": "seed_diabetes_hba1c_targets",
        "preferred_agent": "research_agent",
        "query": "2型糖尿病 HbA1c 控制目标 老年 低血糖风险 ADA",
        "title": "2 型糖尿病 HbA1c 目标与老年/低血糖高风险个体化",
        "triggers": [
            ["HbA1c", "老人"],
            ["HbA1c", "老年"],
            ["HbA1c", "低血糖"],
            ["HbA1c", "控制目标"],
            ["HbA1c", "ADA"],
            ["HbA1c", "EASD"],
            ["2型糖尿病", "HbA1c"],
            ["糖化血红蛋白", "老年"],
            ["ADA", "HbA1c", "older"],
            ["HbA1c", "elderly", "hypoglycemia"],
        ],
        "keywords": ["2型糖尿病", "HbA1c", "糖化血红蛋白", "老人", "老年", "低血糖", "ADA", "older", "hypoglycemia"],
        "answer": (
            "【本地循证记忆】2 型糖尿病 HbA1c 目标通常以约 <7% 作为许多非妊娠成人的常见参考，"
            "但不是人人相同。病程短、预期寿命长、低血糖风险低且能安全达标者可更严格；"
            "老年、衰弱、多病共存、预期寿命有限或使用胰岛素/促泌剂且低血糖风险高者，应放宽目标，"
            "常见可放宽到 <7.5%、<8% 甚至更宽，重点避免低血糖和急性高血糖症状。"
            "目标需要由医生结合病情、用药、并发症和患者偏好个体化制定。"
        ),
        "findings": [
            "多数非妊娠成人 2 型糖尿病常见 HbA1c 参考目标约为 <7%。",
            "目标需根据年龄、病程、合并症、预期寿命和低血糖风险个体化。",
            "老年、衰弱或低血糖高风险患者通常需要放宽目标，优先避免低血糖。",
            "胰岛素或促泌剂使用者需特别关注低血糖风险。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 4,
    },
    {
        "id": "seed_sglt2_cardiorenal",
        "preferred_agent": "research_agent",
        "query": "SGLT2 抑制剂 糖尿病 心衰 肾病 获益 指南 证据",
        "title": "SGLT2 抑制剂在糖尿病合并心衰/慢性肾病中的心肾获益",
        "triggers": [
            ["SGLT2", "心衰"],
            ["SGLT2", "心力衰竭"],
            ["SGLT2", "肾病"],
            ["SGLT2", "慢性肾脏病"],
            ["SGLT2", "糖尿病"],
            ["SGLT2", "ADA"],
            ["SGLT2", "指南"],
            ["SGLT2", "推荐"],
            ["SGLT2", "CKD"],
            ["SGLT2", "heart failure"],
            ["SGLT2", "chronic kidney disease"],
            ["SGLT2 inhibitors", "heart failure", "kidney"],
        ],
        "keywords": ["SGLT2", "糖尿病", "心衰", "心力衰竭", "肾病", "慢性肾脏病", "CKD", "heart failure", "kidney", "cardiorenal", "DAPA-HF", "EMPA", "CREDENCE"],
        "answer": (
            "【本地循证记忆】SGLT2 抑制剂对 2 型糖尿病合并心力衰竭或慢性肾病患者的获益总体是靠谱的，"
            "多项大型随机试验和指南支持其降低心衰住院、延缓肾功能恶化并改善部分心肾复合终点。"
            "这不等于所有人都适合自行使用；是否用药需结合 eGFR、容量状态、酮症酸中毒风险、泌尿生殖感染风险、"
            "合并用药和禁忌证，由医生评估。出现脱水、严重感染、手术/禁食或酮症风险时尤其需要医疗指导。"
        ),
        "findings": [
            "SGLT2 抑制剂在心衰和 CKD 人群中具有明确的心肾保护证据。",
            "获益包括降低心衰住院风险、延缓肾功能下降和改善部分复合终点。",
            "使用前需评估 eGFR、容量状态、感染风险和酮症酸中毒风险。",
            "不能据此建议患者自行购买或调整处方药。",
        ],
        "evidence_level": "A",
        "confidence": "high",
        "sources": 5,
    },
    {
        "id": "seed_preeclampsia_warning",
        "preferred_agent": "diagnostic_agent",
        "query": "孕28周 血压150/95 尿蛋白阳性 头痛 子痫前期 风险",
        "title": "孕期高血压、尿蛋白阳性与头痛的子痫前期风险",
        "triggers": [
            ["孕", "尿蛋白", "头痛"],
            ["妊娠", "高血压", "尿蛋白"],
            ["孕28周", "血压"],
            ["preeclampsia", "headache"],
            ["pregnancy", "hypertension", "proteinuria"],
        ],
        "keywords": ["孕", "妊娠", "血压", "尿蛋白", "头痛", "子痫前期", "preeclampsia", "pregnancy", "proteinuria"],
        "answer": (
            "【本地循证记忆】孕 20 周后出现血压升高并伴尿蛋白阳性，需要高度警惕子痫前期；"
            "若头痛越来越明显、视物异常、上腹痛、气促、明显水肿或血压继续升高，属于需要尽快产科/急诊评估的危险信号。"
            "孕 28 周血压 150/95 mmHg 加尿蛋白阳性并头痛加重，不建议在家观察，应尽快联系产科或急诊，"
            "评估血压、尿蛋白、血小板、肝肾功能、胎儿情况等。不要自行服用或调整降压药。"
        ),
        "findings": [
            "孕 20 周后高血压合并尿蛋白阳性需警惕子痫前期。",
            "头痛加重、视物异常、上腹痛等提示严重风险，应尽快就医。",
            "需要产科评估母体实验室指标与胎儿情况。",
            "不应自行用药或延迟就诊。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 4,
    },
    {
        "id": "seed_headache_visual_redflags",
        "preferred_agent": "diagnostic_agent",
        "query": "反复头痛 恶心 视物模糊 需要就医 红旗征象",
        "title": "头痛伴恶心和视物模糊的红旗风险分诊",
        "triggers": [
            ["头痛", "视物模糊"],
            ["头痛", "恶心", "视物"],
            ["headache", "blurred vision"],
            ["headache", "nausea", "vision"],
        ],
        "keywords": ["头痛", "恶心", "视物模糊", "视力", "高血压", "颅内压", "偏头痛", "眼科", "神经内科", "headache", "vision"],
        "answer": (
            "【本地分诊记忆】反复头痛伴恶心、视物模糊不能在线确诊，需要重视。可能方向包括偏头痛、血压明显升高、"
            "眼科急症、颅内压增高、感染或其他神经系统问题。若出现突发最严重头痛、肢体无力/麻木、说话不清、意识改变、"
            "发热颈强直、持续呕吐、视力明显下降或血压很高，应立即急诊。即使没有这些表现，持续一周反复发作并伴视觉症状，"
            "也建议尽快就诊神经内科/眼科或急诊，记录头痛特点并测量血压。"
        ),
        "findings": [
            "头痛伴恶心和视物模糊属于需要重视的组合。",
            "应优先排除血压明显升高、眼科急症、颅内压增高和神经系统急症。",
            "出现神经功能缺损、意识改变、突发最严重头痛等红旗表现需急诊。",
            "持续反复一周并伴视觉症状，建议尽快线下就医。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 3,
    },
    {
        "id": "seed_copd_chd_dyspnea_redflags",
        "preferred_agent": "diagnostic_agent",
        "query": "慢阻肺 冠心病 咳嗽发热 气短加重 吸氧仍喘 怎么办",
        "title": "COPD/冠心病老人发热咳嗽气短加重的急症风险",
        "triggers": [
            ["慢阻肺", "冠心病", "气短"],
            ["慢阻肺", "发热", "喘"],
            ["吸氧", "还是喘"],
            ["COPD", "heart disease", "dyspnea"],
            ["COPD", "fever", "shortness of breath"],
        ],
        "keywords": ["慢阻肺", "冠心病", "咳嗽", "发热", "气短", "吸氧", "喘", "肺炎", "心衰", "COPD", "dyspnea"],
        "answer": (
            "【本地分诊记忆】70 岁老人有慢阻肺和冠心病，出现咳嗽发热、气短明显加重，且家里吸氧后仍喘，"
            "属于高危甚至急症场景。需要警惕 COPD 急性加重、肺炎、低氧血症、心衰或急性冠脉问题。"
            "不建议继续在家观察，应尽快急诊/呼叫 120，途中保持半坐位、避免活动，携带既往病历和用药清单。"
            "不要自行加大氧流量或自行使用抗生素、激素、心血管药物，除非医生曾明确交代。"
        ),
        "findings": [
            "慢阻肺/冠心病老人发热咳嗽并气短加重属于高危组合。",
            "吸氧后仍喘提示家庭处理不足，应尽快急诊评估血氧、感染和心肺状态。",
            "需警惕 COPD 急性加重、肺炎、心衰和急性冠脉事件。",
            "不建议自行调整氧疗、抗生素、激素或心血管处方药。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 3,
    },
    {
        "id": "seed_diabetes_symptoms_tests",
        "preferred_agent": "research_agent",
        "query": "口渴 尿多 体重下降 糖尿病 检查",
        "title": "多饮多尿体重下降的糖尿病筛查与就医建议",
        "triggers": [
            ["口渴", "尿多", "体重下降"],
            ["多饮", "多尿", "体重下降"],
            ["糖尿病", "做什么检查"],
            ["diabetes", "polyuria", "weight loss"],
        ],
        "keywords": ["口渴", "尿多", "多饮", "多尿", "体重下降", "糖尿病", "血糖", "HbA1c", "尿酮", "酮症酸中毒"],
        "answer": (
            "【本地分诊记忆】口渴、尿多、体重下降是糖尿病的典型可能症状，但不能仅凭症状确诊。"
            "建议尽快到内分泌科或全科就诊，检查空腹血糖、随机血糖、HbA1c，必要时做口服葡萄糖耐量试验；"
            "若症状明显、体重下降快或伴恶心呕吐、腹痛、呼吸深快、乏力/意识变差，应检查尿/血酮、电解质和血气，"
            "警惕糖尿病酮症酸中毒并及时急诊。不要自行购买降糖药或胰岛素。"
        ),
        "findings": [
            "口渴、尿多、体重下降提示糖尿病可能，但不能单凭症状确诊。",
            "核心检查包括空腹血糖、随机血糖、HbA1c，必要时口服葡萄糖耐量试验。",
            "症状明显或伴恶心呕吐、腹痛、呼吸深快、意识差时需排查酮症酸中毒。",
            "不应自行购买或调整降糖药、胰岛素。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 3,
    },
    {
        "id": "seed_exertional_chest_pain",
        "preferred_agent": "diagnostic_agent",
        "query": "胸口闷痛 走路加重 休息缓解 心绞痛 冠心病",
        "title": "活动诱发、休息缓解胸痛的心绞痛风险",
        "triggers": [
            ["胸口", "走路", "休息"],
            ["胸痛", "活动", "缓解"],
            ["胸闷", "活动", "休息"],
            ["chest pain", "exertion", "rest"],
        ],
        "keywords": ["胸口", "胸痛", "胸闷", "走路", "活动", "休息", "心绞痛", "冠心病", "心电图", "肌钙蛋白"],
        "answer": (
            "【本地分诊记忆】活动或走快时胸口闷痛加重、休息缓解，需要警惕心绞痛或冠心病。"
            "应尽快到心内科或急诊评估，通常需要心电图、心肌标志物等检查。若胸痛持续不缓解、伴大汗、气短、恶心、晕厥或放射痛，"
            "应立即拨打 120/急诊。明确前避免剧烈活动，不要自行服用处方心血管药物。"
        ),
        "findings": [
            "活动诱发、休息缓解的胸闷胸痛需优先排除心绞痛/冠心病。",
            "建议尽快心内科或急诊评估心电图和心肌标志物。",
            "持续胸痛或伴大汗、气短、放射痛等需立即急救。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 3,
    },
    {
        "id": "seed_acs_chest_pain_emergency",
        "preferred_agent": "diagnostic_agent",
        "query": "突然剧烈胸痛 大汗 左臂放射痛 急性冠脉综合征",
        "title": "剧烈胸痛伴大汗和左臂放射痛的急救处理",
        "triggers": [
            ["剧烈胸痛", "大汗"],
            ["胸痛", "左臂", "放射"],
            ["胸痛", "大汗", "左臂"],
            ["chest pain", "sweating", "left arm"],
        ],
        "keywords": ["胸痛", "剧烈", "大汗", "左臂", "放射痛", "心梗", "急性冠脉综合征", "120"],
        "answer": (
            "【本地急症记忆】突然剧烈胸痛伴大汗和左臂放射痛，应按疑似急性冠脉综合征/心肌梗死处理。"
            "立即停止活动、就地休息，拨打 120，不要自行开车就医。若医生曾明确开具急救药，可按既往医嘱使用；"
            "不要自行乱服处方药。等待急救时记录发作时间、伴随症状和既往病史。"
        ),
        "findings": [
            "剧烈胸痛伴大汗和左臂放射痛高度提示急性冠脉事件风险。",
            "应立即拨打 120，避免自行开车。",
            "急救药物仅在既往医生明确交代时按医嘱使用。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 3,
    },
    {
        "id": "seed_stroke_fast_emergency",
        "preferred_agent": "diagnostic_agent",
        "query": "口角歪斜 说话含糊 一侧胳膊无力 卒中 FAST",
        "title": "FAST 卒中症状的急救处理",
        "triggers": [
            ["口角歪斜", "说话"],
            ["一侧胳膊", "没力气"],
            ["口角", "一侧", "无力"],
            ["face droop", "arm weakness", "speech"],
        ],
        "keywords": ["口角歪斜", "说话含糊", "一侧", "胳膊", "无力", "卒中", "中风", "FAST", "120"],
        "answer": (
            "【本地急症记忆】突然口角歪斜、说话含糊、一侧胳膊无力符合 FAST 卒中警示信号。"
            "应立即拨打 120/急诊，记录最后正常时间，不要等待自行恢复，不要自行喂水喂药或开车送医。"
            "卒中救治有时间窗，越早评估影像和再灌注治疗可能性越好。"
        ),
        "findings": [
            "口角歪斜、言语含糊、一侧肢体无力是典型卒中警示信号。",
            "应立即拨打 120 并记录最后正常时间。",
            "不要等待观察、喂水喂药或自行开车。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 3,
    },
    {
        "id": "seed_anaphylaxis_child_emergency",
        "preferred_agent": "diagnostic_agent",
        "query": "孩子 花生 喘不上气 嘴唇发紫 风团 过敏性休克",
        "title": "儿童食物过敏后呼吸困难/发绀/风团的急救处理",
        "triggers": [
            ["花生", "喘不上气"],
            ["嘴唇发紫", "风团"],
            ["孩子", "喘不上气", "风团"],
            ["peanut", "wheezing", "hives"],
        ],
        "keywords": ["孩子", "花生", "喘不上气", "嘴唇发紫", "风团", "过敏", "过敏性休克", "肾上腺素", "120"],
        "answer": (
            "【本地急症记忆】孩子吃花生后突然喘不上气、嘴唇发紫、大片风团，需按严重过敏/过敏性休克风险处理。"
            "立即拨打 120；若已有医生开具的肾上腺素自动注射器，应按说明立即使用。让孩子保持安全体位，避免继续进食，"
            "不要自行给口服药后等待观察。即使症状缓解也需要急诊评估。"
        ),
        "findings": [
            "食物暴露后呼吸困难、发绀、风团提示严重过敏风险。",
            "应立即拨打 120；有肾上腺素自动注射器时按医嘱/说明使用。",
            "不要只口服抗过敏药后在家观察。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 3,
    },
    {
        "id": "seed_heart_failure_redflags",
        "preferred_agent": "diagnostic_agent",
        "query": "高血压 糖尿病 夜里憋醒 喘不上气 双腿肿 心衰",
        "title": "夜间憋醒、气短和双腿水肿的心衰风险",
        "triggers": [
            ["夜里憋醒", "双腿"],
            ["喘不上气", "双腿", "肿"],
            ["高血压", "糖尿病", "腿肿"],
            ["orthopnea", "edema", "heart failure"],
        ],
        "keywords": ["夜里憋醒", "喘不上气", "双腿肿", "水肿", "心衰", "冠心病", "高血压", "糖尿病", "orthopnea", "edema"],
        "answer": (
            "【本地分诊记忆】高血压/糖尿病患者出现夜间憋醒、喘不上气、双腿水肿，需要警惕心力衰竭或其他心肺问题。"
            "建议尽快心内科或急诊评估，检查血氧、心电图、BNP/NT-proBNP、胸片/超声心动图、肾功能等。"
            "若气短明显、不能平卧、胸痛、意识差或血氧低，应立即急诊/120。不要自行加减利尿剂或心血管药。"
        ),
        "findings": [
            "夜间憋醒、气短和双腿水肿是心衰相关红旗组合。",
            "合并高血压/糖尿病时心血管风险更高。",
            "需要尽快评估心肺、肾功能和容量状态。",
            "不应自行调整利尿剂或心血管处方药。",
        ],
        "evidence_level": "B",
        "confidence": "high",
        "sources": 3,
    },
]


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_query(query: str) -> str:
    text = (query or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[\W_]+", "", text, flags=re.UNICODE)
    return text


def _ascii_tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _char_bigrams(text: str) -> set:
    normalized = normalize_query(text)
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[i : i + 2] for i in range(len(normalized) - 1)}


def _contains_all(text: str, terms: Iterable[str]) -> bool:
    lowered = (text or "").lower()
    compact = normalize_query(text)
    for term in terms:
        term_lower = str(term or "").lower()
        if not term_lower:
            continue
        if term_lower not in lowered and normalize_query(term_lower) not in compact:
            return False
    return True


class EvidenceMemory:
    """Persistent local cache plus curated seed evidence for medical research."""

    def __init__(self, cache_path: Optional[Path] = None, enabled: Optional[bool] = None):
        env_path = os.getenv("MEDICAL_EVIDENCE_CACHE_PATH")
        self.cache_path = Path(env_path) if env_path else (cache_path or DEFAULT_CACHE_PATH)
        disabled = os.getenv("MEDICAL_EVIDENCE_CACHE_DISABLED", "").lower() in {"1", "true", "yes"}
        self.enabled = (not disabled) if enabled is None else enabled
        self._entries_cache: Optional[List[Dict[str, Any]]] = None

    def lookup(self, query: str, min_score: float = 0.58) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        normalized = normalize_query(query)
        if not normalized:
            return None

        best_entry: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for entry in self._all_entries():
            score = self._score_entry(query, normalized, entry)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry and best_score >= min_score:
            result = dict(best_entry)
            result["cache_hit"] = True
            result["match_score"] = round(best_score, 3)
            result["matched_query"] = best_entry.get("query", "")
            logger.info(
                "Evidence memory hit "
                f"(score={best_score:.2f}, id={best_entry.get('id', 'unknown')})"
            )
            return result
        logger.debug(f"Evidence memory miss for query: {query[:80]}")
        return None

    def store(self, query: str, result: Dict[str, Any]) -> None:
        if not self.enabled or not query or not isinstance(result, dict):
            return
        if result.get("status") not in {None, "completed"}:
            return
        answer = result.get("answer") or ""
        if len(answer) < 80:
            return

        payload = self._load_payload()
        entries = payload.setdefault("entries", [])
        normalized = normalize_query(query)
        now = _now_iso()
        entry = {
            "id": f"cache_{normalized[:32]}",
            "query": query,
            "title": query[:80],
            "normalized_query": normalized,
            "answer": answer,
            "findings": result.get("findings") or [],
            "confidence": result.get("confidence") or "medium",
            "sources": result.get("sources") or 0,
            "evidence_level": result.get("evidence_level") or "C",
            "keywords": self._derive_keywords(query, result),
            "triggers": [],
            "created_at": now,
            "updated_at": now,
            "hit_count": 0,
            "source": "runtime_cache",
        }

        replaced = False
        for idx, existing in enumerate(entries):
            if existing.get("normalized_query") == normalized:
                entry["created_at"] = existing.get("created_at") or now
                entry["hit_count"] = existing.get("hit_count", 0)
                entries[idx] = entry
                replaced = True
                break
        if not replaced:
            entries.append(entry)

        self._write_payload(payload)
        self._entries_cache = None
        logger.info(f"Stored evidence memory entry for query: {query[:80]}")

    def _all_entries(self) -> List[Dict[str, Any]]:
        if self._entries_cache is not None:
            return self._entries_cache
        payload = self._load_payload()
        runtime_entries = payload.get("entries") or []
        entries: List[Dict[str, Any]] = []
        for seed in SEED_EVIDENCE:
            item = dict(seed)
            item.setdefault("source", "seed")
            item.setdefault("normalized_query", normalize_query(item.get("query", "")))
            entries.append(item)
        for entry in runtime_entries:
            if isinstance(entry, dict):
                entry.setdefault("source", "runtime_cache")
                entry.setdefault("normalized_query", normalize_query(entry.get("query", "")))
                entries.append(entry)
        self._entries_cache = entries
        return entries

    def _score_entry(self, query: str, normalized_query: str, entry: Dict[str, Any]) -> float:
        entry_norm = entry.get("normalized_query") or normalize_query(entry.get("query", ""))
        if entry_norm and entry_norm == normalized_query:
            return 1.0

        for trigger in entry.get("triggers") or []:
            if isinstance(trigger, str):
                if _contains_all(query, [trigger]):
                    return 0.92
            elif isinstance(trigger, (list, tuple)) and _contains_all(query, trigger):
                return 0.94

        keywords = [str(k) for k in (entry.get("keywords") or []) if str(k).strip()]
        keyword_score = 0.0
        if keywords:
            hits = sum(1 for keyword in keywords if _contains_all(query, [keyword]))
            keyword_score = hits / max(3, min(len(keywords), 8))

        ascii_query = set(_ascii_tokens(query))
        ascii_entry = set(_ascii_tokens(" ".join([entry.get("query", ""), entry.get("title", "")])))
        ascii_score = (
            len(ascii_query & ascii_entry) / len(ascii_query | ascii_entry)
            if ascii_query and ascii_entry
            else 0.0
        )

        query_bigrams = _char_bigrams(query)
        entry_bigrams = _char_bigrams(" ".join([entry.get("query", ""), entry.get("title", "")]))
        char_score = (
            len(query_bigrams & entry_bigrams) / len(query_bigrams | entry_bigrams)
            if query_bigrams and entry_bigrams
            else 0.0
        )
        return max(keyword_score, ascii_score, char_score)

    def _derive_keywords(self, query: str, result: Dict[str, Any]) -> List[str]:
        text = " ".join([query, " ".join(result.get("findings") or [])])
        tokens = _ascii_tokens(text)
        chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
        seen = set()
        keywords = []
        for token in [*tokens, *chinese_terms]:
            if token in seen:
                continue
            seen.add(token)
            keywords.append(token)
            if len(keywords) >= 16:
                break
        return keywords

    def _load_payload(self) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {"version": 1, "entries": []}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Failed to read evidence cache {self.cache_path}: {exc}")
            return {"version": 1, "entries": []}

    def _write_payload(self, payload: Dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        payload["updated_at"] = _now_iso()
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        try:
            import filelock
            lock_path = self.cache_path.with_suffix(self.cache_path.suffix + ".lock")
            with filelock.FileLock(lock_path):
                tmp_path.replace(self.cache_path)
        except ImportError:
            tmp_path.replace(self.cache_path)
