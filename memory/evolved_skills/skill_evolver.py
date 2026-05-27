"""
SkillEvolver: 基于使用数据的 Evolved Skill 定向进化

检查 skill 使用统计，对表现不佳的 skill 进行 LLM 辅助改进：
- 分析具体失败模式
- 生成定向改进版本
- 归档旧版本，保存新版本
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from loguru import logger

from .skill_store import SkillStore, EvolvedSkill
from .skill_matcher import SkillMatcher
from .usage_logger import UsageLogger


class SkillEvolver:
    """Evolved Skill 进化器"""

    REFINEMENT_PROMPT = """你是医疗智能体系统的知识工程师。一个工作流指南在实际使用中表现不佳，需要改进。

## 当前工作流

{current_skill_markdown}

## 使用统计
- 被匹配次数: {times_matched}
- 被注入次数: {times_used}
- 正面结果: {positive_outcomes} 次
- 负面结果: {negative_outcomes} 次
- 正面率: {positive_rate:.1%}

## 你的任务

分析此工作流表现不佳的可能原因，并生成改进版本。

**改进方向**:
1. 触发条件是否太宽泛或太狭窄？
2. 工作流步骤是否遗漏了关键环节？
3. Tool 使用模式是否合理？
4. 安全注意事项是否充分？
5. 已知陷阱是否需要补充？

**输出要求**：
1. 先用一行说明改进了什么
2. 然后输出改进后的完整工作流 Markdown（保持原格式）
3. 如果需要更新 embedding_text，在最后用 ```yaml ``` 包裹输出新的 embedding_text

改进后的工作流："""

    def __init__(
        self,
        skill_store: SkillStore,
        skill_matcher: SkillMatcher,
        usage_logger: UsageLogger,
        llm_client,
        min_observations: int = 5,
        refine_threshold: float = 0.4,
        cooldown_hours: int = 24,
    ):
        self.skill_store = skill_store
        self.skill_matcher = skill_matcher
        self.usage_logger = usage_logger
        self.llm_client = llm_client
        self.min_observations = min_observations
        self.refine_threshold = refine_threshold
        self.cooldown_hours = cooldown_hours

    async def check_and_evolve(self) -> List[str]:
        """
        检查所有 skills 的使用统计，对表现不佳的进行 evolution

        Returns:
            被进化的 skill IDs 列表
        """
        underperformers = self.usage_logger.get_underperforming_skills(
            min_observations=self.min_observations,
            max_positive_rate=self.refine_threshold,
        )

        if not underperformers:
            return []

        evolved_ids = []
        for entry in underperformers:
            skill_id = entry["skill_id"]
            skill = self.skill_store.load_skill(skill_id)

            if skill is None:
                continue
            if not self._cooldown_passed(skill):
                continue

            try:
                new_id = await self._refine_skill(skill, entry["stats"])
                if new_id:
                    evolved_ids.append(new_id)
            except Exception as e:
                logger.error(f"Failed to evolve skill {skill_id}: {e}")

        return evolved_ids

    def _cooldown_passed(self, skill: EvolvedSkill) -> bool:
        """检查 skill 是否过了进化冷却期"""
        cooldown = timedelta(hours=self.cooldown_hours)
        return datetime.now() - skill.updated_at > cooldown

    async def _refine_skill(
        self,
        skill: EvolvedSkill,
        stats: Dict[str, Any],
    ) -> Optional[str]:
        """对单个 skill 进行 LLM 辅助 refinement"""
        used = stats.get("positive_outcomes", 0) + stats.get("negative_outcomes", 0) + stats.get("neutral_outcomes", 0)
        positive_rate = stats.get("positive_outcomes", 0) / used if used > 0 else 0

        prompt = self.REFINEMENT_PROMPT.format(
            current_skill_markdown=skill.body_markdown,
            times_matched=stats.get("times_matched", 0),
            times_used=stats.get("times_used", 0),
            positive_outcomes=stats.get("positive_outcomes", 0),
            negative_outcomes=stats.get("negative_outcomes", 0),
            positive_rate=positive_rate,
        )

        try:
            response = await self.llm_client.chat([
                {"role": "user", "content": prompt}
            ])
            if not response:
                return None
        except Exception as e:
            logger.error(f"Skill evolution LLM call failed: {e}")
            return None

        # 提取改进后的 body
        improved_body = self._extract_improved_body(response)
        if not improved_body:
            return None

        # 检查是否有新的 embedding_text
        import re
        import yaml as yaml_mod
        new_embedding = skill.embedding_text
        yaml_match = re.search(r"```yaml\s*\n(.*?)```", response, re.DOTALL)
        if yaml_match:
            try:
                yaml_data = yaml_mod.safe_load(yaml_match.group(1))
                if isinstance(yaml_data, dict) and "embedding_text" in yaml_data:
                    new_embedding = yaml_data["embedding_text"]
            except Exception:
                pass

        # 归档旧版本
        self.skill_store.archive_skill(skill.id)

        # 创建新版本
        now = datetime.now()
        new_skill = EvolvedSkill(
            id=skill.id,  # 保持同一 ID
            name=skill.name,
            version=skill.version + 1,
            created_at=skill.created_at,
            updated_at=now,
            source_sessions=skill.source_sessions,
            trigger_keywords=skill.trigger_keywords,
            category=skill.category,
            target_agents=skill.target_agents,
            performance={
                "times_matched": 0,
                "times_used": 0,
                "positive_outcomes": 0,
                "negative_outcomes": 0,
            },
            embedding_text=new_embedding,
            body_markdown=improved_body,
        )

        self.skill_store.save_skill(new_skill)
        self.skill_matcher.invalidate_cache()

        self.usage_logger.log_synthesis(
            session_id=None,
            action="refined",
            skill_id=skill.id,
            skill_name=skill.name,
            reason=f"Positive rate {positive_rate:.1%} < {self.refine_threshold:.1%} after {used} observations. Evolved to v{new_skill.version}.",
        )

        logger.info(f"Evolved skill {skill.id}: v{skill.version} -> v{new_skill.version}")
        return skill.id

    def _extract_improved_body(self, response: str) -> Optional[str]:
        """从 LLM 响应中提取改进后的 workflow body"""
        # 跳过第一行（改进说明）
        lines = response.strip().split("\n")
        if not lines:
            return None

        # 找到第一个 # 开头的行作为 body 起始
        body_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("#"):
                body_start = i
                break

        body = "\n".join(lines[body_start:])

        # 移除末尾可能的 yaml 块
        import re
        body = re.sub(r"\n```yaml\s*\n.*?```\s*$", "", body, flags=re.DOTALL)

        return body.strip() if body.strip() else None
