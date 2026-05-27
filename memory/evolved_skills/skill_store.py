"""
SkillStore: Evolved Skill 的数据模型和文件系统管理

Evolved Skill 是从成功会话中自动提炼的工作流指南，
以 YAML frontmatter + Markdown body 格式存储，
通过语义匹配注入 Agent 系统提示词。
"""
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

import yaml
from loguru import logger


@dataclass
class EvolvedSkill:
    """Evolved Skill 数据对象"""
    id: str
    name: str
    version: int
    created_at: datetime
    updated_at: datetime
    source_sessions: List[str]
    trigger_keywords: List[str]
    category: str
    target_agents: List[str]
    performance: Dict[str, Any]
    embedding_text: str
    body_markdown: str
    file_path: Optional[Path] = None

    @property
    def positive_rate(self) -> float:
        """基于 performance 计算正面率"""
        used = self.performance.get("times_used", 0)
        if used == 0:
            return 0.5
        positive = self.performance.get("positive_outcomes", 0)
        return positive / used

    def to_frontmatter_dict(self) -> Dict[str, Any]:
        """序列化为 YAML frontmatter 字典"""
        return {
            "id": self.id,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_sessions": self.source_sessions,
            "trigger_keywords": self.trigger_keywords,
            "category": self.category,
            "target_agents": self.target_agents,
            "performance": self.performance,
            "embedding_text": self.embedding_text,
        }

    def to_markdown(self) -> str:
        """序列化为完整 markdown 文件内容"""
        frontmatter = yaml.dump(
            self.to_frontmatter_dict(),
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
        return f"---\n{frontmatter}---\n\n{self.body_markdown}"


class SkillStore:
    """
    Evolved Skill 文件存储管理器

    skills/ — 活跃的 evolved skill 文件
    archive/ — 被进化替换的旧版本
    """

    MAX_SKILLS = 100  # 最大 skill 数量，超出时按 last_used 淘汰

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path(__file__).parent
        self.base_dir = base_dir
        self.skills_dir = base_dir / "skills"
        self.archive_dir = base_dir / "archive"

        for d in [self.skills_dir, self.archive_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def _parse_skill_file(self, file_path: Path) -> Optional[EvolvedSkill]:
        """解析 evolved skill markdown 文件"""
        try:
            text = file_path.read_text(encoding="utf-8")
            if not text.startswith("---"):
                return None

            parts = text.split("---", 2)
            if len(parts) < 3:
                return None

            meta = yaml.safe_load(parts[1])
            if not meta or not isinstance(meta, dict):
                return None

            body = parts[2].strip()

            # 解析时间字段
            created_at = meta.get("created_at", "")
            if isinstance(created_at, str):
                created_at = datetime.fromisoformat(created_at)
            elif not isinstance(created_at, datetime):
                created_at = datetime.now()

            updated_at = meta.get("updated_at", "")
            if isinstance(updated_at, str):
                updated_at = datetime.fromisoformat(updated_at)
            elif not isinstance(updated_at, datetime):
                updated_at = datetime.now()

            return EvolvedSkill(
                id=meta.get("id", file_path.stem),
                name=meta.get("name", "Unknown"),
                version=meta.get("version", 1),
                created_at=created_at,
                updated_at=updated_at,
                source_sessions=meta.get("source_sessions", []),
                trigger_keywords=meta.get("trigger_keywords", []),
                category=meta.get("category", "general"),
                target_agents=meta.get("target_agents", []),
                performance=meta.get("performance", {}),
                embedding_text=meta.get("embedding_text", ""),
                body_markdown=body,
                file_path=file_path,
            )
        except Exception as e:
            logger.warning(f"Failed to parse skill file {file_path}: {e}")
            return None

    def save_skill(self, skill: EvolvedSkill) -> Path:
        """保存 skill 到文件系统"""
        file_path = self.skills_dir / f"{skill.id}.md"
        file_path.write_text(skill.to_markdown(), encoding="utf-8")
        skill.file_path = file_path
        logger.debug(f"Saved evolved skill: {skill.id} -> {file_path}")
        return file_path

    def load_skill(self, skill_id: str) -> Optional[EvolvedSkill]:
        """按 ID 加载 skill"""
        file_path = self.skills_dir / f"{skill_id}.md"
        if not file_path.exists():
            return None
        return self._parse_skill_file(file_path)

    def list_all_skills(self) -> List[EvolvedSkill]:
        """列出所有活跃的 evolved skills"""
        skills = []
        for f in self.skills_dir.glob("*.md"):
            skill = self._parse_skill_file(f)
            if skill:
                skills.append(skill)
        return skills

    def archive_skill(self, skill_id: str) -> Optional[Path]:
        """归档 skill（移动到 archive/ 并加时间戳）"""
        src = self.skills_dir / f"{skill_id}.md"
        if not src.exists():
            return None

        skill = self._parse_skill_file(src)
        version = skill.version if skill else 0
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        dst = self.archive_dir / f"{skill_id}_v{version}_{ts}.md"
        shutil.move(str(src), str(dst))
        logger.info(f"Archived skill: {skill_id} -> {dst.name}")
        return dst

    def delete_skill(self, skill_id: str) -> bool:
        """删除 skill 文件"""
        file_path = self.skills_dir / f"{skill_id}.md"
        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def get_all_embedding_texts(self) -> List[Tuple[str, str]]:
        """返回所有 skills 的 (skill_id, embedding_text) 列表"""
        results = []
        for skill in self.list_all_skills():
            if skill.embedding_text:
                results.append((skill.id, skill.embedding_text))
        return results

    def prune_if_needed(self) -> int:
        """如果 skill 数量超过上限，按更新时间淘汰最旧的"""
        skills = self.list_all_skills()
        if len(skills) <= self.MAX_SKILLS:
            return 0

        skills.sort(key=lambda s: s.updated_at)
        to_remove = len(skills) - self.MAX_SKILLS
        removed = 0
        for skill in skills[:to_remove]:
            self.archive_skill(skill.id)
            removed += 1
        logger.info(f"Pruned {removed} oldest skills (limit={self.MAX_SKILLS})")
        return removed

    @staticmethod
    def generate_skill_id() -> str:
        """生成唯一 skill ID"""
        date_str = datetime.now().strftime("%Y%m%d")
        short_uuid = str(uuid.uuid4())[:8]
        return f"es_{date_str}_{short_uuid}"
