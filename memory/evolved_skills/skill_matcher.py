"""
SkillMatcher: 基于语义相似度匹配 Evolved Skills

复用 MedicalKnowledgeBase 的 BAAI/bge-small-zh-v1.5 embedding 模型，
对用户查询与 evolved skill 的 embedding_text 做余弦相似度匹配。
"""
from typing import List, Dict, Any, Optional

import numpy as np
from loguru import logger

from .skill_store import SkillStore


class SkillMatcher:
    """
    Evolved Skill 语义匹配器

    - 复用外部 SentenceTransformer 实例（从 MedicalKnowledgeBase 获取）
    - 维护内存 embedding 缓存，skill 变更时 invalidate
    - embedding model 不可用时静默降级
    """

    def __init__(
        self,
        skill_store: SkillStore,
        embedding_model=None,
        top_k: int = 3,
        threshold: float = 0.65,
    ):
        self.skill_store = skill_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.threshold = threshold

        # 内存缓存: {skill_id: np.ndarray}
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_built = False

    def _ensure_cache(self) -> bool:
        """构建/重建 embedding 缓存"""
        if self._cache_built:
            return True
        if self.embedding_model is None:
            return False

        try:
            entries = self.skill_store.get_all_embedding_texts()
            if not entries:
                self._cache_built = True
                return True

            skill_ids = [e[0] for e in entries]
            texts = [e[1] for e in entries]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)

            self._embedding_cache = {
                sid: emb for sid, emb in zip(skill_ids, embeddings)
            }
            self._cache_built = True
            logger.debug(f"SkillMatcher: cached {len(self._embedding_cache)} skill embeddings")
            return True
        except Exception as e:
            logger.error(f"SkillMatcher: failed to build cache: {e}")
            return False

    def invalidate_cache(self) -> None:
        """使缓存失效（新增/修改/删除 skill 后调用）"""
        self._embedding_cache.clear()
        self._cache_built = False

    def match(
        self,
        query: str,
        target_agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        对用户查询进行语义匹配

        Returns:
            匹配结果列表，按相似度降序：
            [{"skill_id", "skill_name", "score", "skill"}, ...]
        """
        if self.embedding_model is None:
            return []

        try:
            if not self._ensure_cache():
                return []
            if not self._embedding_cache:
                return []

            query_vec = self.embedding_model.encode([query], show_progress_bar=False)[0]

            results = []
            for skill_id, skill_vec in self._embedding_cache.items():
                score = self._cosine_similarity(query_vec, skill_vec)
                if score < self.threshold:
                    continue

                skill = self.skill_store.load_skill(skill_id)
                if skill is None:
                    continue
                if target_agent and target_agent not in skill.target_agents:
                    continue

                results.append({
                    "skill_id": skill_id,
                    "skill_name": skill.name,
                    "score": float(score),
                    "skill": skill,
                })

            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:self.top_k]

        except Exception as e:
            logger.error(f"SkillMatcher.match() failed: {e}")
            return []

    def check_duplicate(self, embedding_text: str, threshold: float = 0.85) -> Optional[str]:
        """
        检查 embedding_text 是否与现有 skill 重复

        Returns:
            重复的 skill_id，或 None
        """
        if self.embedding_model is None:
            return None

        try:
            if not self._ensure_cache():
                return None
            if not self._embedding_cache:
                return None

            query_vec = self.embedding_model.encode([embedding_text], show_progress_bar=False)[0]

            for skill_id, skill_vec in self._embedding_cache.items():
                score = self._cosine_similarity(query_vec, skill_vec)
                if score >= threshold:
                    return skill_id

            return None
        except Exception as e:
            logger.error(f"SkillMatcher.check_duplicate() failed: {e}")
            return None

    def format_for_prompt(
        self,
        matched_skills: List[Dict[str, Any]],
        max_chars: int = 3000,
    ) -> str:
        """将匹配到的 skills 格式化为可注入 system prompt 的文本块"""
        if not matched_skills:
            return ""

        header = (
            "\n---\n"
            "## [自进化工作流指南] 以下是基于历史成功案例提炼的工作流建议，仅供参考：\n"
        )
        footer = "\n注意：以上工作流仅为参考，请根据具体情况灵活调整。\n---"
        total_chars = len(header) + len(footer)

        blocks = []
        for i, match_result in enumerate(matched_skills, 1):
            skill = match_result["skill"]
            score = match_result["score"]

            block = (
                f"\n### 参考工作流 {i}: {skill.name} (匹配度: {score:.0%})\n"
                f"{skill.body_markdown}\n"
            )

            if total_chars + len(block) > max_chars:
                blocks.append("\n(更多工作流因篇幅限制被省略)")
                break

            blocks.append(block)
            total_chars += len(block)

        return header + "".join(blocks) + footer

    @staticmethod
    def _cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
