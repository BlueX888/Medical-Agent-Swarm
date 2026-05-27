"""
UsageLogger: Evolved Skill 使用日志（JSONL 格式）

记录 skill 匹配、注入、会话结果和合成事件，
为 SkillEvolver 提供数据支撑。
"""
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from loguru import logger


class UsageLogger:
    """JSONL 日志记录器"""

    def __init__(self, log_dir: Optional[Path] = None):
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs"
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.usage_log_path = self.log_dir / "skill_usage.jsonl"
        self.synthesis_log_path = self.log_dir / "synthesis_log.jsonl"
        self._write_lock = asyncio.Lock()

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        """追加一条 JSONL 记录（同步写入）"""
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to {path.name}: {e}")

    def log_match(
        self,
        session_id: str,
        query: str,
        matched_skills: List[Dict[str, Any]],
    ) -> None:
        """记录 skill 匹配事件"""
        self._append_jsonl(self.usage_log_path, {
            "event": "match",
            "ts": datetime.now().isoformat(),
            "sid": session_id,
            "query": query[:200],
            "matches": [
                {
                    "skill_id": m.get("skill_id", ""),
                    "name": m.get("skill_name", ""),
                    "score": round(m.get("score", 0), 4),
                    "injected": m.get("injected", False),
                }
                for m in matched_skills
            ],
        })

    def log_session_outcome(
        self,
        session_id: str,
        injected_skill_ids: List[str],
        outcome: str,
        session_time_seconds: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录会话结果（关联到已注入的 skills）"""
        self._append_jsonl(self.usage_log_path, {
            "event": "outcome",
            "ts": datetime.now().isoformat(),
            "sid": session_id,
            "injected": injected_skill_ids,
            "outcome": outcome,
            "time_s": round(session_time_seconds, 2),
            "meta": metadata or {},
        })

    def log_synthesis(
        self,
        session_id: Optional[str],
        action: str,
        skill_id: Optional[str] = None,
        skill_name: Optional[str] = None,
        reason: str = "",
    ) -> None:
        """记录 skill 合成/进化事件"""
        self._append_jsonl(self.synthesis_log_path, {
            "event": "synthesis" if action in ("created", "skipped_duplicate", "failed") else "evolution",
            "ts": datetime.now().isoformat(),
            "sid": session_id,
            "action": action,
            "skill_id": skill_id,
            "name": skill_name,
            "reason": reason,
        })

    def get_skill_usage_stats(self, skill_id: str) -> Dict[str, Any]:
        """聚合某个 skill 的使用统计"""
        stats = {
            "times_matched": 0,
            "times_used": 0,
            "positive_outcomes": 0,
            "negative_outcomes": 0,
            "neutral_outcomes": 0,
            "total_session_time": 0.0,
            "last_used": None,
        }

        if not self.usage_log_path.exists():
            return stats

        try:
            with open(self.usage_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if record.get("event") == "match":
                        for m in record.get("matches", []):
                            if m.get("skill_id") == skill_id:
                                stats["times_matched"] += 1
                                if m.get("injected"):
                                    stats["times_used"] += 1

                    elif record.get("event") == "outcome":
                        if skill_id in record.get("injected", []):
                            outcome = record.get("outcome", "neutral")
                            if outcome == "positive":
                                stats["positive_outcomes"] += 1
                            elif outcome == "negative":
                                stats["negative_outcomes"] += 1
                            else:
                                stats["neutral_outcomes"] += 1
                            stats["total_session_time"] += record.get("time_s", 0)
                            stats["last_used"] = record.get("ts")

        except Exception as e:
            logger.error(f"Failed to read usage log: {e}")

        return stats

    def get_underperforming_skills(
        self,
        min_observations: int = 5,
        max_positive_rate: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """找出表现不佳的 skills"""
        # 先收集所有出现过的 skill_ids
        all_skill_ids = set()
        if self.usage_log_path.exists():
            try:
                with open(self.usage_log_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            record = json.loads(line.strip())
                            if record.get("event") == "outcome":
                                for sid in record.get("injected", []):
                                    all_skill_ids.add(sid)
                        except (json.JSONDecodeError, ValueError):
                            continue
            except Exception as e:
                logger.error(f"Failed to scan usage log: {e}")
                return []

        underperformers = []
        for skill_id in all_skill_ids:
            stats = self.get_skill_usage_stats(skill_id)
            used = stats["positive_outcomes"] + stats["negative_outcomes"] + stats["neutral_outcomes"]
            if used < min_observations:
                continue
            positive_rate = stats["positive_outcomes"] / used if used > 0 else 0
            if positive_rate < max_positive_rate:
                underperformers.append({
                    "skill_id": skill_id,
                    "stats": stats,
                    "positive_rate": positive_rate,
                })

        return underperformers
