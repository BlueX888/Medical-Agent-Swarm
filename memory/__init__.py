"""
记忆系统：Agent 的持久化学习和记忆管理

包含：
- ShortTermMemory：会话级对话历史（默认内存）
- LongTermMemory：可选跨会话记忆接口（默认禁用）
"""

# 短期和长期记忆
from .short_term import (
    ShortTermMemory,
    MemoryMessage,
    create_short_term_memory,
)
from .long_term import (
    LongTermMemory
)
from .evidence_cache import (
    EvidenceMemory
)

# 本地 Markdown 持久化
from .session_summary import (
    SessionSummary,
    SessionSummaryManager,
    AgentParticipation,
    KeyFinding,
    Lesson,
    PerformanceMetrics
)

__all__ = [
    # 短期和长期记忆
    'ShortTermMemory',
    'MemoryMessage',
    'create_short_term_memory',
    'LongTermMemory',
    'EvidenceMemory',
    # 本地持久化类
    'SessionSummary',
    'SessionSummaryManager',
    'AgentParticipation',
    'KeyFinding',
    'Lesson',
    'PerformanceMetrics',
]
