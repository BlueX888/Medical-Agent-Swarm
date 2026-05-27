"""
记忆系统：Agent 的持久化学习和记忆管理

包含：
- ShortTermMemory：会话级对话历史（内存/Redis）
- LongTermMemory：跨会话记忆（Mem0）
"""

# 短期和长期记忆
from .short_term import (
    ShortTermMemory,
    ConversationHistory
)
from .long_term import (
    LongTermMemory
)

# 本地 Markdown 持久化
from .agent_identity import (
    AgentIdentity,
    AgentIdentityManager,
    CollaborationRecord,
    ToolUsageStats
)
from .session_summary import (
    SessionSummary,
    SessionSummaryManager,
    AgentParticipation,
    KeyFinding,
    Lesson,
    PerformanceMetrics
)

# Evolved Skills: 自进化 Skill 系统
try:
    from .evolved_skills import (
        SkillStore,
        EvolvedSkill,
        SkillMatcher,
        SkillSynthesizer,
        SkillEvolver,
        UsageLogger,
    )
    _EVOLVED_SKILLS_EXPORTS = [
        'SkillStore', 'EvolvedSkill', 'SkillMatcher',
        'SkillSynthesizer', 'SkillEvolver', 'UsageLogger',
    ]
except ImportError:
    _EVOLVED_SKILLS_EXPORTS = []

__all__ = [
    # 短期和长期记忆
    'ShortTermMemory',
    'ConversationHistory',
    'LongTermMemory',
    # 本地持久化类
    'AgentIdentity',
    'AgentIdentityManager',
    'LearningRecord',
    'CollaborationRecord',
    'ToolUsageStats',
    'SessionSummary',
    'SessionSummaryManager',
    'AgentParticipation',
    'KeyFinding',
    'Lesson',
    'PerformanceMetrics',
] + _EVOLVED_SKILLS_EXPORTS
