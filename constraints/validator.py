"""
约束验证器
运行时检查 Agent 行为是否违反约束

基于 Harness Engineering 原则：
- 显式化约束
- 运行时验证
- 最终文本安全由 SafetyGuard 统一兜底
"""
from typing import Dict, Any, List, Optional
import yaml
from pathlib import Path
from loguru import logger


class ConstraintValidator:
    """约束验证器"""

    def __init__(
        self,
        agent_constraints_file: str = "constraints/agent_constraints.yaml",
        swarm_constraints_file: str = "constraints/swarm_constraints.yaml"
    ):
        """
        初始化约束验证器

        Args:
            agent_constraints_file: Agent约束定义文件
            swarm_constraints_file: Swarm约束定义文件
        """
        # 加载 Agent 约束
        agent_path = Path(__file__).parent / "agent_constraints.yaml"
        with open(agent_path, 'r', encoding='utf-8') as f:
            self.agent_constraints = yaml.safe_load(f)

        # 加载 Swarm 约束
        swarm_path = Path(__file__).parent / "swarm_constraints.yaml"
        with open(swarm_path, 'r', encoding='utf-8') as f:
            self.swarm_constraints = yaml.safe_load(f)

        logger.info("✅ ConstraintValidator initialized")

    def validate_tool_call(self, agent_id: str, tool_name: str) -> Dict[str, Any]:
        """
        验证工具调用是否允许

        Args:
            agent_id: Agent ID
            tool_name: 工具名称

        Returns:
            {
                "valid": bool,
                "reason": str (如果不允许)
            }
        """
        agent_constraints = self.agent_constraints['agents'].get(agent_id, {})
        allowed_tools = agent_constraints.get('allowed_tools', [])

        # 如果 allowed_tools 为空，表示没有限制
        if not allowed_tools:
            return {"valid": True}

        # 检查工具是否在允许列表中
        if tool_name not in allowed_tools:
            reason = f"工具 {tool_name} 不在 {agent_id} 的推荐工具列表中"
            logger.warning(f"⚠️ {reason}")
            return {
                "valid": False,
                "reason": reason,
                "severity": "warning"  # 警告级别（不阻止执行，只记录）
            }

        return {"valid": True}

    def validate_task_decomposition(
        self,
        question: str,
        subtasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        验证任务分解是否合理（基于 Swarm 约束）

        Args:
            question: 用户问题
            subtasks: LangGraph planning 节点分解的子任务列表

        Returns:
            {
                "valid": bool,
                "issues": List[str],
                "recommendations": List[str]
            }
        """
        rules = self.swarm_constraints['swarm']['task_decomposition_rules']
        issues = []
        recommendations = []

        num_subtasks = len(subtasks)

        # 检查是否匹配规则
        for rule in rules:
            pattern = rule['pattern']
            keywords = pattern.split('|')

            if any(kw in question for kw in keywords):
                max_subtasks = rule.get('max_subtasks')
                min_subtasks = rule.get('min_subtasks', 1)

                if max_subtasks and num_subtasks > max_subtasks:
                    issues.append(
                        f"任务过度分解：{rule['name']} 类型问题最多 {max_subtasks} 个子任务，"
                        f"当前 {num_subtasks} 个"
                    )
                    recommendations.append(f"建议合并为 {max_subtasks} 个任务")

                if min_subtasks and num_subtasks < min_subtasks:
                    issues.append(
                        f"任务分解不足：{rule['name']} 类型问题至少需要 {min_subtasks} 个子任务，"
                        f"当前 {num_subtasks} 个"
                    )

                # 找到匹配规则，停止检查
                break

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "recommendations": recommendations
        }