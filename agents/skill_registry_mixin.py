"""
统一的 Skills 注册（自动发现）
所有 Worker Agents 共享
"""
from core.skill_loader import discover_skills, discover_active_skills
from core.skill_registry import SkillParameter
from pathlib import Path
from loguru import logger
import inspect
import typing
from typing import get_type_hints, Literal


class SkillRegistryMixin:
    """
    统一注册所有 Skills（自动发现模式）

    所有 Worker Agents (ConsultationAgent, DiagnosticAgent, ResearchAgent)
    都继承这个 Mixin，避免重复代码
    """

    def register_all_skills(self):
        """
        自动扫描并注册所有 Skills

        Skills 会从 .claude/skills/ 目录自动发现，
        无需手动维护列表
        """
        allowed = getattr(self, 'allowed_skill_names', None)

        project_root = Path(__file__).parent.parent
        discovered = discover_skills(project_root)
        active = discover_active_skills(project_root, discovered)

        # 自动注册所有 active skills（过滤 auto_execute 或显式禁用的）
        registered_count = 0
        for skill_info in active:
            function_name = skill_info["function_name"]
            metadata = skill_info["metadata"]
            func = skill_info["function"]

            if allowed is not None and function_name not in allowed:
                logger.debug(f"Skipping skill {function_name} (not in allowed list for {self.__class__.__name__})")
                continue

            # 从 metadata 获取描述
            description = metadata.get("description", f"Skill: {skill_info['name']}")

            # 根据函数签名自动推断参数
            parameters = self._infer_skill_parameters(skill_info)

            # 注册到 SkillRegistry
            self.skill_registry.register(
                name=function_name,
                function=func,
                description=description,
                parameters=parameters
            )
            registered_count += 1
            logger.info(f"✅ Registered skill: {function_name}")

        logger.info(
            f"Total {registered_count} skills registered "
            f"(discovered {len(discovered)}, active {len(active)}, skipped {len(discovered) - len(active)})"
        )

    def _infer_skill_parameters(self, skill_info: dict) -> list:
        """
        从 skill 信息推断参数

        Args:
            skill_info: skill 信息字典

        Returns:
            [SkillParameter(...), ...]
        """
        func = skill_info["function"]

        # 获取函数签名和类型提示
        sig = inspect.signature(func)
        try:
            type_hints = get_type_hints(func)
        except Exception:
            type_hints = {}

        parameters = []

        for param_name, param in sig.parameters.items():
            # 跳过 self 和特殊参数
            if param_name in ["self", "args", "kwargs"]:
                continue

            # 判断是否必需
            required = param.default == inspect.Parameter.empty

            # 推断类型和 enum
            param_type = "string"
            enum_values = None

            hint = type_hints.get(param_name)
            if hint is not None and getattr(hint, '__origin__', None) is Literal:
                # typing.Literal["a", "b", "c"] → enum
                enum_values = list(hint.__args__)
                param_type = "string"
            elif "count" in param_name or "limit" in param_name or "max" in param_name or "iterations" in param_name:
                param_type = "number"

            # 生成描述
            param_desc = param_name.replace('_', ' ').title()

            parameters.append(SkillParameter(
                param_name,
                param_type,
                param_desc,
                required,
                enum=enum_values,
            ))

        return parameters
