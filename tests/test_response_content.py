from core.response_content import (
    sanitize_user_visible_answer,
    strip_legacy_knowledge_citations,
    strip_trailing_structured_metadata,
)


def test_strips_trailing_agent_metadata_json_block():
    answer = "\n".join(
        [
            "## 居家建议",
            "请注意休息并持续观察。",
            "```json",
            '{"suggestions":["每4~6小时监测体温"],"risk_level":"medium"}',
            "```",
        ]
    )

    assert strip_trailing_structured_metadata(answer) == (
        "## 居家建议\n请注意休息并持续观察。"
    )


def test_preserves_user_facing_json_examples_and_invalid_blocks():
    json_example = '示例：\n```json\n{"symptom":"headache"}\n```'
    invalid_json = '原文：\n```json\n{"suggestions": [}\n```'

    assert strip_trailing_structured_metadata(json_example) == json_example
    assert strip_trailing_structured_metadata(invalid_json) == invalid_json


def test_removes_legacy_knowledge_citations_without_touching_other_brackets():
    answer = "建议今天就医 [K1]，并携带用药清单 [K23]。保留 [KPI]。"

    assert strip_legacy_knowledge_citations(answer) == (
        "建议今天就医，并携带用药清单。保留 [KPI]。"
    )


def test_user_visible_sanitizer_applies_metadata_and_legacy_citation_cleanup():
    answer = '结论 [K2]。\n```json\n{"risk_level":"low"}\n```'

    assert sanitize_user_visible_answer(answer) == "结论。"
