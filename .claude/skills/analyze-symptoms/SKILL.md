---
name: analyze-symptoms
description: Analyze symptom patterns and potential disease associations. Use when user describes multiple symptoms and needs pattern analysis or differential diagnosis suggestions.
---

# Analyze Symptoms (症状分析)

分析症状模式和潜在疾病关联，用于鉴别诊断。

## When to Use

- 用户描述多个症状，需要模式分析
- 需要鉴别诊断建议
- 评估症状所涉及的身体系统

## 底层实现

- 技术: 症状分类规则引擎 + 症状组合规则
- 数据源: 内置症状系统分类和高危组合规则
- 特点: 输出“可能方向”而不是确诊

## 调用方式

```bash
/analyze-symptoms 头痛,发热,咳嗽
```
