---
name: assess-risk
description: Assess symptom risk level (low/medium/high/emergency). Use when user describes symptoms and needs risk evaluation to determine urgency of medical attention.
---

# Assess Risk (风险评估)

评估症状的风险等级，判断是否需要紧急就医。

## When to Use

- 用户描述症状，需要评估严重程度
- 判断是否需要紧急就医
- 风险分级（低/中/高/紧急）

## 底层实现

- 技术: 风险规则引擎
- 数据源: 内置红旗症状、症状组合和特殊人群加权规则
- 特点: 自包含规则实现，适合稳定风险分诊

## 调用方式

```bash
/assess-risk 胸痛,呼吸困难
```
