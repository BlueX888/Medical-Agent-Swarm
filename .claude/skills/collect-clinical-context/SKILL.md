---
name: collect-clinical-context
description: Extract structured clinical context from a user's message, identify missing triage fields, generate follow-up questions, and flag potential red flags.
---

# Collect Clinical Context (问诊信息补全)

从用户输入中抽取问诊关键信息，并判断还缺哪些信息。

## When to Use

- 用户刚开始描述症状，需要先补全问诊信息
- 需要整理年龄、性别、症状、持续时间、严重程度、伴随症状、既往史、用药史
- 需要生成下一步追问
- 需要提前识别潜在高危信息

## 底层实现

- 技术: 规则抽取 + 高危关键词识别
- 数据源: 内置问诊字段和红旗症状规则
- 特点: 自包含规则实现，优先补全分诊关键字段

## 调用方式

```bash
/collect-clinical-context 52岁男性，胸痛2小时，伴呼吸困难
```
