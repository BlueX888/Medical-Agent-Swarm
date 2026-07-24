---
name: recommend-lifestyle
description: Provide lifestyle and medication guidance based on disease or symptoms. Use when user asks about diet, exercise, sleep advice, or basic medication guidance for specific conditions.
---

# Recommend Lifestyle (生活方式建议)

根据疾病或症状提供生活方式建议，包括饮食、运动、睡眠和基础用药指导。

## When to Use

- 用户问"高血压患者饮食注意什么""糖尿病如何运动"
- 需要生活方式调整建议
- 需要基础用药指导

## 底层实现

- 技术: 内置生活方式模板库
- 数据源: 高血压、糖尿病、感冒/呼吸道症状、睡眠、体重管理、一般健康模板
- 特点: 高危或紧急风险时拒绝用生活方式建议替代就医

## 调用方式

```bash
/recommend-lifestyle 高血压
```
