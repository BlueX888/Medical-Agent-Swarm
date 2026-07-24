# Medical-Agent-Swarm 简历项目经历整理

## 项目指标分析

数据来源：`HEALTHBENCH_ZH18_MEMORY_OPT_V3_REPORT.md` 与同目录优化摘要。

- 评测对象：HealthBench-style 中文医疗 Agent Benchmark，共 18 个病例，覆盖基础医学知识、复杂合并症、急症分诊、慢病与生活方式、指南/循证检索、症状分析 6 类场景。
- 全局 SLO：P95 延迟 <= 180s，超时率 <= 5%。
- V3 全量结果：18/18 成功，0 超时，0 错误，SLO PASS。
- 性能：平均延迟 59.2s，P50 50.9s，P95 100.8s，最大 125.1s。
- 成本与调用：平均 3.6 次 LLM 调用 / case，平均 9,537 tokens / case，平均成本 $0.002088 / case。
- 难度维度：hard 7/7 成功，平均延迟 52.0s，P95 74.4s；medium 9/9 成功，平均延迟 69.1s，P95 113.7s；easy 2/2 成功。
- 优化收益：相比 baseline，成功数从 14/18 提升到 18/18，超时率从 22.2% 降至 0，平均延迟降低 33.6%，P95 降低 26.6%，平均 LLM 调用数降低 36.8%，平均 token 降低 40.7%，平均单次成本降低 47.5%。

> 简历表述建议：这些指标证明的是系统性能、稳定性与 token 成本优化，不建议写成“医疗准确率”或“诊断准确率”。当前每个病例 repetition=1，适合写“自建 / HealthBench-style 评测集上的性能回归结果”。

## 简历版项目经历

**医疗多智能体问答系统 Medical-Agent-Swarm | LLM Agent 工程 / 性能优化**  
技术栈：Python、LangGraph、LangChain Core、OpenAI SDK、FastAPI、Pydantic、pandas / numpy、DDGS / BeautifulSoup、React

- 基于 LangGraph 构建医疗多智能体问答系统，设计咨询、症状分析、医学研究等 Agent 协作链路，接入医疗 Skill 完成问诊信息收集、风险评估、症状模式分析、循证检索与输出安全检查。
- 建设 HealthBench-style 中文医疗 Agent 性能评测集，覆盖基础医学知识、复杂合并症、急症分诊、慢病与生活方式、指南 / 循证检索、症状分析 6 类场景；实现自动化 benchmark，统计成功率、超时率、P50 / P95 / Max 延迟、LLM 调用次数、token 与单次成本。
- 定位系统慢路径主要来自实时指南 / 循证检索与冗余多 Agent 拆解，设计 EvidenceMemory 本地证据缓存（种子证据 + runtime JSON cache）、DeepResearch cache-first 策略与迭代上限，并在高置信证据命中时启用单 Agent 快路径。
- 经过 BadCase 定位、定向回归与全量回归优化，最终 V3 在 18 例全量评测中达到 18/18 成功、0 超时 / 0 错误，P95 延迟 100.8s，平均延迟 59.2s，满足 P95 <= 180s、超时率 <= 5% 的全局 SLO。
- 相比 baseline，超时率由 22.2% 降至 0，平均延迟降低 33.6%，P95 降低 26.6%，平均 LLM 调用数从 5.7 降至 3.6，平均 token 从 16,095 降至 9,537，单次平均成本从 $0.003975 降至 $0.002088。
- 提供 CLI、Python API、FastAPI 接口和 React 本地调试界面，并在回答生成后执行医疗安全检查与就医提醒补充，降低危险建议、过度诊断和用药风险。

## 更精简的简历版本

**Medical-Agent-Swarm 医疗多智能体问答系统 | LLM Agent 性能优化**  
技术栈：Python、LangGraph、OpenAI SDK、FastAPI、Pydantic、React

- 基于 LangGraph 设计医疗多 Agent 协作流程，支持问诊信息收集、症状分析、风险评估、循证检索、结果综合与医疗安全检查。
- 构建 HealthBench-style 中文医疗 Agent 自动化评测集（18 个病例，6 类医疗场景），统计成功率、超时率、P95 延迟、LLM 调用、token 与单次成本等性能指标。
- 通过 EvidenceMemory 本地证据缓存、DeepResearch cache-first、检索迭代上限与单 Agent 快路径优化，减少实时检索和冗余 Agent 调度。
- V3 全量评测达到 18/18 成功、0 超时 / 0 错误，P95 100.8s，平均延迟 59.2s；相比 baseline，超时率 22.2% -> 0，平均 token 降低 40.7%，单次成本降低 47.5%，满足全局 SLO。

## 面试追问口径

- 如果被问“这个 18/18 是准确率吗”：不是医学诊断准确率，是性能回归中的请求成功率 / 无超时结果；医学效果需要独立 judge、专家标注或更大规模数据集验证。
- 如果被问“优化点是什么”：核心是把高频、稳定的指南 / 急症 / 症状证据前置到 EvidenceMemory，命中后减少 live web search 和多 Agent 拆解，让系统走更短、更可控的执行路径。
- 如果被问“为什么成本下降”：LLM 调用次数从 5.7 降到 3.6，平均 token 从 16,095 降到 9,537，减少了长链路检索、规划和综合带来的重复上下文消耗。
