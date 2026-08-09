# Medical-Agent-Swarm 简历项目经历

## 推荐投递版

**Medical-Agent-Swarm 医疗多智能体问答系统｜LLM Agent / Python 后端开发**

**项目类型：** 个人项目 / 研究原型

**技术栈：** Python、LangGraph、LangChain Core、OpenAI SDK、FastAPI、Pydantic、Redis、asyncio、LangSmith、React、TypeScript、Pytest

- 基于 LangGraph 设计 Orchestrator–Worker 多智能体架构，将医疗咨询拆分为通用咨询、症状分诊和循证研究任务；通过强类型 `RoutePlan` 描述意图、风险、任务依赖和 Worker 能力，并按依赖关系动态选择单 Agent、并行或串行执行。
- 构建确定性 Agent 路由与安全控制层，基于运行时 Agent Catalog 校验能力匹配、任务预算、重复任务及循环依赖，对非法规划进行自动修复或安全降级；急症场景强制优先执行风险分诊，避免外部检索阻塞紧急提示。
- 设计可发现、可白名单隔离的医疗 Skill 体系，使 Worker 通过 Function Calling 自主选择问诊信息采集、风险评估、症状分析、生活方式建议和深度研究等工具，同时限制越权调用，降低 LLM 规划和工具执行的不可控性。
- 基于 Redis 实现会话级短期记忆，使用事务 Pipeline 完成原子追加、定长裁剪和 TTL 刷新，默认保留最近 10 轮、24 小时过期；增加同会话串行化、健康检查和故障显式降级，后端异常时通过 HTTP 503 暴露真实状态。
- 建立覆盖 LangGraph、Agent、LLM、Tool 和 SafetyGuard 的可观测链路，支持本地事件追踪与可选 LangSmith Trace；默认脱敏医疗文本和敏感字段，并通过 HMAC 生成不可逆会话标识，兼顾故障定位与隐私保护。
- 搭建 FastAPI 异步服务及 React/TypeScript 调试端与用户端，提供任务创建、状态与事件查询、Agent/Skill 元数据、会话管理和健康检查接口；使用 Pytest 覆盖路由、安全、记忆、API 和可观测性，当前非集成测试 **96 项全部通过**。

## 一页简历精简版

**Medical-Agent-Swarm 医疗多智能体问答系统｜核心开发**

Python、LangGraph、OpenAI SDK、FastAPI、Pydantic、Redis、LangSmith、React

- 基于 LangGraph 实现 Orchestrator–Worker 多 Agent 工作流，由咨询、症状分析和循证研究 Worker 协作完成医疗问答，支持强类型规划及 single / parallel / sequential 依赖调度。
- 设计运行时 Agent Catalog、Skill 自动发现与白名单机制，通过 Function Calling 实现 Worker 自主选工具，并校验能力匹配、任务预算、重复任务及依赖关系，对异常规划自动修复或安全降级。
- 建立“确定性急症预检 + 执行期安全约束 + 输出 SafetyGuard”三层防护，高风险请求强制优先分诊，拦截危险用药建议、过度诊断和缺失急症提示。
- 使用 Redis Pipeline 实现最近 10 轮会话记忆、24 小时 TTL、同会话串行化和故障感知；接入 FastAPI、React 调试界面及 LangSmith 脱敏追踪，96 项非集成自动化测试全部通过。

## 极简版（版面有限时）

**Medical-Agent-Swarm｜医疗多智能体问答系统**

- 基于 Python、LangGraph 和 OpenAI Function Calling 构建 Orchestrator–Worker 医疗 Agent 系统，实现咨询、症状分诊与循证研究任务的动态路由和依赖调度。
- 设计 Agent Catalog、Skill 白名单、规划校验及安全降级机制，并通过急症预检与 SafetyGuard 降低越权工具调用、过度诊断和危险用药建议风险。
- 使用 Redis、FastAPI、LangSmith 和 React 完成会话记忆、异步 API、隐私脱敏追踪及调试界面；96 项非集成测试全部通过。

## 完整简历模板

> 下列个人信息、教育经历和时间需要按实际情况补充。

### 基本信息

**姓名：** [姓名]

**手机：** [手机号] ｜ **邮箱：** [邮箱]

**GitHub：** [项目或个人主页链接]

**求职方向：** LLM Agent 工程师 / AI 应用后端工程师

### 个人优势

- 熟悉 Python 异步编程、FastAPI 服务开发及 Redis 会话状态管理，能够独立完成 LLM 应用从工作流编排到 API 和前端调试工具的端到端落地。
- 掌握 LangGraph 多 Agent 编排、Function Calling、结构化输出、工具权限隔离和确定性 Guardrail，关注 Agent 系统的可控性、可测试性与故障降级。
- 具备医疗 AI 安全意识，能够围绕急症分诊、危险建议拦截、隐私脱敏和可观测性设计工程防线。

### 专业技能

- **语言与后端：** Python、asyncio、FastAPI、Pydantic、RESTful API
- **LLM 与 Agent：** LangGraph、LangChain Core、OpenAI SDK、Function Calling、结构化输出、Prompt Engineering
- **数据与基础设施：** Redis、Docker Compose、Mem0（可选长期记忆）
- **可观测与测试：** LangSmith、结构化事件追踪、Pytest、pytest-asyncio
- **前端：** React、TypeScript、Vite

### 项目经历

使用上方“推荐投递版”或“一页简历精简版”。

### 教育经历

**[学校名称]｜[专业]｜[学历]**　[开始时间] – [结束时间]

- 可填写与人工智能、软件工程、数据结构、数据库、计算机网络相关的课程或成果。

## 面试追问参考

### 1. 为什么需要 Orchestrator–Worker，而不是单 Agent？

单 Agent 容易把意图识别、工具选择、医学分析和最终回答混在一个不可控循环中。本项目让 Orchestrator 只负责“需要完成什么、交给谁以及任务依赖”，Worker 再在白名单内自主选择 Skill，从而分离规划权和工具执行权，便于校验、测试和扩展。

### 2. 如何控制 LLM 生成错误规划？

规划结果先通过 Pydantic 强类型解析，再由确定性策略检查 Agent ID、能力匹配、任务数量、重复任务、依赖引用、循环依赖及高风险分诊要求。可修复问题重新分配或调整依赖，无法修复时进入低置信度安全 fallback，不直接执行非法计划。

### 3. 并行执行如何避免状态冲突？

执行器按依赖关系分波次调度；不同 Worker 的就绪任务可以通过 `asyncio.gather` 并行，同一有状态 Worker 每个波次最多执行一个任务。存在依赖或任务落在同一 Worker 时自动使用串行模式。

### 4. Redis 会话记忆解决了什么问题？

它使同一会话在应用重启或多进程部署后仍能读取最近上下文。每轮对话以 Redis 事务 Pipeline 原子追加，同时裁剪长度并刷新 TTL；同一事件循环内对相同 `session_id` 加锁，避免并发请求造成上下文乱序。

### 5. 医疗安全如何实现？

系统在规划前使用确定性规则识别急症信号，在路由阶段强制加入或提前风险分诊，最终回答还必须经过 SafetyGuard，检查急症提示、危险用药建议、过度诊断和免责声明。该项目仍定位为研究原型，不能替代医生诊疗。

### 6. 为什么强调可观测性脱敏？

医疗问题可能包含症状、病史和个人标识。项目默认只记录低基数元数据、调用耗时和工具结果摘要，医疗文本与敏感字段在发送到 LangSmith 前会被替换；会话 ID 仅在配置密钥后生成 HMAC 引用，避免导出原始标识。

## 指标口径说明

- “96 项测试全部通过”来自当前仓库执行 `python -m pytest -q -m "not integration"` 的结果：`96 passed, 1 deselected`。
- 不要将请求成功率、无超时率或安全规则命中率表述为“医学准确率”或“诊断准确率”。
- 在没有固定模型、硬件、并发量、重复次数和可复现报告前，不建议在简历中填写延迟、Token 或成本优化百分比。
