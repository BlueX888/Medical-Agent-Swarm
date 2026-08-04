# Medical-Agent-Swarm 项目简历（Agent / 后端方向）

## 推荐投递版

**Medical-Agent-Swarm｜医疗多智能体问答系统**  
**角色：LLM Agent / Python 后端开发｜项目类型：个人项目 / 研究原型**  
**技术栈：Python、LangGraph、LangChain Core、OpenAI SDK、FastAPI、Pydantic、Redis、asyncio、LangSmith、Pytest**

**项目简介：** 面向医疗咨询、症状分诊和循证研究场景，构建具备动态规划、工具调用、会话记忆、安全防护和全链路追踪能力的多智能体问答系统。

- 基于 **LangGraph** 设计 Orchestrator–Worker 多 Agent 架构，由 Orchestrator 完成意图识别、任务拆解和 Worker 匹配，咨询、诊断、研究 3 类 Worker 负责具体执行；使用 Pydantic 定义强类型 `RoutePlan`，支持 `single / parallel / sequential` 三种依赖调度模式。
- 构建“**LLM 规划 + 确定性校验**”混合路由机制：运行时 Agent Catalog 按能力匹配 Worker，并校验任务预算、重复任务、依赖引用和循环依赖；对可恢复的错误规划自动重分配或修复，对异常链路执行安全 fallback，降低 LLM 非确定性带来的执行风险。
- 设计可发现、可扩展的医疗 **Skill / Function Calling** 体系，Worker 仅能自主调用白名单内的问诊信息采集、风险评估、症状分析、生活方式建议和深度研究工具，实现规划权与工具执行权分离，限制 Agent 越权调用。
- 建立医疗场景三层安全防线：规划前通过确定性规则识别急症信号，路由阶段强制高风险请求优先分诊，输出阶段由 `SafetyGuard` 检查危险用药建议、过度诊断、急症提示和免责声明，避免外部检索阻塞紧急就医建议。
- 搭建 **FastAPI 异步后端**，通过 `asyncio` 后台执行 Agent 任务，提供任务创建、状态与事件查询、Agent/Skill 元数据、会话管理和健康检查等 9 个 REST API；使用 Pydantic 统一 API 与 Agent 路由的数据契约。
- 基于 **Redis** 实现会话级短期记忆，使用事务 Pipeline 完成消息原子追加、定长裁剪和 TTL 刷新，默认保存最近 20 轮对话、24 小时过期；通过同会话异步锁保证请求顺序，并以健康检查和 HTTP 503 显式暴露存储故障。
- 建立覆盖 LangGraph 节点、Agent、LLM、Tool 和 SafetyGuard 的可观测链路，支持本地结构化事件追踪及可选 LangSmith Trace；对医疗文本和敏感字段进行脱敏，并使用 HMAC 生成不可逆会话标识。使用 Pytest 覆盖路由、依赖调度、Skill 权限、记忆、API 和可观测性，**96 项非集成测试全部通过**。

## 一页简历精简版

**Medical-Agent-Swarm｜医疗多智能体问答系统｜核心开发**  
Python、LangGraph、OpenAI SDK、FastAPI、Pydantic、Redis、asyncio、LangSmith

- 基于 LangGraph 实现 Orchestrator–Worker 多 Agent 工作流，通过强类型 `RoutePlan` 完成意图识别、任务拆解、Worker 能力匹配及单 Agent / 并行 / 串行依赖调度。
- 设计 Agent Catalog、Skill 自动发现与工具白名单机制，使 Worker 通过 Function Calling 自主选择医疗工具；增加任务预算、重复任务、循环依赖和能力匹配校验，对异常规划自动修复或安全降级。
- 建立“急症规则预检 + 高风险优先分诊 + 输出 SafetyGuard”三层防护，降低越权工具调用、危险用药建议和过度诊断风险。
- 使用 FastAPI、asyncio 和 Redis Pipeline 实现异步任务 API、最近 20 轮会话记忆、24 小时 TTL、同会话串行化与故障感知；接入脱敏链路追踪，96 项非集成自动化测试全部通过。

## 极简版

**Medical-Agent-Swarm｜医疗多智能体问答系统**

- 基于 Python、LangGraph 和 OpenAI Function Calling 构建 Orchestrator–Worker 多 Agent 系统，实现医疗咨询、症状分诊和循证研究任务的动态路由与依赖调度。
- 设计强类型规划、Agent Catalog、Skill 白名单、异常规划修复及医疗安全 Guardrail，提升 Agent 工具调用的可控性、可扩展性和可测试性。
- 基于 FastAPI、asyncio、Redis 和 LangSmith 完成异步服务、会话记忆、故障降级及脱敏追踪，96 项非集成测试全部通过。

## 面试表述口径

- **为什么采用多 Agent：** 将“任务规划”和“工具执行”拆开，Orchestrator 只决定做什么、交给谁和依赖顺序，Worker 仅在授权 Skill 内执行，便于独立校验、测试和扩展。
- **如何处理 LLM 错误规划：** 先用 Pydantic 约束结构，再用确定性策略检查 Agent、能力、预算和任务 DAG；可修复问题自动纠正，无法修复时进入安全 fallback。
- **如何实现并发调度：** 执行器按任务依赖分波次调度，不同 Worker 的就绪任务通过 `asyncio.gather` 并行，同一有状态 Worker 或存在前置依赖时串行执行。
- **Redis 记忆如何保证一致性：** 使用事务 Pipeline 原子追加、裁剪并刷新 TTL；在单服务事件循环内按 `session_id` 加锁，避免同一会话的并发请求打乱上下文。
- **测试指标如何解释：** “96 项测试通过”代表工程回归结果，不代表医学诊断准确率；项目定位为技术研究原型，不能替代医生诊疗。
