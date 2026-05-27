# Medical-Agent-Swarm

基于 Skills-Agent 两层架构的多智能体医疗助手系统。通过 Agent Loop、Agent Swarm、RAG 知识检索和记忆管理，提供症状分析、风险评估、鉴别诊断、临床指南检索等医疗咨询服务。

## 架构概览

```
用户问题 → SwarmCoordinator（智能路由）
              │
        ┌─────┴──────┐
        ↓            ↓
    单 Agent      LeadAgent（任务分解）
        │            │
        │     ┌──────┼──────┐
        │     ↓      ↓      ↓
        │  Consult Diag  Research    ← 3 个专业 Agent，并行执行
        │     │      │      │
        │     └──────┼──────┘
        │            ↓
        │     LeadAgent（结果综合）
        ↓            ↓
         最终回答 + 记忆持久化
```

**核心机制：**

- **Skills-Agent 两层架构** — 8 个原子 Skill 自动发现、动态加载，直接转换为 OpenAI Function Calling 格式，Agent 自主选择调用
- **Agent Loop** — Think-Act-Observe 循环，LLM 驱动技能调用与自主推理
- **Agent Swarm** — 去中心化群体协作，SharedContext 共享黑板通信，asyncio.gather 并行执行
- **RAG 知识检索** — Milvus Lite 向量库 + BAAI/bge-small-zh-v1.5 中文嵌入，8 份医学文档语义检索
- **多层记忆** — 短期（会话级/Redis）+ 长期（Mem0 跨会话向量检索）
- **Harness Engineering** — YAML 声明式约束 + 运行时校验 + 输出自动修复（免责声明/高危预警）
- **自进化技能** — 从成功会话中提取工作流模式，语义匹配注入，反馈驱动定向进化

## Skills 与 Agent

### 8 个原子 Skill

| Skill | 功能 | 数据源 |
|-------|------|--------|
| `search_knowledge` | 医学知识语义检索 | Milvus |
| `assess_risk` | 症状风险等级评估（低/中/高/紧急） | 规则引擎 + Milvus |
| `analyze_symptoms` | 症状模式分析与疾病关联 | 规则引擎 + Milvus |
| `recommend_lifestyle` | 饮食、运动、用药生活方式建议 | Milvus |
| `clinical_guideline` | 临床实践指南检索 | Milvus |
| `deep_research` | 深度研究（网络搜索 + 多源证据综合） | DuckDuckGo + Milvus + LLM |
| `search_history` | 当前会话对话历史搜索 | 短期记忆 |
| `search_similar_cases` | 跨会话相似病例检索 | Mem0 长期记忆 |

Skill 通过目录扫描 + YAML 元数据解析 + `importlib` 动态加载自动注册，新增 Skill 只需添加目录和脚本，无需修改 Agent 代码。

### 5 个 Agent

| Agent | 角色 | 擅长场景 |
|-------|------|----------|
| **ConsultationAgent** | 健康咨询 | 通用健康建议、生活方式指导、风险预判 |
| **DiagnosticAgent** | 症状诊断 | 症状分析、鉴别诊断（VINDICATE 框架）、多系统分析 |
| **ResearchAgent** | 医学研究 | 循证医学、临床指南、最新文献检索 |
| **LeadAgent** | 任务分解与综合 | 将复杂问题拆分为子任务，综合多 Agent 贡献 |
| **SwarmCoordinator** | 智能路由 | 简单问题 → 单 Agent；复杂问题 → Swarm 并行协作 |

3 个专业 Agent 共享全部 8 个 Skill，由 LLM 根据任务自主决定调用哪些。

## 记忆系统

```
┌──────────────────────────────────────┐
│  短期记忆（会话级，内存 / Redis）      │
│  当前对话历史，支持多轮追问上下文      │
└──────────────┬───────────────────────┘
               ↕ 会话结束时自动持久化
┌──────────────┴───────────────────────┐
│  长期记忆（跨会话，Mem0 云服务）       │
│  会话摘要向量化存储，相似病例检索      │
└──────────────────────────────────────┘
```

- **会话摘要**：Swarm 协作结束后自动生成 Markdown 报告（参与 Agent、关键发现、性能指标）
- **Agent 身份**：每个 Agent 维护 IDENTITY.md，记录协作历史和工具使用统计
- **优雅降级**：Mem0 / Redis 不可用时自动回退到内存存储，系统持续运行

## Harness Engineering

以"人类设计约束，AI 代理执行"为理念，保障医疗场景安全合规：

| 机制 | 实现 | 说明 |
|------|------|------|
| 约束定义 | `constraints/*.yaml` | 声明式定义 Agent 能力边界和禁止行为 |
| 运行时校验 | `constraints/validator.py` | 在 Agent Loop 中校验工具调用和输出内容 |
| 输出自动修复 | `validation/auto_fixer.py` | 缺免责声明自动补充；高危症状自动插入急救预警；确诊性表述自动软化 |

约束系统以非侵入方式注入 Agent Loop，不修改现有 Agent 代码，模块缺失时自动跳过。

## 快速开始

```bash
# 1. 创建环境
conda create -n medical-agent-swarm python=3.12 -y && conda activate medical-agent-swarm

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API（在项目根目录创建 config.py）
cat > config.py << 'EOF'
LLM_CONFIG = {
    "api_key": "your-api-key",
    "model_name": "your-model",
    "base_url": "https://api.openai.com/v1",
    "temperature": 0.7,
    "max_tokens": 8192,
}
MEM0_CONFIG = {"api_key": "m0-your-mem0-key"}  # 可选，https://app.mem0.ai
REDIS_CONFIG = {"host": "localhost", "port": 6379, "db": 0}  # 可选
EOF

# 4. 初始化知识库
python knowledge/scripts/import_hardcoded_data.py

# 5. 运行
python main.py

# 6. 运行测试（26 个端到端集成测试）
python examples/test_all.py
```

## 项目结构

```
medical-agent-swarm/
├── .claude/skills/          # 8 个原子 Skill（自动发现 + 动态加载）
│   ├── search-knowledge/    #   医学知识检索
│   ├── assess-risk/         #   风险等级评估
│   ├── analyze-symptoms/    #   症状模式分析
│   ├── recommend-lifestyle/ #   生活方式建议
│   ├── clinical-guideline/  #   临床指南检索
│   ├── deep-research/       #   深度研究工作流
│   ├── search-history/      #   会话历史搜索
│   └── search-similar-cases/#   相似病例检索
├── agents/                  # Agent 实现
│   ├── base_agent.py        #   Agent 基类（Agent Loop + Skill 注册）
│   ├── consultation_agent.py#   健康咨询 Agent
│   ├── diagnostic_agent.py  #   症状诊断 Agent
│   ├── research_agent.py    #   医学研究 Agent
│   └── skill_registry_mixin.py # Skill 自动注册混入
├── core/                    # 核心引擎
│   ├── agent_loop.py        #   Agent Loop（Think-Act-Observe + 约束校验）
│   ├── llm_client.py        #   LLM 客户端（OpenAI-compatible）
│   ├── skill_loader.py      #   Skill 动态加载器
│   └── skill_registry.py    #   Skill 注册表 → OpenAI Function Calling 格式
├── swarm/                   # Swarm 协作
│   ├── swarm_coordinator.py #   智能路由（单 Agent / Swarm）
│   ├── lead_agent.py        #   任务分解 + 结果综合
│   ├── shared_context.py    #   共享黑板（信息素通信）
│   └── events.py            #   事件驱动审计
├── memory/                  # 记忆管理
│   ├── short_term.py        #   短期记忆（内存 / Redis）
│   ├── long_term.py         #   长期记忆（Mem0）
│   ├── session_summary.py   #   会话摘要生成
│   ├── agent_identity.py    #   Agent 身份持久化
│   └── evolved_skills/      #   自进化技能子系统
├── constraints/             # 约束系统
│   ├── agent_constraints.yaml  # Agent 能力边界定义
│   ├── swarm_constraints.yaml  # Swarm 协作规则
│   └── validator.py         #   运行时约束校验器
├── validation/              # 输出校验
│   └── auto_fixer.py        #   自动修复器（免责声明 / 高危预警）
├── knowledge/               # 知识库
│   ├── milvus_kb.py         #   Milvus 向量库封装（单例）
│   ├── data/documents/      #   8 份医学文档（生活方式 / 临床指南）
│   └── scripts/             #   数据导入脚本
├── research/                # 深度研究
│   ├── deep_research_workflow.py # 多步研究工作流
│   ├── evidence_synthesizer.py   # 多源证据综合
│   ├── web_search.py        #   DuckDuckGo 搜索
│   └── knowledge_base.py    #   Qdrant 知识库（可选）
├── config.py                # API 配置（LLM / Mem0 / Redis）
├── examples/test_all.py     # 26 个集成测试（8 个阶段全覆盖）
└── main.py                  # 交互式 CLI 入口
```

## 技术栈

| 类别 | 技术 |
|------|------|
| LLM 调用 | OpenAI-compatible API（Function Calling） |
| 向量数据库 | Milvus Lite / Qdrant |
| 嵌入模型 | BAAI/bge-small-zh-v1.5（中文，512 维） |
| 长期记忆 | Mem0 云服务 |
| 短期记忆 | 内存（默认）/ Redis |
| 网络搜索 | DuckDuckGo Search |
| 异步框架 | asyncio / httpx |
| 数据校验 | Pydantic v2 |
| 日志 | Loguru |

## 免责声明

本系统仅供学习和研究使用，不能替代专业医生的诊断和治疗。如有健康问题请及时就医。

## 许可证

MIT License
