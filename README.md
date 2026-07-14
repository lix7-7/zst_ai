# 智扫通 · 扫地机器人智能客服 Agent

> 基于 LangChain 1.x ReAct Agent + 混合 RAG 检索 + SSE 流式输出的垂直领域智能客服系统。
> 覆盖售前咨询、故障排查、环境适配建议等核心客服场景。

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python">
  <img src="https://img.shields.io/badge/LangChain-1.x-orange" alt="LangChain">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-green" alt="FastAPI">
  <img src="https://img.shields.io/badge/LLM-通义千问%20qwen3.7--max-red" alt="LLM">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
</p>

---

## 目录

- [项目背景](#项目背景)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [使用示例](#使用示例)
- [项目结构](#项目结构)
- [配置说明](#配置说明)
- [技术栈](#技术栈)

---

## 项目背景

扫地机器人品类存在「选购决策复杂 + 售后问题专业 + 用户数据难以洞察」三大客服痛点。本项目以 ReAct Agent 为核心，结合垂直知识库 RAG 与用户使用记录检索，构建一个能「思考—行动—观察」的智能客服：

- **售前咨询**：基于私有知识库（产品手册、100 问、故障排除等）的精准 RAG 问答
- **场景适配**：结合用户所在城市实时天气（高德 API），给出适配的机器人使用建议
- **记忆增强**：双层记忆系统（Redis 短期 + Chroma 长期），跨会话上下文召回

---

## 核心特性

### 1. ReAct Agent + 4 工具编排

基于 LangChain 1.x `create_agent` API，Agent 自主决策调用 **4 个工具**：

| 工具 | 类型 | 说明 |
|------|------|------|
| `rag_summarize` | RAG 检索 | 从知识库检索文档并生成总结回答 |
| `get_weather` | 真实 API | 高德天气 API，获取指定城市实时天气 |
| `get_user_location` | 真实 API | 高德 IP 定位，获取用户所在城市 |
| `memory_recall` | 记忆检索 | 从长期记忆库语义检索历史对话 |

### 2. Middleware 中间件机制

4 个中间件解耦横切关注点，按执行顺序：

| # | 中间件 | 装饰器 | 作用 |
|---|--------|--------|------|
| 1 | `monitor_tool` | `@wrap_tool_call` | 工具调用日志/异常捕获/运行时上下文注入 |
| 2 | `log_before_model` | `@before_model` | 每次 LLM 调用前打印消息摘要 |
| 3 | `memory_inject` | `@before_model` | 注入 Redis 短期 + Chroma 长期记忆到上下文 |
| 4 | `token_guard` | `@before_model` | tiktoken 估算 → 超限裁剪 → LLM 摘要压缩 |

### 3. 混合 RAG 检索管道

```
用户查询
  → BM25 关键词检索 (jieba 分词, top 20)
  → Dense Vector 语义检索 (Chroma, top 20)
  → RRF 融合 (Reciprocal Rank Fusion, k=60)
  → Cross-Encoder Reranker 精排 (qwen3-rerank)
  → Top-5 文档 → LLM 总结回答
```

- **MD5 去重**：4KB 流式分块计算，避免重复入库
- **中文分片**：RecursiveCharacterTextSplitter，按中文标点智能切分
- **评测体系**：30 条标注测试集 × 4 路检索器对比 × 多维度指标（Hit Rate@k、MRR、关键词覆盖率）+ RAGAS 生成质量评测

### 4. SSE 逐 Token 流式输出

- 基于 LangGraph `stream_mode="messages"` + `ChatTongyi(streaming=True)` 实现 token 级流式
- FastAPI `StreamingResponse(media_type="text/event-stream")` 推送
- 前端 `EventSource` 消费 + `marked.js` 实时 Markdown 渲染
- 工具调用/结果以 JSON 事件形式嵌入 SSE 流

### 5. 双层记忆系统 + Token 窗口管理

| 层级 | 存储 | 说明 |
|------|------|------|
| 短期记忆 | Redis List | 滑动窗口（LPUSH + LTRIM），每会话 10 轮，TTL 24h |
| 长期记忆 | Chroma 向量库 | 对话摘要 → 语义检索，跨会话记忆召回 |
| Token 守卫 | tiktoken / 启发式 | 超限自动裁剪最早消息 + LLM 摘要压缩 |

### 6. 工程化设计

- **FastAPI 服务化**：lifespan 管理单例，RESTful API + SSE StreamingResponse，自动生成 `/docs`
- **localStorage 会话管理**：前端本地存储会话元数据，Redis 后端持久化消息历史，支持会话切换与历史恢复
- **配置驱动**：5 个 YAML 配置文件管理所有参数
- **工厂模式**：`BaseModelFactory` 抽象基类封装模型实例化
- **优雅降级**：Redis 不可用 → 记忆返回空；Reranker 失败 → 回退 RRF；记忆保存失败 → 不影响响应
- **统一日志**：按日期分文件，DEBUG 到文件 / INFO 到控制台

---

## 系统架构

```
用户浏览器 (index.html)
  │  SSE EventSource
  ▼
FastAPI (api/main.py)
  ├── POST /api/v1/chat/stream     ← SSE 流式对话
  ├── GET  /api/v1/chat/sessions    ← 用户会话列表
  ├── GET  /api/v1/chat/sessions/{id}/messages
  ├── DELETE /api/v1/chat/sessions/{id}
  ├── GET  /api/v1/knowledge/stats
  ├── POST /api/v1/knowledge/reindex
  └── GET  /api/v1/health
  │
  ▼
ReactAgent (agent/react_agent.py)
  │  create_agent(model, tools, middleware)
  │  astream(stream_mode="messages")
  ├── 4 Tools ──── rag / weather / location / memory
  ├── 4 Middleware ──── monitor / log / memory / token guard
  └── Memory System ──── Redis (short-term) + Chroma (long-term)
```

---

## 快速开始

### 环境要求

- Python ≥ 3.11
- 阿里云 DashScope API Key（通义千问 + Embeddings + Reranker）
- Redis（可选，用于多轮对话记忆）
- Docker Desktop（Windows 用户运行 Redis 需要）

### 1. 克隆项目

```bash
git clone https://github.com/lix7-7/zst_ai.git
cd zst_ai
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
# Windows PowerShell
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxx"
# 高德地图 API Key（可选，未设置时天气/定位功能降级）
$env:AMAP_API_KEY = "你的高德Key"

# macOS / Linux
export DASHSCOPE_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
export AMAP_API_KEY="你的高德Key"
```

### 4. 启动 Redis（可选）

```bash
# Windows / macOS / Linux（Docker）
docker run -d --name redis-zst -p 6379:6379 redis:7-alpine

# 之后启动/停止
docker start redis-zst
docker stop redis-zst
```

> Redis 不可用时系统自动降级，对话功能不受影响，仅记忆系统不可用。

### 5. 初始化知识库

将知识库文档（PDF/TXT）放入 `data/` 目录后：

```bash
python -m rag.vector_store
```

### 6. 启动服务

```bash
# 主服务 — FastAPI + 聊天界面 (http://localhost:8000)
uvicorn api.main:app --reload --port 8000
```

浏览器访问 `http://localhost:8000` 即可开始对话。API 文档在 `http://localhost:8000/docs`。

---

## 使用示例

### 场景一：售前 RAG 问答

```
用户：小户型适合哪些扫地机器人？
Agent：[调用 rag_summarize("小户型 扫地机器人 选购")]
       → 混合检索 → Reranker 精排 → LLM 总结
       → 流式输出选购建议...
```

### 场景二：环境适配咨询（多工具编排）

```
用户：深圳今天适合用扫地机器人吗？
Agent：[思考] 需要获取位置和天气
       [调用 get_user_location] → "深圳"
       [调用 get_weather("深圳")] → 晴天 28°C 湿度65%
       [观察] 信息充足 → 流式输出综合建议...
```

---

## 项目结构

```
zst_ai/
├── api/                            # FastAPI 服务
│   ├── main.py                     # 应用入口 + lifespan
│   ├── routers/
│   │   ├── chat.py                 # SSE 对话 + 会话管理 API
│   │   └── knowledge.py            # 知识库管理 API
│   ├── schemas/chat.py             # Pydantic 模型
│   └── static/index.html           # 聊天界面 (vanilla JS)
├── agent/                          # Agent 编排层
│   ├── react_agent.py              # ReactAgent 封装
│   └── tools/
│       ├── agent_tools.py          # 4 个工具定义
│       └── middleware.py           # 4 个中间件
├── rag/                            # RAG 检索层
│   ├── hybrid_retriever.py         # BM25 + Dense + RRF + Reranker
│   ├── rag_service.py              # 检索 + LLM 总结 Chain
│   └── vector_store.py             # Chroma 向量库 + 文档加载
├── memory/                         # 记忆系统
│   ├── short_term.py               # Redis 滑动窗口
│   ├── long_term.py                # Chroma 对话摘要检索
│   └── manager.py                  # 统一入口，双存储协调
├── model/
│   └── factory.py                  # 模型工厂 (ChatTongyi / Embeddings)
├── utils/                          # 基础设施
│   ├── config_handler.py           # YAML 配置加载
│   ├── prompt_loader.py            # 提示词文件加载
│   ├── token_counter.py            # tiktoken 估算 + 启发式 fallback
│   ├── file_handler.py             # 文件 MD5 / PDF / TXT 加载
│   ├── logger_handler.py           # 日志器
│   └── path_tool.py                # 绝对路径管理
├── eval/                           # RAG 评测
│   ├── evaluate.py                 # 4 路检索器对比评测
│   ├── test_queries.json           # 30 条标注测试集
│   └── eval_result.json            # 最新评测结果
├── config/                         # YAML 配置
│   ├── rag.yml                     # 模型名称
│   ├── chroma.yml                  # 向量库 + 混合检索参数
│   ├── prompts.yml                 # 提示词路径
│   ├── agent.yml                   # Agent 业务配置
│   └── memory.yml                  # Redis + Token 窗口配置
├── prompts/                        # 提示词模板
│   ├── main_prompt.txt             # 客服主提示词
│   └── rag_summarize.txt           # RAG 总结提示词
├── data/                           # 知识库文档
└── requirements.txt
```

---

## 配置说明

| 配置文件 | 主要内容 |
|----------|---------|
| `config/rag.yml` | LLM 模型名、Embedding 模型名 |
| `config/chroma.yml` | 分片策略、检索 top-k、混合检索参数（BM25/Dense/RRF/Reranker） |
| `config/prompts.yml` | 2 套提示词文件路径 |
| `config/agent.yml` | Agent 业务配置 |
| `config/memory.yml` | Redis 连接、窗口大小、Token 预算、摘要触发阈值 |

所有配置通过 `utils/config_handler.py` 加载为模块级单例，运行时零 IO。

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| Agent 框架 | LangChain 1.x + LangGraph | `create_agent` + middleware 装饰器 API |
| LLM | 通义千问 qwen3.7-max | DashScope API，原生支持工具调用 |
| Embeddings | text-embedding-v4 | 阿里云 DashScope，中文语义检索 |
| Reranker | qwen3-rerank | Cross-Encoder 交叉编码器精排 |
| 向量库 | Chroma | 嵌入式部署，持久化存储 |
| 关键词检索 | BM25 (rank_bm25) | jieba 中文分词 |
| API 框架 | FastAPI | SSE StreamingResponse + RESTful |
| 缓存 | Redis | 短期记忆滑动窗口 |
| 前端 | Vanilla JS | EventSource SSE + marked.js 渲染 |
| Token 估算 | tiktoken (cl100k_base) | 主方案 + 字符启发式 fallback |
| 日志 | Python logging | 按日期分文件，双输出 |
| 配置 | YAML | 5 文件配置驱动 |

## License

MIT
