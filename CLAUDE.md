# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ZhiSaoTong (智扫通)** — a FastAPI + SSE streaming intelligent customer service agent for robot vacuum cleaners. Uses LangChain 1.x ReAct Agent + RAG (BM25 + Dense + RRF + Cross-Encoder Reranker) over a private knowledge base to handle pre-sales Q&A, environment-aware recommendations, and multi-turn conversations with memory.

## Commands

```bash
# Requires Python >= 3.11

# Install dependencies
pip install -r requirements.txt

# Initialize/re-index the Chroma vector store from data/ (first run, or when documents change)
python -m rag.vector_store

# Start the FastAPI server (primary — SSE streaming API + chat UI at http://localhost:8000)
uvicorn api.main:app --reload --port 8000

# Test hybrid search independently
python -m rag.hybrid_retriever

# Run RAG evaluation (4-way retriever comparison + RAGAS)
python -m eval.evaluate

# Re-index knowledge base via API
curl -X POST http://localhost:8000/api/v1/knowledge/reindex

# Redis (Docker) — start if not running
# Data persisted via AOF in container, mount to ./redis-data for host access
docker start redis-zst
```

Requires `DASHSCOPE_API_KEY` (Alibaba Cloud DashScope — LLM + Embeddings + Reranker).  
Requires `AMAP_API_KEY` (Alibaba Cloud Amap — IP geolocation + weather; default fallback key included).  
Optional: Redis on `localhost:6379` for multi-turn memory (gracefully degrades if unavailable).  
FastAPI docs at `http://localhost:8000/docs`.

## Architecture (v2.0)

```
FastAPI (api/main.py) — lifespan-managed Agent + Memory singletons
  ├── api/routers/chat.py      — POST /api/v1/chat/stream (SSE), session CRUD
  ├── api/routers/knowledge.py — GET /stats, POST /reindex
  ├── api/schemas/chat.py      — Pydantic request/response models
  └── api/static/index.html    — vanilla JS chat UI (SSE EventSource, localStorage sessions, marked.js)
  └── agent/react_agent.py     — ReactAgent wrapping langchain create_agent
       ├── agent/tools/agent_tools.py   — 4 @tool functions
       ├── agent/tools/middleware.py    — 4 middleware hooks
       ├── rag/                         — hybrid retrieval pipeline
       ├── memory/                      — short-term (Redis) + long-term (Chroma) memory
       ├── model/factory.py             — ChatTongyi + DashScopeEmbeddings (factory pattern)
       └── utils/                       — config, prompts, logging, paths, token_counter
  └── eval/
       ├── test_queries.json (30 annotated queries × 4 categories)
       ├── evaluate.py        — 4-way retriever comparison
       └── eval_result.json   — latest run output
```

### SSE streaming pipeline (critical path)

```
User sends query → event_generator() → agent.execute_stream_async()
  → LangGraph astream(stream_mode="messages")
    → AIMessageChunk(content="字")  → format_sse({"type":"content","content":"字"})
    → AIMessageChunk(tool_calls=[]) → format_sse({"type":"tool_call",...})
    → ToolMessage                  → format_sse({"type":"tool_result",...})
  → StreamingResponse(media_type="text/event-stream")
  → Browser EventSource accumulates tokens → marked.parse() renders markdown
```

**Two things required for true token-by-token streaming:**
1. `ChatTongyi(streaming=True)` in `model/factory.py` — without this, LangChain falls back to `ainvoke()` producing a single chunk
2. `stream_mode="messages"` on `agent.astream()` — `"values"` yields entire state per graph node; `"messages"` yields `(AIMessageChunk, metadata)` tuples

### The 4 tools

`rag_summarize`, `get_weather`, `get_user_location`, `memory_recall`.

Real API: `get_weather` + `get_user_location` use Amap APIs. `rag_summarize` performs hybrid RAG retrieval + LLM summarization. `memory_recall` semantically retrieves relevant conversation history from long-term memory.

### The 4 middleware (LangChain 1.x decorator-based, ordered by execution)

| # | Middleware | Decorator | Purpose |
|---|-----------|-----------|---------|
| 1 | `monitor_tool` | `@wrap_tool_call` | Log tool calls/errors |
| 2 | `log_before_model` | `@before_model` | Log message count + last message before each LLM call |
| 3 | `memory_inject` | `@before_model` | Fetches short-term (Redis) + long-term (Chroma) memory, injects into system prompt. Uses `<!--memory_injected-->` guard to prevent repeated injection during ReAct loops |
| 4 | `token_guard` | `@before_model` | Estimates tokens via tiktoken (heuristic fallback), trims oldest history when over budget, triggers LLM summarization of trimmed messages. Uses `<!--token_guarded-->` guard |

**Middleware ordering matters:** `memory_inject` must run before `token_guard` — injection increases token count, so the guard must see the final system prompt.

### Hybrid RAG pipeline (`rag/`)

Documents → MD5 dedup (4KB streaming chunks) → Chinese-aware chunking (RecursiveCharacterTextSplitter) → Chroma DB.
Retrieval: BM25 (jieba tokenization) + Dense Vector (Chroma) → RRF fusion (k=60) → top-N candidates → Cross-Encoder Reranker (qwen3-rerank) → final top-k → LLM summarization.
Config in `config/chroma.yml` → `hybrid_search` section.

### Memory system (`memory/`)

- **ShortTermMemory** — Redis List per session (`session:{id}:messages`), sliding window (LPUSH + LTRIM). Gracefully degrades to empty when Redis unavailable.
- **LongTermMemory** — conversation summaries stored in Chroma (`long_term_memory` collection), semantically retrieved for context. Simple rule-based summarization; designed for future LLM-based summarization.
- **MemoryManager** — singleton (`get_instance()`), initializes both stores, provides `add_interaction()` / `get_context()` / `clear()`. Auto-archives old short-term messages to long-term when threshold exceeded.
- **Token window management** — `utils/token_counter.py` provides tiktoken-based estimation (cl100k_base, Chinese ~1.5 token/char) with heuristic fallback (±30%). `token_guard` middleware enforces budget via history trimming + LLM summarization. Config in `config/memory.yml` → `token_window`.

### Session management

Two-layer architecture:
- **Backend**: REST API (`GET/DELETE /api/v1/chat/sessions`, `GET /sessions/{id}/messages`) scans Redis keys via SCAN (non-blocking), returns session metadata + message history
- **Frontend**: `localStorage` stores session metadata (id, preview, lastActive, messageCount). Survives page refresh. `initApp()` restores last active session. Session switching fetches history from backend API.

## Key Patterns

- **`__init__.py` usage** — `api/`, `memory/`, `eval/` use standard package structure. `agent/`, `rag/`, `model/`, `utils/` omit them; imports work because all code runs from the project root.
- **Absolute path resolution** — `utils/path_tool.py` derives project root from its own location (`__file__` → up 2 levels). Always use `get_abs_path()` for file I/O.
- **YAML-driven config** — `utils/config_handler.py` loads 5 config files as module-level singletons (`rag_conf`, `chroma_conf`, `prompts_conf`, `agent_conf`, `memory_conf`). Model names, chunk settings, prompt paths, data paths, and memory parameters all live in `config/`.
- **Model factory singletons** — `model/factory.py`: `ChatModelFactory` and `EmbeddingsFactory` produce module-level `chat_model` and `embed_model` via factory pattern. `chat_model` has `streaming=True` (critical for SSE).
- **Middleware guard markers** — `@before_model` fires on EVERY LLM call within a single ReAct loop. Use sentinel strings (`<!--memory_injected-->`, `<!--token_guarded-->`) in system prompt content to prevent repeated operations.
- **Logging** — `utils/logger_handler.py` sets up daily log files in `logs/agent_YYYYMMDD.log` (DEBUG to file, INFO to console) with duplicate-handler guard.
- **MD5-based dedup** — `rag/vector_store.py` computes MD5 per file, compares against `md5.text`, only embeds new/changed files.
- **Streaming chunk routing** — `execute_stream_async` yields both plain text tokens and JSON tool events. The SSE handler in `chat.py` routes by prefix: strings starting with `{"type":` are parsed as JSON tool events; everything else is a text token.
- **Graceful degradation** — Redis failure → short-term memory returns empty. Reranker API failure → falls back to RRF ordering. Memory save failure → doesn't affect SSE response.
- **Frontend CDN dependency** — `api/static/index.html` loads `marked.js` from CDN for Markdown rendering. Offline environments won't render chat messages properly.
- **`data/external/` directory** — `records.csv` still exists on disk but is no longer used by any tool (report generation was removed). Safe to ignore or delete.
