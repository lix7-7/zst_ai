# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ZhiSaoTong (智扫通)** — a FastAPI + SSE streaming intelligent customer service agent for robot vacuum cleaners. Uses LangChain 1.x ReAct Agent + RAG (BM25 + Dense + RRF + Cross-Encoder Reranker) over a private knowledge base to handle pre-sales Q&A, environment-aware recommendations, and personalized usage report generation. Streamlit retained as legacy UI.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize/re-index the Chroma vector store from data/ (first run, or when documents change)
python -m rag.vector_store

# Start the FastAPI server (primary — SSE streaming API)
uvicorn api.main:app --reload --port 8000

# Start the Streamlit web app (legacy UI)
streamlit run app.py

# Test hybrid search independently
python -m rag.hybrid_retriever

# Run RAG evaluation (4-way retriever comparison + RAGAS)
python -m eval.evaluate

# Re-index knowledge base via API
curl -X POST http://localhost:8000/knowledge/reindex
```

Requires `DASHSCOPE_API_KEY` (Alibaba Cloud DashScope — LLM + Embeddings + Reranker).  
Requires `AMAP_API_KEY` (Alibaba Cloud Amap — IP geolocation + weather; default fallback key included).  
Optional: Redis on `localhost:6379` for multi-turn memory (gracefully degrades if unavailable).  
FastAPI docs at `http://localhost:8000/docs`.

## Architecture (v2.0)

```
FastAPI (api/main.py + api/routers/) — SSE streaming + REST endpoints
  ├── api/routers/chat.py — POST /api/v1/chat/stream (SSE)
  ├── api/routers/knowledge.py — GET /knowledge/stats, POST /knowledge/reindex
  ├── api/schemas/chat.py — Pydantic models
  └── api/static/index.html — vanilla JS chat UI with SSE consumption, markdown, tool call cards
  └── agent/react_agent.py (ReactAgent — wraps langchain create_agent)
       ├── agent/tools/agent_tools.py (8 @tool functions)
       ├── agent/tools/middleware.py (4 middleware hooks)
       ├── rag/hybrid_retriever.py (BM25 + Dense Vector + RRF → Cross-Encoder Reranker)
       ├── memory/ (Redis short-term + Chroma long-term memory)
       └── model/factory.py (ChatTongyi + DashScopeEmbeddings via factory pattern)
  └── eval/
       ├── test_queries.json (30 annotated queries × 4 categories)
       ├── evaluate.py (4-way retriever comparison)
       └── eval_result.json (latest run output)
```

**The 8 tools:** `rag_summarize`, `get_weather`, `get_user_location`, `get_user_id`, `get_current_month`, `fetch_external_data`, `fill_context_for_report`, `memory_recall`.

**The 5 middleware** (LangChain 1.x decorator-based middleware API):
- `monitor_tool` (`@wrap_tool_call`) — logs tool calls/errors; when `fill_context_for_report` is called, injects `context["report"]=True`
- `log_before_model` (`@before_model`) — logs message history before each LLM call
- `report_prompt_switch` (`@dynamic_prompt`) — detects `context["report"]==True` and switches from `main_prompt.txt` to `report_prompt.txt`
- `memory_inject` (`@before_model`) — injects short-term + long-term memory context into the system prompt before each model call (with dedup guard to prevent repeated injection during ReAct loops)
- `token_guard` (`@before_model`) — [NEW] estimates total token count via tiktoken (with heuristic fallback), trims oldest history messages when over budget, and triggers LLM summarization of trimmed messages to preserve context continuity

**Hybrid RAG pipeline** (`rag/`):
Documents → MD5 dedup → Chinese-aware chunking → Chroma DB.
Retrieval: BM25 keyword search (jieba tokenization) + Dense Vector (Chroma) → RRF fusion → top-N candidates → Cross-Encoder Reranker (qwen3-rerank) → final top-k → LLM summarization. Configurable via `config/chroma.yml` `hybrid_search` section.

**Memory system** (`memory/`):
- `ShortTermMemory` — Redis sliding window (recent N rounds per session)
- `LongTermMemory` — conversation summaries stored in Chroma, semantically retrieved for context
- `MemoryManager` — unified entry point; gracefully degrades when Redis is unavailable
- **Token window management** — `utils/token_counter.py` provides tiktoken-based estimation + heuristic fallback; `token_guard` middleware enforces budget via history trimming + LLM summarization. Config in `config/memory.yml` `token_window` section.

**Runtime context flow for report generation:**
1. User requests a report → Agent calls `fill_context_for_report`
2. `monitor_tool` sets `context["report"] = True`
3. Next model call, `report_prompt_switch` returns `report_prompt.txt` as the system prompt
4. Agent operates in report-generation mode from then on

## Key Patterns

- **`__init__.py` used in new packages** — `api/`, `memory/`, `eval/` use standard Python package structure with `__init__.py`. Older modules (`agent/`, `rag/`, `model/`, `utils/`) omit them; imports work because all code runs from the project root.
- **Absolute path resolution** — `utils/path_tool.py` derives the project root from its own location (`__file__` → up 2 levels). All file I/O should use `get_abs_path()`.
- **YAML-driven config** — `utils/config_handler.py` loads 5 config files at module level (`rag_conf`, `chroma_conf`, `prompts_conf`, `agent_conf`, `memory_conf`). Model names, chunk settings, prompt paths, data paths, and memory parameters all live in `config/`.
- **Model factory** — `model/factory.py`: `ChatModelFactory` and `EmbeddingsFactory` extend `BaseModelFactory`. Module-level singletons `chat_model` and `embed_model` are the canonical instances.
- **Real API tools** — `get_weather` and `get_user_location` use Alibaba Cloud Amap APIs (IP geolocation + weather). `get_user_id` and `get_current_month` return mock data. `fetch_external_data` reads `data/external/records.csv`. These are extensible to real API/SQL calls.
- **Logging** — `utils/logger_handler.py` sets up daily log files in `logs/agent_YYYYMMDD.log` (DEBUG to file, INFO to console) with duplicate-handler guard.
- **MD5-based dedup** — `rag/vector_store.py` computes MD5 per file (4KB streaming chunks), compares against `md5.text`, only embeds new/changed files.
- **RAG evaluation** — `eval/evaluate.py` runs 4-way retriever comparison (Dense / BM25 / Hybrid RRF / Hybrid + Rerank) across 30 annotated queries, measuring Hit Rate@k (loose + strict), MRR, and keyword coverage. RAGAS generation quality evaluation (Faithfulness / Context Relevance / Answer Relevance) also included but depends on `ragas` package.
