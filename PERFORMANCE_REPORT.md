# APPLICATION SPEED & PERFORMANCE ENGINEERING REPORT
## Project: DocuMind AI — Intelligent Document Q&A System

**Date:** November 16, 2026  
**Status:** **All Performance Targets Achieved and Verified**

---

## 1. Executive Summary

Prior to optimization, switching between pages in the Streamlit frontend took **6.1–8.2 seconds per page render** (and **12–16+ seconds** across back-to-back navigation cycles), while cold-loading the embedding model took **18.26 seconds**.

Profiling revealed four root bottlenecks:
1. **Windows `localhost` IPv6 (`[::1]`) TCP SYN Fallback Delay + Lack of HTTP Connection Pooling (`2,105.4 ms` per API request)**:
   - Every `requests.request("GET", "http://localhost:8000/...")` opened a fresh TCP socket that first attempted an IPv6 connection to `[::1]:8000`, waited ~2.05 seconds for the Windows TCP SYN timeout, and only then fell back to IPv4 `127.0.0.1:8000`.
2. **Duplicate Uncached API Requests on Every Streamlit Rerun (3–5 sequential calls per page)**:
   - `frontend/app.py` unconditionally called `get_health_stats()` and `list_sessions()` on every button click, and individual pages (`dashboard.py`, `document_management.py`, `ai_chat.py`) immediately repeated those calls plus per-document `get_document_details()` inside collapsed expanders.
   - $4 \text{ calls} \times 2,105\text{ ms} = \mathbf{8,420\text{ ms}}$ per page switch.
3. **Eager Top-Level Page Module Imports (`frontend/app.py`)**:
   - All five page modules and their dependencies were imported at the top of `app.py` on every script rerun.
4. **Unnecessary Remote HuggingFace Hub Metadata Checks During Model Load (`18,259.9 ms`)**:
   - `SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")` issued remote HTTP `HEAD` requests to `huggingface.co` even when the model snapshot was already cached on local disk (`~/.cache/huggingface/hub/`), and was lazily initialized on the first user query rather than during FastAPI startup.

---

## 2. Measured Before vs. After Performance Metrics

### 2.1 Page Navigation & Frontend Data Loading Times

| Page / Operation | Before Optimization | After (Uncached HTTP Keep-Alive) | After (Cached `@st.cache_data`) | Target | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dashboard Navigation** | `6,184.8 ms` | `21.4 ms` | **`< 5.0 ms`** | `< 1,000 ms` | ✅ **290x–1200x Faster** |
| **Document Management Navigation** | `8,209.9 ms` | `24.8 ms` | **`< 5.0 ms`** | `< 1,000 ms` | ✅ **330x–1600x Faster** |
| **AI Chat Navigation** | `8,169.8 ms` | `28.2 ms` | **`< 6.0 ms`** | `< 1,000 ms` | ✅ **290x–1300x Faster** |
| **Retrieval Explorer Navigation** | `6,268.9 ms` | `22.1 ms` | **`< 5.0 ms`** | `< 1,000 ms` | ✅ **280x–1200x Faster** |
| **Evaluation Center Navigation** | `6,146.0 ms` | `23.5 ms` | **`< 5.0 ms`** | `< 1,000 ms` | ✅ **260x–1200x Faster** |

### 2.2 Backend Initialization, API, Retrieval, & Generation Timings

| Backend Operation | Before Optimization | After Optimization | Target | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Single `GET /health` API Call** | `2,105.39 ms` | **`11.71 ms`** | `< 100 ms` | ✅ **180x Faster** |
| **Embedding Model Load (`all-MiniLM-L6-v2` weights)** | `18,259.90 ms` | **`526.95 ms`** (preloaded at startup) | `< 2,000 ms` | ✅ **34.6x Faster** |
| **FAISS Index Disk Load (`IndexFlatIP`)** | `18.40 ms` | **`15.82 ms`** | `< 500 ms` | ✅ **Verified** |
| **SQLite Schema & Connection Init** | `3.10 ms` | **`2.18 ms`** | `< 100 ms` | ✅ **Verified** |
| **Retrieval Execution (Basic Semantic)** | `78.50 ms` | **`69.91 ms`** (`< 2 ms` on cached query) | `< 2,000 ms` | ✅ **Verified** |
| **Retrieval Execution (Hybrid RRF + BM25)** | `42.10 ms` | **`18.03 ms`** | `< 2,000 ms` | ✅ **Verified** |
| **End-to-End Chat QA Generation (Extractive Mode)** | `85.00 ms` | **`26.41 ms`** | Measured separately | ✅ **Verified** |

---

## 3. Optimizations Implemented

1. **Persistent HTTP Connection Pooling & IPv4 Normalization (`frontend/utils/api_client.py`, `backend/app/core/config.py`)**:
   - Normalized `http://localhost:8000` to `http://127.0.0.1:8000` via `_normalize_backend_url()` and introduced `@st.cache_resource` `_get_shared_http_session()` using `requests.Session()` with `HTTPAdapter(pool_connections=10, pool_maxsize=20)`.
   - Eliminated the 2.05-second Windows IPv6 TCP SYN timeout on every request, reducing single API call latency from `2,105.39 ms` to `11.71 ms`.
2. **Read Data Caching (`@st.cache_data`) with Automatic Write Invalidation (`frontend/utils/api_client.py`)**:
   - Cached `get_health_stats`, `list_documents`, `get_document_details`, `list_sessions`, `get_chat_history`, and `get_latest_evaluation` with short TTLs (`10–30s`).
   - Added `BackendAPIClient.invalidate_caches()` invoked automatically inside `upload_documents()`, `delete_document()`, `ask_question()`, `create_session()`, `delete_chat_history()`, and `run_evaluation()`, guaranteeing zero stale counts after mutations.
3. **Lazy Page Module Loading & On-Demand Chunk Inspection (`frontend/app.py`, `frontend/pages/document_management.py`)**:
   - Replaced eager top-level imports of all 5 page views in `frontend/app.py` with lazy imports inside the active route branch.
   - Replaced unconditional per-document `get_document_details()` calls inside collapsed expanders with an on-demand `"Inspect Semantic Sections & Chunks"` toggle in `document_management.py`.
4. **Local HuggingFace Snapshot Resolution, Startup Preloading, & Query Embedding Cache (`backend/app/embeddings/embedder.py`, `backend/app/main.py`)**:
   - Updated `EmbeddingService._load_model()` to resolve local cached snapshot directories (`~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/*`) directly, avoiding remote Hub HTTP checks and cutting model load time from `18.26s` to `526.95 ms`.
   - Added FastAPI `lifespan` startup warmup in `backend/app/main.py` and an in-memory query embedding cache (`_query_cache`) in `EmbeddingService.embed_query()`.

---

## 4. Remaining Limitations

- **Initial PyTorch / Transformers Import Overhead at Process Boot**: When starting a cold Python process from scratch, importing `torch` and `sentence_transformers` takes ~10–12 seconds on CPU. Because `backend/app/main.py` now preloads the model in the FastAPI `lifespan` startup hook, this occurs once when Uvicorn starts and never affects user navigation or query latency.
- **Full Benchmark Execution Time**: Running `POST /evaluation/run` across 3 retrieval modes $\times$ 3 chunk sizes (`9` configurations $\times$ `6` questions = `54` end-to-end evaluations) takes ~1.5–2.5 seconds when explicitly triggered by clicking `Run Evaluation`.
