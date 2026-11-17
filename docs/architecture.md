# System Architecture — DocuMind AI

**DocuMind AI** is designed as a decoupled, multi-layered Retrieval-Augmented Generation (RAG) system consisting of a **FastAPI REST Backend**, a **Modular RAG & Storage Core**, an **Automated Evaluation Engine**, and a **Streamlit AI SaaS Frontend**.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph ClientLayer["Frontend Presentation Layer (Streamlit SaaS UI :8501)"]
        UI_Dash["Dashboard View"]
        UI_Docs["Document Management"]
        UI_Chat["AI Chat Workspace"]
        UI_Ret["Retrieval Explorer"]
        UI_Eval["Evaluation Center"]
        APIClient["BackendAPIClient (Keep-Alive Pool + @st.cache_data)"]
        UI_Dash & UI_Docs & UI_Chat & UI_Ret & UI_Eval --> APIClient
    end

    subgraph APILayer["REST API Layer (FastAPI + Uvicorn :8000)"]
        MW["CORS + RateLimitMiddleware + SecretRedactingFilter"]
        R_Health["GET /health"]
        R_Docs["/documents (upload, list, detail, delete, reindex)"]
        R_Chat["/chat (ask, sessions, history)"]
        R_Ret["POST /retrieval/search"]
        R_Eval["/evaluation (run, latest)"]
        APIClient -->|HTTP JSON / Multipart| MW
        MW --> R_Health & R_Docs & R_Chat & R_Ret & R_Eval
    end

    subgraph CoreServices["Modular RAG Core Services (backend/app/)"]
        Ingest["IngestionPipeline & PDFExtractor (PyMuPDF)"]
        Chunker["DocumentChunker (Semantic & Section-Aware)"]
        Embedder["EmbeddingService (all-MiniLM-L6-v2, 384-d L2 Norm)"]
        Retriever["RetrievalService (Basic, MMR, Hybrid RRF + BM25)"]
        RAG["RAGPipeline & LLMClient (Live API / Grounded Extractive)"]
        Memory["SessionMemoryManager"]
        Evaluator["EvaluationEngine"]
    end

    subgraph StorageLayer["Persistence Layer"]
        SQLite[("SQLite Repository (documents, chunks, sessions, messages)")]
        FAISS[("FAISS Vector Index (IndexFlatIP Cosine + Metadata JSON)")]
        Uploads[("Local File Store (data/uploads/)")]
    end

    R_Docs --> Ingest
    Ingest --> Chunker --> Embedder --> FAISS
    Ingest --> SQLite & Uploads
    R_Chat --> RAG
    RAG --> Memory & Retriever
    R_Ret --> Retriever
    Retriever --> Embedder & FAISS
    Memory --> SQLite
    R_Eval --> Evaluator --> Retriever & RAG
```

---

## 2. Architectural Design Principles

1. **Strict Separation of Concerns**:
   - **API Controllers (`backend/app/api/`)** handle HTTP status codes, multipart parsing, and Pydantic request/response serialization without embedding business logic.
   - **Domain Services (`ingestion`, `chunking`, `embeddings`, `vectorstore`, `retrieval`, `generation`, `memory`, `evaluation`)** are independently testable and wired via `ServiceContainer` (`backend/app/api/dependencies.py`).
2. **Swappable Storage Interfaces**:
   - `BaseRepository` (`backend/app/database/db.py`) abstracts relational persistence so `SQLiteRepository` can be swapped for PostgreSQL without modifying upstream pipelines.
   - `VectorStoreBase` (`backend/app/vectorstore/base.py`) abstracts vector indexing so `FAISSVectorStore` can be extended with ChromaDB or Qdrant.
3. **Defense-in-Depth Security**:
   - Uploaded files undergo extension, MIME type, file size (`MAX_FILE_SIZE_MB`), and `%PDF-` magic-header validation before parsing.
   - Untrusted document text is sanitized against prompt-injection directives (`sanitize_untrusted_document_text`) and isolated inside `<RETRIEVED_DOCUMENT_CONTEXT>` XML tags.
   - API logs pass through `SecretRedactingFilter` to prevent accidental credential exposure.
