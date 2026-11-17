# REST API Documentation — DocuMind AI

The FastAPI backend exposes a RESTful JSON & Multipart API on `http://localhost:8000`.
Interactive OpenAPI (Swagger) documentation is available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## 1. Health & System Statistics

### `GET /health`
Returns application health, active model configuration, and live counts from SQLite and FAISS.

**Example Response (`200 OK`):**
```json
{
  "status": "healthy",
  "app_name": "DocuMind AI",
  "llm_provider": "mock (grounded-extractive)",
  "llm_model": "extractive-rag-v1",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "total_documents": 2,
  "total_chunks": 5,
  "vector_index_size": 5,
  "total_sessions": 1,
  "total_messages": 2,
  "recent_documents": []
}
```

---

## 2. Document Management Endpoints

### `POST /documents/upload`
Uploads, validates, extracts, chunks, embeds, and indexes one or more PDF files.
- **Content-Type:** `multipart/form-data`
- **Form Fields:**
  - `files`: List of `.pdf` files
  - `chunk_size`: `int` (default `350`)
  - `chunk_overlap`: `int` (default `50`)

**Example `curl` Request:**
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "files=@Enterprise_Cloud_Security_Policy_2026.pdf;type=application/pdf" \
  -F "chunk_size=350" \
  -F "chunk_overlap=50"
```

**Example Response (`201 Created`):**
```json
{
  "total_files": 1,
  "indexed_count": 1,
  "duplicate_count": 0,
  "failed_count": 0,
  "results": [
    {
      "filename": "Enterprise_Cloud_Security_Policy_2026.pdf",
      "document_id": "doc_85a488c4063e",
      "status": "indexed",
      "message": "Successfully extracted 3 pages and indexed 3 chunks.",
      "page_count": 3,
      "chunk_count": 3,
      "file_size_bytes": 4665
    }
  ]
}
```

### `GET /documents`
Lists all uploaded documents, optionally filtered by `status_filter` (`indexed`, `failed`, `duplicate`).

### `GET /documents/{document_id}`
Returns the document metadata and all extracted `DocumentChunk` records (including `section_title`, `page_number`, `token_count`, and `text`).

### `DELETE /documents/{document_id}`
Deletes a document from SQLite, removes its vectors from FAISS, rebuilds the vector index, and removes the stored PDF file from `data/uploads/`.

### `POST /documents/reindex`
Reprocesses all indexed documents using the specified `chunk_size` and `chunk_overlap` and rebuilds the FAISS index.

---

## 3. AI Chat & Conversational Memory Endpoints

### `POST /chat/ask`
Executes the end-to-end RAG pipeline for a user question and stores the turn in isolated session memory.

**Example Request:**
```json
{
  "question": "What encryption standard is required for data at rest and how often must keys be rotated?",
  "session_id": "sess_demo01",
  "document_ids": null,
  "retrieval_mode": "hybrid",
  "top_k": 5,
  "similarity_threshold": 0.05,
  "use_query_rewriting": true,
  "use_context_compression": false,
  "multi_doc_comparison": false,
  "temperature": 0.1
}
```

**Example Response (`200 OK`):**
```json
{
  "session_id": "sess_demo01",
  "question": "What encryption standard is required for data at rest and how often must keys be rotated?",
  "rewritten_query": "What encryption standard is required for data at rest and how often must keys be rotated?",
  "answer": "Based on the retrieved document evidence:\n\n- 2.1 Encryption at Rest: All persistent cloud databases, object storage buckets (AWS S3, GCS), and block volumes must use AES-256-GCM encryption at rest managed through Customer-Managed Keys (CMK) in Cloud KMS. **[1]** *(Source: Enterprise_Cloud_Security_Policy_2026.pdf, Page 2)*\n- 2.3 Key Rotation Schedule: Cryptographic master keys (CMKs) in Cloud KMS must undergo automatic rotation every 90 days. **[1]** *(Source: Enterprise_Cloud_Security_Policy_2026.pdf, Page 2)*",
  "citations": [
    {
      "citation_id": "[1]",
      "chunk_id": "doc_85a488c4063e_p2_c1",
      "document_id": "doc_85a488c4063e",
      "document_name": "Enterprise_Cloud_Security_Policy_2026.pdf",
      "page_number": 2,
      "snippet": "2.1 Encryption at Rest: All persistent cloud databases...",
      "score": 0.8912
    }
  ],
  "retrieved_chunks": [],
  "retrieval_mode": "hybrid",
  "llm_provider": "mock (grounded-extractive)",
  "llm_model": "extractive-rag-v1",
  "insufficient_context": false,
  "grounding_score": 0.974,
  "latency_ms": 26.41
}
```

### `GET /chat/sessions` & `POST /chat/sessions`
Lists all conversation sessions or creates a new isolated session.

### `GET /chat/history/{session_id}` & `DELETE /chat/history/{session_id}`
Retrieves or clears the message history for a specific session.

---

## 4. Retrieval Explorer & Evaluation Endpoints

### `POST /retrieval/search`
Executes standalone semantic (`basic`), `mmr`, or `hybrid` chunk retrieval and returns ranked chunks with similarity scores and section metadata.

### `POST /evaluation/run` & `GET /evaluation/latest`
Runs the reproducible RAG evaluation benchmark across specified `modes_to_compare` (`basic`, `mmr`, `hybrid`) and `chunk_sizes_to_compare` (`250`, `350`, `450`), or fetches the most recent benchmark report.
