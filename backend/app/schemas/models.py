from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ============================================================================
# Document & Chunk Schemas
# ============================================================================

class PageContent(BaseModel):
    page_number: int = Field(..., ge=1, description="1-based page number")
    text: str = Field(..., description="Cleaned text extracted from the page")
    char_count: int = Field(..., ge=0)
    word_count: int = Field(..., ge=0)


class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    document_id: str = Field(..., description="Unique identifier of the parent document")
    filename: str = Field(..., description="Original safe filename of the document")
    page_number: int = Field(..., ge=1, description="Page number where this chunk originated")
    chunk_index: int = Field(..., ge=0, description="Sequential index of chunk within document")
    text: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., ge=0, description="Estimated token count of the chunk")
    section_title: str = Field(default="General", description="Detected document section heading")
    source_text: str = Field(default="", description="Original preserved source text of the chunk")
    document_hash: str = Field(default="", description="SHA-256 hash of parent document")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentMetadata(BaseModel):
    document_id: str
    filename: str
    file_hash: str
    file_size_bytes: int
    page_count: int
    chunk_count: int
    status: Literal["indexed", "failed", "duplicate", "processing"]
    error_message: Optional[str] = None
    chunk_size: int = 350
    chunk_overlap: int = 50
    uploaded_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUploadResult(BaseModel):
    filename: str
    document_id: Optional[str] = None
    status: Literal["indexed", "failed", "duplicate"]
    message: str
    page_count: int = 0
    chunk_count: int = 0
    file_size_bytes: int = 0


class DocumentUploadResponse(BaseModel):
    total_files: int
    indexed_count: int
    duplicate_count: int
    failed_count: int
    results: List[DocumentUploadResult]


class DocumentDetailResponse(BaseModel):
    document: DocumentMetadata
    chunks: List[DocumentChunk]


# ============================================================================
# Retrieval Schemas
# ============================================================================

class RetrievedChunkResult(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    text: str
    score: float = Field(..., description="Similarity or hybrid relevance score")
    retrieval_method: str = Field(default="semantic")
    section_title: str = Field(default="General")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=30)
    similarity_threshold: float = Field(default=0.05, ge=-1.0, le=1.0)
    mode: Literal["basic", "mmr", "hybrid"] = Field(default="basic")
    document_ids: Optional[List[str]] = Field(
        default=None, description="Optional list of document IDs to filter by"
    )
    use_query_rewriting: bool = Field(default=False)
    use_context_compression: bool = Field(default=False)
    mmr_lambda: float = Field(
        default=0.65, ge=0.0, le=1.0, description="Diversity vs relevance trade-off for MMR"
    )


class RetrievalSearchResponse(BaseModel):
    original_query: str
    effective_query: str
    mode: str
    total_found: int
    results: List[RetrievedChunkResult]


# ============================================================================
# Source Citation & Chat / RAG Schemas
# ============================================================================

class SourceCitation(BaseModel):
    citation_id: str = Field(..., description="Citation label, e.g. [1]")
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    snippet: str = Field(..., description="Exact supporting text snippet from chunk")
    score: float


class ChatAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=3000)
    session_id: Optional[str] = Field(
        default=None, description="Conversation session ID; auto-created if omitted"
    )
    document_ids: Optional[List[str]] = Field(
        default=None, description="Optional filter restricting search to specific documents"
    )
    retrieval_mode: Literal["basic", "mmr", "hybrid"] = Field(default="basic")
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    use_query_rewriting: bool = Field(default=True)
    use_context_compression: bool = Field(default=False)
    multi_doc_comparison: bool = Field(default=False)
    temperature: float = Field(default=0.1, ge=0.0, le=1.5)


class ChatMessage(BaseModel):
    message_id: str
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    citations: List[SourceCitation] = Field(default_factory=list)
    rewritten_query: Optional[str] = None
    retrieval_mode: Optional[str] = None
    insufficient_context: bool = False
    created_at: str


class ChatAskResponse(BaseModel):
    session_id: str
    question: str
    rewritten_query: str
    answer: str
    citations: List[SourceCitation]
    retrieved_chunks: List[RetrievedChunkResult]
    retrieval_mode: str
    llm_provider: str
    llm_model: str
    insufficient_context: bool
    grounding_score: float = Field(
        default=1.0, description="Estimated faithfulness/grounding score in [0, 1]"
    )
    latency_ms: float


class ChatSessionSummary(BaseModel):
    session_id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int


class ChatHistoryResponse(BaseModel):
    session_id: str
    title: str
    messages: List[ChatMessage]


# ============================================================================
# Evaluation Schemas
# ============================================================================

class EvaluationItem(BaseModel):
    id: str
    question: str
    expected_answer: str
    expected_keywords: List[str] = Field(default_factory=list)
    relevant_document_names: List[str]
    relevant_pages: List[int]


class EvaluationRunRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    modes_to_compare: List[Literal["basic", "mmr", "hybrid"]] = Field(
        default_factory=lambda: ["basic", "mmr", "hybrid"]
    )
    chunk_sizes_to_compare: Optional[List[int]] = Field(
        default=None,
        description="Optional list of chunk sizes (in tokens) to benchmark",
    )


class QuestionEvalDetail(BaseModel):
    question_id: str
    question: str
    expected_answer: str
    generated_answer: str
    expected_docs: List[str]
    expected_pages: List[int]
    retrieved_docs: List[str]
    retrieved_pages: List[int]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    hit: float
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    citation_correctness: float


class ModeEvaluationSummary(BaseModel):
    mode: str
    chunk_size: int
    top_k: int
    num_questions: int
    # Retrieval metrics
    precision_at_k: float
    recall_at_k: float
    mrr: float
    hit_rate: float
    # Generation metrics
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    citation_correctness: float
    details: List[QuestionEvalDetail]


class EvaluationRunResponse(BaseModel):
    run_id: str
    timestamp: str
    dataset_size: int
    summaries: List[ModeEvaluationSummary]


# ============================================================================
# System / Dashboard Schemas
# ============================================================================

class SystemStatsResponse(BaseModel):
    status: str
    app_name: str
    llm_provider: str
    llm_model: str
    embedding_model: str
    total_documents: int
    total_chunks: int
    vector_index_size: int
    total_sessions: int
    total_messages: int
    recent_documents: List[DocumentMetadata]
