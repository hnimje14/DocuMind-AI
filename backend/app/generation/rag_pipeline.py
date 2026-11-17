import re
import time
import uuid
from typing import List, Optional, Set, Tuple
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.security import detect_prompt_injection_in_query
from backend.app.database.db import BaseRepository, utc_now_iso
from backend.app.generation.llm_client import LLMClient
from backend.app.generation.prompts import INSUFFICIENT_CONTEXT_MESSAGE
from backend.app.retrieval.retriever import RetrievalService, tokenize_for_bm25
from backend.app.schemas.models import (
    ChatAskRequest,
    ChatAskResponse,
    ChatMessage,
    RetrievedChunkResult,
    SourceCitation,
)


def compute_grounding_score(answer: str, chunks: List[RetrievedChunkResult]) -> float:
    """
    Evaluate faithfulness/grounding of the generated answer against the retrieved context.
    Measures the fraction of substantive tokens in the answer that appear in the retrieved chunks.
    """
    if not chunks or not answer or answer == INSUFFICIENT_CONTEXT_MESSAGE:
        return 1.0

    # Strip boilerplate citation labels before checking content grounding
    cleaned_answer = re.sub(r"\*\*\[\d+\]\*\*", "", answer)
    cleaned_answer = re.sub(r"\*\(Source:.*?\)\*", "", cleaned_answer)
    cleaned_answer = cleaned_answer.replace("Based on the retrieved document evidence:", "")

    ans_tokens = tokenize_for_bm25(cleaned_answer)
    if not ans_tokens:
        return 1.0

    context_tokens: Set[str] = set()
    for c in chunks:
        context_tokens.update(tokenize_for_bm25(c.text))
        context_tokens.update(tokenize_for_bm25(c.filename))

    supported = sum(1 for t in ans_tokens if t in context_tokens)
    score = supported / len(ans_tokens)
    return round(min(1.0, max(0.0, score)), 4)


def build_verified_citations(
    answer: str,
    chunks: List[RetrievedChunkResult],
    insufficient_context: bool,
) -> List[SourceCitation]:
    """
    Build verified SourceCitation objects pointing strictly to actual retrieved chunks.
    Never fabricates page numbers or document references.
    """
    if insufficient_context or not chunks:
        return []

    # Check which citation numbers [1], [2], ... are explicitly referenced in the answer
    referenced_indices: Set[int] = set()
    for match in re.finditer(r"\[(\d+)\]", answer):
        idx = int(match.group(1))
        if 1 <= idx <= len(chunks):
            referenced_indices.add(idx)

    # Only fall back to filename + page matching if no bracketed citation index [N] was found
    if not referenced_indices:
        for idx, chunk in enumerate(chunks, start=1):
            if (
                chunk.filename.lower() in answer.lower()
                and f"page {chunk.page_number}" in answer.lower()
            ):
                referenced_indices.add(idx)

    # If still no explicit reference was parsed, attach the top supporting retrieved chunk
    if not referenced_indices:
        referenced_indices = {1}

    citations: List[SourceCitation] = []
    for idx in sorted(referenced_indices):
        chunk = chunks[idx - 1]
        snippet = chunk.text.strip()
        if len(snippet) > 450:
            snippet = snippet[:447] + "..."
        citations.append(
            SourceCitation(
                citation_id=f"[{idx}]",
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=chunk.filename,
                page_number=chunk.page_number,
                snippet=snippet,
                score=chunk.score,
            )
        )
    return citations


class RAGPipeline:
    """
    End-to-end Retrieval-Augmented Generation pipeline with conversational memory,
    query rewriting, multi-mode retrieval, hallucination checks, and verified citations.
    """

    def __init__(
        self,
        repository: BaseRepository,
        retriever: RetrievalService,
        llm_client: Optional[LLMClient] = None,
    ):
        self.repository = repository
        self.retriever = retriever
        self.llm_client = llm_client or LLMClient()

    def answer_question(self, request: ChatAskRequest) -> ChatAskResponse:
        start_time = time.perf_counter()

        session_id = request.session_id or f"sess_{uuid.uuid4().hex[:10]}"
        self.repository.create_session(session_id=session_id)

        # Retrieve isolated session conversation history
        history = self.repository.get_session_messages(session_id)

        # Guard against direct prompt injection in user question
        if detect_prompt_injection_in_query(request.question):
            logger.warning("Blocked prompt injection attempt in session '%s'.", session_id)
            refusal = (
                "Your request contains instructions that attempt to override system safety policies. "
                "Please ask a factual question about the uploaded documents."
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            return ChatAskResponse(
                session_id=session_id,
                question=request.question,
                rewritten_query=request.question,
                answer=refusal,
                citations=[],
                retrieved_chunks=[],
                retrieval_mode=request.retrieval_mode,
                llm_provider=self.llm_client.provider,
                llm_model=self.llm_client.model,
                insufficient_context=True,
                grounding_score=1.0,
                latency_ms=latency_ms,
            )

        # 1. Retrieve relevant chunks (with optional query rewriting & context compression)
        effective_query, retrieved_chunks = self.retriever.search(
            query=request.question,
            mode=request.retrieval_mode,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_ids=request.document_ids,
            history=history,
            use_query_rewriting=request.use_query_rewriting,
            use_context_compression=request.use_context_compression,
            multi_doc_comparison=request.multi_doc_comparison,
        )

        # 2. Generate grounded answer
        answer_text, insufficient_context, active_provider, active_model = (
            self.llm_client.generate_answer(
                question=request.question,
                effective_query=effective_query,
                chunks=retrieved_chunks,
                history=history,
                temperature=request.temperature,
                multi_doc_comparison=request.multi_doc_comparison,
            )
        )

        # 3. Verify grounding & construct citations
        grounding_score = compute_grounding_score(answer_text, retrieved_chunks)
        if not insufficient_context and grounding_score < 0.25:
            logger.warning(
                "Low grounding score (%.2f) detected; replacing with insufficient-context response.",
                grounding_score,
            )
            answer_text = INSUFFICIENT_CONTEXT_MESSAGE
            insufficient_context = True
            grounding_score = 1.0

        citations = build_verified_citations(
            answer=answer_text,
            chunks=retrieved_chunks,
            insufficient_context=insufficient_context,
        )

        # 4. Store user question and assistant response in isolated session history
        user_msg = ChatMessage(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            role="user",
            content=request.question,
            citations=[],
            rewritten_query=effective_query,
            retrieval_mode=request.retrieval_mode,
            insufficient_context=False,
            created_at=utc_now_iso(),
        )
        assistant_msg = ChatMessage(
            message_id=f"msg_{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            role="assistant",
            content=answer_text,
            citations=citations,
            rewritten_query=effective_query,
            retrieval_mode=request.retrieval_mode,
            insufficient_context=insufficient_context,
            created_at=utc_now_iso(),
        )
        self.repository.add_message(user_msg)
        self.repository.add_message(assistant_msg)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return ChatAskResponse(
            session_id=session_id,
            question=request.question,
            rewritten_query=effective_query,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            retrieval_mode=request.retrieval_mode,
            llm_provider=active_provider,
            llm_model=active_model,
            insufficient_context=insufficient_context,
            grounding_score=grounding_score,
            latency_ms=latency_ms,
        )
