import json
import uuid
from pathlib import Path
from typing import List, Optional, Set, Tuple
from backend.app.chunking.chunker import DocumentChunker
from backend.app.core.config import PROJECT_ROOT, settings
from backend.app.core.logging import logger
from backend.app.database.db import BaseRepository, utc_now_iso
from backend.app.embeddings.embedder import EmbeddingService
from backend.app.generation.llm_client import LLMClient
from backend.app.generation.rag_pipeline import (
    build_verified_citations,
    compute_grounding_score,
)
from backend.app.ingestion.extractor import PDFExtractor
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.retrieval.retriever import RetrievalService, tokenize_for_bm25
from backend.app.schemas.models import (
    EvaluationItem,
    EvaluationRunRequest,
    EvaluationRunResponse,
    ModeEvaluationSummary,
    QuestionEvalDetail,
    RetrievedChunkResult,
)
from backend.app.vectorstore.faiss_store import FAISSVectorStore
from scripts.generate_sample_pdfs import (
    EVALUATION_DATASET,
    SAMPLE_DOCS_CONTENT,
    build_pdf_bytes,
)


class EvaluationEngine:
    """
    Reproducible RAG Evaluation Engine measuring:
    - Retrieval: Precision@K, Recall@K, Mean Reciprocal Rank (MRR), Hit Rate
    - Generation: Faithfulness, Answer Relevance, Context Relevance, Citation Correctness
    Supports comparing Basic, MMR, and Hybrid retrieval modes as well as different chunk sizes.
    """

    def __init__(
        self,
        repository: BaseRepository,
        embedder: EmbeddingService,
        retriever: RetrievalService,
        ingestion_pipeline: IngestionPipeline,
        llm_client: Optional[LLMClient] = None,
        evaluation_dir: Optional[str] = None,
    ):
        self.repository = repository
        self.embedder = embedder
        self.retriever = retriever
        self.ingestion_pipeline = ingestion_pipeline
        self.llm_client = llm_client or LLMClient()
        self.evaluation_dir = evaluation_dir or settings.EVALUATION_DIR

    def load_or_create_dataset(self) -> List[EvaluationItem]:
        """Load the verified evaluation dataset without polluting the runtime user document store."""
        dataset_path = Path(settings.EVALUATION_DIR) / "datasets" / "eval_dataset.json"
        if not dataset_path.exists():
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(
                json.dumps(EVALUATION_DATASET, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        raw_items = json.loads(dataset_path.read_text(encoding="utf-8"))
        return [EvaluationItem(**item) for item in raw_items]

    @staticmethod
    def _is_chunk_relevant(chunk: RetrievedChunkResult, item: EvaluationItem) -> bool:
        """
        A retrieved chunk is relevant if its document name matches one of item.relevant_document_names
        AND its page_number is in item.relevant_pages.
        """
        doc_match = any(
            rel_doc.lower() in chunk.filename.lower()
            for rel_doc in item.relevant_document_names
        )
        page_match = chunk.page_number in item.relevant_pages
        return doc_match and page_match

    def _evaluate_single_question(
        self,
        item: EvaluationItem,
        retrieved_chunks: List[RetrievedChunkResult],
        top_k: int,
    ) -> QuestionEvalDetail:
        """Compute retrieval and generation metrics for one evaluation item."""
        k = max(1, top_k)
        chunks_k = retrieved_chunks[:k]

        # 1. Retrieval Metrics
        # Build set of expected (doc, page) targets
        expected_targets: Set[Tuple[str, int]] = set()
        if len(item.relevant_document_names) == len(item.relevant_pages):
            for d_name, p_num in zip(item.relevant_document_names, item.relevant_pages):
                expected_targets.add((d_name.lower(), p_num))
        else:
            for d_name in item.relevant_document_names:
                for p_num in item.relevant_pages:
                    expected_targets.add((d_name.lower(), p_num))

        relevant_retrieved_count = 0
        matched_targets: Set[Tuple[str, int]] = set()
        first_relevant_rank: Optional[int] = None

        for rank_idx, chunk in enumerate(chunks_k, start=1):
            is_rel = self._is_chunk_relevant(chunk, item)
            if is_rel:
                relevant_retrieved_count += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank_idx
                for d_name, p_num in expected_targets:
                    if d_name in chunk.filename.lower() and chunk.page_number == p_num:
                        matched_targets.add((d_name, p_num))

        # Precision@K: proportion of retrieved chunks that are relevant
        # Normalized by min(len(chunks_k), len(expected_targets)) or len(chunks_k)
        precision_at_k = (
            relevant_retrieved_count / len(chunks_k) if chunks_k else 0.0
        )
        recall_at_k = (
            len(matched_targets) / len(expected_targets) if expected_targets else 0.0
        )
        reciprocal_rank = (1.0 / first_relevant_rank) if first_relevant_rank else 0.0
        hit = 1.0 if relevant_retrieved_count > 0 else 0.0

        # 2. Generation Metrics
        multi_doc = len(item.relevant_document_names) > 1
        answer_text, insufficient, _, _ = self.llm_client.generate_answer(
            question=item.question,
            effective_query=item.question,
            chunks=chunks_k,
            history=[],
            multi_doc_comparison=multi_doc,
        )
        citations = build_verified_citations(answer_text, chunks_k, insufficient)

        # Faithfulness: grounding of generated answer tokens in retrieved context
        faithfulness = compute_grounding_score(answer_text, chunks_k) if not insufficient else 0.0

        # Answer Relevance: keyword & token overlap with expected answer + question
        ans_tokens = set(tokenize_for_bm25(answer_text))
        exp_tokens = set(tokenize_for_bm25(item.expected_answer))
        kw_matches = sum(
            1 for kw in item.expected_keywords if kw.lower() in answer_text.lower()
        )
        kw_score = (
            kw_matches / len(item.expected_keywords) if item.expected_keywords else 0.0
        )
        tok_overlap = (
            len(ans_tokens & exp_tokens) / max(1, len(exp_tokens))
            if exp_tokens
            else 0.0
        )
        answer_relevance = round(min(1.0, 0.65 * kw_score + 0.35 * tok_overlap), 4)

        # Context Relevance: how well retrieved chunks cover expected keywords & answer tokens
        ctx_text = " ".join(c.text for c in chunks_k)
        ctx_tokens = set(tokenize_for_bm25(ctx_text))
        ctx_kw_matches = sum(
            1 for kw in item.expected_keywords if kw.lower() in ctx_text.lower()
        )
        ctx_kw_score = (
            ctx_kw_matches / len(item.expected_keywords)
            if item.expected_keywords
            else 0.0
        )
        ctx_tok_score = (
            len(ctx_tokens & exp_tokens) / max(1, len(exp_tokens))
            if exp_tokens
            else 0.0
        )
        context_relevance = round(min(1.0, 0.65 * ctx_kw_score + 0.35 * ctx_tok_score), 4)

        # Citation Correctness: fraction of generated citations that point to ground-truth (doc, page)
        if citations:
            correct_cits = 0
            for cit in citations:
                doc_ok = any(
                    d.lower() in cit.document_name.lower()
                    for d in item.relevant_document_names
                )
                page_ok = cit.page_number in item.relevant_pages
                if doc_ok and page_ok:
                    correct_cits += 1
            citation_correctness = round(correct_cits / len(citations), 4)
        else:
            citation_correctness = 0.0

        return QuestionEvalDetail(
            question_id=item.id,
            question=item.question,
            expected_answer=item.expected_answer,
            generated_answer=answer_text,
            expected_docs=item.relevant_document_names,
            expected_pages=item.relevant_pages,
            retrieved_docs=[c.filename for c in chunks_k],
            retrieved_pages=[c.page_number for c in chunks_k],
            precision_at_k=round(precision_at_k, 4),
            recall_at_k=round(recall_at_k, 4),
            reciprocal_rank=round(reciprocal_rank, 4),
            hit=round(hit, 4),
            faithfulness=round(faithfulness, 4),
            answer_relevance=round(answer_relevance, 4),
            context_relevance=round(context_relevance, 4),
            citation_correctness=round(citation_correctness, 4),
        )

    def _build_temp_retriever_for_chunk_size(self, chunk_size: int) -> RetrievalService:
        """Build an isolated FAISS index in memory/temp for evaluation without touching runtime user data."""
        overlap = max(20, min(150, chunk_size // 5))
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=overlap)
        extractor = PDFExtractor()

        all_chunks = []
        for idx, (filename, pages_data) in enumerate(SAMPLE_DOCS_CONTENT.items()):
            pdf_bytes = build_pdf_bytes(filename, pages_data)
            pages, meta = extractor.extract_from_bytes(pdf_bytes, filename)
            doc_chunks = chunker.chunk_document_pages(
                document_id=f"eval_doc_{idx}",
                filename=filename,
                pages=pages,
                extra_metadata=meta,
            )
            all_chunks.extend(doc_chunks)

        temp_dir = Path(self.evaluation_dir) / "results" / f"temp_vs_{chunk_size}"
        temp_vs = FAISSVectorStore(
            dimension=self.embedder.dimension,
            storage_dir=str(temp_dir),
            auto_load=False,
        )
        embeddings = self.embedder.embed_texts([c.text for c in all_chunks])
        temp_vs.rebuild(all_chunks, embeddings)
        return RetrievalService(embedder=self.embedder, vector_store=temp_vs)

    def run_evaluation(self, request: EvaluationRunRequest) -> EvaluationRunResponse:
        dataset = self.load_or_create_dataset()
        summaries: List[ModeEvaluationSummary] = []

        modes = request.modes_to_compare or ["basic", "mmr", "hybrid"]
        chunk_sizes = request.chunk_sizes_to_compare or [settings.DEFAULT_CHUNK_SIZE]

        for chunk_sz in chunk_sizes:
            active_retriever = self._build_temp_retriever_for_chunk_size(chunk_sz)

            for mode in modes:
                details: List[QuestionEvalDetail] = []
                for item in dataset:
                    multi_doc = len(item.relevant_document_names) > 1
                    _, retrieved = active_retriever.search(
                        query=item.question,
                        mode=mode,
                        top_k=request.top_k,
                        similarity_threshold=request.similarity_threshold,
                        multi_doc_comparison=multi_doc,
                    )
                    q_detail = self._evaluate_single_question(
                        item=item,
                        retrieved_chunks=retrieved,
                        top_k=request.top_k,
                    )
                    details.append(q_detail)

                n = max(1, len(details))
                summary = ModeEvaluationSummary(
                    mode=mode,
                    chunk_size=chunk_sz,
                    top_k=request.top_k,
                    num_questions=len(details),
                    precision_at_k=round(sum(d.precision_at_k for d in details) / n, 4),
                    recall_at_k=round(sum(d.recall_at_k for d in details) / n, 4),
                    mrr=round(sum(d.reciprocal_rank for d in details) / n, 4),
                    hit_rate=round(sum(d.hit for d in details) / n, 4),
                    faithfulness=round(sum(d.faithfulness for d in details) / n, 4),
                    answer_relevance=round(sum(d.answer_relevance for d in details) / n, 4),
                    context_relevance=round(sum(d.context_relevance for d in details) / n, 4),
                    citation_correctness=round(
                        sum(d.citation_correctness for d in details) / n, 4
                    ),
                    details=details,
                )
                summaries.append(summary)

        run_id = f"eval_{uuid.uuid4().hex[:8]}"
        response = EvaluationRunResponse(
            run_id=run_id,
            timestamp=utc_now_iso(),
            dataset_size=len(dataset),
            summaries=summaries,
        )

        # Persist evaluation report to disk for reproducibility
        results_dir = Path(self.evaluation_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        latest_path = results_dir / "latest_evaluation.json"
        latest_path.write_text(
            response.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return response

    def get_latest_evaluation(self) -> Optional[EvaluationRunResponse]:
        latest_path = Path(self.evaluation_dir) / "results" / "latest_evaluation.json"
        if not latest_path.exists():
            return None
        try:
            raw = json.loads(latest_path.read_text(encoding="utf-8"))
            return EvaluationRunResponse(**raw)
        except Exception as exc:
            logger.warning("Failed to load latest evaluation file: %s", str(exc))
            return None
