import math
import re
from typing import Dict, List, Literal, Optional, Set, Tuple
import numpy as np
from rank_bm25 import BM25Okapi
from backend.app.chunking.chunker import estimate_token_count
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.embeddings.embedder import EmbeddingService
from backend.app.schemas.models import (
    ChatMessage,
    DocumentChunk,
    RetrievedChunkResult,
)
from backend.app.vectorstore.base import VectorStoreBase

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "of", "over", "under", "again",
    "further", "once", "here", "there", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "should", "now", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "what", "which", "who", "whom", "person", "candidate", "applicant",
}

FOLLOWUP_PRONOUNS: Set[str] = {
    "it", "its", "they", "them", "their", "theirs", "that", "those",
    "this", "these", "he", "she", "his", "her", "there", "also",
}

HEADER_INTENT_TOKENS: Set[str] = {
    "name", "person", "candidate", "author", "profile", "resume",
    "course", "pursuing", "pursue", "degree", "major", "branch",
    "cgpa", "gpa", "grade", "graduation", "graduate", "graduat",
    "expected", "expect", "year", "email", "mobile", "phone", "contact",
}


def stem_token(token: str) -> str:
    """Lightweight suffix normalizer for English technical words."""
    if len(token) <= 4 or any(ch.isdigit() for ch in token):
        return token
    for suffix in (
        "ation", "ition", "ment", "ness", "ingly", "edly",
        "ing", "ted", "ed", "ies", "es", "s",
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize_for_bm25(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric tokens excluding stopwords."""
    tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [t for t in tokens if len(t) > 1 and t not in STOPWORDS]


def is_profile_header_chunk(chunk: DocumentChunk) -> bool:
    """Return True if a Page 1 chunk represents Personal Information or Academic Details header."""
    if chunk.page_number != 1:
        return False
    sec = (chunk.section_title or chunk.metadata.get("section_title", "")).lower()
    if "personal information" in sec or "academic details" in sec:
        return True
    if chunk.chunk_index != 0:
        return False
    lower = chunk.text.lower()
    markers = ("email:", "mobile:", "phone:", "course:", "cgpa:", "linkedin", "github")
    return sum(1 for m in markers if m in lower) >= 2


QueryIntentType = Literal["simple_factual", "entity_or_section", "complex_broad"]


def classify_query_intent(query: str) -> Tuple[QueryIntentType, Optional[str]]:
    """
    Classify the user's question into:
    - 'simple_factual': Direct attribute lookup (name, CGPA, graduation year, course, email, SLA number, etc.)
    - 'entity_or_section': Explanation of a specific named project, section, or component
    - 'complex_broad': Multi-aspect or multi-document comparison/synthesis
    Also returns an optional target section hint:
      'Personal Information', 'Academic Details', 'Technical Proficiency', 'Projects', 'Internships', 'Certifications'.
    """
    q_lower = query.lower().strip()
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", q_lower)

    # 1. Academic Details (CGPA, Class 10/12, School/College, Electives)
    if any(
        w in q_lower
        for w in (
            "cgpa", "gpa", "class xii", "class 12", "class x", "class 10",
            "school", "college", "institute", "board", "msbshse", "academic",
            "elective", "subject",
        )
    ):
        return "simple_factual", "Academic Details"

    # 2. Personal Information (Name, Course, Graduation Year, Contact, or absent personal attributes)
    if any(
        phrase in q_lower
        for phrase in (
            "full name", "name of the person", "person's name", "name of person", "who is",
            "graduation year", "expected graduation", "which course", "what course", "pursuing",
            "email", "mobile", "phone", "father's name", "mother's name", "date of birth",
        )
    ):
        return "simple_factual", "Personal Information"

    # 3. Technical Proficiency (Programming languages, frameworks, libraries, tools)
    if any(
        w in q_lower
        for w in (
            "programming language", "languages", "technical proficiency", "technical skill",
            "skills", "framework", "library", "libraries", "tools",
        )
    ):
        return "entity_or_section", "Technical Proficiency"

    # 4. Projects
    if any(w in q_lower for w in ("project", "orchestration", "multi-agent", "securecart", "churn")):
        return "entity_or_section", "Projects"

    # 5. Internships
    if any(w in q_lower for w in ("internship", "intern", "extion", "clustor", "work experience")):
        return "entity_or_section", "Internships"

    # 6. Certifications
    if any(w in q_lower for w in ("certification", "certificate", "coursera", "mht-cet", "percentile")):
        return "entity_or_section", "Certifications"

    # 7. Complex comparative / multi-part queries
    if (
        any(w in q_lower for w in ("compare", "difference", "versus", "vs", "both", "summarize all", "entire resume", "overview"))
        or (" and " in q_lower and len(words) >= 10)
    ):
        return "complex_broad", None

    if len(words) <= 8 and q_lower.startswith(("what is", "what are", "when ", "how often", "how much", "which ")):
        return "simple_factual", None

    return "entity_or_section", None


class QueryRewriter:
    """
    Rewrites ambiguous follow-up questions into standalone retrieval queries
    ONLY when conversation history is actually needed to resolve pronouns or ellipsis.
    """

    @staticmethod
    def needs_rewriting(question: str, history: List[ChatMessage]) -> bool:
        if not history:
            return False
        q_lower = question.lower().strip()
        words = re.findall(r"[a-z]+", q_lower)
        word_set = set(words)

        # Standalone questions that explicitly refer to "the person", "the candidate", "the document",
        # or named entities do NOT need history rewriting
        if any(
            phrase in q_lower
            for phrase in ("the person", "the candidate", "the resume", "project", "cgpa", "graduation", "name of")
        ):
            if not (word_set & {"it", "its", "they", "them", "their", "that", "those"}):
                return False

        if word_set & FOLLOWUP_PRONOUNS:
            return True
        if len(words) <= 4 and q_lower.startswith(("what about", "how about", "and ", "also ")):
            return True
        return False

    @classmethod
    def rewrite(
        cls,
        question: str,
        history: List[ChatMessage],
        llm_rewrite_fn=None,
    ) -> str:
        cleaned_q = question.strip()
        if not history or not cls.needs_rewriting(cleaned_q, history):
            return cleaned_q

        if llm_rewrite_fn is not None:
            try:
                rewritten = llm_rewrite_fn(cleaned_q, history)
                if rewritten and len(rewritten.strip()) > 3:
                    return rewritten.strip()
            except Exception as exc:
                logger.warning("LLM query rewrite failed, using context expansion: %s", str(exc))

        recent_user_msgs = [m.content for m in history if m.role == "user"][-2:]
        if not recent_user_msgs:
            return cleaned_q

        context_tokens: List[str] = []
        seen: Set[str] = set(tokenize_for_bm25(cleaned_q))
        for prev_q in reversed(recent_user_msgs):
            for tok in tokenize_for_bm25(prev_q):
                if tok not in seen and tok not in FOLLOWUP_PRONOUNS and len(tok) >= 3:
                    context_tokens.append(tok)
                    seen.add(tok)
                if len(context_tokens) >= 6:
                    break

        if not context_tokens:
            return cleaned_q

        expanded = f"{cleaned_q} (Context: {' '.join(context_tokens)})"
        logger.info("Rewrote follow-up query '%s' -> '%s'", cleaned_q, expanded)
        return expanded


class ContextCompressor:
    """
    Implements Part D:
    - Duplicate chunk removal
    - Sentence-level relevance filtering
    - Context token budget enforcement
    - Relevant context ordering
    """

    @staticmethod
    def deduplicate_chunks(
        results: List[RetrievedChunkResult], jaccard_threshold: float = 0.80
    ) -> List[RetrievedChunkResult]:
        """Remove near-duplicate chunks based on token Jaccard similarity."""
        if len(results) <= 1:
            return results

        unique_results: List[RetrievedChunkResult] = []
        seen_token_sets: List[Set[str]] = []

        for item in results:
            tok_set = set(tokenize_for_bm25(item.text))
            if not tok_set:
                continue

            is_dup = False
            for existing_set in seen_token_sets:
                intersection = len(tok_set & existing_set)
                union = len(tok_set | existing_set)
                if union > 0 and (intersection / union) >= jaccard_threshold:
                    is_dup = True
                    break

            if not is_dup:
                unique_results.append(item)
                seen_token_sets.append(tok_set)

        return unique_results

    @staticmethod
    def compress_chunks(
        query: str,
        results: List[RetrievedChunkResult],
        max_total_tokens: int = 900,
    ) -> List[RetrievedChunkResult]:
        """
        Filter sentences/lines within retrieved chunks to retain only supporting evidence
        and cohesive block context while respecting a total token budget.
        """
        if not results:
            return []

        intent, target_section = classify_query_intent(query)
        q_stems = {stem_token(t) for t in tokenize_for_bm25(query)}
        compressed_list: List[RetrievedChunkResult] = []
        used_tokens = 0

        for item in results:
            # If the chunk is already compact (<= 160 tokens) or is a cohesive project/personal block, keep intact
            chunk_tokens = estimate_token_count(item.text)
            if chunk_tokens <= 160 or (
                intent == "entity_or_section" and chunk_tokens <= 280
            ):
                if used_tokens + chunk_tokens <= max_total_tokens or not compressed_list:
                    compressed_list.append(item)
                    used_tokens += chunk_tokens
                continue

            units = [u.strip() for u in re.split(r"(?<=[.!?])\s+|\n+", item.text) if u.strip()]
            if len(units) <= 3:
                if used_tokens + chunk_tokens <= max_total_tokens or not compressed_list:
                    compressed_list.append(item)
                    used_tokens += chunk_tokens
                continue

            keep_indices: Set[int] = set()
            for idx, unit in enumerate(units):
                u_stems = {stem_token(t) for t in tokenize_for_bm25(unit)}
                if q_stems & u_stems:
                    keep_indices.add(idx)
                    if idx > 0:
                        keep_indices.add(idx - 1)
                    if idx + 1 < len(units):
                        keep_indices.add(idx + 1)

            if not keep_indices:
                filtered_text = "\n".join(units[:3])
            else:
                filtered_text = "\n".join(units[i] for i in sorted(keep_indices))

            new_toks = estimate_token_count(filtered_text)
            if used_tokens + new_toks > max_total_tokens and compressed_list:
                break

            new_meta = dict(item.metadata)
            new_meta["uncompressed_text"] = item.text
            compressed_list.append(
                item.model_copy(update={"text": filtered_text, "metadata": new_meta})
            )
            used_tokens += new_toks

        return compressed_list


class RetrievalService:
    """
    Implements Part B & Part C:
    - Basic Semantic Retrieval (FAISS)
    - MMR Retrieval (Maximum Marginal Relevance)
    - Principled Hybrid Retrieval using Reciprocal Rank Fusion (RRF) + Lexical/Semantic Alignment
    - Query-Aware Candidate Retrieval (candidate_k=10 -> rerank/filter -> final 1..top_k)
    - Strict Document Scope Filtering before retrieval
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: VectorStoreBase,
    ):
        self.embedder = embedder
        self.vector_store = vector_store

    def _compute_section_alignment_bonus(
        self,
        query: str,
        chunk: DocumentChunk,
        intent: QueryIntentType,
        target_section: Optional[str],
    ) -> float:
        """
        Compute a principled query-to-section alignment bonus in [0.0, 0.45]
        when a chunk's section_title matches the query's target section or specific project.
        """
        sec_title = (chunk.section_title or chunk.metadata.get("section_title", "General")).lower()
        q_stems = {stem_token(t) for t in tokenize_for_bm25(query)}
        sec_stems = {stem_token(t) for t in tokenize_for_bm25(sec_title)}

        if target_section == "Projects":
            if not sec_title.startswith("projects"):
                return 0.0
            specific_q = q_stems - {"project", "explain", "describe", "tell", "detail", "work"}
            specific_sec = sec_stems - {"project"}
            if specific_q:
                overlap = len(specific_q & specific_sec)
                return min(0.45, 0.22 * overlap) if overlap > 0 else 0.0
            return 0.25

        if target_section:
            t_lower = target_section.lower()
            if t_lower in sec_title:
                return 0.38
            if t_lower == "personal information" and chunk.page_number == 1 and chunk.chunk_index == 0:
                return 0.38
            return 0.0

        if q_stems and sec_stems:
            sec_overlap = len(q_stems & sec_stems)
            if sec_overlap > 0:
                return min(0.30, 0.12 * sec_overlap)

        return 0.0

    def retrieve_basic(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        document_ids: Optional[List[str]] = None,
        candidate_k: int = 10,
    ) -> List[RetrievedChunkResult]:
        """Semantic retrieval using FAISS cosine similarity with candidate pool filtering."""
        chunks, vectors = self.vector_store.get_all_chunks_and_vectors(document_ids=document_ids)
        if not chunks or vectors.size == 0:
            return []

        q_vec = self.embedder.embed_query(query).reshape(-1).astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        intent, target_sec = classify_query_intent(query)
        sem_scores = np.dot(vectors, q_vec)

        scored_candidates: List[Tuple[float, int]] = []
        for idx, chunk in enumerate(chunks):
            base_sim = float(sem_scores[idx])
            sec_bonus = self._compute_section_alignment_bonus(query, chunk, intent, target_sec)
            final_score = min(1.0, base_sim + sec_bonus)
            if final_score >= similarity_threshold:
                scored_candidates.append((final_score, idx))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        pool = scored_candidates[: max(candidate_k, top_k)]
        return self._finalize_candidates(query, chunks, pool, top_k, intent, method="semantic")

    def retrieve_mmr(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        document_ids: Optional[List[str]] = None,
        mmr_lambda: float = 0.65,
        candidate_k: int = 10,
    ) -> List[RetrievedChunkResult]:
        """Maximum Marginal Relevance (MMR) retrieval over filtered candidate pool."""
        chunks, vectors = self.vector_store.get_all_chunks_and_vectors(document_ids=document_ids)
        if not chunks or vectors.size == 0:
            return []

        q_vec = self.embedder.embed_query(query).reshape(-1).astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        intent, target_sec = classify_query_intent(query)
        query_sims = np.dot(vectors, q_vec).astype(np.float32)
        for idx, c in enumerate(chunks):
            bonus = self._compute_section_alignment_bonus(query, c, intent, target_sec)
            if bonus > 0:
                query_sims[idx] = min(1.0, float(query_sims[idx]) + bonus)

        valid_indices = [
            i for i, sim in enumerate(query_sims) if float(sim) >= similarity_threshold
        ]
        if not valid_indices:
            return []

        valid_indices.sort(key=lambda idx: float(query_sims[idx]), reverse=True)
        candidate_indices = valid_indices[: max(candidate_k, top_k * 2)]

        selected_pairs: List[Tuple[float, int]] = []
        selected_indices: List[int] = []
        while len(selected_indices) < top_k and candidate_indices:
            best_idx = None
            best_mmr_score = -1e9

            for cand_idx in candidate_indices:
                rel_score = float(query_sims[cand_idx])
                if not selected_indices:
                    redundancy = 0.0
                else:
                    sel_vecs = vectors[selected_indices]
                    cand_vec = vectors[cand_idx]
                    redundancy = float(np.max(np.dot(sel_vecs, cand_vec)))

                mmr_score = mmr_lambda * rel_score - (1.0 - mmr_lambda) * redundancy
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = cand_idx

            if best_idx is None:
                break
            selected_indices.append(best_idx)
            selected_pairs.append((float(query_sims[best_idx]), best_idx))
            candidate_indices.remove(best_idx)

        return self._finalize_candidates(query, chunks, selected_pairs, top_k, intent, method="mmr")

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = settings.DEFAULT_TOP_K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        document_ids: Optional[List[str]] = None,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ) -> List[RetrievedChunkResult]:
        """
        Principled Hybrid Retrieval combining:
        1. Dense Semantic Search (FAISS cosine similarity rank)
        2. Sparse Lexical Search (Lucene-IDF BM25 rank)
        via Reciprocal Rank Fusion (RRF) + Section Alignment Reranking.
        """
        chunks, vectors = self.vector_store.get_all_chunks_and_vectors(document_ids=document_ids)
        if not chunks or vectors.size == 0:
            return []

        q_vec = self.embedder.embed_query(query).reshape(-1).astype(np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        intent, target_sec = classify_query_intent(query)
        semantic_scores = np.dot(vectors, q_vec).astype(np.float32)

        # Build stemmed corpus for BM25 including section_title and filename
        tokenized_corpus: List[List[str]] = []
        for c in chunks:
            sec_title = c.section_title or c.metadata.get("section_title", "")
            toks = [stem_token(t) for t in tokenize_for_bm25(c.text)]
            toks.extend(stem_token(t) for t in tokenize_for_bm25(sec_title))
            toks.extend(stem_token(t) for t in tokenize_for_bm25(c.filename))
            if is_profile_header_chunk(c):
                toks.extend(stem_token(t) for t in HEADER_INTENT_TOKENS)
            tokenized_corpus.append(toks or ["empty"])

        query_tokens = [stem_token(t) for t in tokenize_for_bm25(query)]
        n_chunks = len(chunks)

        if query_tokens:
            bm25 = BM25Okapi(tokenized_corpus)
            for word in list(bm25.idf.keys()):
                freq = sum(1 for doc_dict in bm25.doc_freqs if word in doc_dict)
                bm25.idf[word] = math.log(
                    1.0 + (bm25.corpus_size - freq + 0.5) / (freq + 0.5)
                )
            raw_bm25 = np.array(bm25.get_scores(query_tokens), dtype=np.float32)
            max_bm25 = float(np.max(raw_bm25)) if raw_bm25.size > 0 else 0.0
            if max_bm25 > 0:
                q_set = set(query_tokens)
                coverage = np.array(
                    [
                        len(q_set & set(doc_toks)) / max(1, len(q_set))
                        for doc_toks in tokenized_corpus
                    ],
                    dtype=np.float32,
                )
                norm_bm25 = (raw_bm25 / max_bm25) * coverage
            else:
                norm_bm25 = np.zeros(n_chunks, dtype=np.float32)
        else:
            raw_bm25 = np.zeros(n_chunks, dtype=np.float32)
            norm_bm25 = np.zeros(n_chunks, dtype=np.float32)

        # Compute 1-based ranks for Reciprocal Rank Fusion (RRF)
        sem_order = np.argsort(semantic_scores)[::-1]
        sem_ranks = np.empty(n_chunks, dtype=np.int32)
        for rank_pos, idx in enumerate(sem_order, start=1):
            sem_ranks[idx] = rank_pos

        bm25_order = np.argsort(raw_bm25)[::-1]
        bm25_ranks = np.empty(n_chunks, dtype=np.int32)
        for rank_pos, idx in enumerate(bm25_order, start=1):
            bm25_ranks[idx] = rank_pos

        # Max theoretical RRF score (when rank=1 in both lists) is 2 / (rrf_k + 1)
        max_rrf = 2.0 / (rrf_k + 1.0)
        candidates: List[Tuple[float, int]] = []

        for idx, chunk in enumerate(chunks):
            sem_s = max(0.0, float(semantic_scores[idx]))
            lex_s = float(norm_bm25[idx])

            rrf_raw = (1.0 / (rrf_k + int(sem_ranks[idx]))) + (
                (1.0 / (rrf_k + int(bm25_ranks[idx]))) if lex_s > 0 else 0.0
            )
            rrf_norm = rrf_raw / max_rrf  # Normalized in [0, 1]

            sec_bonus = self._compute_section_alignment_bonus(query, chunk, intent, target_sec)
            # Principled fusion: calibrated combination of RRF rank consensus + magnitude + section alignment
            calibrated_score = min(
                1.0,
                (0.45 * sem_s) + (0.35 * lex_s) + (0.20 * rrf_norm) + sec_bonus,
            )

            # Require actual evidence above similarity_threshold (not pure RRF rank floor on empty matches)
            evidence_signal = max(sem_s, lex_s, sec_bonus)
            if evidence_signal < similarity_threshold and calibrated_score < similarity_threshold:
                continue

            chunk.metadata["semantic_score"] = round(float(semantic_scores[idx]), 4)
            chunk.metadata["bm25_score"] = round(lex_s, 4)
            chunk.metadata["rrf_score"] = round(rrf_norm, 4)
            candidates.append((calibrated_score, idx))

        candidates.sort(key=lambda x: x[0], reverse=True)
        pool = candidates[: max(candidate_k, top_k)]
        return self._finalize_candidates(query, chunks, pool, top_k, intent, method="hybrid")

    def _finalize_candidates(
        self,
        query: str,
        chunks: List[DocumentChunk],
        scored_pool: List[Tuple[float, int]],
        top_k: int,
        intent: QueryIntentType,
        method: str,
    ) -> List[RetrievedChunkResult]:
        """
        Stage 2 Candidate Filtering & Relevance Pruning:
        - Never returns irrelevant tail chunks simply to fill top_k.
        - For simple_factual queries, prunes low-scoring unrelated sections when a strong top match exists.
        """
        if not scored_pool:
            return []

        top_score = scored_pool[0][0]
        _, target_sec = classify_query_intent(query)

        # When a specific target resume/document section is identified, retrieve ONLY that matching section
        # (unless multiple chunks belong to that same target section and score closely)
        if target_sec is not None:
            rel_cutoff = max(0.15, top_score * 0.80)
            effective_limit = 1 if intent == "simple_factual" else min(top_k, 2)
        elif intent == "simple_factual":
            rel_cutoff = max(0.10, top_score * 0.72)
            effective_limit = 1
        elif intent == "entity_or_section":
            rel_cutoff = max(0.08, top_score * 0.65)
            effective_limit = min(top_k, 2)
        else:
            rel_cutoff = max(0.05, top_score * 0.35)
            effective_limit = top_k

        results: List[RetrievedChunkResult] = []
        for score, idx in scored_pool:
            if results and score < rel_cutoff:
                break
            chunk = chunks[idx]
            sec_title = chunk.section_title or chunk.metadata.get("section_title", "General")
            results.append(
                RetrievedChunkResult(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    score=round(float(score), 4),
                    retrieval_method=method,
                    section_title=sec_title,
                    metadata=chunk.metadata,
                )
            )
            if len(results) >= effective_limit:
                break

        return ContextCompressor.deduplicate_chunks(results)

    def search(
        self,
        query: str,
        mode: str = "basic",
        top_k: int = settings.DEFAULT_TOP_K,
        similarity_threshold: float = settings.DEFAULT_SIMILARITY_THRESHOLD,
        document_ids: Optional[List[str]] = None,
        history: Optional[List[ChatMessage]] = None,
        use_query_rewriting: bool = False,
        use_context_compression: bool = True,
        multi_doc_comparison: bool = False,
        mmr_lambda: float = 0.65,
        candidate_k: int = 10,
    ) -> Tuple[str, List[RetrievedChunkResult]]:
        """
        Unified query-aware retrieval entrypoint returning (effective_query, retrieved_chunks).
        Applies document scope filtering first, retrieves up to candidate_k=10 candidates,
        reranks/filters to top relevant chunks, and applies sentence-level context compression.
        """
        effective_query = query.strip()
        if use_query_rewriting and history:
            effective_query = QueryRewriter.rewrite(effective_query, history)

        fetch_k = top_k * 2 if multi_doc_comparison else top_k

        if mode == "mmr":
            results = self.retrieve_mmr(
                query=effective_query,
                top_k=fetch_k,
                similarity_threshold=similarity_threshold,
                document_ids=document_ids,
                mmr_lambda=mmr_lambda,
                candidate_k=candidate_k,
            )
        elif mode == "hybrid":
            results = self.retrieve_hybrid(
                query=effective_query,
                top_k=fetch_k,
                similarity_threshold=similarity_threshold,
                document_ids=document_ids,
                candidate_k=candidate_k,
            )
        else:
            results = self.retrieve_basic(
                query=effective_query,
                top_k=fetch_k,
                similarity_threshold=similarity_threshold,
                document_ids=document_ids,
                candidate_k=candidate_k,
            )

        # Multi-document comparison: balance across distinct documents
        if multi_doc_comparison and len(results) > 1:
            by_doc: Dict[str, List[RetrievedChunkResult]] = {}
            for r in results:
                by_doc.setdefault(r.document_id, []).append(r)
            if len(by_doc) > 1:
                balanced: List[RetrievedChunkResult] = []
                max_len = max(len(v) for v in by_doc.values())
                for i in range(max_len):
                    for doc_id in by_doc:
                        if i < len(by_doc[doc_id]) and len(balanced) < top_k:
                            balanced.append(by_doc[doc_id][i])
                results = balanced
            else:
                results = results[:top_k]
        else:
            results = results[:top_k]

        if use_context_compression and results:
            results = ContextCompressor.compress_chunks(effective_query, results)

        return effective_query, results
