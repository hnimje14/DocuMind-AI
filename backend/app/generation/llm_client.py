import re
from typing import Dict, List, Optional, Set, Tuple
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.generation.prompts import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    SECURITY_BOUNDARY_NOTE,
    SYSTEM_INSTRUCTION,
    build_rag_user_prompt,
)
from backend.app.retrieval.retriever import (
    HEADER_INTENT_TOKENS,
    classify_query_intent,
    is_profile_header_chunk,
    stem_token,
    tokenize_for_bm25,
)
from backend.app.schemas.models import ChatMessage, DocumentChunk, RetrievedChunkResult


PROVIDER_DEFAULT_URLS: Dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


class LLMClient:
    """
    Configurable LLM client supporting API-based providers (OpenAI, Groq, OpenRouter, Gemini)
    via environment variables, plus a deterministic Grounded Extractive Mode ("mock")
    that synthesizes citation-backed answers directly from retrieved evidence when no
    paid API key is configured.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = (provider or settings.LLM_PROVIDER or "mock").lower().strip()
        self.api_key = api_key if api_key is not None else settings.LLM_API_KEY
        self.base_url = base_url if base_url is not None else settings.LLM_BASE_URL
        self.model = model or settings.LLM_MODEL

    def is_live_api_configured(self) -> bool:
        """Return True if a non-mock provider and a non-empty API key are configured."""
        return (
            self.provider != "mock"
            and bool(self.api_key)
            and len(self.api_key.strip()) > 5
        )

    def generate_answer(
        self,
        question: str,
        effective_query: str,
        chunks: List[RetrievedChunkResult],
        history: List[ChatMessage],
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        multi_doc_comparison: bool = False,
    ) -> Tuple[str, bool, str, str]:
        """
        Generate a grounded RAG answer.
        Returns (answer_text, is_insufficient_context, active_provider, active_model).
        """
        if not chunks:
            return INSUFFICIENT_CONTEXT_MESSAGE, True, self.provider, self.model

        # Verify evidence relevance before generating
        if not self._has_sufficient_evidence(question, effective_query, chunks):
            return INSUFFICIENT_CONTEXT_MESSAGE, True, self.provider, self.model

        if self.is_live_api_configured():
            try:
                ans = self._call_live_api(
                    question=question,
                    chunks=chunks,
                    history=history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    multi_doc_comparison=multi_doc_comparison,
                )
                lower_ans = ans.lower()
                insufficient = (
                    "cannot be found in the uploaded documents" in lower_ans
                    or "cannot find sufficient information" in lower_ans
                    or "not present in the context" in lower_ans
                )
                return ans, insufficient, self.provider, self.model
            except Exception as exc:
                logger.error(
                    "Live LLM API call (%s) failed: %s. Falling back to Grounded Extractive Mode.",
                    self.provider,
                    str(exc),
                )

        # Grounded Extractive Mode (Mock / Development mode)
        ans, insufficient = self._generate_extractive_grounded_answer(
            question=question,
            effective_query=effective_query,
            chunks=chunks,
            multi_doc_comparison=multi_doc_comparison,
        )
        return ans, insufficient, "mock (grounded-extractive)", "extractive-rag-v1"

    def _has_sufficient_evidence(
        self,
        question: str,
        effective_query: str,
        chunks: List[RetrievedChunkResult],
    ) -> bool:
        """
        Check whether retrieved chunks actually contain evidence related to the query.
        Prevents hallucinating answers when the user asks about topics absent from the PDFs.
        """
        if not chunks:
            return False

        top_score = max(c.score for c in chunks)
        raw_q_tokens = {stem_token(t) for t in tokenize_for_bm25(question)}
        eff_q_tokens = {stem_token(t) for t in tokenize_for_bm25(effective_query)}
        if not eff_q_tokens:
            return top_score >= 0.20

        real_chunk_tokens: Set[str] = set()
        has_profile_chunk = False
        for c in chunks:
            real_chunk_tokens.update(stem_token(t) for t in tokenize_for_bm25(c.text))
            real_chunk_tokens.update(stem_token(t) for t in tokenize_for_bm25(c.filename))
            dummy = DocumentChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                text=c.text,
                token_count=1,
                section_title=c.section_title,
                metadata=c.metadata,
            )
            if is_profile_header_chunk(dummy):
                has_profile_chunk = True

        header_stems = {
            stem_token(t)
            for t in (
                HEADER_INTENT_TOKENS
                | {
                    "full", "which", "when", "student", "applicant",
                    "programming", "language", "know", "skill", "proficiency",
                    "certification", "certificate", "internship",
                }
            )
        }

        q_lower = question.lower()
        asks_unmentioned_relative = any(
            rel in q_lower
            for rel in ("father", "mother", "parent", "spouse", "brother", "sister", "salary", "passport")
        ) and not any(
            rel in real_chunk_tokens
            for rel in ("father", "mother", "parent", "spouse", "brother", "sister", "salary", "passport")
        )
        if asks_unmentioned_relative:
            return False

        intent, target_sec = classify_query_intent(question)
        if target_sec and chunks:
            top_sec = (chunks[0].section_title or chunks[0].metadata.get("section_title", "")).lower()
            if target_sec.lower() in top_sec and top_score >= 0.18:
                return True

        # If the question asks about header attributes on a profile chunk, verify that any
        # specific non-header qualifier is actually in the text
        if has_profile_chunk and (raw_q_tokens & header_stems):
            non_header_q = raw_q_tokens - header_stems
            if not non_header_q or (non_header_q & real_chunk_tokens):
                return True
            return False

        # Check raw question overlap first (prevents history context from masking an unrelated new question)
        if len(raw_q_tokens) >= 4:
            raw_overlap = raw_q_tokens & real_chunk_tokens
            if len(raw_overlap) <= 1 and top_score < 0.45:
                return False

        overlap = eff_q_tokens & real_chunk_tokens
        overlap_ratio = len(overlap) / max(1, len(eff_q_tokens))

        if len(overlap) == 0 and top_score < 0.45:
            return False
        if top_score < 0.15 and overlap_ratio < 0.25:
            return False
        return True

    def _call_live_api(
        self,
        question: str,
        chunks: List[RetrievedChunkResult],
        history: List[ChatMessage],
        temperature: float,
        max_tokens: int,
        multi_doc_comparison: bool,
    ) -> str:
        from openai import OpenAI

        resolved_base_url = self.base_url or PROVIDER_DEFAULT_URLS.get(self.provider)
        client = OpenAI(api_key=self.api_key, base_url=resolved_base_url)

        system_msg = f"{SYSTEM_INSTRUCTION}\n\n{SECURITY_BOUNDARY_NOTE}"
        user_msg = build_rag_user_prompt(
            question=question,
            chunks=chunks,
            history=history,
            multi_doc_comparison=multi_doc_comparison,
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        return content.strip()

    @staticmethod
    def _extract_structured_profile_lines(
        question: str, chunks: List[RetrievedChunkResult]
    ) -> List[Tuple[int, str, RetrievedChunkResult]]:
        """
        Dynamically extract structured profile/header attributes (person name, course/degree,
        CGPA/GPA, graduation year, email/phone) directly from retrieved document chunks
        when the question queries those attributes.
        """
        q_lower = question.lower()
        q_stems = {stem_token(t) for t in tokenize_for_bm25(question)}
        matched: List[Tuple[int, str, RetrievedChunkResult]] = []

        # Do not match candidate's own name if the question asks for a relative/other person (e.g. father's/mother's name)
        asks_other_person = any(
            rel in q_lower
            for rel in ("father", "mother", "parent", "spouse", "brother", "sister", "manager", "supervisor", "referee")
        )
        asks_name = (
            not asks_other_person
            and (
                ("name" in q_lower and any(w in q_lower for w in ("person", "candidate", "applicant", "student", "author", "who", "full")))
                or re.search(r"\bwho\s+is\b", q_lower) is not None
                or ("name" in q_stems and len(q_stems) <= 3)
            )
        )
        asks_cgpa = any(w in q_lower for w in ("cgpa", "gpa", "grade point"))
        asks_grad_year = (
            ("year" in q_lower or "when" in q_lower or "batch" in q_lower)
            and any(w in q_lower for w in ("graduat", "expect", "pass", "complet", "finish", "degree", "course"))
        )
        asks_course = (
            any(w in q_lower for w in ("course", "pursu", "degree", "major", "branch", "program", "studying"))
            and not asks_grad_year
        )
        asks_contact = any(w in q_lower for w in ("email", "mobile", "phone", "contact"))

        for cit_idx, chunk in enumerate(chunks, start=1):
            text = chunk.text
            # 1. Name extraction from explicit Name: field or Page 1 header line
            if asks_name:
                name_field = re.search(r"(?:^|\n)\s*(?:Full\s+)?Name\s*:\s*([^\n]+)", text, re.IGNORECASE)
                if name_field:
                    matched.append((cit_idx, f"**{name_field.group(1).strip()}**", chunk))
                    break
                if chunk.page_number == 1 and chunk.chunk_index == 0:
                    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                    for candidate_line in lines[:2]:
                        if re.match(r"^(?:RESUME|CURRICULUM\s+VITAE|CV)$", candidate_line, re.IGNORECASE):
                            continue
                        # Strip trailing inline fields if attached on the first line
                        candidate_clean = re.split(
                            r"\s+(?:Course|Degree|Email|Mobile|Phone|CGPA|GPA|Social)\s*:",
                            candidate_line,
                            maxsplit=1,
                        )[0].strip()
                        if (
                            ":" not in candidate_clean
                            and re.match(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z.]+){1,4}$", candidate_clean)
                        ):
                            matched.append((cit_idx, f"**{candidate_clean}**", chunk))
                            break
                    if matched:
                        break

            # 2. CGPA / GPA extraction
            if asks_cgpa:
                cgpa_match = re.search(r"\b(CGPA|GPA)\s*:\s*([0-9]+(?:\.[0-9]+)?(?:\s*/\s*[0-9.]+)?)", text, re.IGNORECASE)
                if cgpa_match:
                    label = cgpa_match.group(1).upper()
                    val = cgpa_match.group(2).strip()
                    matched.append((cit_idx, f"**{label}: {val}**", chunk))
                    break

            # 3. Expected Graduation Year extraction
            if asks_grad_year:
                course_year_match = re.search(
                    r"(?:Course|Degree|Graduation|Batch|Program)\s*:\s*([^\n]*?\b(20\d{2})\b[^\n]*)",
                    text,
                    re.IGNORECASE,
                )
                if course_year_match:
                    full_val = course_year_match.group(1).strip()
                    year_val = course_year_match.group(2).strip()
                    matched.append((cit_idx, f"**{year_val}** (Course: {full_val})", chunk))
                    break

            # 4. Course / Degree Pursuing extraction
            if asks_course:
                course_match = re.search(r"(?:^|\n)\s*(?:Course|Degree|Program)\s*:\s*([^\n]+)", text, re.IGNORECASE)
                if course_match:
                    raw_course = course_match.group(1).strip()
                    # Strip trailing graduation year ", 20XX" to highlight the course while keeping full context
                    course_clean = re.sub(r",\s*20\d{2}\s*$", "", raw_course).strip()
                    if course_clean != raw_course:
                        matched.append((cit_idx, f"**{course_clean}** (Course: {raw_course})", chunk))
                    else:
                        matched.append((cit_idx, f"**{raw_course}**", chunk))
                    break

            # 5. Contact (Email / Mobile) extraction
            if asks_contact:
                contact_lines = re.findall(
                    r"(?:Email|Mobile|Phone)\s*:\s*[^\n]+", text, re.IGNORECASE
                )
                if contact_lines:
                    for cl in contact_lines:
                        matched.append((cit_idx, f"**{cl.strip()}**", chunk))
                    break

        return matched

    def _generate_extractive_grounded_answer(
        self,
        question: str,
        effective_query: str,
        chunks: List[RetrievedChunkResult],
        multi_doc_comparison: bool = False,
    ) -> Tuple[str, bool]:
        """
        Synthesizes an accurate, citation-backed answer strictly from the retrieved sentences
        without inventing any external facts. Also detects multi-document comparisons/conflicts.
        """
        # First check for direct structured header/key-value matches (e.g., resume/form attributes)
        structured_matches = self._extract_structured_profile_lines(question, chunks)
        if structured_matches:
            lines: List[str] = ["Based on the retrieved document evidence:\n"]
            for cit_idx, fact_text, chunk in structured_matches:
                lines.append(
                    f"- {fact_text} **[{cit_idx}]** *(Source: {chunk.filename}, Page {chunk.page_number})*"
                )
            return "\n".join(lines), False

        q_tokens = {
            stem_token(t)
            for t in (tokenize_for_bm25(effective_query) + tokenize_for_bm25(question))
        } - {"explain", "describe", "tell", "detail"}

        # If the top retrieved chunk is a dedicated semantic section (e.g. Technical Proficiency, Certifications, or a specific Project)
        # that matches the query's target section or entity tokens, return its lines cleanly
        intent, target_sec = classify_query_intent(question)
        if chunks and not multi_doc_comparison:
            top_chunk = chunks[0]
            sec_title = (top_chunk.section_title or top_chunk.metadata.get("section_title", "")).lower()
            sec_stems = {stem_token(t) for t in tokenize_for_bm25(sec_title)}
            section_matched = (
                (sec_stems and len(q_tokens & sec_stems) >= 2)
                or (
                    target_sec in ("Technical Proficiency", "Certifications", "Internships")
                    and target_sec.lower() in sec_title
                )
            )
            if section_matched and len(top_chunk.text) <= 1200:
                lines_out = ["Based on the retrieved document evidence:\n"]
                for ln in top_chunk.text.splitlines():
                    cleaned_ln = ln.strip().lstrip("●•-").strip()
                    if cleaned_ln and not re.match(r"^(?:CERTIFICATIONS|INTERNSHIPS|PROJECTS)$", cleaned_ln, re.IGNORECASE):
                        lines_out.append(
                            f"- {cleaned_ln} **[1]** *(Source: {top_chunk.filename}, Page {top_chunk.page_number})*"
                        )
                if len(lines_out) > 1:
                    return "\n".join(lines_out), False

        scored_sentences: List[Tuple[float, int, str, RetrievedChunkResult]] = []
        for citation_idx, chunk in enumerate(chunks, start=1):
            raw_sentences = re.split(r"(?<=[.!?])\s+(?=[0-9]+\.[0-9]+|[A-Z])|\n+", chunk.text)
            for sent in raw_sentences:
                cleaned = re.sub(r"\s+", " ", sent).strip()
                # Strip leading "Page X: ... " header if attached to the first sentence
                cleaned = re.sub(r"^Page\s+\d+\s*:[^0-9]+?(?=[0-9]+\.[0-9]+)", "", cleaned).strip()
                # Skip standalone page title headings that have no sentence-ending period
                if len(cleaned) < 12 or re.match(r"^Page\s+\d+\s*:[^.!?]+$", cleaned):
                    continue
                s_tokens = {stem_token(t) for t in tokenize_for_bm25(cleaned)}
                if not s_tokens:
                    continue
                overlap = len(q_tokens & s_tokens)
                # Combine token overlap with chunk similarity score
                sent_score = (overlap * 2.0) + chunk.score
                if overlap > 0:
                    scored_sentences.append((sent_score, citation_idx, cleaned, chunk))

        if not scored_sentences:
            return INSUFFICIENT_CONTEXT_MESSAGE, True

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        top_sent_score = scored_sentences[0][0]
        min_acceptable_score = max(2.0, top_sent_score * 0.45)

        # Deduplicate selected sentences while preserving multi-document coverage if requested
        selected: List[Tuple[int, str, RetrievedChunkResult]] = []
        seen_texts: Set[str] = set()
        docs_represented: Set[str] = set()

        for s_score, cit_idx, sent_text, chunk in scored_sentences:
            if selected and s_score < min_acceptable_score:
                continue
            norm_key = re.sub(r"\W+", "", sent_text.lower())
            if norm_key in seen_texts:
                continue
            seen_texts.add(norm_key)
            selected.append((cit_idx, sent_text, chunk))
            docs_represented.add(chunk.filename)
            if len(selected) >= (4 if multi_doc_comparison else 3):
                break

        # If multi-doc comparison is requested, ensure second doc is included if relevant
        distinct_chunk_docs = {c.filename for c in chunks}
        if multi_doc_comparison and len(distinct_chunk_docs) > 1 and len(docs_represented) == 1:
            for _, cit_idx, sent_text, chunk in scored_sentences:
                if chunk.filename not in docs_represented:
                    norm_key = re.sub(r"\W+", "", sent_text.lower())
                    if norm_key not in seen_texts:
                        selected.append((cit_idx, sent_text, chunk))
                        docs_represented.add(chunk.filename)
                        break

        lines: List[str] = ["Based on the retrieved document evidence:\n"]
        for cit_idx, sent_text, chunk in selected:
            lines.append(
                f"- {sent_text} **[{cit_idx}]** *(Source: {chunk.filename}, Page {chunk.page_number})*"
            )

        if len(docs_represented) > 1:
            doc_list = ", ".join(sorted(docs_represented))
            lines.append(
                f"\n**Multi-Document Note:** The evidence above synthesizes information across multiple sources ({doc_list}). Please review the respective page citations for document-specific context."
            )

        return "\n".join(lines), False
