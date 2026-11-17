import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.app.core.config import settings
from backend.app.schemas.models import DocumentChunk, PageContent


def estimate_token_count(text: str) -> int:
    """
    Estimate token count for English/technical text (~1.3 tokens per whitespace-delimited word,
    or ~4 characters per token).
    """
    if not text:
        return 0
    words = text.split()
    by_words = int(round(len(words) * 1.3))
    by_chars = max(1, len(text) // 4)
    return max(by_words, by_chars)


RESUME_SECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Academic Details", re.compile(r"^(?:CGPA\s*:|GPA\s*:|ACADEMIC\s+DETAILS\b|EDUCATION\b|Subjects\s*/\s*Electives\s*:)", re.IGNORECASE)),
    ("Technical Proficiency", re.compile(r"^(?:Technical\s+Proficiency|Technical\s+Skills|Skills)\s*:?", re.IGNORECASE)),
    ("Projects", re.compile(r"^PROJECTS\b", re.IGNORECASE)),
    ("Internships", re.compile(r"^(?:INTERNSHIPS|WORK\s+EXPERIENCE|EXPERIENCE)\b", re.IGNORECASE)),
    ("Certifications", re.compile(r"^(?:CERTIFICATIONS|ACHIEVEMENTS)\b", re.IGNORECASE)),
]

RESEARCH_SECTION_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("Abstract", re.compile(r"^Abstract\b", re.IGNORECASE)),
    ("Executive Summary", re.compile(r"^Executive\s+Summary\b", re.IGNORECASE)),
    ("Introduction", re.compile(r"^(?:\d+\.?\s*)?Introduction\b", re.IGNORECASE)),
    ("Methodology", re.compile(r"^(?:\d+\.?\s*)?(?:Methodology|Methods|Architecture)\b", re.IGNORECASE)),
    ("Results", re.compile(r"^(?:\d+\.?\s*)?(?:Results|Benchmark|Evaluation|Findings)\b", re.IGNORECASE)),
    ("Conclusion", re.compile(r"^(?:\d+\.?\s*)?(?:Conclusion|Conclusions|Summary)\b", re.IGNORECASE)),
    ("References", re.compile(r"^References\b", re.IGNORECASE)),
]


def detect_document_type(pages: List[PageContent], filename: str = "") -> str:
    """
    Classify document structure into 'resume', 'research', or 'general'
    without forcing non-resume documents into resume sections.
    """
    combined = "\n".join(p.text for p in pages[:2]).lower()
    fname_lower = filename.lower()

    resume_signals = sum(
        1
        for kw in (
            "cgpa",
            "academic details",
            "technical proficiency",
            "internships",
            "certifications",
            "linkedin",
            "github",
            "curriculum vitae",
        )
        if kw in combined
    )
    if "resume" in fname_lower or "cv" in fname_lower:
        resume_signals += 2

    if resume_signals >= 3:
        return "resume"

    research_signals = sum(
        1
        for kw in (
            "abstract",
            "introduction",
            "methodology",
            "conclusion",
            "references",
            "executive summary",
            "benchmark report",
        )
        if kw in combined
    )
    if research_signals >= 2 or re.search(r"\b1\.1\s+[A-Z]", combined):
        return "research"

    return "general"


def _split_projects_block(projects_text: str) -> List[Tuple[str, str]]:
    """
    Split a 'PROJECTS' section into individual project blocks when multiple projects
    with 'Statement:' / 'Explanation:' / 'Skills used:' structure are present.
    Keeps each project's title, statement, explanation, and skills together.
    """
    body = re.sub(r"^PROJECTS\s*", "", projects_text.strip(), flags=re.IGNORECASE).strip()
    if not body:
        return []

    # Detect boundaries before "<Project Title> — <Category>\nStatement:"
    parts = re.split(
        r"(?=(?:^|\n+)[^\n]{5,110}(?:—|·)[^\n]*\n+Statement\s*:)",
        body,
    )
    blocks: List[Tuple[str, str]] = []
    for part in parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        first_line = cleaned.splitlines()[0].strip()
        proj_title = first_line.split("—")[0].split("·")[0].strip()[:60]
        section_name = f"Projects: {proj_title}" if proj_title else "Projects"
        blocks.append((section_name, f"PROJECTS — {cleaned}"))

    return blocks if blocks else [("Projects", projects_text.strip())]


def segment_page_into_sections(
    page_text: str, page_number: int, doc_type: str, carry_section: str = "General"
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Segment a single page's cleaned text into (section_title, section_text) blocks
    based on document type and structural headings.
    Returns (sections_list, last_active_section_title).
    """
    paragraphs = [p.strip() for p in re.split(r"\n\n+", page_text) if p.strip()]
    if not paragraphs:
        return [], carry_section

    if doc_type == "resume":
        active_section = "Personal Information" if page_number == 1 and carry_section == "General" else carry_section
        grouped: List[Tuple[str, List[str]]] = []

        for para in paragraphs:
            matched_title: Optional[str] = None
            for sec_title, pattern in RESUME_SECTION_PATTERNS:
                if pattern.search(para):
                    matched_title = sec_title
                    break

            if matched_title:
                active_section = matched_title

            if grouped and grouped[-1][0] == active_section:
                grouped[-1][1].append(para)
            else:
                grouped.append((active_section, [para]))

        final_sections: List[Tuple[str, str]] = []
        for sec_title, para_list in grouped:
            joined = "\n\n".join(para_list).strip()
            if sec_title == "Projects" and "Statement:" in joined:
                final_sections.extend(_split_projects_block(joined))
            else:
                final_sections.append((sec_title, joined))

        return final_sections, active_section

    if doc_type == "research":
        active_section = carry_section
        grouped_res: List[Tuple[str, List[str]]] = []

        for para in paragraphs:
            first_line = para.splitlines()[0].strip()
            matched_title = None

            page_hdr = re.match(r"^Page\s+\d+\s*:\s*(.+)$", first_line, re.IGNORECASE)
            num_hdr = re.match(r"^(\d+\.\d+\s+[^:.\n]{3,65})\s*:", first_line)
            if page_hdr:
                matched_title = page_hdr.group(1).strip()
            elif num_hdr:
                matched_title = num_hdr.group(1).strip()
            else:
                for sec_title, pattern in RESEARCH_SECTION_PATTERNS:
                    if pattern.search(first_line):
                        matched_title = sec_title
                        break

            if matched_title:
                active_section = matched_title

            if grouped_res and grouped_res[-1][0] == active_section:
                grouped_res[-1][1].append(para)
            else:
                grouped_res.append((active_section, [para]))

        return [(t, "\n\n".join(ps)) for t, ps in grouped_res], active_section

    # General documents: detect explicit headings or numbered sections
    active_section = carry_section
    grouped_gen: List[Tuple[str, List[str]]] = []
    for para in paragraphs:
        first_line = para.splitlines()[0].strip()
        if (
            len(first_line) <= 75
            and not first_line.endswith(".")
            and (
                first_line.isupper()
                or re.match(r"^(?:Section|Chapter|Part|Unit|Module)\s+\d+", first_line, re.IGNORECASE)
            )
        ):
            active_section = first_line.title()

        if grouped_gen and grouped_gen[-1][0] == active_section:
            grouped_gen[-1][1].append(para)
        else:
            grouped_gen.append((active_section, [para]))

    return [(t, "\n\n".join(ps)) for t, ps in grouped_gen], active_section


class DocumentChunker:
    """
    Semantic-aware, document-aware text chunker.
    Detects document structure (resumes, research/technical reports, general docs),
    segments by semantic headings & section boundaries, and splits oversized sections
    along paragraph/sentence boundaries while preserving full metadata.
    """

    def __init__(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ):
        if chunk_size < 50:
            raise ValueError("chunk_size must be at least 50 tokens.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=estimate_token_count,
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            keep_separator=True,
        )

    def chunk_document_pages(
        self,
        document_id: str,
        filename: str,
        pages: List[PageContent],
        extra_metadata: Optional[Dict[str, Any]] = None,
        document_hash: str = "",
    ) -> List[DocumentChunk]:
        """
        Segment extracted pages into semantic sections and token-bounded chunks
        while preserving document_id, filename, page_number, chunk_id, chunk_index,
        section_title, source_text, and document_hash.
        """
        chunks: List[DocumentChunk] = []
        chunk_idx = 0
        base_meta = dict(extra_metadata or {})
        doc_hash = document_hash or str(base_meta.get("file_hash", ""))
        doc_type = detect_document_type(pages, filename=filename)
        carry_section = "General"

        for page in pages:
            page_text = page.text.strip()
            if not page_text:
                continue

            sections, carry_section = segment_page_into_sections(
                page_text=page_text,
                page_number=page.page_number,
                doc_type=doc_type,
                carry_section=carry_section,
            )

            for section_title, section_text in sections:
                cleaned_section = section_text.strip()
                if not cleaned_section:
                    continue

                sec_tokens = estimate_token_count(cleaned_section)
                if sec_tokens <= self.chunk_size:
                    split_pieces = [cleaned_section]
                else:
                    split_pieces = self._splitter.split_text(cleaned_section)

                for sub_idx, piece in enumerate(split_pieces):
                    raw_source = piece.strip()
                    if not raw_source:
                        continue

                    # Prepend section context header on continuation sub-chunks if not already present
                    if (
                        section_title != "General"
                        and sub_idx > 0
                        and section_title.lower() not in raw_source[:80].lower()
                    ):
                        enriched_text = f"[{section_title}]\n{raw_source}"
                    else:
                        enriched_text = raw_source

                    tok_count = estimate_token_count(enriched_text)
                    chunk_id = f"{document_id}_p{page.page_number}_c{chunk_idx}"
                    chunk_meta = {
                        **base_meta,
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page.page_number,
                        "chunk_index": chunk_idx,
                        "section_title": section_title,
                        "source_text": raw_source,
                        "document_hash": doc_hash,
                        "document_type": doc_type,
                        "chunk_size_config": self.chunk_size,
                        "chunk_overlap_config": self.chunk_overlap,
                    }
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            filename=filename,
                            page_number=page.page_number,
                            chunk_index=chunk_idx,
                            text=enriched_text,
                            token_count=tok_count,
                            section_title=section_title,
                            source_text=raw_source,
                            document_hash=doc_hash,
                            metadata=chunk_meta,
                        )
                    )
                    chunk_idx += 1

        return chunks
