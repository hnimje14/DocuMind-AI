import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pymupdf as fitz  # PyMuPDF
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.security import (
    compute_sha256,
    sanitize_filename,
    sanitize_untrusted_document_text,
    validate_pdf_upload,
)
from backend.app.schemas.models import PageContent


class PDFExtractionError(Exception):
    """Raised when a PDF is corrupted, encrypted, unreadable, or contains no extractable text."""


SECTION_HEADER_PATTERNS = [
    r"ACADEMIC\s+DETAILS\b",
    r"EDUCATION\b",
    r"SUBJECTS\s*/\s*ELECTIVES\s*:",
    r"TECHNICAL\s+PROFICIENCY\s*:",
    r"TECHNICAL\s+SKILLS\b",
    r"PROJECTS\b",
    r"INTERNSHIPS\b",
    r"WORK\s+EXPERIENCE\b",
    r"EXPERIENCE\b",
    r"CERTIFICATIONS\b",
    r"ACHIEVEMENTS\b",
    r"ABSTRACT\b",
    r"INTRODUCTION\b",
    r"METHODOLOGY\b",
    r"RESULTS\b",
    r"CONCLUSION\b",
    r"REFERENCES\b",
    r"EXECUTIVE\s+SUMMARY\b",
]


def clean_extracted_text(raw_text: str) -> str:
    """
    Clean raw PDF text while preserving structural section headings, key-value lines,
    bullet lists, and project/item boundaries.
    """
    if not raw_text:
        return ""

    text = raw_text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Rejoin hyphenated words split across line breaks (e.g. "architec-\nture" -> "architecture")
    text = re.sub(r"([a-zA-Z])-\n([a-zA-Z])", r"\1\2", text)

    # Collapse multiple horizontal spaces/tabs into a single space
    text = re.sub(r"[ \t\f\v]+", " ", text)

    # Strip spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Ensure Page headers stay on their own paragraph break
    text = re.sub(r"(^|\n)(Page\s+\d+\s*:[^\n]+)\n", r"\1\2\n\n", text)

    # Insert explicit paragraph breaks before major section headers
    header_union = "|".join(SECTION_HEADER_PATTERNS)
    text = re.sub(
        rf"(?:^|\n|\s+)(?=(?:{header_union}))",
        "\n\n",
        text,
        flags=re.IGNORECASE,
    )

    # Separate project titles that precede a "Statement:" line (with 1 or 2 newlines) onto their own paragraph break
    text = re.sub(
        r"\n([^\n]{5,110}(?:—|·)\s*[^\n]{3,65})\n+(?=Statement\s*:)",
        r"\n\n\1\n",
        text,
    )

    # Join single line-wrap newlines inside prose paragraphs where the line does not end with terminal punctuation
    # and the next line is a continuation (not a numbered section, bullet, Key: Value label, or uppercase header)
    text = re.sub(
        r"(?<![.!?:])\n(?![0-9]+\.[0-9]+|Page \d+|•|●|-|[A-Z][A-Za-z0-9 /_-]{1,28}:|[A-Z]{2,}(?:\s+[A-Z/&-]+)*\b|\n)",
        " ",
        text,
    )

    # Separate inline glued Key: Value labels such as "Social: Github Linkedin CGPA: 6.95"
    # Place paragraph break BEFORE CGPA: so CGPA is grouped with Academic Details rather than Contact Info
    text = re.sub(r"\s+(?=(?:Email|Mobile|Phone|Course|Social)\s*:)", "\n", text)
    text = re.sub(r"\s+(?=(?:CGPA|GPA)\s*:)", "\n\n", text)
    text = re.sub(r"(\b(?:CGPA|GPA)\s*:\s*[0-9.]+)\s*\n+(?=ACADEMIC\s+DETAILS|EDUCATION)", r"\1\n", text, flags=re.IGNORECASE)

    # Collapse 3+ consecutive newlines into 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    # Neutralize embedded prompt injection strings
    text = sanitize_untrusted_document_text(text)

    return text.strip()


class PDFExtractor:
    """Extracts and cleans page-level text and metadata from PDF files using PyMuPDF."""

    def extract_from_bytes(
        self, file_bytes: bytes, filename: str = "document.pdf"
    ) -> Tuple[List[PageContent], Dict[str, Any]]:
        """
        Extract cleaned text per page along with PDF document metadata.
        Raises PDFExtractionError if unreadable or empty of text.
        """
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as exc:
            logger.error("Failed to open PDF '%s': %s", filename, str(exc))
            raise PDFExtractionError(
                f"Failed to read PDF file '{filename}'. The file may be corrupted or invalid."
            ) from exc

        if doc.is_encrypted:
            doc.close()
            raise PDFExtractionError(
                f"PDF file '{filename}' is password-protected/encrypted and cannot be processed."
            )

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            raise PDFExtractionError(f"PDF file '{filename}' has 0 pages.")

        raw_meta = doc.metadata or {}
        pdf_metadata: Dict[str, Any] = {
            "title": (raw_meta.get("title") or "").strip(),
            "author": (raw_meta.get("author") or "").strip(),
            "subject": (raw_meta.get("subject") or "").strip(),
            "creator": (raw_meta.get("creator") or "").strip(),
            "total_pages": total_pages,
        }

        pages: List[PageContent] = []
        for page_idx in range(total_pages):
            page = doc.load_page(page_idx)
            raw_text = page.get_text("text") or ""
            cleaned = clean_extracted_text(raw_text)
            if cleaned:
                words = cleaned.split()
                pages.append(
                    PageContent(
                        page_number=page_idx + 1,
                        text=cleaned,
                        char_count=len(cleaned),
                        word_count=len(words),
                    )
                )

        doc.close()

        if not pages:
            raise PDFExtractionError(
                f"PDF file '{filename}' contains no extractable text (it may be a blank or scanned image-only PDF)."
            )

        return pages, pdf_metadata

    def save_uploaded_file(
        self, file_bytes: bytes, document_id: str, safe_name: str, upload_dir: Optional[str] = None
    ) -> Path:
        """Persist the validated PDF file to disk with a collision-safe name."""
        target_dir = Path(upload_dir or settings.UPLOAD_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)
        dest_path = target_dir / f"{document_id}_{safe_name}"
        dest_path.write_bytes(file_bytes)
        return dest_path


def generate_document_id() -> str:
    """Generate a unique document identifier."""
    return f"doc_{uuid.uuid4().hex[:12]}"
