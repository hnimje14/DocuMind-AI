import hashlib
import os
import re
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Tuple
from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from backend.app.core.config import settings
from backend.app.core.logging import logger

ALLOWED_EXTENSIONS = {".pdf"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",  # Validated further via %PDF- magic header
}

# Patterns that attempt prompt injection or instruction override inside documents/queries
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|system)\s+instructions", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(in\s+)?developer\s+mode", re.IGNORECASE),
    re.compile(r"system\s+override\s*:", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"<\s*/\s*system\s*>", re.IGNORECASE),
]


def sanitize_filename(filename: str) -> str:
    """Return a safe filename stripped of path traversal and special characters."""
    if not filename:
        return "unnamed_document.pdf"
    base_name = Path(filename).name
    # Replace unsafe characters with underscores
    cleaned = re.sub(r"[^a-zA-Z0-9._\-\s]", "_", base_name).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:180]


def compute_sha256(file_bytes: bytes) -> str:
    """Compute the SHA-256 hex digest of raw file bytes."""
    return hashlib.sha256(file_bytes).hexdigest()


def validate_pdf_upload(
    filename: str,
    content_type: str,
    file_bytes: bytes,
    max_file_size_mb: int = settings.MAX_FILE_SIZE_MB,
) -> Tuple[bool, str]:
    """
    Validate uploaded file extension, MIME type, size limit, and PDF magic bytes.
    Returns (is_valid, message).
    """
    if not filename:
        return False, "Filename is missing."

    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid file extension '{ext}'. Only PDF files (.pdf) are allowed."

    if content_type and content_type.lower() not in ALLOWED_MIME_TYPES:
        return False, f"Invalid MIME type '{content_type}'. Expected 'application/pdf'."

    if not file_bytes or len(file_bytes) == 0:
        return False, "Uploaded file is empty (0 bytes)."

    max_bytes = max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        return (
            False,
            f"File size ({size_mb:.2f} MB) exceeds maximum limit of {max_file_size_mb} MB.",
        )

    # Check PDF magic bytes (%PDF-) within the first 1024 bytes
    header_window = file_bytes[:1024]
    if b"%PDF-" not in header_window:
        return False, "File does not contain a valid '%PDF-' header signature."

    return True, "Valid PDF file."


def sanitize_untrusted_document_text(text: str) -> str:
    """
    Neutralize potential prompt injection directives inside untrusted PDF content
    without destroying legitimate factual content.
    """
    if not text:
        return ""
    sanitized = text
    for pattern in PROMPT_INJECTION_PATTERNS:
        sanitized = pattern.sub("[NEUTRALIZED_DIRECTIVE]", sanitized)
    return sanitized


def detect_prompt_injection_in_query(query: str) -> bool:
    """Detect whether a user query contains obvious prompt-injection overrides."""
    if not query:
        return False
    return any(pattern.search(query) for pattern in PROMPT_INJECTION_PATTERNS)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple sliding-window in-memory rate limiter per client IP."""

    def __init__(self, app, requests_per_minute: int = 120):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60.0
        self.client_requests: Dict[str, Deque[float]] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self.client_requests[client_ip]

        while window and (now - window[0]) > self.window_seconds:
            window.popleft()

        if len(window) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded for client IP %s", client_ip)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please wait before sending more requests."
                },
            )

        window.append(now)
        return await call_next(request)
