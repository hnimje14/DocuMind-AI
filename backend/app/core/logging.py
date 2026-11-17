import logging
import re
import sys
from backend.app.core.config import settings


class SecretRedactingFilter(logging.Filter):
    """Filter that redacts potential API keys or bearer tokens from log messages."""

    SECRET_PATTERNS = [
        re.compile(r"(sk-[a-zA-Z0-9_-]{12,})"),
        re.compile(r"(gsk_[a-zA-Z0-9_-]{12,})"),
        re.compile(r"(Bearer\s+[a-zA-Z0-9._-]{12,})", re.IGNORECASE),
        re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?[a-zA-Z0-9._-]{8,}['\"]?)", re.IGNORECASE),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for pattern in self.SECRET_PATTERNS:
                record.msg = pattern.sub("[REDACTED_SECRET]", record.msg)
        if record.args:
            new_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern in self.SECRET_PATTERNS:
                        arg = pattern.sub("[REDACTED_SECRET]", arg)
                new_args.append(arg)
            record.args = tuple(new_args)
        return True


def setup_logger(name: str = "rag_system") -> logging.Logger:
    """Configure and return a structured application logger with secret redaction."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        handler.addFilter(SecretRedactingFilter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


logger = setup_logger()
