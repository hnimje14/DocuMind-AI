import abc
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.models import (
    ChatMessage,
    ChatSessionSummary,
    DocumentChunk,
    DocumentMetadata,
    SourceCitation,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaseRepository(abc.ABC):
    """
    Abstract database interface for storing document metadata, chunks,
    conversation history, and evaluation results.
    Can be implemented by SQLiteRepository or PostgreSQLRepository.
    """

    @abc.abstractmethod
    def save_document(self, doc: DocumentMetadata) -> None:
        pass

    @abc.abstractmethod
    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        pass

    @abc.abstractmethod
    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentMetadata]:
        pass

    @abc.abstractmethod
    def list_documents(self, status_filter: Optional[str] = None) -> List[DocumentMetadata]:
        pass

    @abc.abstractmethod
    def delete_document(self, document_id: str) -> bool:
        pass

    @abc.abstractmethod
    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        pass

    @abc.abstractmethod
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        pass

    @abc.abstractmethod
    def get_all_chunks(self) -> List[DocumentChunk]:
        pass

    @abc.abstractmethod
    def create_session(self, session_id: str, title: str = "New Conversation") -> ChatSessionSummary:
        pass

    @abc.abstractmethod
    def get_session(self, session_id: str) -> Optional[ChatSessionSummary]:
        pass

    @abc.abstractmethod
    def list_sessions(self) -> List[ChatSessionSummary]:
        pass

    @abc.abstractmethod
    def delete_session(self, session_id: str) -> bool:
        pass

    @abc.abstractmethod
    def add_message(self, message: ChatMessage) -> None:
        pass

    @abc.abstractmethod
    def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        pass

    @abc.abstractmethod
    def count_messages(self) -> int:
        pass


class SQLiteRepository(BaseRepository):
    """Thread-safe SQLite implementation of BaseRepository."""

    def __init__(self, db_path: str = settings.SQLITE_DB_PATH):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    chunk_size INTEGER NOT NULL DEFAULT 800,
                    chunk_overlap INTEGER NOT NULL DEFAULT 150,
                    uploaded_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );

                CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(file_hash);
                CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    rewritten_query TEXT,
                    retrieval_mode TEXT,
                    insufficient_context INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
                """
            )
            conn.commit()

    # ------------------------------------------------------------------------
    # Document Operations
    # ------------------------------------------------------------------------

    def save_document(self, doc: DocumentMetadata) -> None:
        with self._lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    document_id, filename, file_hash, file_size_bytes,
                    page_count, chunk_count, status, error_message,
                    chunk_size, chunk_overlap, uploaded_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.document_id,
                    doc.filename,
                    doc.file_hash,
                    doc.file_size_bytes,
                    doc.page_count,
                    doc.chunk_count,
                    doc.status,
                    doc.error_message,
                    doc.chunk_size,
                    doc.chunk_overlap,
                    doc.uploaded_at,
                    json.dumps(doc.metadata),
                ),
            )
            conn.commit()

    def _row_to_document(self, row: sqlite3.Row) -> DocumentMetadata:
        return DocumentMetadata(
            document_id=row["document_id"],
            filename=row["filename"],
            file_hash=row["file_hash"],
            file_size_bytes=row["file_size_bytes"],
            page_count=row["page_count"],
            chunk_count=row["chunk_count"],
            status=row["status"],
            error_message=row["error_message"],
            chunk_size=row["chunk_size"],
            chunk_overlap=row["chunk_overlap"],
            uploaded_at=row["uploaded_at"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
            return self._row_to_document(row) if row else None

    def get_document_by_hash(self, file_hash: str) -> Optional[DocumentMetadata]:
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ? AND status = 'indexed' LIMIT 1",
                (file_hash,),
            ).fetchone()
            return self._row_to_document(row) if row else None

    def list_documents(self, status_filter: Optional[str] = None) -> List[DocumentMetadata]:
        with self._lock, self._get_connection() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT * FROM documents WHERE status = ? ORDER BY uploaded_at DESC",
                    (status_filter,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM documents ORDER BY uploaded_at DESC"
                ).fetchall()
            return [self._row_to_document(r) for r in rows]

    def delete_document(self, document_id: str) -> bool:
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            cursor = conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------------
    # Chunk Operations
    # ------------------------------------------------------------------------

    def save_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            return
        with self._lock, self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks (
                    chunk_id, document_id, filename, page_number,
                    chunk_index, text, token_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.filename,
                        c.page_number,
                        c.chunk_index,
                        c.text,
                        c.token_count,
                        json.dumps(c.metadata),
                    )
                    for c in chunks
                ],
            )
            conn.commit()

    def _row_to_chunk(self, row: sqlite3.Row) -> DocumentChunk:
        meta = json.loads(row["metadata_json"] or "{}")
        return DocumentChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            filename=row["filename"],
            page_number=row["page_number"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            token_count=row["token_count"],
            section_title=meta.get("section_title", "General"),
            source_text=meta.get("source_text", row["text"]),
            document_hash=meta.get("document_hash", ""),
            metadata=meta,
        )

    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
                (document_id,),
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    def get_all_chunks(self) -> List[DocumentChunk]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chunks ORDER BY document_id, chunk_index ASC"
            ).fetchall()
            return [self._row_to_chunk(r) for r in rows]

    # ------------------------------------------------------------------------
    # Conversational Session & Message Operations
    # ------------------------------------------------------------------------

    def create_session(
        self, session_id: str, title: str = "New Conversation"
    ) -> ChatSessionSummary:
        now = utc_now_iso()
        with self._lock, self._get_connection() as conn:
            existing = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (session_id, title, now, now),
                )
                conn.commit()
        return self.get_session(session_id)  # type: ignore

    def get_session(self, session_id: str) -> Optional[ChatSessionSummary]:
        with self._lock, self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.message_id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                WHERE s.session_id = ?
                GROUP BY s.session_id
                """,
                (session_id,),
            ).fetchone()
            if not row:
                return None
            return ChatSessionSummary(
                session_id=row["session_id"],
                title=row["title"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                message_count=row["message_count"],
            )

    def list_sessions(self) -> List[ChatSessionSummary]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.message_id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON s.session_id = m.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                """
            ).fetchall()
            return [
                ChatSessionSummary(
                    session_id=r["session_id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    message_count=r["message_count"],
                )
                for r in rows
            ]

    def delete_session(self, session_id: str) -> bool:
        with self._lock, self._get_connection() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def add_message(self, message: ChatMessage) -> None:
        now = utc_now_iso()
        with self._lock, self._get_connection() as conn:
            existing = conn.execute(
                "SELECT session_id, title FROM sessions WHERE session_id = ?",
                (message.session_id,),
            ).fetchone()
            if not existing:
                initial_title = (
                    message.content[:50] + "..."
                    if len(message.content) > 50
                    else message.content
                ) if message.role == "user" else "Conversation"
                conn.execute(
                    "INSERT INTO sessions (session_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (message.session_id, initial_title, now, now),
                )
            else:
                # Update session title on first user message if it's still 'New Conversation'
                if existing["title"] == "New Conversation" and message.role == "user":
                    new_title = (
                        message.content[:50] + "..."
                        if len(message.content) > 50
                        else message.content
                    )
                    conn.execute(
                        "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
                        (new_title, now, message.session_id),
                    )
                else:
                    conn.execute(
                        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                        (now, message.session_id),
                    )

            citations_data = [c.model_dump() for c in message.citations]
            conn.execute(
                """
                INSERT INTO messages (
                    message_id, session_id, role, content,
                    citations_json, rewritten_query, retrieval_mode,
                    insufficient_context, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.message_id,
                    message.session_id,
                    message.role,
                    message.content,
                    json.dumps(citations_data),
                    message.rewritten_query,
                    message.retrieval_mode,
                    1 if message.insufficient_context else 0,
                    message.created_at,
                ),
            )
            conn.commit()

    def get_session_messages(self, session_id: str) -> List[ChatMessage]:
        with self._lock, self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC, rowid ASC",
                (session_id,),
            ).fetchall()
            messages: List[ChatMessage] = []
            for r in rows:
                raw_citations = json.loads(r["citations_json"] or "[]")
                citations = [SourceCitation(**c) for c in raw_citations]
                messages.append(
                    ChatMessage(
                        message_id=r["message_id"],
                        session_id=r["session_id"],
                        role=r["role"],
                        content=r["content"],
                        citations=citations,
                        rewritten_query=r["rewritten_query"],
                        retrieval_mode=r["retrieval_mode"],
                        insufficient_context=bool(r["insufficient_context"]),
                        created_at=r["created_at"],
                    )
                )
            return messages

    def count_messages(self) -> int:
        with self._lock, self._get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM messages").fetchone()
            return int(row["cnt"]) if row else 0
