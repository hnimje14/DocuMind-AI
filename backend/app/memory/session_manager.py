import uuid
from typing import List, Optional
from backend.app.database.db import BaseRepository
from backend.app.schemas.models import (
    ChatHistoryResponse,
    ChatMessage,
    ChatSessionSummary,
)


class SessionMemoryManager:
    """
    Manages isolated chat sessions, message history retrieval, session creation,
    and conversation clearing. Ensures zero cross-session leakage.
    """

    def __init__(self, repository: BaseRepository):
        self.repository = repository

    def create_session(
        self, session_id: Optional[str] = None, title: str = "New Conversation"
    ) -> ChatSessionSummary:
        sid = session_id or f"sess_{uuid.uuid4().hex[:10]}"
        return self.repository.create_session(session_id=sid, title=title)

    def list_sessions(self) -> List[ChatSessionSummary]:
        return self.repository.list_sessions()

    def get_history(self, session_id: str) -> Optional[ChatHistoryResponse]:
        session = self.repository.get_session(session_id)
        if not session:
            return None
        messages = self.repository.get_session_messages(session_id)
        return ChatHistoryResponse(
            session_id=session.session_id,
            title=session.title,
            messages=messages,
        )

    def clear_session(self, session_id: str) -> bool:
        return self.repository.delete_session(session_id)
