from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from backend.app.api.dependencies import ServiceContainer, get_container
from backend.app.schemas.models import (
    ChatAskRequest,
    ChatAskResponse,
    ChatHistoryResponse,
    ChatSessionSummary,
)

router = APIRouter(prefix="/chat", tags=["Chat & Conversational Memory"])


@router.post("/ask", response_model=ChatAskResponse)
def ask_question(
    request: ChatAskRequest,
    container: ServiceContainer = Depends(get_container),
) -> ChatAskResponse:
    """Answer a user question grounded in retrieved document chunks and record conversation history."""
    return container.rag_pipeline.answer_question(request)


@router.get("/sessions", response_model=List[ChatSessionSummary])
def list_chat_sessions(
    container: ServiceContainer = Depends(get_container),
) -> List[ChatSessionSummary]:
    """List all chat sessions ordered by most recent update."""
    return container.session_manager.list_sessions()


@router.post("/sessions", response_model=ChatSessionSummary, status_code=status.HTTP_201_CREATED)
def create_chat_session(
    title: Optional[str] = "New Conversation",
    container: ServiceContainer = Depends(get_container),
) -> ChatSessionSummary:
    """Create a new isolated conversation session."""
    return container.session_manager.create_session(title=title or "New Conversation")


@router.get("/history/{session_id}", response_model=ChatHistoryResponse)
def get_chat_history(
    session_id: str,
    container: ServiceContainer = Depends(get_container),
) -> ChatHistoryResponse:
    """Retrieve message history and citations for a specific conversation session."""
    history = container.session_manager.get_history(session_id)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found.",
        )
    return history


@router.delete("/history/{session_id}")
def delete_chat_history(
    session_id: str,
    container: ServiceContainer = Depends(get_container),
):
    """Clear and delete a specific conversation session."""
    deleted = container.session_manager.clear_session(session_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found.",
        )
    return {
        "status": "deleted",
        "session_id": session_id,
        "message": f"Chat session '{session_id}' cleared.",
    }
