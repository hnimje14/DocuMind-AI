from fastapi import APIRouter, Depends
from backend.app.api.dependencies import ServiceContainer, get_container
from backend.app.core.config import settings
from backend.app.schemas.models import SystemStatsResponse

router = APIRouter(tags=["Health & Stats"])


@router.get("/health", response_model=SystemStatsResponse)
def get_health_and_stats(
    container: ServiceContainer = Depends(get_container),
) -> SystemStatsResponse:
    """Return application health status, LLM/embedding config, and dashboard statistics."""
    docs = container.repository.list_documents()
    indexed_docs = [d for d in docs if d.status == "indexed"]
    total_chunks = sum(d.chunk_count for d in indexed_docs)
    sessions = container.repository.list_sessions()
    total_messages = container.repository.count_messages()

    active_provider = (
        container.llm_client.provider
        if container.llm_client.is_live_api_configured()
        else "mock (grounded-extractive)"
    )
    active_model = (
        container.llm_client.model
        if container.llm_client.is_live_api_configured()
        else "extractive-rag-v1"
    )

    return SystemStatsResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        llm_provider=active_provider,
        llm_model=active_model,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        total_documents=len(indexed_docs),
        total_chunks=total_chunks,
        vector_index_size=container.vector_store.count(),
        total_sessions=len(sessions),
        total_messages=total_messages,
        recent_documents=docs[:10],
    )
