from fastapi import APIRouter, Depends
from backend.app.api.dependencies import ServiceContainer, get_container
from backend.app.schemas.models import (
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)

router = APIRouter(prefix="/retrieval", tags=["Retrieval Explorer"])


@router.post("/search", response_model=RetrievalSearchResponse)
def search_chunks(
    request: RetrievalSearchRequest,
    container: ServiceContainer = Depends(get_container),
) -> RetrievalSearchResponse:
    """Execute semantic, MMR, or hybrid chunk retrieval and return ranked chunks with scores."""
    effective_query, results = container.retriever.search(
        query=request.query,
        mode=request.mode,
        top_k=request.top_k,
        similarity_threshold=request.similarity_threshold,
        document_ids=request.document_ids,
        use_query_rewriting=request.use_query_rewriting,
        use_context_compression=request.use_context_compression,
        mmr_lambda=request.mmr_lambda,
    )
    return RetrievalSearchResponse(
        original_query=request.query,
        effective_query=effective_query,
        mode=request.mode,
        total_found=len(results),
        results=results,
    )
