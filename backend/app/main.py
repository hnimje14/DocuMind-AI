from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.app.api import (
    routes_chat,
    routes_documents,
    routes_evaluation,
    routes_health,
    routes_retrieval,
)
from backend.app.api.dependencies import get_container
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.security import RateLimitMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    container = get_container()
    if not settings.USE_HASH_EMBEDDINGS_FALLBACK:
        container.embedder._load_model()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "Production-oriented Retrieval-Augmented Generation (RAG) API with "
            "multi-PDF ingestion, FAISS vector search, MMR & Hybrid retrieval, "
            "verified page citations, isolated session memory, and automated evaluation."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=settings.RATE_LIMIT_REQUESTS_PER_MINUTE,
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled exception on %s %s: %s", request.method, request.url.path, str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal server error: {str(exc)}"},
        )

    app.include_router(routes_health.router)
    app.include_router(routes_documents.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_retrieval.router)
    app.include_router(routes_evaluation.router)

    return app


app = create_app()
