from functools import lru_cache
from backend.app.core.config import settings
from backend.app.database.db import SQLiteRepository
from backend.app.embeddings.embedder import EmbeddingService
from backend.app.evaluation.evaluator import EvaluationEngine
from backend.app.generation.llm_client import LLMClient
from backend.app.generation.rag_pipeline import RAGPipeline
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.memory.session_manager import SessionMemoryManager
from backend.app.retrieval.retriever import RetrievalService
from backend.app.vectorstore.faiss_store import FAISSVectorStore


class ServiceContainer:
    """Singleton service container wiring repositories, embeddings, vector store, and pipelines."""

    def __init__(self):
        self.repository = SQLiteRepository(db_path=settings.SQLITE_DB_PATH)
        self.embedder = EmbeddingService(
            model_name=settings.EMBEDDING_MODEL_NAME,
            device=settings.EMBEDDING_DEVICE,
            normalize=settings.NORMALIZE_EMBEDDINGS,
            use_fallback=settings.USE_HASH_EMBEDDINGS_FALLBACK,
        )
        self.vector_store = FAISSVectorStore(
            dimension=self.embedder.dimension,
            storage_dir=settings.VECTORSTORE_DIR,
            auto_load=True,
        )
        self.ingestion_pipeline = IngestionPipeline(
            repository=self.repository,
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        self.retriever = RetrievalService(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        self.llm_client = LLMClient()
        self.rag_pipeline = RAGPipeline(
            repository=self.repository,
            retriever=self.retriever,
            llm_client=self.llm_client,
        )
        self.session_manager = SessionMemoryManager(repository=self.repository)
        self.evaluator = EvaluationEngine(
            repository=self.repository,
            embedder=self.embedder,
            retriever=self.retriever,
            ingestion_pipeline=self.ingestion_pipeline,
            llm_client=self.llm_client,
        )


@lru_cache(maxsize=1)
def get_container() -> ServiceContainer:
    return ServiceContainer()
