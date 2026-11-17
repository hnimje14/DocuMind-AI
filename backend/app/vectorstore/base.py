import abc
from typing import List, Optional, Tuple
import numpy as np
from backend.app.schemas.models import DocumentChunk, RetrievedChunkResult


class VectorStoreBase(abc.ABC):
    """
    Modular vector database interface.
    Implementations (e.g., FAISSVectorStore, ChromaVectorStore) must adhere to this contract.
    """

    @abc.abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> int:
        """Insert document chunks and their corresponding embeddings into the index."""

    @abc.abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunkResult]:
        """Perform similarity search with optional metadata filtering."""

    @abc.abstractmethod
    def get_all_chunks_and_vectors(
        self, document_ids: Optional[List[str]] = None
    ) -> Tuple[List[DocumentChunk], np.ndarray]:
        """Return stored chunks and their vectors (used for MMR and hybrid search)."""

    @abc.abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Delete all vectors belonging to a document and rebuild index."""

    @abc.abstractmethod
    def rebuild(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Rebuild the entire vector index from scratch."""

    @abc.abstractmethod
    def save(self) -> None:
        """Persist index and metadata mapping to disk."""

    @abc.abstractmethod
    def load(self) -> bool:
        """Load index and metadata mapping from disk if present."""

    @abc.abstractmethod
    def count(self) -> int:
        """Return total number of indexed vectors."""
