import json
import threading
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import faiss
import numpy as np
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.schemas.models import DocumentChunk, RetrievedChunkResult
from backend.app.vectorstore.base import VectorStoreBase


class FAISSVectorStore(VectorStoreBase):
    """
    FAISS-backed vector store using IndexFlatIP on L2-normalized vectors
    (exact cosine similarity). Maintains 1-to-1 metadata alignment,
    supports document deletion, metadata filtering, rebuilding, and disk persistence.
    """

    def __init__(
        self,
        dimension: int = 384,
        storage_dir: str = settings.VECTORSTORE_DIR,
        auto_load: bool = True,
    ):
        self.dimension = dimension
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.storage_dir / "faiss_index.bin"
        self.meta_path = self.storage_dir / "faiss_metadata.json"

        self._lock = threading.RLock()
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self._chunks: List[DocumentChunk] = []
        self._vectors: np.ndarray = np.empty((0, self.dimension), dtype=np.float32)
        self._chunk_id_set: Set[str] = set()

        if auto_load:
            self.load()

    @staticmethod
    def _ensure_normalized(vectors: np.ndarray) -> np.ndarray:
        if vectors.size == 0:
            return vectors.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return np.ascontiguousarray((vectors / norms).astype(np.float32))

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> int:
        """Add new chunks and vectors, skipping any chunk_id already indexed."""
        if not chunks or embeddings.size == 0:
            return 0
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between chunks count ({len(chunks)}) and embeddings rows ({embeddings.shape[0]})."
            )

        with self._lock:
            if embeddings.shape[1] != self.dimension:
                if self._index.ntotal == 0:
                    self.dimension = embeddings.shape[1]
                    self._index = faiss.IndexFlatIP(self.dimension)
                    self._vectors = np.empty((0, self.dimension), dtype=np.float32)
                else:
                    raise ValueError(
                        f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}."
                    )

            new_chunks: List[DocumentChunk] = []
            new_rows: List[int] = []
            for idx, chunk in enumerate(chunks):
                if chunk.chunk_id not in self._chunk_id_set:
                    new_chunks.append(chunk)
                    new_rows.append(idx)
                    self._chunk_id_set.add(chunk.chunk_id)

            if not new_chunks:
                return 0

            norm_vecs = self._ensure_normalized(embeddings[new_rows])
            self._index.add(norm_vecs)
            self._chunks.extend(new_chunks)
            if self._vectors.size == 0:
                self._vectors = norm_vecs
            else:
                self._vectors = np.vstack([self._vectors, norm_vecs])

            self.save()
            return len(new_chunks)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0,
        document_ids: Optional[List[str]] = None,
    ) -> List[RetrievedChunkResult]:
        """
        Cosine similarity search across indexed chunks with optional document_ids filter
        and minimum similarity threshold. Never returns irrelevant chunks below threshold.
        """
        with self._lock:
            if self._index.ntotal == 0 or not self._chunks:
                return []

            q_vec = query_embedding.reshape(1, -1).astype(np.float32)
            q_vec = self._ensure_normalized(q_vec)

            # Search all vectors when filtering or up to ntotal so metadata filter never starves results
            search_k = self._index.ntotal if document_ids else min(max(top_k * 3, top_k), self._index.ntotal)
            scores, indices = self._index.search(q_vec, search_k)

            doc_filter_set = set(document_ids) if document_ids else None
            results: List[RetrievedChunkResult] = []

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._chunks):
                    continue
                sim_score = float(score)
                if sim_score < similarity_threshold:
                    continue

                chunk = self._chunks[idx]
                if doc_filter_set is not None and chunk.document_id not in doc_filter_set:
                    continue

                results.append(
                    RetrievedChunkResult(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        filename=chunk.filename,
                        page_number=chunk.page_number,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        score=round(sim_score, 4),
                        retrieval_method="semantic",
                        metadata=chunk.metadata,
                    )
                )
                if len(results) >= top_k:
                    break

            return results

    def get_all_chunks_and_vectors(
        self, document_ids: Optional[List[str]] = None
    ) -> Tuple[List[DocumentChunk], np.ndarray]:
        with self._lock:
            if not self._chunks or self._vectors.size == 0:
                return [], np.empty((0, self.dimension), dtype=np.float32)

            if not document_ids:
                return list(self._chunks), self._vectors.copy()

            allowed = set(document_ids)
            keep_indices = [
                i for i, c in enumerate(self._chunks) if c.document_id in allowed
            ]
            if not keep_indices:
                return [], np.empty((0, self.dimension), dtype=np.float32)

            filtered_chunks = [self._chunks[i] for i in keep_indices]
            filtered_vectors = self._vectors[keep_indices].copy()
            return filtered_chunks, filtered_vectors

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks/vectors for a document_id and rebuild the FAISS index."""
        with self._lock:
            if not self._chunks:
                return 0

            keep_indices = [
                i for i, c in enumerate(self._chunks) if c.document_id != document_id
            ]
            removed_count = len(self._chunks) - len(keep_indices)
            if removed_count == 0:
                return 0

            remaining_chunks = [self._chunks[i] for i in keep_indices]
            remaining_vectors = (
                self._vectors[keep_indices]
                if keep_indices
                else np.empty((0, self.dimension), dtype=np.float32)
            )
            self.rebuild(remaining_chunks, remaining_vectors)
            return removed_count

    def rebuild(self, chunks: List[DocumentChunk], embeddings: np.ndarray) -> None:
        """Rebuild the FAISS index from provided chunks and embeddings."""
        with self._lock:
            self._index = faiss.IndexFlatIP(self.dimension)
            self._chunks = []
            self._vectors = np.empty((0, self.dimension), dtype=np.float32)
            self._chunk_id_set = set()

            if chunks and embeddings.size > 0:
                norm_vecs = self._ensure_normalized(embeddings)
                self._index.add(norm_vecs)
                self._chunks = list(chunks)
                self._vectors = norm_vecs
                self._chunk_id_set = {c.chunk_id for c in chunks}

            self.save()

    def save(self) -> None:
        """Persist FAISS index and chunk metadata to disk."""
        with self._lock:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(self.index_path))
            meta_payload = {
                "dimension": self.dimension,
                "chunks": [c.model_dump() for c in self._chunks],
            }
            self.meta_path.write_text(
                json.dumps(meta_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    def load(self) -> bool:
        """Load FAISS index and chunk metadata from disk if available."""
        with self._lock:
            if not self.index_path.exists() or not self.meta_path.exists():
                return False
            try:
                loaded_index = faiss.read_index(str(self.index_path))
                raw_meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
                loaded_chunks = [
                    DocumentChunk(**item) for item in raw_meta.get("chunks", [])
                ]
                if loaded_index.ntotal != len(loaded_chunks):
                    logger.warning(
                        "FAISS index count (%d) != metadata count (%d); skipping corrupt load.",
                        loaded_index.ntotal,
                        len(loaded_chunks),
                    )
                    return False

                self.dimension = int(raw_meta.get("dimension", loaded_index.d))
                self._index = loaded_index
                self._chunks = loaded_chunks
                self._chunk_id_set = {c.chunk_id for c in loaded_chunks}

                if loaded_index.ntotal > 0:
                    reconstructed = np.vstack(
                        [loaded_index.reconstruct(i) for i in range(loaded_index.ntotal)]
                    ).astype(np.float32)
                    self._vectors = reconstructed
                else:
                    self._vectors = np.empty((0, self.dimension), dtype=np.float32)

                logger.info(
                    "Loaded FAISS index from disk with %d vectors.", self._index.ntotal
                )
                return True
            except Exception as exc:
                logger.error("Failed to load FAISS index from disk: %s", str(exc))
                return False

    def count(self) -> int:
        with self._lock:
            return int(self._index.ntotal)
