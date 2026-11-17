import hashlib
import re
from typing import List, Optional
import numpy as np
from backend.app.core.config import settings
from backend.app.core.logging import logger


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


class EmbeddingService:
    """
    Generates normalized vector embeddings for document chunks and user queries
    using HuggingFace sentence-transformers (default: sentence-transformers/all-MiniLM-L6-v2).
    """

    def __init__(
        self,
        model_name: str = settings.EMBEDDING_MODEL_NAME,
        device: str = settings.EMBEDDING_DEVICE,
        normalize: bool = settings.NORMALIZE_EMBEDDINGS,
        use_fallback: bool = settings.USE_HASH_EMBEDDINGS_FALLBACK,
    ):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.use_fallback = use_fallback
        self.dimension: int = 384  # all-MiniLM-L6-v2 dimension
        self._model = None
        self._query_cache: dict[str, np.ndarray] = {}

    def _load_model(self) -> None:
        if self._model is not None or self.use_fallback:
            return
        try:
            logger.info("Loading embedding model '%s' on device '%s'...", self.model_name, self.device)
            from pathlib import Path
            from sentence_transformers import SentenceTransformer

            # Resolve local HuggingFace snapshot directory if cached on disk to avoid network roundtrips
            cache_slug = f"models--{self.model_name.replace('/', '--')}"
            snap_dir = Path.home() / ".cache" / "huggingface" / "hub" / cache_slug / "snapshots"
            snaps = sorted(snap_dir.glob("*")) if snap_dir.exists() else []
            model_target = str(snaps[-1]) if snaps else self.model_name

            self._model = SentenceTransformer(model_target, device=self.device)

            if hasattr(self._model, "get_embedding_dimension"):
                dim = self._model.get_embedding_dimension()
            else:
                dim = self._model.get_sentence_embedding_dimension()
            if dim:
                self.dimension = int(dim)
            logger.info("Loaded embedding model '%s' (dim=%d).", self.model_name, self.dimension)
        except Exception as exc:
            logger.warning(
                "Could not load SentenceTransformer '%s' (%s). Falling back to deterministic feature hashing embeddings.",
                self.model_name,
                str(exc),
            )
            self.use_fallback = True

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (vectors / norms).astype(np.float32)

    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """
        Deterministic semantic-token feature hashing embedding (dim=384) used only
        when USE_HASH_EMBEDDINGS_FALLBACK=True or in offline test environments.
        Preserves strong cosine similarity between texts sharing words/bigrams.
        """
        mat = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            if not tokens:
                mat[row_idx, 0] = 1.0
                continue
            # Unigrams + bigrams
            features = list(tokens) + [
                f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)
            ]
            for feat in features:
                digest = hashlib.md5(feat.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if (digest[4] & 1) == 0 else -1.0
                mat[row_idx, idx] += sign
        return self._normalize_vectors(mat)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Convert a list of document chunk strings into a 2D float32 numpy array (N, D).
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        cleaned_texts = [t.strip() if t and t.strip() else " " for t in texts]
        self._load_model()

        if self.use_fallback or self._model is None:
            return self._fallback_encode(cleaned_texts)

        try:
            embeddings = self._model.encode(
                cleaned_texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=self.normalize,
            )
            arr = np.asarray(embeddings, dtype=np.float32)
            if self.normalize:
                arr = self._normalize_vectors(arr)
            return arr
        except Exception as exc:
            logger.error("Error generating chunk embeddings: %s", str(exc))
            raise EmbeddingError(f"Failed to generate embeddings: {exc}") from exc

    def embed_query(self, query: str) -> np.ndarray:
        """
        Convert a single user query string into a 1D float32 numpy vector (D,).
        Uses an in-memory LRU-style cache to avoid redundant encoding of identical queries.
        """
        if not query or not query.strip():
            raise EmbeddingError("Query string cannot be empty for embedding generation.")
        cleaned = query.strip()
        if cleaned in self._query_cache:
            return self._query_cache[cleaned]
        matrix = self.embed_texts([cleaned])
        vec = matrix[0]
        if len(self._query_cache) >= 256:
            self._query_cache.clear()
        self._query_cache[cleaned] = vec
        return vec
