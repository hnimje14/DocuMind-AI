from typing import List, Optional, Tuple
from backend.app.chunking.chunker import DocumentChunker
from backend.app.core.config import settings
from backend.app.core.logging import logger
from backend.app.core.security import (
    compute_sha256,
    sanitize_filename,
    validate_pdf_upload,
)
from backend.app.database.db import BaseRepository, utc_now_iso
from backend.app.embeddings.embedder import EmbeddingService
from backend.app.ingestion.extractor import (
    PDFExtractionError,
    PDFExtractor,
    generate_document_id,
)
from backend.app.schemas.models import (
    DocumentMetadata,
    DocumentUploadResponse,
    DocumentUploadResult,
)
from backend.app.vectorstore.base import VectorStoreBase


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline:
    PDF Upload -> Validation -> Duplicate Hash Check -> Extraction -> Cleaning ->
    Chunking -> Embedding Generation -> FAISS Vector Indexing + SQLite Persistence.
    """

    def __init__(
        self,
        repository: BaseRepository,
        embedder: EmbeddingService,
        vector_store: VectorStoreBase,
        extractor: Optional[PDFExtractor] = None,
        upload_dir: Optional[str] = None,
    ):
        self.repository = repository
        self.embedder = embedder
        self.vector_store = vector_store
        self.extractor = extractor or PDFExtractor()
        self.upload_dir = upload_dir or settings.UPLOAD_DIR

    def ingest_single_file(
        self,
        filename: str,
        content_type: str,
        file_bytes: bytes,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ) -> DocumentUploadResult:
        """Process and index a single uploaded PDF file."""
        safe_name = sanitize_filename(filename)
        file_size = len(file_bytes) if file_bytes else 0

        # 1. Validate extension, MIME, size, and PDF magic bytes
        is_valid, validation_msg = validate_pdf_upload(
            filename=filename,
            content_type=content_type,
            file_bytes=file_bytes,
            max_file_size_mb=settings.MAX_FILE_SIZE_MB,
        )
        if not is_valid:
            logger.warning("Upload validation failed for '%s': %s", filename, validation_msg)
            return DocumentUploadResult(
                filename=safe_name,
                document_id=None,
                status="failed",
                message=validation_msg,
                page_count=0,
                chunk_count=0,
                file_size_bytes=file_size,
            )

        # 2. Check SHA-256 hash to prevent duplicate ingestion and avoid regenerating embeddings
        file_hash = compute_sha256(file_bytes)
        existing_doc = self.repository.get_document_by_hash(file_hash)
        if existing_doc is not None:
            logger.info(
                "Skipping duplicate upload '%s' (matches indexed document_id='%s').",
                safe_name,
                existing_doc.document_id,
            )
            return DocumentUploadResult(
                filename=safe_name,
                document_id=existing_doc.document_id,
                status="duplicate",
                message=(
                    f"Duplicate file detected: already indexed as '{existing_doc.filename}' "
                    f"(ID: {existing_doc.document_id}). Reusing existing embeddings."
                ),
                page_count=existing_doc.page_count,
                chunk_count=existing_doc.chunk_count,
                file_size_bytes=file_size,
            )

        document_id = generate_document_id()

        # 3. Extract text and page metadata
        try:
            pages, pdf_meta = self.extractor.extract_from_bytes(
                file_bytes=file_bytes, filename=safe_name
            )
        except PDFExtractionError as exc:
            err_msg = str(exc)
            logger.warning("Extraction failed for '%s': %s", safe_name, err_msg)
            failed_doc = DocumentMetadata(
                document_id=document_id,
                filename=safe_name,
                file_hash=file_hash,
                file_size_bytes=file_size,
                page_count=0,
                chunk_count=0,
                status="failed",
                error_message=err_msg,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                uploaded_at=utc_now_iso(),
                metadata={},
            )
            self.repository.save_document(failed_doc)
            return DocumentUploadResult(
                filename=safe_name,
                document_id=document_id,
                status="failed",
                message=err_msg,
                page_count=0,
                chunk_count=0,
                file_size_bytes=file_size,
            )

        # 4. Chunk document pages
        try:
            pdf_meta["file_hash"] = file_hash
            chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            chunks = chunker.chunk_document_pages(
                document_id=document_id,
                filename=safe_name,
                pages=pages,
                extra_metadata=pdf_meta,
                document_hash=file_hash,
            )
            if not chunks:
                raise PDFExtractionError(f"No valid text chunks could be created from '{safe_name}'.")
        except Exception as exc:
            err_msg = f"Chunking error: {str(exc)}"
            logger.error("Chunking failed for '%s': %s", safe_name, err_msg)
            return DocumentUploadResult(
                filename=safe_name,
                document_id=document_id,
                status="failed",
                message=err_msg,
                page_count=len(pages),
                chunk_count=0,
                file_size_bytes=file_size,
            )

        # 5. Generate embeddings & store in FAISS + SQLite
        try:
            chunk_texts = [c.text for c in chunks]
            embeddings = self.embedder.embed_texts(chunk_texts)
            self.vector_store.add_chunks(chunks, embeddings)

            # Save physical file on disk
            saved_path = self.extractor.save_uploaded_file(
                file_bytes=file_bytes,
                document_id=document_id,
                safe_name=safe_name,
                upload_dir=self.upload_dir,
            )
            pdf_meta["stored_path"] = str(saved_path)

            doc_meta = DocumentMetadata(
                document_id=document_id,
                filename=safe_name,
                file_hash=file_hash,
                file_size_bytes=file_size,
                page_count=len(pages),
                chunk_count=len(chunks),
                status="indexed",
                error_message=None,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                uploaded_at=utc_now_iso(),
                metadata=pdf_meta,
            )
            self.repository.save_document(doc_meta)
            self.repository.save_chunks(chunks)

            logger.info(
                "Successfully indexed '%s' (id=%s, pages=%d, chunks=%d).",
                safe_name,
                document_id,
                len(pages),
                len(chunks),
            )
            return DocumentUploadResult(
                filename=safe_name,
                document_id=document_id,
                status="indexed",
                message=f"Successfully extracted {len(pages)} pages and indexed {len(chunks)} chunks.",
                page_count=len(pages),
                chunk_count=len(chunks),
                file_size_bytes=file_size,
            )
        except Exception as exc:
            err_msg = f"Indexing error: {str(exc)}"
            logger.error("Indexing failed for '%s': %s", safe_name, err_msg)
            return DocumentUploadResult(
                filename=safe_name,
                document_id=document_id,
                status="failed",
                message=err_msg,
                page_count=len(pages),
                chunk_count=0,
                file_size_bytes=file_size,
            )

    def ingest_multiple_files(
        self,
        files: List[Tuple[str, str, bytes]],
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ) -> DocumentUploadResponse:
        """
        Process a batch of uploaded files.
        Each item in `files` is (filename, content_type, file_bytes).
        """
        results: List[DocumentUploadResult] = []
        indexed_count = 0
        duplicate_count = 0
        failed_count = 0

        for filename, content_type, file_bytes in files:
            res = self.ingest_single_file(
                filename=filename,
                content_type=content_type,
                file_bytes=file_bytes,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            results.append(res)
            if res.status == "indexed":
                indexed_count += 1
            elif res.status == "duplicate":
                duplicate_count += 1
            else:
                failed_count += 1

        return DocumentUploadResponse(
            total_files=len(files),
            indexed_count=indexed_count,
            duplicate_count=duplicate_count,
            failed_count=failed_count,
            results=results,
        )

    def delete_document(self, document_id: str) -> bool:
        """Delete a document from SQLite, FAISS vector store, and disk."""
        doc = self.repository.get_document(document_id)
        if not doc:
            return False

        self.vector_store.delete_document(document_id)
        deleted_db = self.repository.delete_document(document_id)

        stored_path = doc.metadata.get("stored_path")
        if stored_path:
            try:
                from pathlib import Path
                p = Path(stored_path)
                if p.exists():
                    p.unlink()
            except Exception as exc:
                logger.warning("Failed to remove uploaded file from disk: %s", str(exc))

        return deleted_db

    def reindex_all_documents(
        self,
        chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    ) -> int:
        """
        Reprocess all existing indexed documents from their stored PDF files on disk,
        regenerate semantic chunks and embeddings, and rebuild the FAISS vector index
        without duplicate vectors or obsolete chunks.
        """
        from pathlib import Path
        from backend.app.schemas.models import DocumentChunk

        docs = self.repository.list_documents(status_filter="indexed")
        all_new_chunks: List[DocumentChunk] = []
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for doc in docs:
            stored_path_str = doc.metadata.get("stored_path")
            pdf_path = Path(stored_path_str) if stored_path_str else None
            if not pdf_path or not pdf_path.exists():
                candidate = Path(self.upload_dir) / f"{doc.document_id}_{doc.filename}"
                if candidate.exists():
                    pdf_path = candidate

            if not pdf_path or not pdf_path.exists():
                logger.warning("Stored PDF not found for '%s'; skipping reindex.", doc.document_id)
                continue

            file_bytes = pdf_path.read_bytes()
            pages, pdf_meta = self.extractor.extract_from_bytes(file_bytes, doc.filename)
            pdf_meta["stored_path"] = str(pdf_path)
            pdf_meta["file_hash"] = doc.file_hash

            new_chunks = chunker.chunk_document_pages(
                document_id=doc.document_id,
                filename=doc.filename,
                pages=pages,
                extra_metadata=pdf_meta,
                document_hash=doc.file_hash,
            )

            # Remove old chunks for this document in SQLite and insert new ones
            self.repository.delete_document(doc.document_id)
            updated_doc = doc.model_copy(
                update={
                    "page_count": len(pages),
                    "chunk_count": len(new_chunks),
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "metadata": pdf_meta,
                }
            )
            self.repository.save_document(updated_doc)
            self.repository.save_chunks(new_chunks)
            all_new_chunks.extend(new_chunks)

        if all_new_chunks:
            embeddings = self.embedder.embed_texts([c.text for c in all_new_chunks])
            self.vector_store.rebuild(all_new_chunks, embeddings)
        else:
            import numpy as np
            self.vector_store.rebuild([], np.empty((0, self.embedder.dimension), dtype=np.float32))

        logger.info(
            "Reindexed %d document(s) into %d semantic chunks (size=%d, overlap=%d).",
            len(docs),
            len(all_new_chunks),
            chunk_size,
            chunk_overlap,
        )
        return len(all_new_chunks)
