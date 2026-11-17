from typing import List, Optional, Tuple
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from backend.app.api.dependencies import ServiceContainer, get_container
from backend.app.core.config import settings
from backend.app.schemas.models import (
    DocumentDetailResponse,
    DocumentMetadata,
    DocumentUploadResponse,
)

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="One or more PDF files to upload"),
    chunk_size: int = Form(default=settings.DEFAULT_CHUNK_SIZE, ge=100, le=4000),
    chunk_overlap: int = Form(default=settings.DEFAULT_CHUNK_OVERLAP, ge=0, le=1000),
    container: ServiceContainer = Depends(get_container),
) -> DocumentUploadResponse:
    """Upload, validate, extract, chunk, and index one or more PDF documents."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files were provided in the upload request.",
        )
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="chunk_overlap must be strictly less than chunk_size.",
        )

    file_tuples: List[Tuple[str, str, bytes]] = []
    for upload in files:
        content = await upload.read()
        file_tuples.append(
            (
                upload.filename or "unnamed.pdf",
                upload.content_type or "application/pdf",
                content,
            )
        )

    return container.ingestion_pipeline.ingest_multiple_files(
        files=file_tuples,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@router.get("", response_model=List[DocumentMetadata])
def list_documents(
    status_filter: Optional[str] = None,
    container: ServiceContainer = Depends(get_container),
) -> List[DocumentMetadata]:
    """List all uploaded documents and their processing status."""
    return container.repository.list_documents(status_filter=status_filter)


@router.get("/{document_id}", response_model=DocumentDetailResponse)
def get_document_details(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
) -> DocumentDetailResponse:
    """Retrieve metadata and all extracted chunks for a specific document."""
    doc = container.repository.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    chunks = container.repository.get_document_chunks(document_id)
    return DocumentDetailResponse(document=doc, chunks=chunks)


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    container: ServiceContainer = Depends(get_container),
):
    """Delete a document, its chunks in SQLite, and its vectors in FAISS."""
    deleted = container.ingestion_pipeline.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return {
        "status": "deleted",
        "document_id": document_id,
        "message": f"Document '{document_id}' and its vectors were successfully removed.",
    }


@router.post("/reindex")
def reindex_all_documents(
    chunk_size: int = settings.DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = settings.DEFAULT_CHUNK_OVERLAP,
    container: ServiceContainer = Depends(get_container),
):
    """Reprocess all indexed documents with semantic-aware chunking and rebuild the FAISS index."""
    total_chunks = container.ingestion_pipeline.reindex_all_documents(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return {
        "status": "reindexed",
        "total_chunks": total_chunks,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
