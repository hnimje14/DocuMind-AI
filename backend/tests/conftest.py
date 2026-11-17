import io
from pathlib import Path
from typing import Tuple
import pymupdf as fitz
import pytest
from reportlab.lib.pagesizes import letter
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from backend.app.database.db import SQLiteRepository
from backend.app.embeddings.embedder import EmbeddingService
from backend.app.evaluation.evaluator import EvaluationEngine
from backend.app.generation.llm_client import LLMClient
from backend.app.generation.rag_pipeline import RAGPipeline
from backend.app.ingestion.pipeline import IngestionPipeline
from backend.app.memory.session_manager import SessionMemoryManager
from backend.app.retrieval.retriever import RetrievalService
from backend.app.vectorstore.faiss_store import FAISSVectorStore


def make_test_pdf_bytes(pages_text: list[str]) -> bytes:
    """Create a valid PDF in memory with the given text on each page."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, title="Test PDF Document")
    styles = getSampleStyleSheet()
    story = []
    for idx, text in enumerate(pages_text):
        if text.strip():
            for line in text.split("\n"):
                if line.strip():
                    story.append(Paragraph(line.strip(), styles["BodyText"]))
        if idx < len(pages_text) - 1:
            story.append(PageBreak())
    if not story:
        # Blank page PDF using PyMuPDF
        pdf = fitz.open()
        pdf.new_page()
        raw = pdf.tobytes()
        pdf.close()
        return raw
    doc.build(story)
    return buf.getvalue()


@pytest.fixture
def temp_services(tmp_path: Path):
    """Provide isolated SQLite, EmbeddingService (fast deterministic fallback), FAISS, and RAG services."""
    db_path = str(tmp_path / "test_rag.db")
    vs_dir = str(tmp_path / "test_vectorstore")

    repo = SQLiteRepository(db_path=db_path)
    embedder = EmbeddingService(use_fallback=True)
    vector_store = FAISSVectorStore(
        dimension=embedder.dimension,
        storage_dir=vs_dir,
        auto_load=False,
    )
    ingestion = IngestionPipeline(
        repository=repo,
        embedder=embedder,
        vector_store=vector_store,
        upload_dir=str(tmp_path / "test_uploads"),
    )
    retriever = RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
    )
    llm_client = LLMClient(provider="mock")
    rag = RAGPipeline(
        repository=repo,
        retriever=retriever,
        llm_client=llm_client,
    )
    memory = SessionMemoryManager(repository=repo)
    evaluator = EvaluationEngine(
        repository=repo,
        embedder=embedder,
        retriever=retriever,
        ingestion_pipeline=ingestion,
        llm_client=llm_client,
        evaluation_dir=str(tmp_path / "test_eval"),
    )
    return {
        "repo": repo,
        "embedder": embedder,
        "vector_store": vector_store,
        "ingestion": ingestion,
        "retriever": retriever,
        "llm_client": llm_client,
        "rag": rag,
        "memory": memory,
        "evaluator": evaluator,
    }
