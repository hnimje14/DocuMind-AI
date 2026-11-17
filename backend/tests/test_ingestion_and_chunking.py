import pytest
from backend.app.chunking.chunker import DocumentChunker
from backend.app.core.security import validate_pdf_upload
from backend.app.ingestion.extractor import PDFExtractionError, PDFExtractor
from backend.tests.conftest import make_test_pdf_bytes


def test_pdf_extraction_preserves_pages_and_metadata():
    extractor = PDFExtractor()
    pdf_bytes = make_test_pdf_bytes(
        [
            "Page one introduces the Zero Trust mTLS authentication architecture.",
            "Page two specifies AES-256-GCM encryption at rest and 90-day key rotation.",
        ]
    )
    pages, meta = extractor.extract_from_bytes(pdf_bytes, filename="security.pdf")
    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Zero Trust" in pages[0].text
    assert pages[1].page_number == 2
    assert "AES-256-GCM" in pages[1].text
    assert meta["total_pages"] == 2


def test_empty_or_unreadable_pdf_handling( temp_services):
    extractor = PDFExtractor()
    blank_pdf = make_test_pdf_bytes([""])

    with pytest.raises(PDFExtractionError) as exc_info:
        extractor.extract_from_bytes(blank_pdf, filename="blank.pdf")
    assert "no extractable text" in str(exc_info.value).lower()

    # Corrupt bytes should fail validation or extraction gracefully without silent discard
    ingestion = temp_services["ingestion"]
    res = ingestion.ingest_single_file(
        filename="blank.pdf",
        content_type="application/pdf",
        file_bytes=blank_pdf,
    )
    assert res.status == "failed"
    assert "no extractable text" in res.message.lower()

    # Verify failed document was recorded in SQLite rather than silently discarded
    failed_docs = temp_services["repo"].list_documents(status_filter="failed")
    assert len(failed_docs) == 1
    assert failed_docs[0].filename == "blank.pdf"


def test_file_validation_rejects_invalid_extension_and_magic_bytes():
    valid, msg = validate_pdf_upload("malicious.exe", "application/pdf", b"%PDF-1.4 content")
    assert not valid
    assert "extension" in msg.lower()

    valid2, msg2 = validate_pdf_upload("fake.pdf", "application/pdf", b"not a real pdf file")
    assert not valid2
    assert "%pdf-" in msg2.lower()


def test_duplicate_prevention_by_file_hash(temp_services):
    ingestion = temp_services["ingestion"]
    pdf_bytes = make_test_pdf_bytes(
        ["Quantum cryptography protocol specifications for satellite links."]
    )
    first = ingestion.ingest_single_file("quantum.pdf", "application/pdf", pdf_bytes)
    assert first.status == "indexed"

    second = ingestion.ingest_single_file("quantum_copy.pdf", "application/pdf", pdf_bytes)
    assert second.status == "duplicate"
    assert second.document_id == first.document_id


def test_configurable_text_chunking_preserves_metadata():
    extractor = PDFExtractor()
    long_text = " ".join(
        [f"Sentence {i} discusses distributed vector indexing and semantic search." for i in range(120)]
    )
    pdf_bytes = make_test_pdf_bytes([long_text])
    pages, meta = extractor.extract_from_bytes(pdf_bytes, "long_doc.pdf")

    chunker = DocumentChunker(chunk_size=200, chunk_overlap=40)
    chunks = chunker.chunk_document_pages(
        document_id="doc_test123",
        filename="long_doc.pdf",
        pages=pages,
        extra_metadata=meta,
    )
    assert len(chunks) >= 3
    for idx, ch in enumerate(chunks):
        assert ch.document_id == "doc_test123"
        assert ch.filename == "long_doc.pdf"
        assert ch.page_number in (1, 2)
        assert ch.chunk_index == idx
        assert ch.token_count > 0
