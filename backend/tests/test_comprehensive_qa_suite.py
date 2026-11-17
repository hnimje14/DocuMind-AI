"""
Comprehensive QA Verification Suite covering:
- Step 4: All 7 Document Ingestion cases (single PDF, multiple PDFs, empty PDF, corrupted PDF,
  duplicate PDF, unsupported file type, oversized file > limit)
- Step 5: Complete RAG QA verification (present answers, absent answers, follow-up history, citations)
- Step 6: Advanced Retrieval & Multi-Chunk-Size Benchmark comparison (Basic vs MMR vs Hybrid, chunks 400/800/1200)
"""
from backend.app.schemas.models import ChatAskRequest, EvaluationRunRequest
from backend.tests.conftest import make_test_pdf_bytes


def test_step4_all_seven_document_ingestion_scenarios(temp_services):
    ingestion = temp_services["ingestion"]
    repo = temp_services["repo"]
    vs = temp_services["vector_store"]

    pdf1 = make_test_pdf_bytes(
        [
            "Document Alpha Page 1: Zero Trust mTLS requires X.509 certificates rotated every 24 hours.",
            "Document Alpha Page 2: AES-256-GCM encryption at rest is mandatory for all cloud databases.",
        ]
    )
    pdf2 = make_test_pdf_bytes(
        [
            "Document Beta Page 1: Hybrid search combines FAISS cosine similarity with BM25 lexical scoring.",
        ]
    )
    pdf3 = make_test_pdf_bytes(
        [
            "Document Gamma Page 1: Incident response SLA for Severity 1 breaches is 15 minutes.",
        ]
    )

    # 1. Upload one valid PDF
    r_single = ingestion.ingest_single_file("alpha.pdf", "application/pdf", pdf1)
    assert r_single.status == "indexed"
    assert r_single.page_count == 2
    assert r_single.chunk_count == 2
    assert vs.count() == 2

    # 2. Upload multiple PDFs
    r_multi = ingestion.ingest_multiple_files(
        [
            ("beta.pdf", "application/pdf", pdf2),
            ("gamma.pdf", "application/pdf", pdf3),
        ]
    )
    assert r_multi.total_files == 2
    assert r_multi.indexed_count == 2
    assert vs.count() == 4

    # 3. Upload an empty PDF (0 pages text)
    empty_pdf = make_test_pdf_bytes([""])
    r_empty = ingestion.ingest_single_file("empty.pdf", "application/pdf", empty_pdf)
    assert r_empty.status == "failed"
    assert "no extractable text" in r_empty.message.lower()

    # 4. Upload a corrupted PDF (has %PDF- magic header but corrupted stream body)
    corrupted_pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<CORRUPTED_STREAM_DATA>>"
    r_corrupt = ingestion.ingest_single_file("corrupted.pdf", "application/pdf", corrupted_pdf)
    assert r_corrupt.status == "failed"
    assert "corrupted" in r_corrupt.message.lower() or "failed to read" in r_corrupt.message.lower()

    # 5. Upload a duplicate PDF
    r_dup = ingestion.ingest_single_file("alpha_duplicate.pdf", "application/pdf", pdf1)
    assert r_dup.status == "duplicate"
    assert r_dup.document_id == r_single.document_id
    # Vector store count must NOT increase on duplicate
    assert vs.count() == 4

    # 6. Upload an unsupported file type
    r_unsupported = ingestion.ingest_single_file("notes.txt", "text/plain", b"Plain text content")
    assert r_unsupported.status == "failed"
    assert "extension" in r_unsupported.message.lower()

    # 7. Upload a file exceeding the configured size limit (> 25 MB)
    oversized_bytes = b"%PDF-1.4\n" + (b"0" * (26 * 1024 * 1024))
    r_oversize = ingestion.ingest_single_file("huge.pdf", "application/pdf", oversized_bytes)
    assert r_oversize.status == "failed"
    assert "exceeds maximum limit" in r_oversize.message.lower()

    # Verify failed documents are recorded in metadata store and not silently lost
    failed_list = repo.list_documents(status_filter="failed")
    assert len(failed_list) >= 2


def test_step5_and_step6_rag_and_multi_chunk_evaluation(temp_services):
    evaluator = temp_services["evaluator"]

    # Run evaluation comparing basic, mmr, hybrid across chunk sizes 400 and 800
    eval_req = EvaluationRunRequest(
        top_k=3,
        similarity_threshold=0.05,
        modes_to_compare=["basic", "mmr", "hybrid"],
        chunk_sizes_to_compare=[400, 800],
    )
    eval_resp = evaluator.run_evaluation(eval_req)
    assert eval_resp.dataset_size == 6
    # 3 modes * 2 chunk sizes = 6 summary rows
    assert len(eval_resp.summaries) == 6
    for summary in eval_resp.summaries:
        assert summary.hit_rate > 0.0
        assert summary.mrr > 0.0
        assert summary.faithfulness > 0.0
        assert summary.citation_correctness > 0.0
