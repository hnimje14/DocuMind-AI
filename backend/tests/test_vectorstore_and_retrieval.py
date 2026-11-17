import numpy as np
from backend.tests.conftest import make_test_pdf_bytes


def test_embedding_normalization_and_faiss_operations(temp_services):
    embedder = temp_services["embedder"]
    vs = temp_services["vector_store"]
    ingestion = temp_services["ingestion"]

    pdf_a = make_test_pdf_bytes(
        [
            "PostgreSQL relational database replication uses write-ahead logging (WAL) and synchronous standby nodes.",
            "Kubernetes horizontal pod autoscaler scales deployments based on CPU and custom Prometheus metrics.",
        ]
    )
    pdf_b = make_test_pdf_bytes(
        [
            "AES-256-GCM hardware encryption protects cloud storage buckets and requires 90-day key rotation.",
        ]
    )

    res_a = ingestion.ingest_single_file("infra_guide.pdf", "application/pdf", pdf_a)
    res_b = ingestion.ingest_single_file("security_policy.pdf", "application/pdf", pdf_b)

    assert res_a.status == "indexed"
    assert res_b.status == "indexed"
    assert vs.count() == 3

    # Verify query embedding is L2-normalized
    q_vec = embedder.embed_query("AES-256-GCM encryption key rotation")
    norm = float(np.linalg.norm(q_vec))
    assert abs(norm - 1.0) < 1e-4

    # Basic search should rank security_policy.pdf page 1 first
    retriever = temp_services["retriever"]
    basic_hits = retriever.retrieve_basic("AES-256-GCM encryption key rotation", top_k=2)
    assert len(basic_hits) >= 1
    assert basic_hits[0].filename == "security_policy.pdf"
    assert basic_hits[0].page_number == 1

    # Metadata filter by document_id should restrict results to infra_guide.pdf
    filtered_hits = retriever.retrieve_basic(
        "database replication WAL",
        top_k=5,
        similarity_threshold=0.0,
        document_ids=[res_a.document_id],
    )
    assert all(h.document_id == res_a.document_id for h in filtered_hits)

    # MMR and Hybrid modes should also return relevant results with source metadata
    mmr_hits = retriever.retrieve_mmr("Kubernetes autoscaler Prometheus metrics", top_k=2, similarity_threshold=0.05)
    assert len(mmr_hits) >= 1
    assert mmr_hits[0].filename == "infra_guide.pdf"
    assert mmr_hits[0].page_number == 2

    hybrid_hits = retriever.retrieve_hybrid("write-ahead logging WAL", top_k=2, similarity_threshold=0.05)
    assert len(hybrid_hits) >= 1
    assert hybrid_hits[0].filename == "infra_guide.pdf"
    assert hybrid_hits[0].page_number == 1

    # Document deletion should remove vectors and rebuild FAISS index
    deleted = ingestion.delete_document(res_b.document_id)
    assert deleted is True
    assert vs.count() == 2
