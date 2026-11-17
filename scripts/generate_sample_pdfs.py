"""
Generates realistic multi-page sample PDF documents and a ground-truth evaluation dataset
for testing, benchmarking, and demonstrating the Intelligent Document Q&A System.
"""
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer


SAMPLE_DOCS_CONTENT: Dict[str, List[Tuple[str, List[str]]]] = {
    "Enterprise_Cloud_Security_Policy_2026.pdf": [
        (
            "Page 1: Executive Summary & Access Control Standards",
            [
                "Document ID: SEC-POL-2026-V4. Effective Date: January 15, 2026. Owner: Chief Information Security Officer (CISO).",
                "1.1 Multi-Factor Authentication (MFA): All production cloud environments require phishing-resistant FIDO2/WebAuthn hardware security keys for privileged access. Standard employee accounts must use MFA with a maximum session timeout of 8 hours.",
                "1.2 Zero Trust Network Architecture: Every internal microservice request must authenticate using mutual TLS (mTLS) with short-lived X.509 certificates rotated automatically every 24 hours via HashiCorp Vault.",
                "1.3 Password & Credential Policy: Service account API keys are strictly prohibited in source code repositories. Automated secret scanning runs on every pull request, and human user passwords must be at least 16 characters long.",
            ],
        ),
        (
            "Page 2: Data Encryption & Key Management",
            [
                "2.1 Encryption at Rest: All persistent cloud databases, object storage buckets (AWS S3, GCS), and block volumes must use AES-256-GCM encryption at rest managed through Customer-Managed Keys (CMK) in Cloud KMS.",
                "2.2 Encryption in Transit: External public endpoints must enforce TLS 1.3 exclusively. Legacy TLS 1.0 and TLS 1.1 protocols are permanently disabled across all load balancers.",
                "2.3 Key Rotation Schedule: Cryptographic master keys (CMKs) in Cloud KMS must undergo automatic rotation every 90 days. Compromised keys must be revoked within 15 minutes of detection.",
                "2.4 Data Classification Tiers: Tier 1 (Restricted PII/Financial), Tier 2 (Confidential Internal), and Tier 3 (Public). Tier 1 data must never be replicated to non-production staging environments without irreversible tokenization.",
            ],
        ),
        (
            "Page 3: Incident Response & Disaster Recovery SLAs",
            [
                "3.1 Severity 1 (Critical Breach) Response SLA: The Security Operations Center (SOC) must acknowledge Severity 1 incidents within 15 minutes, initiate containment within 30 minutes, and notify executive leadership within 1 hour.",
                "3.2 Disaster Recovery Objectives: For Tier 1 mission-critical services, the Recovery Time Objective (RTO) is 30 minutes and the Recovery Point Objective (RPO) is 5 minutes using multi-region active-active replication.",
                "3.3 Audit Logging Retention: Immutable security audit logs must be retained in hot searchable storage for 90 days and cold WORM (Write-Once-Read-Many) archive storage for 7 years to satisfy SOC2 Type II and ISO 27001 compliance.",
            ],
        ),
    ],
    "NexusAI_RAG_Architecture_Benchmark_Report.pdf": [
        (
            "Page 1: RAG System Architecture & Embedding Benchmark",
            [
                "Report Title: NexusAI Applied AI Engineering Benchmark Report (Q3 2026). Authors: Dr. Elena Rostova and Marcus Vance.",
                "1.1 Embedding Model Evaluation: We benchmarked sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) against larger 1024-dimension models. all-MiniLM-L6-v2 achieved an average CPU query latency of 14.2 ms while maintaining 91.4% Recall@5 on technical documentation.",
                "1.2 Chunking Strategy Findings: Across 1,200 enterprise PDFs, a chunk size of 800 tokens with a 150-token overlap reduced boundary context loss by 34% compared to naive fixed 500-character splitting without overlap.",
                "1.3 Vector Indexing Engine: FAISS IndexFlatIP with L2-normalized vectors provides exact cosine similarity search up to 250,000 chunks with sub-5ms index lookup latency on standard 4-vCPU instances.",
            ],
        ),
        (
            "Page 2: Hybrid Retrieval, MMR, and Hallucination Mitigation",
            [
                "2.1 Hybrid Search Performance: Combining dense semantic search (FAISS cosine similarity) with sparse lexical search (BM25) using Reciprocal Rank Fusion improved Mean Reciprocal Rank (MRR) from 0.76 to 0.89 on queries containing exact product codes, acronyms, and policy numbers.",
                "2.2 Maximum Marginal Relevance (MMR): Applying MMR with lambda = 0.65 reduced redundant chunk retrieval by 41% on multi-page financial reports while preserving top-3 answer recall.",
                "2.3 Hallucination Mitigation & Grounding: Enforcing strict system prompt grounding with explicit page-level citation verification reduced unsupported factual claims from 18.5% (baseline ungrounded LLM) to 1.8% in audited responses.",
                "2.4 Disaster Recovery Comparison Note: Unlike the Enterprise Cloud Security Policy (which sets RTO at 30 minutes), legacy on-premise analytics clusters described in this report operate under a relaxed RTO of 4 hours and RPO of 60 minutes.",
            ],
        ),
    ],
}


EVALUATION_DATASET = [
    {
        "id": "eval_q1",
        "question": "What encryption standard is required for data at rest and how often must cryptographic master keys be rotated?",
        "expected_answer": "All persistent cloud databases and storage must use AES-256-GCM encryption at rest with Customer-Managed Keys (CMK), and cryptographic master keys must undergo automatic rotation every 90 days.",
        "expected_keywords": ["AES-256-GCM", "90 days", "Customer-Managed Keys", "CMK"],
        "relevant_document_names": ["Enterprise_Cloud_Security_Policy_2026.pdf"],
        "relevant_pages": [2],
    },
    {
        "id": "eval_q2",
        "question": "What are the Severity 1 incident response SLAs and audit log retention requirements?",
        "expected_answer": "Severity 1 incidents must be acknowledged within 15 minutes, contained within 30 minutes, and reported to leadership within 1 hour. Immutable audit logs must be retained in hot storage for 90 days and cold WORM archive storage for 7 years.",
        "expected_keywords": ["15 minutes", "30 minutes", "90 days", "7 years", "WORM"],
        "relevant_document_names": ["Enterprise_Cloud_Security_Policy_2026.pdf"],
        "relevant_pages": [3],
    },
    {
        "id": "eval_q3",
        "question": "How are internal microservice requests authenticated in the Zero Trust Network Architecture?",
        "expected_answer": "Every internal microservice request must authenticate using mutual TLS (mTLS) with short-lived X.509 certificates rotated automatically every 24 hours via HashiCorp Vault.",
        "expected_keywords": ["mutual TLS", "mTLS", "24 hours", "HashiCorp Vault", "X.509"],
        "relevant_document_names": ["Enterprise_Cloud_Security_Policy_2026.pdf"],
        "relevant_pages": [1],
    },
    {
        "id": "eval_q4",
        "question": "How much did Hybrid Search improve Mean Reciprocal Rank (MRR) and what were the MMR diversity results?",
        "expected_answer": "Hybrid search combining FAISS semantic search with BM25 improved Mean Reciprocal Rank (MRR) from 0.76 to 0.89. Applying MMR with lambda = 0.65 reduced redundant chunk retrieval by 41%.",
        "expected_keywords": ["0.76", "0.89", "BM25", "41%", "0.65"],
        "relevant_document_names": ["NexusAI_RAG_Architecture_Benchmark_Report.pdf"],
        "relevant_pages": [2],
    },
    {
        "id": "eval_q5",
        "question": "What latency and Recall@5 did sentence-transformers/all-MiniLM-L6-v2 achieve in the benchmark?",
        "expected_answer": "sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) achieved an average CPU query latency of 14.2 ms while maintaining 91.4% Recall@5 on technical documentation.",
        "expected_keywords": ["14.2 ms", "91.4%", "384", "Recall@5"],
        "relevant_document_names": ["NexusAI_RAG_Architecture_Benchmark_Report.pdf"],
        "relevant_pages": [1],
    },
    {
        "id": "eval_q6",
        "question": "Compare the Recovery Time Objective (RTO) and Recovery Point Objective (RPO) between the Cloud Security Policy and the legacy analytics clusters.",
        "expected_answer": "In the Enterprise Cloud Security Policy, Tier 1 mission-critical services have an RTO of 30 minutes and RPO of 5 minutes. In contrast, the NexusAI Benchmark Report notes that legacy on-premise analytics clusters have a relaxed RTO of 4 hours and RPO of 60 minutes.",
        "expected_keywords": ["30 minutes", "5 minutes", "4 hours", "60 minutes", "RTO", "RPO"],
        "relevant_document_names": [
            "Enterprise_Cloud_Security_Policy_2026.pdf",
            "NexusAI_RAG_Architecture_Benchmark_Report.pdf",
        ],
        "relevant_pages": [3, 2],
    },
]


def build_pdf_bytes(title: str, pages_data: List[Tuple[str, List[str]]]) -> bytes:
    """Create a clean multi-page PDF in memory using ReportLab and return raw bytes."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        title=title,
        author="Intelligent Document Q&A System Benchmark Suite",
    )
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "PageHeading",
        parent=styles["Heading2"],
        spaceAfter=12,
    )
    body_style = ParagraphStyle(
        "PageBody",
        parent=styles["BodyText"],
        spaceAfter=10,
        leading=15,
    )

    story = []
    for idx, (page_heading, paragraphs) in enumerate(pages_data):
        story.append(Paragraph(page_heading, heading_style))
        story.append(Spacer(1, 8))
        for para in paragraphs:
            story.append(Paragraph(para, body_style))
            story.append(Spacer(1, 6))
        if idx < len(pages_data) - 1:
            story.append(PageBreak())

    doc.build(story)
    return buffer.getvalue()


def generate_samples_and_dataset(project_root: Path) -> Tuple[List[Path], Path]:
    """Write sample PDFs to data/sample_docs/ and evaluation dataset to evaluation/datasets/."""
    sample_dir = project_root / "data" / "sample_docs"
    sample_dir.mkdir(parents=True, exist_ok=True)

    created_pdfs: List[Path] = []
    for filename, pages_data in SAMPLE_DOCS_CONTENT.items():
        pdf_bytes = build_pdf_bytes(filename, pages_data)
        out_path = sample_dir / filename
        out_path.write_bytes(pdf_bytes)
        created_pdfs.append(out_path)

    eval_dir = project_root / "evaluation" / "datasets"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / "eval_dataset.json"
    eval_path.write_text(
        json.dumps(EVALUATION_DATASET, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return created_pdfs, eval_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    pdfs, dataset_file = generate_samples_and_dataset(root)
    print(f"Generated {len(pdfs)} sample PDFs: {[p.name for p in pdfs]}")
    print(f"Generated evaluation dataset: {dataset_file}")
