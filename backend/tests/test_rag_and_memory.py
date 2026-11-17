from backend.app.schemas.models import ChatAskRequest
from backend.tests.conftest import make_test_pdf_bytes


def test_rag_answer_generation_and_verified_citations(temp_services):
    ingestion = temp_services["ingestion"]
    rag = temp_services["rag"]

    pdf_bytes = make_test_pdf_bytes(
        [
            "Section 1: All internal microservice calls require mutual TLS (mTLS) with X.509 certificates rotated every 24 hours.",
            "Section 2: Disaster recovery requires a Recovery Time Objective (RTO) of 30 minutes and Recovery Point Objective (RPO) of 5 minutes.",
        ]
    )
    ing_res = ingestion.ingest_single_file("dr_policy.pdf", "application/pdf", pdf_bytes)
    assert ing_res.status == "indexed"

    ask_req = ChatAskRequest(
        question="What are the Recovery Time Objective (RTO) and Recovery Point Objective (RPO) requirements?",
        session_id="sess_alpha",
        retrieval_mode="hybrid",
        top_k=2,
        similarity_threshold=0.05,
    )
    resp = rag.answer_question(ask_req)

    assert resp.insufficient_context is False
    assert "30 minutes" in resp.answer
    assert len(resp.citations) >= 1
    # Verify citation points to actual retrieved document and page 2
    assert any(
        c.document_name == "dr_policy.pdf" and c.page_number == 2
        for c in resp.citations
    )


def test_insufficient_context_behavior_reduces_hallucinations(temp_services):
    ingestion = temp_services["ingestion"]
    rag = temp_services["rag"]

    pdf_bytes = make_test_pdf_bytes(
        ["Corporate travel policy requires booking economy class flights 14 days in advance."]
    )
    ingestion.ingest_single_file("travel_policy.pdf", "application/pdf", pdf_bytes)

    # Ask an unrelated question not present in the document
    unrelated_req = ChatAskRequest(
        question="What is the melting point of tungsten in degrees Celsius on Mars?",
        session_id="sess_unrelated",
        retrieval_mode="basic",
        top_k=3,
        similarity_threshold=0.20,
    )
    resp = rag.answer_question(unrelated_req)

    assert resp.insufficient_context is True
    assert "cannot find sufficient information" in resp.answer.lower()
    assert len(resp.citations) == 0


def test_conversational_session_isolation_and_followup_rewriting(temp_services):
    ingestion = temp_services["ingestion"]
    rag = temp_services["rag"]
    memory = temp_services["memory"]

    pdf_bytes = make_test_pdf_bytes(
        [
            "HashiCorp Vault rotates mutual TLS X.509 certificates every 24 hours automatically.",
        ]
    )
    ingestion.ingest_single_file("vault_spec.pdf", "application/pdf", pdf_bytes)

    # Session 1
    r1 = rag.answer_question(
        ChatAskRequest(
            question="How does HashiCorp Vault handle mutual TLS X.509 certificates?",
            session_id="session_one",
            similarity_threshold=0.05,
        )
    )
    assert not r1.insufficient_context

    # Follow-up question in Session 1 with pronoun reference ("they")
    r2 = rag.answer_question(
        ChatAskRequest(
            question="How often are they rotated?",
            session_id="session_one",
            use_query_rewriting=True,
            similarity_threshold=0.05,
        )
    )
    assert "Context:" in r2.rewritten_query

    # Session 2 must be completely isolated from Session 1
    s2 = memory.create_session(session_id="session_two", title="Second Session")
    hist1 = memory.get_history("session_one")
    hist2 = memory.get_history("session_two")

    assert hist1 is not None and len(hist1.messages) == 4
    assert hist2 is not None and len(hist2.messages) == 0

    # Clearing session 1 removes only session 1
    assert memory.clear_session("session_one") is True
    assert memory.get_history("session_one") is None
    assert memory.get_history("session_two") is not None


def test_resume_document_qa_regression(temp_services):
    """
    Regression test verifying semantic section chunking, metadata preservation,
    and query-aware QA on resume documents across all 6 required cases:
      1. "What is the full name of the person?" -> "ALEXANDER VANCE"
      2. "What is the person's CGPA?" -> "8.92"
      3. "What is the expected graduation year?" -> "2027"
      4. "Which course is the person pursuing?" -> "B.Tech, Computer Engineering"
      5. "Explain the Multi-Agent Task Orchestration project." -> Complete project explanation
      6. "What is the person's father's name?" -> Insufficient information
    """
    ingestion = temp_services["ingestion"]
    rag = temp_services["rag"]
    repo = temp_services["repo"]

    resume_page_1 = (
        "ALEXANDER VANCE\n\n"
        "Course: B.Tech, Computer Engineering, 2027\n"
        "Email: alex.vance@example.org\n"
        "Mobile: 5550192834\n"
        "Social: Github Linkedin\n"
        "CGPA: 8.92\n"
        "ACADEMIC DETAILS\n"
        "Course Institute/College Board/University Score Year\n"
        "Class XII Horizon Institute Of Science and Technology National Board 91.40% 2023\n"
        "Class X Crestview Academy National Board 95.80% 2021\n"
        "Subjects / Electives: Design and Analysis of Algorithms, Operating System, Computer Networks\n"
        "Technical Proficiency: Python, FastAPI, SQL, Docker, Git, Scikit-Learn, TensorFlow\n"
        "PROJECTS\n"
        "SecureCart Pay: Secure Order & Payment Service — Backend Development / Distributed Systems\n\n"
        "Statement: Built a secure order and wallet payment backend that prevents overselling and duplicate charges.\n"
        "Explanation: Designed a PostgreSQL schema with an append-only transaction ledger and row-level locking.\n"
        "Skills used: FastAPI, PostgreSQL, SQLAlchemy, JWT authentication, Docker\n"
        "Multi-Agent Task Orchestration System — Artificial Intelligence / Multi-Agent Systems\n\n"
        "Statement: Built a multi-agent orchestration system that decomposes goals into a dependency-aware task graph, enabling parallel execution with automatic failure recovery.\n"
        "Explanation: Designed a scheduler that runs independent subtasks concurrently, coordinates isolated agents through a shared memory layer, and triggers re-planning and output verification.\n"
        "Skills used: Multi-agent systems, task scheduling & DAG design, Python, concurrent/parallel execution"
    )
    resume_page_2 = (
        "CERTIFICATIONS\n"
        "Machine Learning with Python — IBM (offered through Coursera)\n"
        "Test Score: National Engineering Aptitude — May 11, 2023 — 99.40 Percentile"
    )

    other_pdf = make_test_pdf_bytes(["Unrelated document about cloud network firewalls."])
    ingestion.ingest_single_file("other_doc.pdf", "application/pdf", other_pdf)

    resume_pdf = make_test_pdf_bytes([resume_page_1, resume_page_2])
    ing_res = ingestion.ingest_single_file("Synthetic_Candidate_Resume.pdf", "application/pdf", resume_pdf)
    assert ing_res.status == "indexed"
    resume_doc_id = ing_res.document_id

    # Verify Part A.4 metadata preservation on chunks
    doc_chunks = repo.get_document_chunks(resume_doc_id)
    assert len(doc_chunks) >= 5
    for ch in doc_chunks:
        assert ch.document_id == resume_doc_id
        assert ch.filename == "Synthetic_Candidate_Resume.pdf"
        assert ch.page_number in (1, 2)
        assert ch.chunk_id
        assert ch.section_title
        assert ch.source_text
        assert ch.document_hash

    qa_cases = [
        ("What is the person's full name?", "ALEXANDER VANCE", "Personal Information", 1),
        ("What is the person's CGPA?", "8.92", "Academic Details", 1),
        ("Which course is the person pursuing?", "B.Tech, Computer Engineering", "Personal Information", 1),
        ("What programming languages does the person know?", "Python", "Technical Proficiency", 1),
        ("Explain the Multi-Agent Task Orchestration project.", "dependency-aware task graph", "Projects: Multi-Agent Task Orchestration System", 1),
        ("What are the person's certifications?", "Machine Learning with Python", "Certifications", 2),
        ("What is the expected graduation year?", "2027", "Personal Information", 1),
    ]

    for mode in ("hybrid", "basic", "mmr"):
        for question, expected_substring, expected_section, expected_page in qa_cases:
            resp = rag.answer_question(
                ChatAskRequest(
                    question=question,
                    session_id=f"resume_reg_{mode}",
                    document_ids=[resume_doc_id],
                    retrieval_mode=mode,
                )
            )
            assert resp.insufficient_context is False, f"Failed on {question} in {mode} mode"
            assert expected_substring in resp.answer, (
                f"Expected '{expected_substring}' in answer for '{question}', got: {resp.answer}"
            )
            assert len(resp.citations) >= 1
            assert resp.citations[0].document_name == "Synthetic_Candidate_Resume.pdf"
            assert resp.citations[0].page_number == expected_page
            assert resp.retrieved_chunks[0].section_title == expected_section

        # Case 6: "What is the person's father's name?" -> Insufficient information
        father_resp = rag.answer_question(
            ChatAskRequest(
                question="What is the person's father's name?",
                session_id=f"resume_reg_father_{mode}",
                document_ids=[resume_doc_id],
                retrieval_mode=mode,
            )
        )
        assert father_resp.insufficient_context is True
        assert len(father_resp.citations) == 0
