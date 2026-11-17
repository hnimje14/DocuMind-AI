from typing import Any, Dict, List
import streamlit as st
from frontend.components.styles import render_empty_state, render_metric_card
from frontend.utils.api_client import BackendAPIClient


def render_dashboard_page(client: BackendAPIClient) -> None:
    # Hero Header with AI Visual Element
    st.markdown(
        """
        <div class="dm-hero">
            <div>
                <div style="
                    display:inline-flex; align-items:center; gap:0.45rem;
                    padding:0.24rem 0.7rem; border-radius:999px;
                    background:rgba(53,198,216,0.12); border:1px solid rgba(53,198,216,0.28);
                    color:#35C6D8; font-size:0.74rem; font-weight:600; margin-bottom:0.75rem;
                ">✦ INTELLIGENT RAG WORKSPACE</div>
                <h1 class="dm-hero-title">Welcome to DocuMind AI</h1>
                <p class="dm-hero-sub">
                    Your intelligent workspace for document understanding and AI-powered knowledge discovery.
                </p>
            </div>
            <div style="display:flex; align-items:center; justify-content:center; padding-left:1.5rem;">
                <svg width="136" height="96" viewBox="0 0 136 96" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="8" y="12" width="120" height="72" rx="16" fill="#202531" stroke="rgba(53,198,216,0.35)" stroke-width="1.5"/>
                    <circle cx="42" cy="48" r="18" fill="rgba(53,198,216,0.15)" stroke="#35C6D8" stroke-width="1.5"/>
                    <circle cx="42" cy="48" r="6" fill="#35C6D8"/>
                    <path d="M68 34H110" stroke="#6478FF" stroke-width="3" stroke-linecap="round"/>
                    <path d="M68 48H102" stroke="#35C6D8" stroke-width="3" stroke-linecap="round"/>
                    <path d="M68 62H92" stroke="rgba(244,246,250,0.35)" stroke-width="3" stroke-linecap="round"/>
                </svg>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        stats: Dict[str, Any] = client.get_health_stats()
    except Exception as exc:
        st.error(f"Unable to connect to backend: {exc}")
        return

    raw_provider = stats.get("llm_provider", "mock")
    llm_display = (
        "Mock Mode"
        if raw_provider.lower().startswith("mock")
        else raw_provider.upper()
    )
    total_docs = int(stats.get("total_documents", 0))
    total_chunks = int(stats.get("total_chunks", 0))
    total_sessions = int(stats.get("total_sessions", 0))
    vector_count = int(stats.get("vector_index_size", 0))

    # 4 Dynamic Metric Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            label="Uploaded Documents",
            value=str(total_docs),
            subtitle="Indexed PDF knowledge base",
            icon="▤",
        )
    with c2:
        render_metric_card(
            label="Indexed Chunks",
            value=str(total_chunks),
            subtitle=f"{vector_count} vectors in FAISS",
            icon="◈",
        )
    with c3:
        render_metric_card(
            label="Chat Sessions",
            value=str(total_sessions),
            subtitle=f"{stats.get('total_messages', 0)} total messages",
            icon="💬",
        )
    with c4:
        render_metric_card(
            label="LLM Engine Status",
            value=llm_display,
            subtitle=f"Model: {stats.get('llm_model', 'extractive-rag-v1')}",
            icon="✦",
        )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # Main Grid: Recent Documents (Left) + Quick Actions & System Status (Right)
    left_col, right_col = st.columns([1.65, 1])

    with left_col:
        with st.container(border=True):
            st.markdown(
                """
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem;">
                    <span style="font-size:1.05rem; font-weight:600; color:#F4F6FA;">Recent Documents</span>
                    <span style="font-size:0.78rem; color:#A0A7B4;">Real-time repository state</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            recent_docs: List[Dict[str, Any]] = stats.get("recent_documents", [])
            if not recent_docs:
                render_empty_state(
                    title="No documents uploaded yet",
                    subtitle="Upload PDF files in Document Management to build your searchable vector index.",
                    icon="▤",
                )
            else:
                for doc in recent_docs:
                    status_badge = (
                        '<span class="dm-badge-indexed">INDEXED</span>'
                        if doc["status"] == "indexed"
                        else f'<span class="dm-badge-failed">{doc["status"].upper()}</span>'
                    )
                    upload_date = doc["uploaded_at"][:16].replace("T", " ")
                    row_l, row_r = st.columns([3.4, 1.1])
                    with row_l:
                        st.markdown(
                            f"""
                            <div style="
                                background:#262B36; border:1px solid rgba(255,255,255,0.07);
                                border-radius:12px; padding:0.75rem 1rem; margin-bottom:0.45rem;
                            ">
                                <div style="display:flex; align-items:center; justify-content:space-between;">
                                    <span style="font-weight:600; color:#F4F6FA; font-size:0.9rem;">📄 {doc['filename']}</span>
                                    {status_badge}
                                </div>
                                <div style="font-size:0.78rem; color:#A0A7B4; margin-top:0.3rem;">
                                    Uploaded: {upload_date} &nbsp;•&nbsp; {doc['chunk_count']} chunks &nbsp;•&nbsp; {doc['page_count']} pages
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with row_r:
                        st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
                        if st.button(
                            "Open →",
                            key=f"dash_open_{doc['document_id']}",
                            use_container_width=True,
                        ):
                            st.session_state.selected_inspect_doc_id = doc["document_id"]
                            st.session_state.current_page = "Document Management"
                            st.rerun()

    with right_col:
        # B. Quick Actions
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:1.05rem; font-weight:600; color:#F4F6FA; margin-bottom:0.75rem;">Quick Actions</div>',
                unsafe_allow_html=True,
            )
            qa1, qa2 = st.columns(2)
            with qa1:
                if st.button("▤  Upload Documents", use_container_width=True, key="qa_upload"):
                    st.session_state.current_page = "Document Management"
                    st.rerun()
                if st.button("⌕  Explore Retrieval", use_container_width=True, key="qa_retrieval"):
                    st.session_state.current_page = "Retrieval Explorer"
                    st.rerun()
            with qa2:
                if st.button("✦  Start New Chat", use_container_width=True, key="qa_chat"):
                    st.session_state.current_page = "AI Chat"
                    st.rerun()
                if st.button("◧  View Evaluation", use_container_width=True, key="qa_eval"):
                    st.session_state.current_page = "Evaluation Center"
                    st.rerun()

        st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

        # C. System Status
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:1.05rem; font-weight:600; color:#F4F6FA; margin-bottom:0.75rem;">System Status</div>',
                unsafe_allow_html=True,
            )
            index_status = (
                f"Active ({vector_count} vectors)"
                if vector_count > 0
                else "Ready (Empty Index)"
            )
            st.markdown(
                f"""
                <div style="display:flex; flex-direction:column; gap:0.58rem; font-size:0.84rem;">
                    <div style="display:flex; justify-content:space-between; padding-bottom:0.45rem; border-bottom:1px solid rgba(255,255,255,0.06);">
                        <span style="color:#A0A7B4;">Backend Health</span>
                        <span style="color:#35C6D8; font-weight:600;">● {stats.get('status', 'healthy').capitalize()}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding-bottom:0.45rem; border-bottom:1px solid rgba(255,255,255,0.06);">
                        <span style="color:#A0A7B4;">Embedding Model</span>
                        <span style="color:#F4F6FA; font-weight:500;">all-MiniLM-L6-v2</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding-bottom:0.45rem; border-bottom:1px solid rgba(255,255,255,0.06);">
                        <span style="color:#A0A7B4;">Vector Database</span>
                        <span style="color:#F4F6FA; font-weight:500;">FAISS (Cosine IP)</span>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding-bottom:0.45rem; border-bottom:1px solid rgba(255,255,255,0.06);">
                        <span style="color:#A0A7B4;">LLM Provider</span>
                        <span style="color:#6478FF; font-weight:600;">{llm_display}</span>
                    </div>
                    <div style="display:flex; justify-content:space-between;">
                        <span style="color:#A0A7B4;">Index Status</span>
                        <span style="color:#F4F6FA; font-weight:500;">{index_status}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
