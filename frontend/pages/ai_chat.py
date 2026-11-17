from typing import Any, Dict, List, Optional
import streamlit as st
from frontend.components.styles import render_empty_state, render_page_header
from frontend.utils.api_client import BackendAPIClient


def _render_citations_and_actions(
    answer_text: str,
    citations: List[Dict[str, Any]],
    retrieved_chunks: Optional[List[Dict[str, Any]]],
    key_prefix: str,
) -> None:
    if citations:
        st.markdown(
            f'<div style="font-size:0.8rem; font-weight:600; color:#35C6D8; margin:0.75rem 0 0.45rem 0;">📌 Verified Source Citations ({len(citations)})</div>',
            unsafe_allow_html=True,
        )
        for cit in citations:
            st.markdown(
                f"""
                <div class="dm-citation-card">
                    <div class="dm-citation-meta">
                        <span>{cit['citation_id']} &nbsp;📄 {cit['document_name']} &nbsp;•&nbsp; Page {cit['page_number']}</span>
                        <span style="color:#A0A7B4; font-weight:500;">Chunk: <code>{cit['chunk_id']}</code> &nbsp;|&nbsp; Score: {cit['score']:.4f}</span>
                    </div>
                    <div class="dm-citation-snippet">{cit['snippet']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    act_col1, act_col2 = st.columns([1.4, 2.6])
    with act_col1:
        with st.popover("📋 Copy Answer", use_container_width=True):
            st.caption("Click the copy icon in the top-right corner of the box below:")
            st.code(answer_text, language="markdown")
    with act_col2:
        if retrieved_chunks:
            with st.expander(f"🔬 Expand Retrieved Context ({len(retrieved_chunks)} chunks)"):
                for idx, ch in enumerate(retrieved_chunks, start=1):
                    st.markdown(
                        f"**[{idx}] {ch['filename']} — Page {ch['page_number']}** "
                        f"*(Score: `{ch['score']:.4f}` • Mode: `{ch['retrieval_method']}`)*"
                    )
                    st.code(ch["text"], language="text")


def render_ai_chat_page(client: BackendAPIClient) -> None:
    render_page_header(
        title="Ask your Documents",
        subtitle="Get accurate answers from your uploaded knowledge base.",
    )

    try:
        docs = client.list_documents(status_filter="indexed")
        sessions = client.list_sessions()
    except Exception as exc:
        st.error(f"Unable to reach backend: {exc}")
        return

    # Sync active_session_id without auto-creating dummy sessions on initial load
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = sessions[0]["session_id"] if sessions else None
    elif st.session_state.active_session_id and not any(
        s["session_id"] == st.session_state.active_session_id for s in sessions
    ):
        st.session_state.active_session_id = sessions[0]["session_id"] if sessions else None

    # Top Control Strip: Document Selection, Retrieval Mode, Advanced Options, Clear Conversation
    with st.container(border=True):
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.8, 1.3, 1.1, 0.9])
        doc_map = {d["document_id"]: d["filename"] for d in docs}
        with ctrl1:
            selected_doc_ids = st.multiselect(
                "Document Scope",
                options=list(doc_map.keys()),
                format_func=lambda did: doc_map.get(did, did),
                placeholder="All indexed documents" if docs else "No indexed documents",
            )
        with ctrl2:
            retrieval_mode = st.selectbox(
                "Retrieval Mode",
                options=["hybrid", "basic", "mmr"],
                format_func=lambda m: {
                    "hybrid": "Hybrid (FAISS + BM25)",
                    "basic": "Basic Semantic (FAISS)",
                    "mmr": "MMR (Diversity)",
                }[m],
            )
        with ctrl3:
            with st.popover("⚙️ RAG Tuning", use_container_width=True):
                top_k = st.slider("Top-K Chunks", min_value=1, max_value=12, value=5)
                sim_threshold = st.slider(
                    "Similarity Threshold", min_value=0.0, max_value=0.8, value=0.05, step=0.05
                )
                use_query_rewriting = st.checkbox("Follow-Up Query Rewriting", value=True)
                use_compression = st.checkbox("Context Compression", value=False)
                multi_doc_comp = st.checkbox("Multi-Document Comparison", value=False)
        with ctrl4:
            st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
            if st.button(
                "🧹 Clear Chat",
                use_container_width=True,
                disabled=not st.session_state.active_session_id,
            ):
                if st.session_state.active_session_id:
                    client.delete_chat_history(st.session_state.active_session_id)
                    st.session_state.active_session_id = None
                    st.rerun()

    # Load Conversation History
    messages: List[Dict[str, Any]] = []
    if st.session_state.active_session_id:
        try:
            history_data = client.get_chat_history(st.session_state.active_session_id)
            messages = history_data.get("messages", [])
        except Exception:
            messages = []

    # Elegant Empty State when no conversation history is present
    if not messages:
        render_empty_state(
            title="Start a conversation with your documents.",
            subtitle="Upload documents and ask questions to discover insights.",
            icon="✦",
        )
        if not docs:
            c_left, c_mid, c_right = st.columns([1.2, 1.2, 1.2])
            with c_mid:
                if st.button("▤  Go to Document Management", use_container_width=True):
                    st.session_state.current_page = "Document Management"
                    st.rerun()
    else:
        for idx, msg in enumerate(messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    if msg.get("rewritten_query") and msg["rewritten_query"] != msg["content"]:
                        st.caption(
                            f"🔎 Rewritten Query: `{msg['rewritten_query']}` • Mode: `{msg.get('retrieval_mode', 'hybrid').upper()}`"
                        )
                    _render_citations_and_actions(
                        answer_text=msg["content"],
                        citations=msg.get("citations", []),
                        retrieved_chunks=None,
                        key_prefix=f"hist_{idx}",
                    )

        # Follow-up question suggestions grounded in conversation context
        if docs and messages and messages[-1]["role"] == "assistant":
            st.markdown(
                '<div style="font-size:0.76rem; color:#A0A7B4; margin:0.4rem 0 0.3rem 0.2rem;">Suggested follow-ups:</div>',
                unsafe_allow_html=True,
            )
            fcol1, fcol2, fcol3 = st.columns(3)
            followup_prompts = [
                "Can you elaborate on the specific requirements mentioned on that page?",
                "Are there any exceptions or additional conditions noted in the document?",
                "Summarize the key takeaways from the cited section.",
            ]
            for f_idx, (f_col, f_prompt) in enumerate(zip([fcol1, fcol2, fcol3], followup_prompts)):
                with f_col:
                    if st.button(f"↳ {f_prompt[:42]}…", key=f"followup_{f_idx}", use_container_width=True):
                        st.session_state.pending_followup_question = f_prompt
                        st.rerun()

    # Handle either direct chat input or clicked follow-up chip
    pending_q = st.session_state.pop("pending_followup_question", None)
    chat_box_q = st.chat_input("Ask a question about your uploaded documents...")
    user_question = pending_q or chat_box_q

    if user_question:
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Searching vector index and generating grounded response..."):
                try:
                    resp = client.ask_question(
                        {
                            "question": user_question,
                            "session_id": st.session_state.active_session_id,
                            "document_ids": selected_doc_ids or None,
                            "retrieval_mode": retrieval_mode,
                            "top_k": top_k,
                            "similarity_threshold": sim_threshold,
                            "use_query_rewriting": use_query_rewriting,
                            "use_context_compression": use_compression,
                            "multi_doc_comparison": multi_doc_comp,
                        }
                    )
                    st.session_state.active_session_id = resp["session_id"]
                    st.markdown(resp["answer"])
                    st.caption(
                        f"⏱️ `{resp['latency_ms']} ms` &nbsp;•&nbsp; "
                        f"🛡️ Grounding: `{resp['grounding_score']:.0%}` &nbsp;•&nbsp; "
                        f"🔎 Effective Query: `{resp['rewritten_query']}` &nbsp;•&nbsp; "
                        f"🤖 `{resp['llm_provider']}`"
                    )
                    _render_citations_and_actions(
                        answer_text=resp["answer"],
                        citations=resp.get("citations", []),
                        retrieved_chunks=resp.get("retrieved_chunks", []),
                        key_prefix="latest",
                    )
                except Exception as exc:
                    st.error(f"Error generating response: {exc}")
