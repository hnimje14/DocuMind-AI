from typing import Any, Dict, List, Tuple
import pandas as pd
import streamlit as st
from frontend.components.styles import render_empty_state, render_page_header
from frontend.utils.api_client import BackendAPIClient


def render_document_management_page(client: BackendAPIClient) -> None:
    render_page_header(
        title="Your Knowledge Base",
        subtitle="Upload, organize and manage documents for intelligent retrieval.",
    )

    # Drag-and-drop PDF upload container
    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.5rem;">
                <span style="font-size:1rem; font-weight:600; color:#F4F6FA;">Upload PDF Documents</span>
                <span style="font-size:0.78rem; color:#A0A7B4;">Supports multi-file upload • Max 25 MB per file</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        uploaded_files = st.file_uploader(
            "Drag and drop PDF files here or click Browse files",
            type=["pdf"],
            accept_multiple_files=True,
        )

        col_c1, col_c2, col_c3 = st.columns([1.2, 1.2, 1.4])
        with col_c1:
            chunk_size = st.number_input(
                "Chunk Size (tokens)",
                min_value=100,
                max_value=3000,
                value=350,
                step=50,
            )
        with col_c2:
            chunk_overlap = st.number_input(
                "Chunk Overlap (tokens)",
                min_value=0,
                max_value=800,
                value=50,
                step=10,
            )
        with col_c3:
            st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
            ingest_btn = st.button(
                "✦  Process & Index Documents",
                type="primary",
                use_container_width=True,
                disabled=not uploaded_files,
            )

        if ingest_btn and uploaded_files:
            if chunk_overlap >= chunk_size:
                st.error("Chunk overlap must be strictly less than chunk size.")
            else:
                progress = st.progress(0, text="Validating and extracting PDF pages...")
                payload: List[Tuple[str, bytes, str]] = [
                    (uf.name, uf.getvalue(), uf.type or "application/pdf")
                    for uf in uploaded_files
                ]
                try:
                    progress.progress(45, text="Generating embeddings and indexing in FAISS...")
                    resp = client.upload_documents(
                        files_payload=payload,
                        chunk_size=int(chunk_size),
                        chunk_overlap=int(chunk_overlap),
                    )
                    progress.progress(100, text="Ingestion complete!")
                    st.success(
                        f"Processed {resp['total_files']} file(s): "
                        f"{resp['indexed_count']} indexed, {resp['duplicate_count']} duplicate(s), {resp['failed_count']} failed."
                    )
                    for item in resp.get("results", []):
                        badge = (
                            "✅ INDEXED"
                            if item["status"] == "indexed"
                            else ("♻️ DUPLICATE" if item["status"] == "duplicate" else "❌ FAILED")
                        )
                        st.markdown(f"- **{item['filename']}** (`{badge}`) — {item['message']}")
                except Exception as exc:
                    progress.empty()
                    st.error(f"Upload failed: {exc}")

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

    # Search and Status Filter Bar
    s_col, f_col = st.columns([2.5, 1.2])
    with s_col:
        search_term = st.text_input(
            "Search documents",
            value="",
            placeholder="⌕  Search by filename...",
            label_visibility="collapsed",
        )
    with f_col:
        status_choice = st.selectbox(
            "Filter by status",
            options=["all", "indexed", "failed", "duplicate"],
            format_func=lambda s: f"Status: {s.capitalize()}",
            label_visibility="collapsed",
        )

    try:
        docs: List[Dict[str, Any]] = client.list_documents(
            status_filter=None if status_choice == "all" else status_choice
        )
    except Exception as exc:
        st.error(f"Failed to fetch documents: {exc}")
        return

    if search_term.strip():
        q_lower = search_term.strip().lower()
        docs = [d for d in docs if q_lower in d["filename"].lower()]

    if not docs:
        render_empty_state(
            title="Your knowledge base is empty",
            subtitle="Drag and drop PDF documents into the uploader above to extract text, generate embeddings, and start asking questions.",
            icon="📄",
        )
        return

    # Render Document Cards with Confirmation Before Deletion
    if "confirm_delete_doc_id" not in st.session_state:
        st.session_state.confirm_delete_doc_id = None

    selected_inspect_id = st.session_state.get("selected_inspect_doc_id")

    for doc in docs:
        doc_id = doc["document_id"]
        size_kb = round(doc["file_size_bytes"] / 1024, 1)
        upload_date = doc["uploaded_at"][:16].replace("T", " ")
        status_pill = (
            '<span class="dm-badge-indexed">● INDEXED</span>'
            if doc["status"] == "indexed"
            else f'<span class="dm-badge-failed">● {doc["status"].upper()}</span>'
        )

        with st.container(border=True):
            c_info, c_stats, c_actions = st.columns([2.8, 2.2, 1.3])
            with c_info:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        <div style="
                            width:42px; height:42px; border-radius:11px;
                            background:rgba(53,198,216,0.12); border:1px solid rgba(53,198,216,0.28);
                            display:flex; align-items:center; justify-content:center;
                            font-size:1.15rem; color:#35C6D8; flex-shrink:0;
                        ">PDF</div>
                        <div>
                            <div style="font-size:1rem; font-weight:600; color:#F4F6FA;">{doc['filename']}</div>
                            <div style="font-size:0.76rem; color:#A0A7B4; margin-top:0.15rem;">
                                ID: <code>{doc_id}</code> &nbsp;•&nbsp; Uploaded: {upload_date}
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_stats:
                st.markdown(
                    f"""
                    <div style="display:flex; align-items:center; gap:1.3rem; padding-top:0.35rem;">
                        <div>
                            <div style="font-size:0.72rem; color:#A0A7B4; text-transform:uppercase;">Size</div>
                            <div style="font-size:0.9rem; font-weight:600; color:#F4F6FA;">{size_kb} KB</div>
                        </div>
                        <div>
                            <div style="font-size:0.72rem; color:#A0A7B4; text-transform:uppercase;">Chunks</div>
                            <div style="font-size:0.9rem; font-weight:600; color:#F4F6FA;">{doc['chunk_count']} ({doc['page_count']}p)</div>
                        </div>
                        <div>
                            <div style="font-size:0.72rem; color:#A0A7B4; text-transform:uppercase;">Status</div>
                            <div style="margin-top:0.1rem;">{status_pill}</div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c_actions:
                if st.session_state.confirm_delete_doc_id == doc_id:
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        if st.button("Confirm", key=f"confirm_del_{doc_id}", type="primary", use_container_width=True):
                            client.delete_document(doc_id)
                            st.session_state.confirm_delete_doc_id = None
                            st.toast(f"Deleted {doc['filename']}")
                            st.rerun()
                    with dc2:
                        if st.button("Cancel", key=f"cancel_del_{doc_id}", use_container_width=True):
                            st.session_state.confirm_delete_doc_id = None
                            st.rerun()
                else:
                    if st.button("🗑️ Delete", key=f"req_del_{doc_id}", use_container_width=True):
                        st.session_state.confirm_delete_doc_id = doc_id
                        st.rerun()

            if doc.get("error_message"):
                st.error(f"Processing Error: {doc['error_message']}")

            is_inspected = selected_inspect_id == doc_id
            if st.checkbox(
                "Inspect Semantic Sections & Chunks",
                value=is_inspected,
                key=f"inspect_chk_{doc_id}",
            ):
                try:
                    detail = client.get_document_details(doc_id)
                    meta_col1, meta_col2 = st.columns(2)
                    with meta_col1:
                        st.caption(f"**SHA-256 Hash:** `{doc['file_hash']}`")
                        st.caption(f"**Chunk Settings:** `{doc['chunk_size']} tokens / {doc['chunk_overlap']} overlap`")
                    with meta_col2:
                        st.json(detail["document"].get("metadata", {}), expanded=False)

                    chunks = detail.get("chunks", [])
                    if chunks:
                        chunk_df = pd.DataFrame(
                            [
                                {
                                    "Chunk ID": c["chunk_id"],
                                    "Section": c.get("section_title", c.get("metadata", {}).get("section_title", "General")),
                                    "Page": c["page_number"],
                                    "Tokens": c["token_count"],
                                    "Content Preview": c["text"][:200] + ("..." if len(c["text"]) > 200 else ""),
                                }
                                for c in chunks
                            ]
                        )
                        st.dataframe(chunk_df, use_container_width=True, hide_index=True)
                except Exception as exc:
                    st.warning(f"Unable to load document details: {exc}")
