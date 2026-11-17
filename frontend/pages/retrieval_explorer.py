from typing import Any, Dict, List
import streamlit as st
from frontend.components.styles import render_empty_state, render_page_header
from frontend.utils.api_client import BackendAPIClient


def _relevance_badge(score: float) -> str:
    if score >= 0.45:
        return '<span class="dm-badge-indexed">HIGH RELEVANCE</span>'
    if score >= 0.25:
        return (
            '<span style="display:inline-block; padding:0.18rem 0.6rem; border-radius:999px; '
            'font-size:0.72rem; font-weight:600; background:rgba(100,120,255,0.16); '
            'color:#6478FF; border:1px solid rgba(100,120,255,0.35);">MODERATE RELEVANCE</span>'
        )
    return (
        '<span style="display:inline-block; padding:0.18rem 0.6rem; border-radius:999px; '
        'font-size:0.72rem; font-weight:600; background:rgba(160,167,180,0.14); '
        'color:#A0A7B4; border:1px solid rgba(160,167,180,0.28);">THRESHOLD MATCH</span>'
    )


def render_retrieval_explorer_page(client: BackendAPIClient) -> None:
    render_page_header(
        title="Retrieval Intelligence",
        subtitle="Inspect how your AI system finds relevant information.",
    )

    try:
        docs = client.list_documents(status_filter="indexed")
    except Exception:
        docs = []
    doc_map = {d["document_id"]: d["filename"] for d in docs}

    with st.container(border=True):
        query = st.text_input(
            "Search Query",
            value="",
            placeholder="⌕  Enter a semantic or keyword search query to inspect chunk retrieval...",
        )

        c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
        with c1:
            mode = st.selectbox(
                "Retrieval Mode",
                options=["basic", "mmr", "hybrid"],
                format_func=lambda m: {
                    "basic": "Basic retrieval (Semantic FAISS)",
                    "mmr": "MMR retrieval (Max Marginal Relevance)",
                    "hybrid": "Hybrid retrieval (Semantic + BM25)",
                }[m],
            )
        with c2:
            top_k = st.slider("Top-K Control", min_value=1, max_value=15, value=5)
        with c3:
            threshold = st.slider(
                "Similarity Threshold",
                min_value=0.0,
                max_value=0.85,
                value=0.05,
                step=0.05,
            )
        with c4:
            mmr_lambda = st.slider(
                "MMR Diversity (λ)",
                min_value=0.1,
                max_value=1.0,
                value=0.65,
                step=0.05,
                disabled=(mode != "mmr"),
            )

        b_col1, b_col2, b_col3 = st.columns([2.2, 1.1, 1.1])
        with b_col1:
            selected_docs = st.multiselect(
                "Filter by Document",
                options=list(doc_map.keys()),
                format_func=lambda did: doc_map.get(did, did),
                placeholder="All indexed documents",
            )
        with b_col2:
            st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
            use_compression = st.checkbox("Context Compression", value=False)
        with b_col3:
            st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
            run_btn = st.button(
                "⌕  Inspect Retrieval",
                type="primary",
                use_container_width=True,
                disabled=not query.strip(),
            )

    if not run_btn and "last_retrieval_resp" not in st.session_state:
        render_empty_state(
            title="Enter a search query to inspect vector retrieval",
            subtitle="Compare Basic semantic similarity, Maximum Marginal Relevance (MMR), and Hybrid BM25+FAISS fusion across your indexed chunks.",
            icon="⌕",
        )
        return

    if run_btn and query.strip():
        with st.spinner("Executing retrieval query against FAISS & BM25..."):
            try:
                st.session_state.last_retrieval_resp = client.search_retrieval(
                    {
                        "query": query,
                        "mode": mode,
                        "top_k": top_k,
                        "similarity_threshold": threshold,
                        "document_ids": selected_docs or None,
                        "use_context_compression": use_compression,
                        "mmr_lambda": mmr_lambda,
                    }
                )
            except Exception as exc:
                st.error(f"Retrieval query failed: {exc}")
                return

    resp: Dict[str, Any] = st.session_state.get("last_retrieval_resp", {})
    results: List[Dict[str, Any]] = resp.get("results", [])

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin:1.1rem 0 0.75rem 0;">
            <span style="font-size:1.05rem; font-weight:600; color:#F4F6FA;">
                Retrieved Chunks ({len(results)})
            </span>
            <span style="font-size:0.8rem; color:#A0A7B4;">
                Mode: <strong style="color:#35C6D8;">{resp.get('mode', mode).upper()}</strong>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not results:
        render_empty_state(
            title="No chunks matched above the similarity threshold",
            subtitle="Try lowering the similarity threshold, broadening your search query, or uploading relevant PDF documents.",
            icon="◈",
        )
        return

    for rank, item in enumerate(results, start=1):
        score = float(item["score"])
        pct = min(100, max(5, int(round(score * 100))))
        badge_html = _relevance_badge(score)
        st.markdown(
            f"""
            <div style="
                background:#202531; border:1px solid rgba(255,255,255,0.09);
                border-radius:16px; padding:1.2rem 1.4rem; margin-bottom:0.85rem;
            ">
                <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.65rem;">
                    <div style="display:flex; align-items:center; gap:0.65rem;">
                        <span style="
                            background:rgba(53,198,216,0.14); color:#35C6D8;
                            border:1px solid rgba(53,198,216,0.3); border-radius:8px;
                            padding:0.2rem 0.55rem; font-size:0.78rem; font-weight:700;
                        ">#{rank}</span>
                        <span style="font-size:0.98rem; font-weight:600; color:#F4F6FA;">📄 {item['filename']}</span>
                        <span style="font-size:0.8rem; color:#A0A7B4;">• Page {item['page_number']} • Section: <strong style="color:#F4F6FA;">{item.get('section_title', item.get('metadata', {}).get('section_title', 'General'))}</strong></span>
                    </div>
                    <div style="display:flex; align-items:center; gap:0.75rem;">
                        {badge_html}
                        <span style="font-size:0.88rem; font-weight:700; color:#35C6D8;">Score: {score:.4f}</span>
                    </div>
                </div>
                <div style="width:100%; height:5px; background:#262B36; border-radius:999px; overflow:hidden; margin-bottom:0.85rem;">
                    <div style="width:{pct}%; height:100%; background:linear-gradient(90deg, #35C6D8 0%, #6478FF 100%);"></div>
                </div>
                <div style="
                    background:#262B36; border:1px solid rgba(255,255,255,0.06);
                    border-radius:10px; padding:0.85rem 1rem; font-size:0.88rem;
                    color:#F4F6FA; line-height:1.6; margin-bottom:0.55rem;
                ">{item['text']}</div>
                <div style="display:flex; align-items:center; justify-content:space-between; font-size:0.75rem; color:#A0A7B4;">
                    <span>Chunk ID: <code>{item['chunk_id']}</code></span>
                    <span>Method: <code>{item['retrieval_method']}</code></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
