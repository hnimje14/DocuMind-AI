from typing import Any, Dict, List
import pandas as pd
import streamlit as st
from frontend.components.styles import (
    render_empty_state,
    render_metric_card,
    render_page_header,
)
from frontend.utils.api_client import BackendAPIClient


def render_evaluation_dashboard_page(client: BackendAPIClient) -> None:
    render_page_header(
        title="RAG Performance Analytics",
        subtitle="Measure retrieval quality and answer reliability.",
    )

    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([1.1, 1.5, 1.5, 1.2])
        with c1:
            top_k = st.slider("Evaluation Top-K", min_value=1, max_value=10, value=3)
        with c2:
            modes = st.multiselect(
                "Retrieval Modes",
                options=["basic", "mmr", "hybrid"],
                default=["basic", "mmr", "hybrid"],
            )
        with c3:
            compare_chunks = st.multiselect(
                "Chunk Sizes (tokens)",
                options=[250, 350, 450, 800],
                default=[250, 350, 450],
            )
        with c4:
            st.markdown("<div style='height:1.72rem;'></div>", unsafe_allow_html=True)
            run_eval_btn = st.button(
                "✦  Run Evaluation",
                type="primary",
                use_container_width=True,
                disabled=(not modes or not compare_chunks),
            )

    if run_eval_btn:
        with st.spinner("Running ground-truth retrieval & generation benchmark..."):
            try:
                eval_data = client.run_evaluation(
                    {
                        "top_k": top_k,
                        "similarity_threshold": 0.10,
                        "modes_to_compare": modes,
                        "chunk_sizes_to_compare": compare_chunks,
                    }
                )
                st.session_state.latest_eval_data = eval_data
                st.toast(f"Completed evaluation run {eval_data['run_id']}")
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

    eval_data: Dict[str, Any] = st.session_state.get("latest_eval_data")
    if not eval_data:
        try:
            eval_data = client.get_latest_evaluation()
        except Exception:
            eval_data = None

    if not eval_data or not eval_data.get("summaries"):
        render_empty_state(
            title="No evaluation benchmark has been executed yet",
            subtitle="Select your desired retrieval modes and chunk sizes above, then click Run Evaluation to compute real Precision@K, Recall@K, MRR, Hit Rate, Faithfulness, Answer Relevance, and Citation Correctness metrics.",
            icon="◧",
        )
        return

    summaries: List[Dict[str, Any]] = eval_data["summaries"]
    best = max(
        summaries,
        key=lambda s: (s["mrr"] + s["recall_at_k"] + s["faithfulness"] + s["citation_correctness"]),
    )

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin:1.1rem 0 0.75rem 0;">
            <span style="font-size:1.05rem; font-weight:600; color:#F4F6FA;">
                Core Performance Metrics — <span style="color:#35C6D8;">{best['mode'].upper()}</span> (Chunk Size: {best['chunk_size']}, Top-K: {best['top_k']})
            </span>
            <span style="font-size:0.78rem; color:#A0A7B4;">Run ID: <code>{eval_data.get('run_id', 'latest')}</code> • Dataset: {eval_data.get('dataset_size', 6)} QA pairs</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 7 Metric Cards: Row 1 (4 Retrieval Metrics) + Row 2 (3 Generation Metrics)
    r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
    with r1_c1:
        render_metric_card("Precision@K", f"{best['precision_at_k']:.2%}", f"Top-{best['top_k']} precision", "⌕")
    with r1_c2:
        render_metric_card("Recall@K", f"{best['recall_at_k']:.2%}", "Ground-truth page recall", "◈")
    with r1_c3:
        render_metric_card("Mean Reciprocal Rank", f"{best['mrr']:.2%}", "First relevant rank score", "◧")
    with r1_c4:
        render_metric_card("Hit Rate", f"{best['hit_rate']:.2%}", "Queries with ≥1 relevant hit", "✦")

    st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)

    r2_c1, r2_c2, r2_c3 = st.columns(3)
    with r2_c1:
        render_metric_card("Faithfulness", f"{best['faithfulness']:.2%}", f"Context relevance: {best['context_relevance']:.2%}", "🛡️")
    with r2_c2:
        render_metric_card("Answer Relevance", f"{best['answer_relevance']:.2%}", "Alignment with ground-truth answer", "💬")
    with r2_c3:
        render_metric_card("Citation Correctness", f"{best['citation_correctness']:.2%}", "Verified document & page accuracy", "📌")

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

    # Charts comparing Basic, MMR, and Hybrid retrieval + Chunk-Size Comparison
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:1rem; font-weight:600; color:#F4F6FA; margin-bottom:0.65rem;">Retrieval Mode Comparison (Basic vs. MMR vs. Hybrid)</div>',
                unsafe_allow_html=True,
            )
            mode_rows = []
            seen_modes = set()
            for s in summaries:
                if s["mode"] not in seen_modes:
                    seen_modes.add(s["mode"])
                    mode_rows.append(
                        {
                            "Mode": s["mode"].upper(),
                            "Recall@K": s["recall_at_k"],
                            "MRR": s["mrr"],
                            "Faithfulness": s["faithfulness"],
                            "Answer Relevance": s["answer_relevance"],
                            "Citation Correctness": s["citation_correctness"],
                        }
                    )
            mode_df = pd.DataFrame(mode_rows).set_index("Mode")
            st.bar_chart(mode_df, color=["#35C6D8", "#6478FF", "#4AD991", "#F59E0B", "#EC4899"])

    with chart_col2:
        with st.container(border=True):
            st.markdown(
                '<div style="font-size:1rem; font-weight:600; color:#F4F6FA; margin-bottom:0.65rem;">Chunk-Size Comparison Section</div>',
                unsafe_allow_html=True,
            )
            chunk_rows = []
            for s in summaries:
                chunk_rows.append(
                    {
                        "Chunk Size (tokens)": s["chunk_size"],
                        "Mode": s["mode"].upper(),
                        "Precision@K": s["precision_at_k"],
                        "Recall@K": s["recall_at_k"],
                        "MRR": s["mrr"],
                        "Hit Rate": s["hit_rate"],
                        "Faithfulness": s["faithfulness"],
                        "Citation Correctness": s["citation_correctness"],
                    }
                )
            st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height:0.85rem;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:1rem; font-weight:600; color:#F4F6FA; margin-bottom:0.65rem;">Ground-Truth Question Inspection</div>',
            unsafe_allow_html=True,
        )
        sel_idx = st.selectbox(
            "Select Benchmark Configuration",
            options=list(range(len(summaries))),
            format_func=lambda i: f"{summaries[i]['mode'].upper()} • Chunk Size: {summaries[i]['chunk_size']} tokens • Top-K: {summaries[i]['top_k']}",
        )
        for qd in summaries[sel_idx].get("details", []):
            with st.expander(
                f"{qd['question_id'].upper()} — {qd['question']}  (MRR: {qd['reciprocal_rank']:.2f} | Citation Correctness: {qd['citation_correctness']:.2f})"
            ):
                qa_c1, qa_c2 = st.columns(2)
                with qa_c1:
                    st.markdown("**Expected Ground-Truth Answer**")
                    st.info(qd["expected_answer"])
                    st.caption(f"Expected Sources: `{', '.join(qd['expected_docs'])}` • Pages: `{qd['expected_pages']}`")
                with qa_c2:
                    st.markdown("**Generated RAG Answer**")
                    st.success(qd["generated_answer"])
                    st.caption(f"Retrieved Sources: `{', '.join(qd['retrieved_docs'])}` • Pages: `{qd['retrieved_pages']}`")
