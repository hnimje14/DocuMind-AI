import sys
from pathlib import Path
from typing import Any, Dict, List
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.components.styles import apply_custom_styles, render_top_navbar
from frontend.utils.api_client import BackendAPIClient


st.set_page_config(
    page_title="DocuMind AI — Intelligent Document Q&A",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_styles()


@st.cache_resource
def get_api_client() -> BackendAPIClient:
    return BackendAPIClient()


NAV_ITEMS = [
    ("Dashboard", "◈"),
    ("Document Management", "▤"),
    ("AI Chat", "✦"),
    ("Retrieval Explorer", "⌕"),
    ("Evaluation Center", "◧"),
]


def main() -> None:
    client = get_api_client()

    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"

    # Fetch live stats and session list from backend
    try:
        stats: Dict[str, Any] = client.get_health_stats()
        sessions: List[Dict[str, Any]] = client.list_sessions()
        backend_healthy = True
    except Exception:
        stats = {
            "status": "offline",
            "llm_provider": "mock",
            "llm_model": "extractive-rag-v1",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "total_documents": 0,
            "total_chunks": 0,
            "vector_index_size": 0,
            "total_sessions": 0,
            "total_messages": 0,
            "recent_documents": [],
        }
        sessions = []
        backend_healthy = False

    raw_provider = stats.get("llm_provider", "mock")
    llm_display = (
        "Mock Mode"
        if raw_provider.lower().startswith("mock")
        else raw_provider.upper()
    )

    # ========================================================================
    # A. FIXED LEFT SIDEBAR
    # ========================================================================
    with st.sidebar:
        # Top Branding + Search Popover
        brand_col, search_col = st.columns([3.8, 1])
        with brand_col:
            st.markdown(
                """
                <div style="display:flex; align-items:center; gap:0.65rem; padding:0.2rem 0 0.75rem 0;">
                    <div style="
                        width:36px; height:36px; border-radius:10px;
                        background:linear-gradient(135deg, #35C6D8 0%, #6478FF 100%);
                        display:flex; align-items:center; justify-content:center;
                        color:#0b0e14; font-weight:800; font-size:1.1rem;
                        box-shadow: 0 4px 14px rgba(53,198,216,0.28);
                    ">✦</div>
                    <div>
                        <div style="font-size:1.05rem; font-weight:700; color:#F4F6FA; letter-spacing:-0.01em; line-height:1.15;">
                            DocuMind AI
                        </div>
                        <div style="font-size:0.72rem; color:#A0A7B4;">
                            Knowledge Workspace
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with search_col:
            with st.popover("⌕", help="Quick Search Conversations & Pages"):
                q_nav = st.text_input(
                    "Jump to page or session",
                    placeholder="Filter...",
                    key="sidebar_quick_filter",
                )
                if q_nav:
                    for p_name, _ in NAV_ITEMS:
                        if q_nav.lower() in p_name.lower():
                            if st.button(f"Go to {p_name}", key=f"jump_{p_name}"):
                                st.session_state.current_page = p_name
                                st.rerun()

        # Primary Action: + New Chat Button
        if st.button("＋  New Chat", type="primary", use_container_width=True, key="sidebar_new_chat"):
            try:
                created = client.create_session("New Conversation")
                st.session_state.active_session_id = created["session_id"]
            except Exception:
                st.session_state.active_session_id = None
            st.session_state.current_page = "AI Chat"
            st.rerun()

        st.markdown(
            '<div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#A0A7B4; margin:0.9rem 0 0.35rem 0.3rem;">Navigation</div>',
            unsafe_allow_html=True,
        )

        # Navigation Section
        for page_name, icon_char in NAV_ITEMS:
            is_active = st.session_state.current_page == page_name
            label = f"{icon_char}   {page_name}"
            if is_active:
                st.markdown(
                    f"""
                    <div style="
                        background: linear-gradient(90deg, rgba(53,198,216,0.16) 0%, rgba(38,43,54,0.6) 100%);
                        border: 1px solid rgba(53,198,216,0.35);
                        border-left: 3px solid #35C6D8;
                        border-radius: 10px;
                        padding: 0.52rem 0.85rem;
                        font-size: 0.88rem;
                        font-weight: 600;
                        color: #F4F6FA;
                        margin-bottom: 0.25rem;
                    ">{icon_char}&nbsp;&nbsp;&nbsp;{page_name}</div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                if st.button(label, key=f"nav_{page_name}", use_container_width=True):
                    st.session_state.current_page = page_name
                    st.rerun()

        # History Section: Recent Conversations
        st.markdown(
            '<div style="font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; color:#A0A7B4; margin:1.15rem 0 0.35rem 0.3rem;">Recent Conversations</div>',
            unsafe_allow_html=True,
        )
        if not sessions:
            st.markdown(
                '<div style="font-size:0.8rem; color:#A0A7B4; padding:0.35rem 0.55rem; opacity:0.75;">No recent conversations</div>',
                unsafe_allow_html=True,
            )
        else:
            for sess in sessions[:6]:
                title_short = (
                    sess["title"][:26] + "…"
                    if len(sess["title"]) > 26
                    else sess["title"]
                )
                if st.button(
                    f"💬  {title_short}",
                    key=f"hist_nav_{sess['session_id']}",
                    use_container_width=True,
                    help=f"{sess['title']} ({sess['message_count']} messages)",
                ):
                    st.session_state.active_session_id = sess["session_id"]
                    st.session_state.current_page = "AI Chat"
                    st.rerun()

        st.divider()

        # Bottom Section: Backend Health, LLM Status, Settings, Version
        status_dot_color = "#35C6D8" if backend_healthy else "#f87171"
        status_text = f"Backend Healthy ({client.mode.upper()})" if backend_healthy else "Backend Offline"
        st.markdown(
            f"""
            <div style="
                background:#202531;
                border:1px solid rgba(255,255,255,0.09);
                border-radius:12px;
                padding:0.75rem 0.9rem;
                font-size:0.78rem;
            ">
                <div style="display:flex; align-items:center; gap:0.45rem; font-weight:600; color:#F4F6FA; margin-bottom:0.35rem;">
                    <span style="color:{status_dot_color}; font-size:0.9rem;">●</span>
                    <span>{status_text}</span>
                </div>
                <div style="color:#A0A7B4; line-height:1.5;">
                    <div><strong>LLM:</strong> {llm_display}</div>
                    <div><strong>Vectors:</strong> {stats.get('vector_index_size', 0)} chunks</div>
                    <div style="margin-top:0.25rem; font-size:0.7rem; opacity:0.75;">DocuMind AI • v1.0.0</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ========================================================================
    # B. TOP NAVIGATION BAR
    # ========================================================================
    render_top_navbar(
        current_page=st.session_state.current_page,
        llm_label=llm_display,
        embedding_model=stats.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
    )

    # ========================================================================
    # C. MAIN CONTENT AREA (LAZY PAGE LOADING)
    # ========================================================================
    active_page = st.session_state.current_page
    if active_page == "Dashboard":
        from frontend.pages.dashboard import render_dashboard_page
        render_dashboard_page(client)
    elif active_page == "Document Management":
        from frontend.pages.document_management import render_document_management_page
        render_document_management_page(client)
    elif active_page == "AI Chat":
        from frontend.pages.ai_chat import render_ai_chat_page
        render_ai_chat_page(client)
    elif active_page == "Retrieval Explorer":
        from frontend.pages.retrieval_explorer import render_retrieval_explorer_page
        render_retrieval_explorer_page(client)
    elif active_page == "Evaluation Center":
        from frontend.pages.evaluation_dashboard import render_evaluation_dashboard_page
        render_evaluation_dashboard_page(client)


if __name__ == "__main__":
    main()
