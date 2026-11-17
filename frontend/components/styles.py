from typing import Optional
import streamlit as st


def apply_custom_styles() -> None:
    """
    Inject the complete DocuMind AI dark SaaS design system CSS.
    Palette:
      - Main background:      #111319
      - Sidebar background:   #1C202B
      - Card background:      #202531
      - Secondary background: #262B36
      - Primary accent:       #35C6D8
      - Secondary accent:     #6478FF
      - Primary text:         #F4F6FA
      - Secondary text:       #A0A7B4
      - Border:               rgba(255,255,255,0.09)
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --bg-main: #111319;
            --bg-sidebar: #1C202B;
            --bg-card: #202531;
            --bg-secondary: #262B36;
            --accent-primary: #35C6D8;
            --accent-secondary: #6478FF;
            --text-primary: #F4F6FA;
            --text-secondary: #A0A7B4;
            --border-subtle: rgba(255, 255, 255, 0.09);
            --border-hover: rgba(53, 198, 216, 0.38);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
            color: var(--text-primary);
        }

        /* Hide default Streamlit multipage nav and header clutter */
        [data-testid="stSidebarNav"] {
            display: none !important;
        }
        header[data-testid="stHeader"] {
            background: transparent !important;
            height: 2.2rem !important;
        }
        #MainMenu, footer {
            visibility: hidden;
        }

        /* Main App Container with subtle teal & indigo ambient radial gradients */
        .stApp {
            background-color: var(--bg-main) !important;
            background-image:
                radial-gradient(circle at 14% 12%, rgba(53, 198, 216, 0.065) 0%, transparent 38%),
                radial-gradient(circle at 86% 18%, rgba(100, 120, 255, 0.06) 0%, transparent 42%),
                radial-gradient(circle at 50% 92%, rgba(53, 198, 216, 0.04) 0%, transparent 45%) !important;
        }

        .main .block-container {
            padding-top: 1.1rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2.2rem !important;
            padding-right: 2.2rem !important;
            max-width: 1320px !important;
        }

        /* Fixed Left Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--border-subtle) !important;
        }
        section[data-testid="stSidebar"] > div {
            padding-top: 1rem !important;
        }

        /* Sidebar Buttons (Navigation & History items) */
        section[data-testid="stSidebar"] .stButton > button {
            width: 100%;
            text-align: left !important;
            justify-content: flex-start !important;
            background-color: transparent !important;
            color: var(--text-secondary) !important;
            border: 1px solid transparent !important;
            border-radius: 10px !important;
            padding: 0.52rem 0.85rem !important;
            font-size: 0.88rem !important;
            font-weight: 500 !important;
            transition: all 0.18s ease !important;
            box-shadow: none !important;
        }
        section[data-testid="stSidebar"] .stButton > button:hover {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border-color: var(--border-subtle) !important;
        }
        /* Primary button inside sidebar = "+ New Chat" */
        section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, rgba(53, 198, 216, 0.16) 0%, rgba(100, 120, 255, 0.16) 100%) !important;
            color: var(--text-primary) !important;
            border: 1px solid rgba(53, 198, 216, 0.42) !important;
            justify-content: center !important;
            text-align: center !important;
            font-weight: 600 !important;
            padding: 0.6rem 1rem !important;
            margin-bottom: 0.4rem !important;
        }
        section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
            border-color: var(--accent-primary) !important;
            box-shadow: 0 0 16px rgba(53, 198, 216, 0.22) !important;
            transform: translateY(-1px);
        }

        /* Main Area Buttons */
        .main .stButton > button {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 10px !important;
            font-weight: 500 !important;
            padding: 0.5rem 1rem !important;
            transition: all 0.18s ease !important;
        }
        .main .stButton > button:hover {
            border-color: var(--border-hover) !important;
            background-color: #2c3240 !important;
            transform: translateY(-1px);
        }
        .main .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #35C6D8 0%, #6478FF 100%) !important;
            color: #0b0e14 !important;
            font-weight: 600 !important;
            border: none !important;
        }
        .main .stButton > button[kind="primary"]:hover {
            opacity: 0.94;
            box-shadow: 0 4px 18px rgba(53, 198, 216, 0.28) !important;
        }

        /* Inputs, Selectboxes, File Uploader, Expanders, Containers */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 16px !important;
            padding: 1.15rem !important;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(255, 255, 255, 0.14) !important;
        }

        [data-testid="stExpander"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 12px !important;
        }

        [data-testid="stFileUploader"] {
            background-color: var(--bg-secondary) !important;
            border: 1px dashed rgba(53, 198, 216, 0.4) !important;
            border-radius: 14px !important;
            padding: 1rem !important;
        }

        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
            background-color: var(--bg-secondary) !important;
            color: var(--text-primary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 10px !important;
        }

        /* Chat Input Bar */
        [data-testid="stChatInput"] {
            background-color: var(--bg-card) !important;
            border: 1px solid rgba(53, 198, 216, 0.32) !important;
            border-radius: 16px !important;
        }

        /* Chat Messages */
        [data-testid="stChatMessage"] {
            background-color: var(--bg-card) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: 14px !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.75rem !important;
        }

        /* Custom SaaS Components */
        .dm-topbar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.65rem 1.15rem;
            background: rgba(32, 37, 49, 0.78);
            backdrop-filter: blur(10px);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            margin-bottom: 1.35rem;
        }
        .dm-breadcrumb {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.86rem;
            color: var(--text-secondary);
        }
        .dm-breadcrumb strong {
            color: var(--text-primary);
            font-weight: 600;
        }
        .dm-provider-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.28rem 0.75rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            background: rgba(53, 198, 216, 0.12);
            color: var(--accent-primary);
            border: 1px solid rgba(53, 198, 216, 0.28);
        }

        /* Hero Banner */
        .dm-hero {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, #202531 0%, #191E29 55%, rgba(53, 198, 216, 0.12) 100%);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 1.85rem 2.1rem;
            margin-bottom: 1.4rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .dm-hero-title {
            font-size: 1.65rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            margin: 0 0 0.45rem 0;
        }
        .dm-hero-sub {
            font-size: 0.95rem;
            color: var(--text-secondary);
            margin: 0;
            max-width: 620px;
            line-height: 1.5;
        }

        /* Metric Card */
        .dm-metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 1.2rem 1.35rem;
            transition: all 0.2s ease;
            height: 100%;
        }
        .dm-metric-card:hover {
            border-color: var(--border-hover);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        }
        .dm-metric-top {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.65rem;
        }
        .dm-metric-label {
            font-size: 0.78rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }
        .dm-metric-icon {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            background: rgba(53, 198, 216, 0.1);
            border: 1px solid rgba(53, 198, 216, 0.22);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--accent-primary);
            font-size: 1rem;
        }
        .dm-metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            line-height: 1.15;
        }
        .dm-metric-sub {
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin-top: 0.4rem;
        }

        /* Page Section Header */
        .dm-page-header {
            margin-bottom: 1.3rem;
        }
        .dm-page-title {
            font-size: 1.55rem;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.02em;
            margin: 0 0 0.3rem 0;
        }
        .dm-page-subtitle {
            font-size: 0.92rem;
            color: var(--text-secondary);
            margin: 0;
        }

        /* Citation Card */
        .dm-citation-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--accent-primary);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.65rem;
        }
        .dm-citation-meta {
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--accent-primary);
            margin-bottom: 0.4rem;
        }
        .dm-citation-snippet {
            font-size: 0.86rem;
            color: var(--text-primary);
            line-height: 1.55;
        }

        /* Empty State Container */
        .dm-empty-state {
            background: var(--bg-card);
            border: 1px dashed rgba(255, 255, 255, 0.12);
            border-radius: 18px;
            padding: 2.6rem 1.8rem;
            text-align: center;
            margin: 1rem 0;
        }
        .dm-empty-icon {
            width: 56px;
            height: 56px;
            border-radius: 16px;
            margin: 0 auto 1rem auto;
            background: linear-gradient(135deg, rgba(53, 198, 216, 0.14) 0%, rgba(100, 120, 255, 0.14) 100%);
            border: 1px solid rgba(53, 198, 216, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: var(--accent-primary);
        }
        .dm-empty-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 0.35rem;
        }
        .dm-empty-sub {
            font-size: 0.88rem;
            color: var(--text-secondary);
            max-width: 460px;
            margin: 0 auto;
        }

        /* Status Pills */
        .dm-badge-indexed {
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            background: rgba(53, 198, 216, 0.14);
            color: #35C6D8;
            border: 1px solid rgba(53, 198, 216, 0.32);
        }
        .dm-badge-failed {
            display: inline-block;
            padding: 0.18rem 0.6rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 600;
            background: rgba(239, 68, 68, 0.15);
            color: #f87171;
            border: 1px solid rgba(239, 68, 68, 0.32);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_navbar(current_page: str, llm_label: str, embedding_model: str) -> None:
    """Render the top SaaS navigation bar with breadcrumb, LLM indicator, Settings, and Help."""
    left_col, right_col = st.columns([3, 1.4])
    with left_col:
        st.markdown(
            f"""
            <div class="dm-topbar">
                <div class="dm-breadcrumb">
                    <span>DocuMind AI</span>
                    <span>/</span>
                    <strong>{current_page}</strong>
                </div>
                <div class="dm-provider-pill">
                    <span>●</span>
                    <span>LLM: {llm_label}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right_col:
        c1, c2 = st.columns(2)
        with c1:
            with st.popover("⚙️ Settings", use_container_width=True):
                st.markdown("##### Workspace Configuration")
                st.caption(f"**Active LLM:** `{llm_label}`")
                st.caption(f"**Embedding Model:** `{embedding_model}`")
                st.caption("**Vector Store:** `FAISS (IndexFlatIP / Cosine)`")
                st.caption("**Default Chunk Size:** `800 tokens (150 overlap)`")
        with c2:
            with st.popover("❓ Help", use_container_width=True):
                st.markdown("##### DocuMind AI Guide")
                st.caption(
                    "1. **Document Management**: Upload PDF files to extract, chunk, and index embeddings in FAISS.\n"
                    "2. **AI Chat**: Ask questions grounded in your documents with verified page citations.\n"
                    "3. **Retrieval Explorer**: Compare Basic, MMR, and Hybrid search scores.\n"
                    "4. **Evaluation Center**: Run ground-truth RAG quality benchmarks."
                )


def render_page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="dm-page-header">
            <h1 class="dm-page-title">{title}</h1>
            <p class="dm-page-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str, value: str, subtitle: str = "", icon: str = "◈"
) -> None:
    st.markdown(
        f"""
        <div class="dm-metric-card">
            <div class="dm-metric-top">
                <span class="dm-metric-label">{label}</span>
                <span class="dm-metric-icon">{icon}</span>
            </div>
            <div class="dm-metric-value">{value}</div>
            <div class="dm-metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, subtitle: str, icon: str = "✦") -> None:
    st.markdown(
        f"""
        <div class="dm-empty-state">
            <div class="dm-empty-icon">{icon}</div>
            <div class="dm-empty-title">{title}</div>
            <div class="dm-empty-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
