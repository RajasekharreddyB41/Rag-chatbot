"""Full Streamlit chat UI with sidebar upload, settings panel, streaming responses, and source display."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

from app.config import get_config
from app.core.rag_chain import RAGChain
from app.services.document_ingestor import DocumentIngestor
from app.services.vector_store import VectorStoreManager
from app.utils.exceptions import (
    DocumentIngestionError,
    FileValidationError,
    LLMError,
    RAGChatbotError,
    RetrievalError,
)
from app.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Session-state helpers
# ──────────────────────────────────────────────────────────────────────

def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "rag_chain": None,
        "vector_store": None,
        "ingestor": None,
        "config": None,
        "indexed_files": [],
        "groq_api_key": "",
        "model_name": "llama-3.3-70b-versatile",
        "temperature": 0.1,
        "k_docs": 4,
        "show_sources": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        /* ── Global ───────────────────────────────────────────────── */
        .stApp {
            background-color: #F7F7F8;
        }

        /* ── Sidebar ─────────────────────────────────────────────── */
        [data-testid="stSidebar"] {
            background-color: #F3F4F6 !important;
            border-right: 1px solid #E5E7EB !important;
        }
        [data-testid="stSidebar"] .stMarkdown p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stTextInput label {
            color: #374151 !important;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: #111827 !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: #E5E7EB !important;
        }
        [data-testid="stSidebar"] .stCaption {
            color: #6B7280 !important;
        }

        /* ── Main content area ───────────────────────────────────── */
        .main .block-container {
            padding-top: 2rem;
            max-width: 860px;
        }

        /* ── Page title ──────────────────────────────────────────── */
        h1 {
            color: #111827 !important;
            font-weight: 700 !important;
        }

        /* ── Chat messages ───────────────────────────────────────── */
        [data-testid="stChatMessage"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 12px !important;
            padding: 1rem 1.25rem !important;
            margin-bottom: 0.75rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
        }
        [data-testid="stChatMessage"] p {
            color: #111827 !important;
        }

        /* ── Chat input ──────────────────────────────────────────── */
        [data-testid="stChatInput"] {
            border: 1.5px solid #E5E7EB !important;
            border-radius: 12px !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.06) !important;
        }
        [data-testid="stChatInput"]:focus-within {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
        }
        [data-testid="stChatInput"] textarea {
            color: #111827 !important;
        }

        /* ── Primary buttons ─────────────────────────────────────── */
        .stButton > button[kind="primary"],
        .stButton > button[data-testid*="primary"] {
            background-color: #6366F1 !important;
            border: none !important;
            color: #FFFFFF !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: background-color 0.2s, box-shadow 0.2s !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #4F46E5 !important;
            box-shadow: 0 4px 12px rgba(99,102,241,0.3) !important;
        }

        /* ── Secondary / default buttons (suggestion grid + sidebar) */
        .stButton > button {
            background-color: #FFFFFF !important;
            border: 1.5px solid #E5E7EB !important;
            color: #374151 !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            transition: border-color 0.2s, box-shadow 0.2s, color 0.2s !important;
        }
        .stButton > button:hover {
            border-color: #6366F1 !important;
            color: #6366F1 !important;
            box-shadow: 0 2px 8px rgba(99,102,241,0.15) !important;
            background-color: #F5F3FF !important;
        }

        /* ── File uploader ───────────────────────────────────────── */
        [data-testid="stFileUploader"] {
            border: 2px dashed #E5E7EB !important;
            border-radius: 10px !important;
            background-color: #FAFAFA !important;
            padding: 0.75rem !important;
            transition: border-color 0.2s !important;
        }
        [data-testid="stFileUploader"]:hover {
            border-color: #6366F1 !important;
            background-color: #F5F3FF !important;
        }
        [data-testid="stFileUploader"] label {
            color: #6B7280 !important;
        }

        /* ── Selectbox / dropdowns ───────────────────────────────── */
        [data-testid="stSelectbox"] > div > div {
            background-color: #FFFFFF !important;
            border: 1.5px solid #E5E7EB !important;
            border-radius: 8px !important;
            color: #111827 !important;
        }

        /* ── Sliders ─────────────────────────────────────────────── */
        [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
            background-color: #6366F1 !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stTickBar"] {
            background: linear-gradient(to right, #6366F1, #E5E7EB) !important;
        }

        /* ── Toggle ──────────────────────────────────────────────── */
        [data-testid="stToggle"] [data-checked="true"] {
            background-color: #6366F1 !important;
        }

        /* ── Source expander ─────────────────────────────────────── */
        [data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 8px !important;
            box-shadow: none !important;
        }
        [data-testid="stExpander"] summary {
            color: #6B7280 !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stExpander"] summary:hover {
            color: #6366F1 !important;
        }
        [data-testid="stExpander"] .streamlit-expanderContent p {
            color: #374151 !important;
            font-size: 0.88rem !important;
        }

        /* ── Info / success / warning / error alerts ─────────────── */
        [data-testid="stAlert"][kind="info"],
        div[data-baseweb="notification"][kind="info"] {
            background-color: #EEF2FF !important;
            border-left: 4px solid #6366F1 !important;
            color: #3730A3 !important;
            border-radius: 8px !important;
        }
        [data-testid="stAlert"][kind="success"],
        div[data-baseweb="notification"][kind="positive"] {
            background-color: #ECFDF5 !important;
            border-left: 4px solid #10B981 !important;
            color: #065F46 !important;
            border-radius: 8px !important;
        }
        [data-testid="stAlert"][kind="warning"] {
            background-color: #FFFBEB !important;
            border-left: 4px solid #F59E0B !important;
            color: #92400E !important;
            border-radius: 8px !important;
        }
        [data-testid="stAlert"][kind="error"],
        div[data-baseweb="notification"][kind="negative"] {
            background-color: #FEF2F2 !important;
            border-left: 4px solid #EF4444 !important;
            color: #991B1B !important;
            border-radius: 8px !important;
        }

        /* ── Metric cards ────────────────────────────────────────── */
        [data-testid="stMetric"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E5E7EB !important;
            border-radius: 10px !important;
            padding: 1rem 1.25rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
        }
        [data-testid="stMetricLabel"] {
            color: #6B7280 !important;
            font-size: 0.8rem !important;
        }
        [data-testid="stMetricValue"] {
            color: #111827 !important;
            font-weight: 700 !important;
        }

        /* ── Text input (API key) ────────────────────────────────── */
        [data-testid="stTextInput"] input {
            background-color: #FFFFFF !important;
            border: 1.5px solid #E5E7EB !important;
            border-radius: 8px !important;
            color: #111827 !important;
        }
        [data-testid="stTextInput"] input:focus {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
        }

        /* ── Progress bar ────────────────────────────────────────── */
        [data-testid="stProgress"] > div > div {
            background-color: #6366F1 !important;
        }

        /* ── Suggestion subtitle ─────────────────────────────────── */
        .suggestion-label {
            text-align: center;
            color: #6B7280;
            font-size: 0.9rem;
            margin-bottom: 0.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _get_services() -> tuple[DocumentIngestor, VectorStoreManager, RAGChain]:
    cfg = get_config()

    # Override with user-provided settings
    if st.session_state.groq_api_key:
        cfg.groq.api_key = st.session_state.groq_api_key
    cfg.groq.model = st.session_state.model_name
    cfg.groq.temperature = st.session_state.temperature
    cfg.retriever.k = st.session_state.k_docs

    if st.session_state.ingestor is None:
        st.session_state.ingestor = DocumentIngestor(cfg)
    if st.session_state.vector_store is None:
        st.session_state.vector_store = VectorStoreManager(cfg)
    if st.session_state.rag_chain is None:
        st.session_state.rag_chain = RAGChain(
            vector_store=st.session_state.vector_store, config=cfg
        )

    return (
        st.session_state.ingestor,
        st.session_state.vector_store,
        st.session_state.rag_chain,
    )


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────

def render_sidebar() -> None:
    with st.sidebar:
        st.title("🤖 RAG Chatbot")
        st.markdown("---")

        # ── API Key ──────────────────────────────────────────────────
        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state.groq_api_key,
            placeholder="gsk_...",
            help="Get your free API key at console.groq.com",
        )
        if api_key != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key
            st.session_state.rag_chain = None  # Reset chain

        # ── Model Settings ───────────────────────────────────────────
        st.subheader("⚙️ Model Settings")
        model = st.selectbox(
            "Groq Model",
            options=[
                "llama-3.3-70b-versatile",
                "llama3-70b-8192",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=["llama-3.3-70b-versatile", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"].index(
                st.session_state.model_name
            ),
        )
        if model != st.session_state.model_name:
            st.session_state.model_name = model
            st.session_state.rag_chain = None

        temperature = st.slider(
            "Temperature", min_value=0.0, max_value=1.0,
            value=st.session_state.temperature, step=0.05
        )
        if temperature != st.session_state.temperature:
            st.session_state.temperature = temperature
            st.session_state.rag_chain = None

        k_docs = st.slider(
            "Retrieved Documents (k)", min_value=1, max_value=10,
            value=st.session_state.k_docs
        )
        if k_docs != st.session_state.k_docs:
            st.session_state.k_docs = k_docs
            st.session_state.rag_chain = None

        st.session_state.show_sources = st.toggle(
            "Show Sources", value=st.session_state.show_sources
        )

        st.markdown("---")

        # ── Document Upload ──────────────────────────────────────────
        st.subheader("📁 Document Upload")
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt", "md", "csv"],
            accept_multiple_files=True,
            help="Supported: PDF, DOCX, TXT, MD, CSV (max 50 MB each)",
        )

        if uploaded_files and st.button("📤 Index Documents", type="primary"):
            _index_documents(uploaded_files)

        # ── Indexed Files ────────────────────────────────────────────
        if st.session_state.indexed_files:
            st.markdown("**Indexed Files:**")
            for fname in st.session_state.indexed_files:
                st.markdown(f"- ✅ {fname}")

        # ── Vector Store Stats ───────────────────────────────────────
        if st.button("📊 Collection Stats"):
            try:
                _, vs, _ = _get_services()
                stats = vs.collection_stats()
                st.json(stats)
            except RAGChatbotError as e:
                st.error(str(e))

        if st.button("🗑️ Clear Collection", type="secondary"):
            try:
                _, vs, _ = _get_services()
                vs.delete_collection()
                st.session_state.indexed_files = []
                st.session_state.rag_chain = None
                st.session_state.vector_store = None
                st.success("Collection cleared.")
                st.rerun()
            except RAGChatbotError as e:
                st.error(str(e))

        st.markdown("---")

        # ── Chat Controls ────────────────────────────────────────────
        st.subheader("💬 Chat")
        if st.button("🔄 New Conversation"):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            if st.session_state.rag_chain:
                st.session_state.rag_chain.clear_all_sessions()
            st.rerun()

        st.markdown("---")
        cfg = get_config()
        st.caption(f"v{cfg.app_version} | Model: {st.session_state.model_name}")


def _index_documents(uploaded_files: list) -> None:
    ingestor, vs, _ = _get_services()
    progress = st.progress(0, text="Indexing documents...")
    new_files: list[str] = []

    for i, uf in enumerate(uploaded_files):
        progress.progress((i + 1) / len(uploaded_files), text=f"Indexing {uf.name}…")
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(uf.name).suffix
            ) as tmp:
                tmp.write(uf.read())
                tmp_path = Path(tmp.name)

            chunks = ingestor.ingest_file(tmp_path)

            # Patch source metadata back to original filename
            for chunk in chunks:
                chunk.metadata["source"] = uf.name

            vs.add_documents(chunks)
            new_files.append(uf.name)
            tmp_path.unlink(missing_ok=True)

        except (DocumentIngestionError, FileValidationError) as exc:
            st.error(f"❌ {uf.name}: {exc}")
        except Exception as exc:
            st.error(f"❌ Unexpected error for {uf.name}: {exc}")
            logger.exception("Unexpected error indexing %s", uf.name)

    progress.empty()

    # Reset chain so it picks up new docs
    st.session_state.rag_chain = None

    for fname in new_files:
        if fname not in st.session_state.indexed_files:
            st.session_state.indexed_files.append(fname)

    if new_files:
        st.success(f"✅ Indexed {len(new_files)} file(s): {', '.join(new_files)}")


# ──────────────────────────────────────────────────────────────────────
# Main chat UI
# ──────────────────────────────────────────────────────────────────────

def render_chat() -> None:
    st.title("💬 RAG Chatbot")

    cfg = get_config()

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and st.session_state.show_sources:
                _render_sources(msg["sources"])

    # Guard: require API key
    if not st.session_state.groq_api_key and not cfg.groq.api_key:
        st.info("👈 Enter your **Groq API Key** in the sidebar to start chatting.")
        return

    # Guard: require indexed documents
    try:
        _, vs, _ = _get_services()
        stats = vs.collection_stats()
        if stats["document_count"] == 0:
            st.info("👈 **Upload and index documents** in the sidebar to begin.")
            return
    except RAGChatbotError:
        st.info("👈 Upload and index documents to begin.")
        return

    # Suggestion buttons on empty chat
    if not st.session_state.messages:
        _render_suggestions()

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents…"):
        _handle_user_query(prompt)


_SUGGESTIONS = [
    "📝 Summarize this document",
    "🔑 What are the key points?",
    "❓ What is this document about?",
    "📊 List all important facts and figures",
    "🔍 What are the main conclusions?",
    "💡 Explain this in simple terms",
]


def _render_suggestions() -> None:
    st.markdown(
        "<p class='suggestion-label'>Try one of these to get started</p>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    for i, suggestion in enumerate(_SUGGESTIONS):
        if cols[i % 3].button(suggestion, use_container_width=True, key=f"suggestion_{i}"):
            _handle_user_query(suggestion)

    st.markdown("<div style='margin-bottom:1rem;'></div>", unsafe_allow_html=True)


def _handle_user_query(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        _stream_response(prompt)


def _stream_response(question: str) -> None:
    """Stream a response and display it, then save to history."""
    _, _, chain = _get_services()
    session_id = st.session_state.session_id

    # First retrieve sources (before streaming so we can display them)
    sources: list[dict] = []
    try:
        source_docs = chain.get_sources_for_query(question)
        sources = chain._extract_sources(source_docs)
    except RetrievalError as exc:
        st.error(f"Retrieval error: {exc}")
        return

    # Stream response
    answer_placeholder = st.empty()
    full_answer = ""

    try:
        for token in chain.stream(question, session_id=session_id):
            full_answer += token
            answer_placeholder.markdown(full_answer + "▌")
        answer_placeholder.markdown(full_answer)
    except LLMError as exc:
        st.error(f"LLM error: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        logger.exception("Unexpected error during streaming")
        return

    # Show sources
    if sources and st.session_state.show_sources:
        _render_sources(sources)

    # Save to session
    st.session_state.messages.append(
        {"role": "assistant", "content": full_answer, "sources": sources}
    )


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("📚 Sources", expanded=False):
        for src in sources:
            page_info = f" (page {src['page']})" if src.get("page") != "" else ""
            file_type = f" [{src.get('file_type', '').upper()}]" if src.get("file_type") else ""
            st.markdown(f"- **{src['source']}**{page_info}{file_type}")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()
    _inject_css()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
