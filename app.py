# ============================================================
# app.py — Main Streamlit Application
# ============================================================
# Research Paper Agent — Professional Edition
#
# A RAG-powered AI tool for academic research that helps you:
#   - Upload and process multiple PDF research papers
#   - Ask natural language questions and get cited answers
#   - Discover research gaps and thesis opportunities
#   - Generate structured paper summaries
#   - Compare papers across different dimensions
#   - Generate academic citations (APA, IEEE, BibTeX)
#
# Architecture:
#   app.py → parser.py → embedder.py → vectorstore.py → agent.py → Groq LLM
#
# Run with: streamlit run app.py
# ============================================================

import streamlit as st

# ── Page Configuration ────────────────────────────────────────────────────────
# Must be the FIRST Streamlit command called — configures browser tab and layout
st.set_page_config(
    page_title="Research Paper Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Research Paper Agent — AI-powered academic research tool"
    }
)

# ── Local Module Imports ──────────────────────────────────────────────────────
# Import after set_page_config to avoid Streamlit ordering warnings
from parser import extract_text_from_pdf, chunk_text, get_paper_metadata
from vectorstore import (
    get_or_create_collection,
    add_paper,
    delete_paper,
    get_collection_stats
)
from agent import (
    ask_question,
    find_research_gaps,
    summarize_papers,
    compare_papers,
    generate_citation
)


# ── Custom CSS Styling ────────────────────────────────────────────────────────
def inject_custom_css():
    """
    Inject custom CSS to improve the UI appearance.
    Streamlit's default styling is functional but basic — this adds polish.
    """
    st.markdown("""
    <style>
        /* Sidebar header styling */
        .sidebar-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }

        /* Paper card in sidebar */
        .paper-card {
            background: #f0f2f6;
            border-left: 4px solid #1f77b4;
            padding: 8px 12px;
            border-radius: 4px;
            margin: 4px 0;
            font-size: 0.85rem;
        }

        /* Metric cards */
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
        }

        /* Section dividers */
        .section-divider {
            border-top: 2px solid #e0e0e0;
            margin: 1.5rem 0;
        }

        /* Hide Streamlit default footer */
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────
def init_session_state():
    """
    Initialize all Streamlit session state variables on first load.

    Session state persists data across reruns (user interactions).
    Without this, all data would reset every time the user clicks anything.

    Variables:
        collection      → ChromaDB collection holding all paper embeddings
        papers_loaded   → dict of {filename: metadata} for uploaded papers
        chat_history    → list of {role, content} dicts for Q&A tab
        total_chunks    → running count of stored embedding chunks
    """
    if "collection" not in st.session_state:
        # Create the vector database collection for this session
        st.session_state.collection = get_or_create_collection()

    if "papers_loaded" not in st.session_state:
        # Store paper metadata: {filename: {title, author, page_count, word_count}}
        st.session_state.papers_loaded = {}

    if "chat_history" not in st.session_state:
        # Chat messages: [{"role": "user"/"assistant", "content": "..."}]
        st.session_state.chat_history = []

    if "total_chunks" not in st.session_state:
        # Total number of text chunks stored in ChromaDB
        st.session_state.total_chunks = 0


# ── Sidebar: Paper Upload & Management ───────────────────────────────────────
def render_sidebar():
    """
    Render the sidebar with paper upload functionality and session statistics.

    The sidebar is always visible and lets users:
        1. Upload PDF papers
        2. See which papers are loaded
        3. Remove individual papers
        4. View session statistics
    """
    with st.sidebar:
        # App branding
        st.markdown("## 🔬 Research Agent")
        st.markdown("*AI-powered academic research tool*")
        st.divider()

        # ── Upload Section ────────────────────────────────────────────────────
        st.markdown("### 📄 Upload Papers")
        st.caption("Supports PDF files. Multiple files allowed.")

        uploaded_files = st.file_uploader(
            label="Drop PDF files here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"  # hide redundant label since we have caption
        )

        # Process newly uploaded files
        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Skip if this paper is already in the session
                if uploaded_file.name not in st.session_state.papers_loaded:
                    _process_uploaded_paper(uploaded_file)

        # ── Loaded Papers List ────────────────────────────────────────────────
        if st.session_state.papers_loaded:
            st.divider()
            st.markdown("### 📚 Loaded Papers")
            st.caption(f"{len(st.session_state.papers_loaded)} paper(s) in session")

            for paper_name, meta in st.session_state.papers_loaded.items():
                # Show each paper with its stats and a remove button
                with st.expander(f"📄 {paper_name[:30]}...", expanded=False):
                    st.write(f"**Pages:** {meta.get('page_count', '?')}")
                    st.write(f"**Words:** {meta.get('word_count', '?'):,}")
                    st.write(f"**Author:** {meta.get('author', 'Unknown')}")

                    # Remove button — deletes paper's chunks from ChromaDB
                    if st.button(
                        "🗑️ Remove",
                        key=f"remove_{paper_name}",
                        use_container_width=True
                    ):
                        _remove_paper(paper_name)
                        st.rerun()

            # ── Session Statistics ────────────────────────────────────────────
            st.divider()
            st.markdown("### 📊 Session Stats")

            stats = get_collection_stats(st.session_state.collection)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Papers", len(st.session_state.papers_loaded))
            with col2:
                st.metric("Chunks", stats["total_chunks"])

            # Clear all button — resets the entire session
            st.divider()
            if st.button(
                "🔄 Clear All & Start Over",
                use_container_width=True,
                type="secondary"
            ):
                # Reset session state — app will reinitialize on rerun
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()


def _process_uploaded_paper(uploaded_file):
    """
    Process a single uploaded PDF: extract, chunk, embed, and store.

    This is the core ingestion pipeline:
        PDF bytes → text extraction → cleaning → chunking → embedding → ChromaDB

    Args:
        uploaded_file: Streamlit UploadedFile object from st.file_uploader.
    """
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            # Read raw bytes from the uploaded file
            file_bytes = uploaded_file.read()

            # Step 1: Extract metadata (title, author, page count)
            metadata = get_paper_metadata(file_bytes, uploaded_file.name)

            # Step 2: Extract full text from all pages
            text = extract_text_from_pdf(file_bytes)

            if not text.strip():
                st.warning(
                    f"⚠️ Could not extract text from {uploaded_file.name}. "
                    "This may be a scanned/image PDF. Please use a text-based PDF."
                )
                return

            # Step 3: Split text into overlapping chunks
            # chunk_size=400: ~400 words per chunk (balanced context size)
            # overlap=60: 60-word overlap prevents information loss at boundaries
            chunks = chunk_text(text, chunk_size=400, overlap=60)

            # Step 4: Embed chunks and store in ChromaDB
            num_chunks = add_paper(
                st.session_state.collection,
                chunks,
                uploaded_file.name
            )

            # Track the paper in session state
            st.session_state.papers_loaded[uploaded_file.name] = metadata
            st.session_state.total_chunks += num_chunks

            st.success(
                f"✅ {uploaded_file.name} loaded! "
                f"({metadata['page_count']} pages, {num_chunks} chunks indexed)"
            )

        except Exception as e:
            # Show user-friendly error — don't expose raw exception in production
            st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")


def _remove_paper(paper_name: str):
    """
    Remove a paper from the session and delete its chunks from ChromaDB.

    Args:
        paper_name (str): Filename of the paper to remove.
    """
    # Delete from ChromaDB vector store
    deleted_count = delete_paper(st.session_state.collection, paper_name)

    # Remove from session state tracking dict
    if paper_name in st.session_state.papers_loaded:
        del st.session_state.papers_loaded[paper_name]

    st.session_state.total_chunks = max(
        0, st.session_state.total_chunks - deleted_count
    )

    st.toast(f"Removed {paper_name}", icon="🗑️")


# ── Main Content Area ─────────────────────────────────────────────────────────
def render_main_content():
    """
    Render the main content area with feature tabs.

    Shows either:
        A) Welcome screen — when no papers are loaded
        B) Feature tabs — when papers are loaded
    """
    if not st.session_state.papers_loaded:
        # No papers loaded yet — show welcome/instructions
        render_welcome_screen()
    else:
        # Papers loaded — show all features
        render_feature_tabs()


def render_welcome_screen():
    """
    Render the welcome screen shown when no papers are uploaded yet.
    """
    st.title("🔬 Research Paper Agent")
    st.markdown("### AI-powered academic research assistant")
    st.divider()

    st.info("👈 **Upload PDF research papers** from the sidebar to get started")

    st.markdown("### What can this tool do?")

    # Feature cards in 3 columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 💬 Ask Questions")
        st.write(
            "Ask any natural language question about your papers. "
            "Get accurate, cited answers grounded in the paper content."
        )

    with col2:
        st.markdown("#### 🔍 Find Research Gaps")
        st.write(
            "Automatically identify limitations, research gaps, "
            "and potential thesis topics from your papers."
        )

    with col3:
        st.markdown("#### 📝 Summarize Papers")
        st.write(
            "Get structured summaries covering objectives, methodology, "
            "findings, and cross-paper comparisons."
        )

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("#### 🔄 Compare Papers")
        st.write(
            "Compare multiple papers on methodology, results, "
            "datasets, or limitations side by side."
        )

    with col5:
        st.markdown("#### 📎 Generate Citations")
        st.write(
            "Auto-generate APA, IEEE, and BibTeX citations "
            "for your uploaded papers."
        )

    with col6:
        st.markdown("#### ⚡ Powered by")
        st.write(
            "Groq (free LLM API) + ChromaDB (local vector DB) "
            "+ HuggingFace embeddings. 100% free to run."
        )

    st.divider()
    st.markdown(
        "**Tip:** For best results, upload 2-5 papers on related topics. "
        "The agent compares and cross-references across all uploaded papers."
    )


def render_feature_tabs():
    """
    Render the main feature tabs when papers are loaded.

    Tabs:
        1. 💬 Ask Questions — RAG-powered Q&A chat interface
        2. 🔍 Research Gaps — Gap analysis and thesis topic finder
        3. 📝 Summarize — Structured paper summaries
        4. 🔄 Compare — Cross-paper comparison
        5. 📎 Citations — Academic citation generator
    """
    # Header with loaded paper count
    paper_count = len(st.session_state.papers_loaded)
    st.title("🔬 Research Paper Agent")
    st.caption(
        f"Analyzing {paper_count} paper(s): "
        + ", ".join(list(st.session_state.papers_loaded.keys())[:3])
        + ("..." if paper_count > 3 else "")
    )

    # Create all feature tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Ask Questions",
        "🔍 Research Gaps",
        "📝 Summarize",
        "🔄 Compare Papers",
        "📎 Citations"
    ])

    # ── Tab 1: Q&A Chat ───────────────────────────────────────────────────────
    with tab1:
        render_qa_tab()

    # ── Tab 2: Research Gap Analysis ─────────────────────────────────────────
    with tab2:
        render_gaps_tab()

    # ── Tab 3: Paper Summaries ────────────────────────────────────────────────
    with tab3:
        render_summary_tab()

    # ── Tab 4: Paper Comparison ───────────────────────────────────────────────
    with tab4:
        render_comparison_tab()

    # ── Tab 5: Citation Generator ─────────────────────────────────────────────
    with tab5:
        render_citation_tab()


def render_qa_tab():
    """
    Render the Q&A chat tab with full conversation history.

    Implements a chat-style interface where:
        - Previous messages are shown above
        - New messages appear in real-time
        - History persists within the session
    """
    st.subheader("Ask anything about your papers")
    st.caption(
        "Questions are answered using RAG — the AI searches your papers "
        "first, then generates a grounded, cited answer."
    )

    # ── Chat History Display ──────────────────────────────────────────────────
    # Render all previous messages in the conversation
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ── New Message Input ─────────────────────────────────────────────────────
    # st.chat_input() appears at the bottom of the chat area
    user_question = st.chat_input(
        "Ask a question about your papers...",
        key="qa_input"
    )

    if user_question:
        # Display user's message immediately
        with st.chat_message("user"):
            st.markdown(user_question)

        # Save user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        # Generate and display AI response
        with st.chat_message("assistant"):
            with st.spinner("Searching papers and generating answer..."):
                try:
                    answer = ask_question(
                        st.session_state.collection,
                        user_question
                    )
                    st.markdown(answer)

                    # Save assistant response to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    error_msg = f"❌ Error generating answer: {str(e)}"
                    st.error(error_msg)

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


def render_gaps_tab():
    """
    Render the research gap analysis tab.

    Helps researchers identify:
        - What problems remain unsolved
        - Where existing work has limitations
        - Specific thesis topic opportunities
    """
    st.subheader("🔍 Research Gap Analysis")
    st.write(
        "Automatically identify research gaps, limitations, and potential "
        "thesis topics from your uploaded papers."
    )
    st.info(
        "💡 **Tip for HIT Master's students:** This feature directly helps you "
        "identify what to research. Run this on papers related to your thesis topic!"
    )

    if st.button(
        "🔍 Analyze Research Gaps",
        type="primary",
        use_container_width=True,
        key="gaps_btn"
    ):
        with st.spinner("Analyzing papers for gaps and opportunities..."):
            try:
                gaps_result = find_research_gaps(st.session_state.collection)
                st.markdown(gaps_result)

                # Download button for the analysis
                st.download_button(
                    label="📥 Download Gap Analysis",
                    data=gaps_result,
                    file_name="research_gaps_analysis.md",
                    mime="text/markdown",
                    key="download_gaps"
                )

            except Exception as e:
                st.error(f"❌ Error during gap analysis: {str(e)}")


def render_summary_tab():
    """
    Render the paper summarization tab.

    Generates structured summaries covering:
        - Research objectives and contributions
        - Methodology
        - Key findings
        - Cross-paper relationships
    """
    st.subheader("📝 Paper Summaries")
    st.write(
        "Generate structured summaries of all uploaded papers, "
        "including cross-paper analysis and relationship mapping."
    )

    if st.button(
        "📝 Generate Summaries",
        type="primary",
        use_container_width=True,
        key="summary_btn"
    ):
        with st.spinner("Summarizing papers..."):
            try:
                summary_result = summarize_papers(st.session_state.collection)
                st.markdown(summary_result)

                # Download button
                st.download_button(
                    label="📥 Download Summaries",
                    data=summary_result,
                    file_name="paper_summaries.md",
                    mime="text/markdown",
                    key="download_summary"
                )

            except Exception as e:
                st.error(f"❌ Error generating summaries: {str(e)}")


def render_comparison_tab():
    """
    Render the paper comparison tab.

    Allows comparing papers on specific dimensions:
        - Methodology
        - Results/Performance
        - Datasets used
        - Limitations
        - Theoretical assumptions
    """
    st.subheader("🔄 Compare Papers")
    st.write(
        "Compare your uploaded papers on a specific dimension. "
        "Useful for writing the literature review section of your thesis."
    )

    # Aspect selector — what dimension to compare
    comparison_aspect = st.selectbox(
        "What would you like to compare?",
        options=[
            "methodology",
            "results and performance",
            "datasets and benchmarks",
            "limitations and weaknesses",
            "theoretical assumptions",
            "practical applications"
        ],
        key="comparison_aspect"
    )

    if st.button(
        f"🔄 Compare {comparison_aspect.title()}",
        type="primary",
        use_container_width=True,
        key="compare_btn"
    ):
        with st.spinner(f"Comparing {comparison_aspect} across papers..."):
            try:
                comparison_result = compare_papers(
                    st.session_state.collection,
                    aspect=comparison_aspect
                )
                st.markdown(comparison_result)

                # Download button
                st.download_button(
                    label="📥 Download Comparison",
                    data=comparison_result,
                    file_name=f"comparison_{comparison_aspect.replace(' ', '_')}.md",
                    mime="text/markdown",
                    key="download_comparison"
                )

            except Exception as e:
                st.error(f"❌ Error during comparison: {str(e)}")


def render_citation_tab():
    """
    Render the citation generator tab.

    Generates academic citations in multiple formats:
        - APA (Psychology, Social Sciences)
        - IEEE (Engineering, Computer Science)
        - BibTeX (LaTeX documents)
    """
    st.subheader("📎 Citation Generator")
    st.write(
        "Generate properly formatted citations for your uploaded papers. "
        "Supports APA, IEEE, and BibTeX formats."
    )

    # Paper selector dropdown
    paper_options = list(st.session_state.papers_loaded.keys())

    selected_paper = st.selectbox(
        "Select a paper to cite:",
        options=paper_options,
        key="citation_paper_select"
    )

    if st.button(
        "📎 Generate Citation",
        type="primary",
        use_container_width=True,
        key="citation_btn"
    ):
        with st.spinner(f"Extracting metadata and generating citations..."):
            try:
                citation_result = generate_citation(
                    selected_paper,
                    st.session_state.collection
                )
                st.markdown(citation_result)

                # Download button
                st.download_button(
                    label="📥 Download Citations",
                    data=citation_result,
                    file_name=f"citations_{selected_paper.replace('.pdf', '')}.md",
                    mime="text/markdown",
                    key="download_citation"
                )

            except Exception as e:
                st.error(f"❌ Error generating citation: {str(e)}")


# ── App Entry Point ───────────────────────────────────────────────────────────
def main():
    """
    Main application entry point.

    Orchestrates the app by:
        1. Injecting custom CSS
        2. Initializing session state
        3. Rendering sidebar
        4. Rendering main content
    """
    # Apply custom styling
    inject_custom_css()

    # Initialize session variables on first load
    init_session_state()

    # Render sidebar (always visible — upload papers here)
    render_sidebar()

    # Render main content area (welcome screen or feature tabs)
    render_main_content()


# ── Run the Application ───────────────────────────────────────────────────────
# Standard Python entry point guard
# Streamlit runs this file directly, so main() is always called
if __name__ == "__main__":
    main()
else:
    # Also call main() when Streamlit imports this file (standard behavior)
    main()
