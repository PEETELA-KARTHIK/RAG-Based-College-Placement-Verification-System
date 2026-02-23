"""
Streamlit UI for the College RAG System.
Provides document upload, query interface, and source attribution display.
"""

import os
import sys
import streamlit as st
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline.rag_pipeline import RAGPipeline
from config import CATEGORIES, UPLOAD_DIR

# ── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="College RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5f8a 50%, #3a7cbd 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 58, 95, 0.3);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #b8d4ed;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 300;
    }

    /* Cards */
    .info-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    }

    /* Stats */
    .stat-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .stat-box {
        flex: 1;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3);
    }
    .stat-box .stat-number {
        font-size: 1.8rem;
        font-weight: 700;
        display: block;
    }
    .stat-box .stat-label {
        font-size: 0.75rem;
        opacity: 0.85;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.25rem;
    }

    /* Source badge */
    .source-badge {
        display: inline-block;
        background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
        color: #0c4a6e;
        border: 1px solid #7dd3fc;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0.2rem;
    }

    .answer-box {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        border: 1px solid #86efac;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.7;
        color: #166534;
    }
    .quota-warning-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fcd34d;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        font-size: 1rem;
        line-height: 1.7;
        color: #92400e;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    .quota-warning-box::before {
        content: "⚠️";
        font-size: 1.5rem;
    }

    /* Sidebar */
    .sidebar-section {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }

    /* Success/error animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .answer-box, .info-card {
        animation: fadeIn 0.4s ease-out;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize ───────────────────────────────────────────────────────────────

@st.cache_resource
def get_pipeline():
    return RAGPipeline()

pipeline = get_pipeline()
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🎓 College RAG Assistant</h1>
    <p>Ask questions about placements, academics, notices & more — answered strictly from uploaded documents</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Document Upload ────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📄 Document Upload")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt", "xlsx"],
        help="Supported: PDF, DOCX, TXT, XLSX",
    )

    category = st.selectbox(
        "Document Category",
        options=CATEGORIES,
        index=0,
        help="Categorize the document for better retrieval accuracy",
    )

    if uploaded_file is not None:
        if st.button("📥 Ingest Document", use_container_width=True):
            # Save uploaded file
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Processing **{uploaded_file.name}**..."):
                try:
                    result = pipeline.ingest_document(file_path, category)
                    st.success(
                        f"✅ **{result['document_name']}** ingested!\n\n"
                        f"• **{result['chunks_created']}** chunks created\n\n"
                        f"• Took **{result['time_seconds']}s**"
                    )
                except Exception as e:
                    st.error(f"❌ Ingestion failed: {str(e)}")

    # ── Document Store Stats ─────────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### 📚 Document Store")

    docs = pipeline.get_documents()
    total_chunks = pipeline.get_total_chunks()

    col_a, col_b = st.columns(2)
    col_a.metric("Documents", len(docs))
    col_b.metric("Total Chunks", total_chunks)

    if docs:
        st.markdown("**Ingested Documents:**")
        for doc in docs:
            with st.expander(f"📄 {doc['document_name']}", expanded=False):
                st.markdown(f"- **Category:** {doc['category']}")
                st.markdown(f"- **Chunks:** {doc['chunk_count']}")
                st.markdown(f"- **Uploaded:** {doc['upload_date']}")
                if st.button(
                    f"🗑️ Remove",
                    key=f"del_{doc['document_name']}",
                    use_container_width=True,
                ):
                    pipeline.remove_document(doc["document_name"])
                    st.rerun()
    else:
        st.info("No documents ingested yet. Upload a document above.")


# ── Main Area: Query Interface ──────────────────────────────────────────────

st.markdown("### 💬 Ask a Question")

# Stats row
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f'<div class="stat-box"><span class="stat-number">{len(docs)}</span>'
        f'<span class="stat-label">Documents</span></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="stat-box"><span class="stat-number">{total_chunks}</span>'
        f'<span class="stat-label">Chunks Indexed</span></div>',
        unsafe_allow_html=True,
    )
with col3:
    categories_used = len(set(d["category"] for d in docs)) if docs else 0
    st.markdown(
        f'<div class="stat-box"><span class="stat-number">{categories_used}</span>'
        f'<span class="stat-label">Categories</span></div>',
        unsafe_allow_html=True,
    )

# Query form
with st.form("query_form", clear_on_submit=False):
    user_query = st.text_area(
        "Your question",
        placeholder="e.g., What are the eligibility criteria for campus placements?",
        height=100,
    )

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        category_filter = st.selectbox(
            "Filter by category (optional)",
            options=["All Categories"] + CATEGORIES,
            index=0,
        )
    with col_q2:
        top_k = st.slider("Results to retrieve", min_value=3, max_value=10, value=5)

    submitted = st.form_submit_button(
        "🔍 Search & Answer",
        use_container_width=True,
    )


# ── Results ─────────────────────────────────────────────────────────────────

if submitted and user_query.strip():
    if total_chunks == 0:
        st.warning("⚠️ No documents have been ingested yet. Please upload a document first.")
    else:
        cat_filter = None if category_filter == "All Categories" else category_filter

        with st.spinner("Searching documents and generating answer..."):
            try:
                result = pipeline.query(
                    question=user_query,
                    top_k=top_k,
                    category_filter=cat_filter,
                )

                # Answer
                st.markdown("### 📝 Answer")
                
                # Check for quota error
                is_quota_error = "unavailable" in result["answer"] or "disabled" in result["answer"]
                box_class = "quota-warning-box" if is_quota_error else "answer-box"
                
                st.markdown(
                    f'<div class="{box_class}">{result["answer"]}</div>',
                    unsafe_allow_html=True,
                )

                # Metadata
                st.caption(f"⏱️ Response time: **{result['time_seconds']}s**")

                # Sources
                if result["sources"]:
                    st.markdown("### 📎 Sources")
                    for src in result["sources"]:
                        st.markdown(
                            f'<span class="source-badge">'
                            f'📄 {src["document_name"]} ({src["category"]})'
                            f'</span>',
                            unsafe_allow_html=True,
                        )

                # Retrieved chunks (expandable)
                if result["retrieval_results"]:
                    # Auto-expand if quota error occurred so user see context
                    # (is_quota_error is calculated above)
                    
                    with st.expander("🔍 View Retrieved Chunks (Context)", expanded=is_quota_error):
                        for i, chunk in enumerate(result["retrieval_results"], 1):
                            st.markdown(f"**Chunk {i}** — Distance: `{chunk['distance']:.4f}`")
                            st.markdown(f"> {chunk['text']}")
                            st.markdown(
                                f"Source: `{chunk['metadata'].get('document_name', 'N/A')}` | "
                                f"Category: `{chunk['metadata'].get('category', 'N/A')}`"
                            )
                            st.markdown("---")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

elif submitted:
    st.warning("Please enter a question.")


# ── Footer ──────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.8rem;'>"
    "College RAG Assistant • Answers strictly from uploaded documents • "
    "Powered by Google Gemini + ChromaDB"
    "</div>",
    unsafe_allow_html=True,
)
