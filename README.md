# 🎓 College RAG Assistant

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://peetela-karthik-rag-based-college-placement-verifica-app-tcvn4r.streamlit.app/)

A production-ready Retrieval-Augmented Generation system for college placements and academic queries. Answers student questions **strictly** from uploaded institutional documents — no hallucination, no external knowledge.

**Live Demo:** [Click here to view the app!](https://peetela-karthik-rag-based-college-placement-verifica-app-tcvn4r.streamlit.app/)

---

## Architecture

```
Document Upload
     │
     ▼
Text Extraction (PDF / DOCX / TXT / XLSX)
     │
     ▼
Text Cleaning (noise, headers, footers removed)
     │
     ▼
Token-Aware Chunking (500 tokens, 18% overlap)
     │
     ▼
Gemini Embeddings (models/embedding-001)
     │
     ▼
ChromaDB Storage (persistent, cosine similarity)
     │
     ║
     ║  User Query
     ║       │
     ║       ▼
     ║  Query Embedding (retrieval_query task type)
     ║       │
     ║       ▼
     ╠══► Top-k Retrieval (k=5, metadata filtering)
            │
            ▼
     Context Assembly (with source attribution)
            │
            ▼
     Gemini Generation (gemini-2.0-flash, temp=0.1)
            │
            ▼
       Final Answer
```

---

## Design Decisions

### Why ChromaDB?
- **Zero-config**: Runs in-process, no external database server needed
- **Persistent**: Data survives restarts (stored in `data/chroma_db/`)
- **Metadata filtering**: Filter by category (placements/academics/notices) at query time
- **Cosine similarity**: Built-in, no manual distance computation
- **Scale**: Ideal for small-to-medium institutional datasets (thousands of documents)

### Why Gemini Embeddings?
- Unified API: Same provider for embeddings and generation
- **Task-type differentiation**: `retrieval_document` for indexing, `retrieval_query` for search — improves alignment
- No separate embedding service to manage

### Token-Aware Chunking
- **500 token chunks** with **18% overlap**: Balances context completeness vs retrieval precision
- Uses `tiktoken` (`cl100k_base`) for accurate token counting
- Overlap ensures information at chunk boundaries isn't lost

### Strict RAG Prompt
- **Temperature 0.1**: Near-deterministic, minimizes creative output
- **System instruction**: Hardcoded, non-overridable constraint to answer only from context
- **Hard failure**: Returns explicit "not available" message when no relevant context exists

---

## Project Structure

```
RAG-CHAT-v3/
├── app.py                     # Streamlit frontend
├── config.py                  # Central configuration
├── requirements.txt           # Dependencies
├── .env.example               # API key template
├── README.md
├── data/chroma_db/            # Persistent vector store
├── uploads/                   # Uploaded files staging
├── ingestion/
│   ├── extractors.py          # PDF, DOCX, TXT, XLSX extraction
│   ├── cleaner.py             # Text cleaning pipeline
│   └── chunker.py             # Token-aware chunking
├── embeddings/
│   └── gemini_embedder.py     # Gemini embedding wrapper
├── vectorstore/
│   └── chroma_store.py        # ChromaDB operations
├── retrieval/
│   └── retriever.py           # Top-k retrieval + context formatting
├── generation/
│   └── generator.py           # Gemini answer generation
└── pipeline/
    └── rag_pipeline.py        # End-to-end orchestrator
```

---

## Setup & Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Key
```bash
cp .env.example .env
# Edit .env and add your Google Gemini API key:
# GOOGLE_API_KEY=your-actual-key-here
```

### 3. Launch
```bash
streamlit run app.py
```

### 4. Usage
1. **Upload** a document (PDF/DOCX/TXT/XLSX) via the sidebar
2. **Select** a category (placements, academics, notices, etc.)
3. **Click** "Ingest Document" to process and index it
4. **Ask** a question in the main area
5. **Review** the answer with source citations

---

## Evaluation & Safety

### Metrics Tracked
| Metric | How |
|--------|-----|
| **Response latency** | Displayed per query |
| **Source attribution** | Every answer shows source documents |
| **Retrieval distance** | Raw cosine distances shown in expandable panel |
| **Context coverage** | Retrieved chunks viewable for verification |

### Failure Handling
| Case | Mitigation |
|------|-----------|
| No relevant docs | Hard failure: "not available in uploaded documents" |
| Conflicting notices | Metadata includes upload timestamp for prioritization |
| Unsupported format | Clear error with supported formats listed |
| Empty extraction | Explicit error before embedding |

---

## Security Considerations

- **API Key**: Stored in `.env`, never committed to version control. Add `.env` to `.gitignore`.
- **Data Residency**: All documents stay on your server. Only text chunks are sent to Google's Gemini API for embedding/generation.
- **No Persistence of Queries**: User queries are not logged or stored.
- **Input Validation**: File type validation at upload; text cleaning removes injection-prone artifacts.

---

## Future Extensions

- **Role-Based Access**: Admin vs student roles; admin panel for document management
- **FastAPI Backend**: REST API for mobile apps and programmatic access
- **Hybrid Search**: Keyword (BM25) + dense (embedding) retrieval fusion
- **Multi-Tenant**: Separate vector stores per department
- **Audit Logging**: Track queries and responses for compliance
- **Auto-Expiry**: Documents with expiration dates automatically flagged or removed
- **Batch Upload**: Bulk document ingestion with progress tracking

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Embeddings | Google Gemini `embedding-001` |
| Generation | Google Gemini `gemini-2.0-flash` |
| Vector Store | ChromaDB (persistent) |
| UI | Streamlit |
| Tokenizer | tiktoken (`cl100k_base`) |
| PDF | PyPDF2 |
| DOCX | python-docx |
| XLSX | openpyxl |
"# datha" 
"# datha" 
