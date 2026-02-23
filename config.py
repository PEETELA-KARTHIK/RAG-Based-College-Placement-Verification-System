"""
Central configuration for the RAG system.
Optimized for Google Gemini Free Tier constraints.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
UPLOAD_DIR = str(BASE_DIR / "uploads")

# ── Google Gemini (Free Tier) ────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
EMBEDDING_MODEL = "models/gemini-embedding-001"
LOCAL_LLM_MODEL = "phi3:mini"

# ── Chunking (Strict Limits) ────────────────────────────────────────────────
CHUNK_SIZE = 300          # Reduced to save tokens
CHUNK_OVERLAP = 0.15      # 15% overlap
TIKTOKEN_ENCODING = "cl100k_base"

# ── Retrieval (Minimal Cost) ────────────────────────────────────────────────
TOP_K = 3                 # Process fewer chunks
COLLECTION_NAME = "college_documents"

# ── Document Categories ─────────────────────────────────────────────────────
CATEGORIES = [
    "placements",
    "academics",
    "notices",
    "training",
    "general",
]

# ── Fallbacks ───────────────────────────────────────────────────────────────
NO_ANSWER_RESPONSE = "Not available in the provided documents."
QUOTA_ERROR_RESPONSE = (
    "Answer generation is temporarily disabled due to free-tier limits.\n"
    "Please check the retrieved context below for your answer."
)
