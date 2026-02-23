"""
Google Gemini embedding wrapper using the new google-genai SDK.
Strict Free-Tier version: No Retries, Fail Fast.
"""

from google import genai
from google.genai import types
from config import GOOGLE_API_KEY, EMBEDDING_MODEL


def _get_client() -> genai.Client:
    """Get a configured Gemini client."""
    return genai.Client(api_key=GOOGLE_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of document texts.
    Returns empty list if API fails (Quota limit).
    """
    if not texts:
        return []

    client = _get_client()
    try:
        # Batch embed for documents
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return [e.values for e in result.embeddings]
    except Exception as e:
        # User constraint: Do NOT retry. Do NOT crash.
        print(f"❌ Embedding failed (Quota or Error): {e}")
        return []


def embed_query(query: str) -> list[float]:
    """
    Generate an embedding for a user query.
    Returns None if API fails.
    """
    client = _get_client()
    try:
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"❌ Query embedding failed (Quota or Error): {e}")
        return None
