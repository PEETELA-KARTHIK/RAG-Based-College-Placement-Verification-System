"""
Top-k retriever with optional metadata-based filtering.
Embeds the user query and retrieves the most relevant chunks from ChromaDB.
"""

from embeddings.gemini_embedder import embed_query
from vectorstore.chroma_store import query as chroma_query
from config import TOP_K


def retrieve(
    user_query: str,
    top_k: int = TOP_K,
    category_filter: str | None = None,
) -> list[dict]:
    """
    Retrieve the most relevant document chunks.
    Returns empty list if embedding fails (e.g. quota limit).
    """
    # Step 1: Embed the query
    query_embedding = embed_query(user_query)
    
    # Handle quota failure (returns None)
    if query_embedding is None:
        return []

    # Step 2+3: Search ChromaDB with optional filter
    results = chroma_query(
        embedding=query_embedding,
        top_k=top_k,
        category_filter=category_filter,
    )

    return results


def format_context(results: list[dict]) -> str:
    """
    Format retrieved chunks into a context string.
    """
    if not results:
        return ""

    context_parts = []
    for i, result in enumerate(results, 1):
        source = result["metadata"].get("document_name", "Unknown")
        context_parts.append(
            f"[Source {i}: {source}]\n{result['text']}"
        )

    return "\n\n---\n\n".join(context_parts)
