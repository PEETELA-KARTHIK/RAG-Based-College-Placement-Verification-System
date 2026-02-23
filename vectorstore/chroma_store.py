"""
ChromaDB persistent vector store with metadata filtering.

Why ChromaDB:
  - Lightweight, zero-config, runs in-process (no external DB server)
  - Built-in persistent storage to disk
  - Native metadata filtering for scoped retrieval
  - Cosine similarity out of the box
  - Ideal for small-to-medium institutional datasets (thousands of docs)

Metadata filtering:
  - Enables category-scoped queries (e.g., only "placements" docs)
  - Dramatically reduces irrelevant results without post-processing
  - Supports timestamp-based prioritization for conflicting notices
"""

import hashlib
import chromadb
from config import CHROMA_DB_PATH, COLLECTION_NAME


def _get_client() -> chromadb.PersistentClient:
    """Get a persistent ChromaDB client."""
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


def _get_collection(client: chromadb.PersistentClient):
    """Get or create the document collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _generate_chunk_id(document_name: str, chunk_index: int) -> str:
    """Generate a deterministic ID for a chunk to enable deduplication."""
    raw = f"{document_name}::chunk_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def add_documents(
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    """
    Add document chunks to ChromaDB with embeddings and metadata.
    Uses upsert for deduplication — re-uploading the same document
    overwrites its chunks.
    
    Args:
        chunks: List of chunk dicts with 'text' and 'metadata' keys.
        embeddings: Corresponding embedding vectors.
        
    Returns:
        Number of chunks added.
    """
    client = _get_client()
    collection = _get_collection(client)

    ids = []
    documents = []
    metadatas = []

    for chunk, embedding in zip(chunks, embeddings):
        chunk_id = _generate_chunk_id(
            chunk["metadata"]["document_name"],
            chunk["metadata"]["chunk_index"],
        )
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(ids)


def query(
    embedding: list[float],
    top_k: int = 5,
    category_filter: str | None = None,
) -> list[dict]:
    """
    Query ChromaDB for similar chunks.
    
    Args:
        embedding: Query embedding vector.
        top_k: Number of results to return.
        category_filter: Optional category to filter by.
        
    Returns:
        List of result dicts with 'text', 'metadata', 'distance' keys,
        sorted by relevance (lowest distance first).
    """
    client = _get_client()
    collection = _get_collection(client)

    where_filter = None
    if category_filter:
        where_filter = {"category": category_filter}

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten the nested lists from ChromaDB's batch API
    items = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            items.append({
                "text": doc,
                "metadata": meta,
                "distance": dist,
            })

    return items


def delete_document(document_name: str) -> int:
    """
    Remove all chunks for a specific document.
    
    Args:
        document_name: The document name to delete.
        
    Returns:
        Number of chunks deleted.
    """
    client = _get_client()
    collection = _get_collection(client)

    # Get all chunk IDs for this document
    existing = collection.get(
        where={"document_name": document_name},
        include=[],
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        return len(existing["ids"])
    return 0


def list_documents() -> list[dict]:
    """
    List all unique documents in the store with their metadata.
    
    Returns:
        List of dicts with 'document_name', 'category', 'upload_date', 'chunk_count'.
    """
    client = _get_client()
    collection = _get_collection(client)

    all_items = collection.get(include=["metadatas"])
    if not all_items["metadatas"]:
        return []

    doc_info: dict[str, dict] = {}
    for meta in all_items["metadatas"]:
        name = meta.get("document_name", "unknown")
        if name not in doc_info:
            doc_info[name] = {
                "document_name": name,
                "category": meta.get("category", "general"),
                "upload_date": meta.get("upload_date", ""),
                "chunk_count": 0,
            }
        doc_info[name]["chunk_count"] += 1

    return list(doc_info.values())


def get_collection_count() -> int:
    """Get the total number of chunks in the collection."""
    client = _get_client()
    collection = _get_collection(client)
    return collection.count()
