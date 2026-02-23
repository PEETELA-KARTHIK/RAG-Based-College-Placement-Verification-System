"""
End-to-end RAG pipeline orchestrator.
Connects ingestion, embedding, storage, retrieval, and generation.
Optimized for Free Tier robustness.
"""

import os
import time
from pathlib import Path

from ingestion.extractors import extract_text
from ingestion.cleaner import clean_text
from ingestion.chunker import chunk_text
from ingestion.excel_parser import parse_excel
from embeddings.gemini_embedder import embed_texts
from vectorstore.chroma_store import (
    add_documents,
    delete_document,
    list_documents,
    get_collection_count,
)
from retrieval.retriever import retrieve, format_context
from retrieval.verification import classify_query, verify_match
from generation.generator import generate_answer
from config import UPLOAD_DIR


class RAGPipeline:
    def __init__(self):
        """Initialize the pipeline and ensure directories exist."""
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    def ingest_document(
        self,
        file_path: str,
        category: str = "general",
    ) -> dict:
        start = time.time()
        document_name = Path(file_path).name

        if file_path.lower().endswith(".xlsx"):
            # Excel Handling: Parse row-by-row into structured chunks
            chunks = parse_excel(
                file_path=file_path,
                document_name=document_name,
                category=category
            )
        else:
            # Standard Text Handling: Extract -> Clean -> Chunk
            raw_text = extract_text(file_path)
            if not raw_text.strip():
                raise ValueError(f"No text could be extracted from: {document_name}")

            cleaned_text = clean_text(raw_text)

            chunks = chunk_text(
                text=cleaned_text,
                document_name=document_name,
                category=category,
            )
            
        if not chunks:
            raise ValueError(f"No chunks generated from: {document_name}")

        # Step 4: Generate embeddings (might fail due to quota)
        chunk_texts = [c["text"] for c in chunks]
        embeddings = embed_texts(chunk_texts)

        if not embeddings:
            raise RuntimeError(f"Embedding failed for {document_name}. Check API quota.")

        if len(embeddings) != len(chunks):
             raise RuntimeError(f"Embedding mismatch: {len(embeddings)} vectors for {len(chunks)} chunks.")

        # Step 5: Store in ChromaDB
        count = add_documents(chunks, embeddings)

        elapsed = time.time() - start
        return {
            "document_name": document_name,
            "chunks_created": count,
            "time_seconds": round(elapsed, 2),
        }

    def query(
        self,
        question: str,
        top_k: int = 5,
        category_filter: str | None = None,
    ) -> dict:
        start = time.time()

        # Step 1: Retrieve relevant chunks
        results = retrieve(
            user_query=question,
            top_k=top_k,
            category_filter=category_filter,
        )

        # Step 2: Check for Deterministic Verification Match
        # If query is asking for specifics (HTNO, Name, Status) and we have a definitive metadata match
        # we bypass the LLM to avoid hallucinations.
        
        is_verification = classify_query(question)
        deterministic_answer = None
        
        if is_verification:
            deterministic_answer = verify_match(question, results)
            
        if deterministic_answer:
            answer = deterministic_answer
        else:
            # Step 3: Generate answer using LLM (if no deterministic match found)
            context = format_context(results)
            answer = generate_answer(question=question, context=context)

        # Step 4: Extract source info
        sources = []
        seen = set()
        for r in results:
            doc = r["metadata"].get("document_name", "Unknown")
            if doc not in seen:
                seen.add(doc)
                sources.append({
                    "document_name": doc,
                    "category": r["metadata"].get("category", ""),
                    "upload_date": r["metadata"].get("upload_date", ""),
                })

        elapsed = time.time() - start
        return {
            "answer": answer,
            "sources": sources,
            "retrieval_results": results,
            "time_seconds": round(elapsed, 2),
        }

    def remove_document(self, document_name: str) -> int:
        return delete_document(document_name)

    def get_documents(self) -> list[dict]:
        return list_documents()

    def get_total_chunks(self) -> int:
        return get_collection_count()
