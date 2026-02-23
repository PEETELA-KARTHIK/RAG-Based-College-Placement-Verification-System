"""
Token-aware text chunking with configurable overlap and metadata preservation.
Uses tiktoken for accurate token counting aligned with LLM tokenization.
"""

import tiktoken
from datetime import datetime, timezone
from config import CHUNK_SIZE, CHUNK_OVERLAP, TIKTOKEN_ENCODING


def _get_encoder():
    """Get the tiktoken encoder (cached by tiktoken internally)."""
    return tiktoken.get_encoding(TIKTOKEN_ENCODING)


def count_tokens(text: str) -> int:
    """Count the number of tokens in a text string."""
    encoder = _get_encoder()
    return len(encoder.encode(text))


def chunk_text(
    text: str,
    document_name: str,
    category: str,
    chunk_size: int = CHUNK_SIZE,
    overlap_ratio: float = CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split text into token-aware chunks with overlap and metadata.
    
    Strategy:
      1. Split text into sentences (by newlines and periods).
      2. Greedily accumulate sentences until chunk_size is reached.
      3. Overlap is achieved by carrying the last N% of tokens into the next chunk.
    
    Args:
        text: Cleaned document text.
        document_name: Original filename for metadata.
        category: Document category (placements / academics / notices / ...).
        chunk_size: Target tokens per chunk (default from config).
        overlap_ratio: Fraction of chunk to overlap (default from config).
        
    Returns:
        List of chunk dicts: {text, metadata: {document_name, category, upload_date, chunk_index}}
    """
    if not text.strip():
        return []

    encoder = _get_encoder()
    overlap_tokens = int(chunk_size * overlap_ratio)

    # Split into sentences — by double newlines, then single newlines, then periods
    import re
    sentences = re.split(r"(?<=\.)\s+|\n{2,}|\n", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_sentences: list[str] = []
    current_token_count = 0
    upload_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    for sentence in sentences:
        sentence_tokens = len(encoder.encode(sentence))

        # If a single sentence exceeds chunk_size, split it by words
        if sentence_tokens > chunk_size:
            words = sentence.split()
            sub_sentence = []
            sub_count = 0
            for word in words:
                word_tokens = len(encoder.encode(word + " "))
                if sub_count + word_tokens > chunk_size and sub_sentence:
                    current_sentences.append(" ".join(sub_sentence))
                    current_token_count += sub_count
                    # Flush
                    chunk_text_str = " ".join(current_sentences)
                    chunks.append({
                        "text": chunk_text_str,
                        "metadata": {
                            "document_name": document_name,
                            "category": category,
                            "upload_date": upload_date,
                            "chunk_index": len(chunks),
                        }
                    })
                    # Overlap: keep tail sentences
                    current_sentences, current_token_count = _compute_overlap(
                        current_sentences, encoder, overlap_tokens
                    )
                    sub_sentence = [word]
                    sub_count = word_tokens
                else:
                    sub_sentence.append(word)
                    sub_count += word_tokens
            if sub_sentence:
                current_sentences.append(" ".join(sub_sentence))
                current_token_count += sub_count
            continue

        if current_token_count + sentence_tokens > chunk_size and current_sentences:
            # Flush current chunk
            chunk_text_str = " ".join(current_sentences)
            chunks.append({
                "text": chunk_text_str,
                "metadata": {
                    "document_name": document_name,
                    "category": category,
                    "upload_date": upload_date,
                    "chunk_index": len(chunks),
                }
            })
            # Overlap: keep tail sentences
            current_sentences, current_token_count = _compute_overlap(
                current_sentences, encoder, overlap_tokens
            )

        current_sentences.append(sentence)
        current_token_count += sentence_tokens

    # Final chunk
    if current_sentences:
        chunk_text_str = " ".join(current_sentences)
        chunks.append({
            "text": chunk_text_str,
            "metadata": {
                "document_name": document_name,
                "category": category,
                "upload_date": upload_date,
                "chunk_index": len(chunks),
            }
        })

    return chunks


def _compute_overlap(
    sentences: list[str],
    encoder,
    overlap_tokens: int,
) -> tuple[list[str], int]:
    """
    Keep the tail sentences whose total tokens <= overlap_tokens.
    Returns the overlap sentences and their token count.
    """
    if not sentences:
        return [], 0

    kept = []
    token_count = 0
    for sentence in reversed(sentences):
        s_tokens = len(encoder.encode(sentence))
        if token_count + s_tokens > overlap_tokens:
            break
        kept.insert(0, sentence)
        token_count += s_tokens

    return kept, token_count
