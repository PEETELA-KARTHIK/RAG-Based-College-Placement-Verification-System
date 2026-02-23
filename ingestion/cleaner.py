"""
Text cleaning utilities for extracted document content.
Removes noise such as repeated headers/footers, excessive whitespace,
page numbers, and normalizes unicode characters.
"""

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to their canonical forms."""
    # NFKC normalization: compatibility decomposition + canonical composition
    text = unicodedata.normalize("NFKC", text)
    # Replace common unicode artifacts
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = text.replace("\u2026", "...")
    text = text.replace("\xa0", " ")  # non-breaking space
    return text


def remove_page_numbers(text: str) -> str:
    """Remove standalone page numbers (e.g., 'Page 1', '- 2 -', '1 of 10')."""
    # "Page X" or "Page X of Y"
    text = re.sub(r"(?im)^\s*page\s+\d+(\s+of\s+\d+)?\s*$", "", text)
    # "- X -" style
    text = re.sub(r"(?m)^\s*-\s*\d+\s*-\s*$", "", text)
    # Standalone numbers on a line (likely page numbers)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    return text


def remove_repeated_headers_footers(text: str, min_repeats: int = 3) -> str:
    """
    Detect and remove lines that repeat frequently across the document,
    which are likely headers or footers.
    """
    lines = text.split("\n")
    line_counts: dict[str, int] = {}

    for line in lines:
        stripped = line.strip()
        if stripped and len(stripped) > 3:  # ignore very short lines
            line_counts[stripped] = line_counts.get(stripped, 0) + 1

    # Lines that repeat >= min_repeats times are likely headers/footers
    repeated = {line for line, count in line_counts.items() if count >= min_repeats}

    cleaned = [line for line in lines if line.strip() not in repeated]
    return "\n".join(cleaned)


def collapse_whitespace(text: str) -> str:
    """Collapse excessive whitespace and blank lines."""
    # Replace multiple spaces/tabs with single space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Replace 3+ consecutive newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines)


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline: normalize → remove noise → collapse whitespace.
    
    Args:
        text: Raw extracted text.
        
    Returns:
        Cleaned text ready for chunking.
    """
    text = normalize_unicode(text)
    text = remove_page_numbers(text)
    text = remove_repeated_headers_footers(text)
    text = collapse_whitespace(text)
    return text.strip()
