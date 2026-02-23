"""
Deterministic verification logic for RAG.
Bypasses LLM generation when exact matches are found in Excel metadata.
Enforces Atomic Row Logic: One Query -> One Row.
"""

import re

# Keywords that trigger verification logic
VERIFICATION_KEYWORDS = {
    "htno", "roll no", "id", "ticket", "number",
    "status", "placed", "selected", "company",
    "salary", "package", "cpa", "cgpa",
    "listed", "present", "check", "verify",
    "panel", "time", "skill", "reference"
}

def classify_query(query: str) -> bool:
    """
    Check if the query is asking for specific verificational data.
    """
    cleaned = query.lower()
    return any(kw in cleaned for kw in VERIFICATION_KEYWORDS)


def verify_match(query: str, results: list[dict]) -> str | None:
    """
    Check if retrieved chunks provide a SINGLE deterministic answer.
    
    Args:
        query: The user's question.
        results: List of retrieved chunks from ChromaDB.
        
    Returns:
        Formatted string answer if a SINGLE match is definitive.
        Error string if multiple matches found.
        None if no match found.
    """
    if not results:
        return None
        
    query_tokens = set(re.findall(r"\w+", query.lower()))
    
    # Remove common verification keywords to focus on the entity name (e.g. "panel", "for")
    # We also remove stop words manually if needed, but starting with our keyword list is good.
    query_tokens = {t for t in query_tokens if t not in VERIFICATION_KEYWORDS and t not in {"of", "for", "in", "the", "is", "a", "an"}}

    
    # Fields to check for identity match
    identity_fields = ["name", "student_name", "candidate_name", "htno", "roll_no", "roll_number", "id", "reference_id"]
    
    matched_chunks = []
    
    for result in results:
        meta = result.get("metadata", {})
        
        # Only apply to Excel rows which have rich structure
        if meta.get("source_type") != "excel_row":
            continue
            
        match_confidence = 0
        
        for field in identity_fields:
            if field in meta:
                val = str(meta[field]).lower()
                val_tokens = set(re.findall(r"\w+", val))
                
                # If the field value is in the query (e.g. "Peetela Karthik" in "is Peetela Karthik placed?")
                # OR if the query is a subset of the field value (e.g. "Karthik" in "Peetela Karthik")
                
                if val_tokens and (val_tokens.issubset(query_tokens) or query_tokens.issubset(val_tokens)):
                    match_confidence += 1

        # If we have a strong match (at least one identity field matched exactly)
        if match_confidence > 0:
            matched_chunks.append(meta)
            
    # Deduplicate matches based on row identifier (Sheet + Row)
    unique_matches = {}
    for meta in matched_chunks:
        key = f"{meta.get('sheet_name')}_{meta.get('row_number')}"
        unique_matches[key] = meta
        
    final_matches = list(unique_matches.values())
    
    # Strict matching logic
    if len(final_matches) == 0:
        return None  # Let caller decide (heuristic: "No exact match found" if verification intent is strong)
        
    if len(final_matches) > 1:
        return "❌ Multiple records found. Please refine your query with more specific details (e.g., full name, ID)."
        
    # Exactly one match
    return _format_deterministic_answer(final_matches[0])


def _format_deterministic_answer(meta: dict) -> str:
    """Format the metadata into a clean, evidence-based answer."""
    
    lines = ["**✔ Record Found:**"]
    
    # Priority Fields (Panel, Time, Skill, ID)
    priority_order = [
         "name", "student_name", "candidate_name",
         "reference_id", "ref_id", "registration_id",
         "htno", "roll_no", "id",
         "skill", "technology", "domain",
         "panel", "panel_no",
         "reporting_time", "time", "slot",
         "status", "company", "package",
    ]
    
    seen = set()
    
    for key in priority_order:
        # Check for clean keys in metadata (parser lowercases them)
        # We try to match flexibility
        
        # Direct match
        if key in meta:
            label = key.replace('_', ' ').title()
            # Custom labels
            if key == "htno": label = "HTNO"
            if key == "reference_id": label = "Reference ID"
            
            lines.append(f"- **{label}**: `{meta[key]}`")
            seen.add(key)
            
    # Add remaining interesting fields
    internal_keys = {
        "document_name", "category", "upload_date", "sheet_name", 
        "row_number", "source_type", "chunk_index"
    }
    
    for k, v in meta.items():
        if k not in seen and k not in internal_keys:
             lines.append(f"- **{k.replace('_', ' ').title()}**: {v}")
             
    # Add Source footer
    lines.append(f"\n*Source: {meta.get('document_name')} (Sheet: {meta.get('sheet_name')}, Row: {meta.get('row_number')})*")
    
    return "\n".join(lines)
