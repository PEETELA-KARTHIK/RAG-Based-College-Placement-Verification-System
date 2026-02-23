"""
Google Gemini-powered answer generator.
Model: Configurable via GENERATION_MODEL
Strict RAG behavior.
"""

from google import genai
from google.genai import types
from config import (
    GOOGLE_API_KEY,
    GENERATION_MODEL,
    NO_ANSWER_RESPONSE
)

# ── RAG Logic ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Retrieval-Augmented Generation (RAG) assistant for a college placement system.

Rules:
1. Use ONLY the information provided in the context below.
2. If the answer is not present, respond with: "Not available in the provided documents."
3. You are NOT allowed to generate or modify numbers (HTNO, Roll No, IDs).
4. Copy numbers EXACTLY as shown in the context.
"""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question:
{question}

Answer:"""


def _get_client() -> genai.Client:
    """Get a configured Gemini client."""
    return genai.Client(api_key=GOOGLE_API_KEY)


def generate_answer(
    question: str,
    context: str,
) -> str:
    """
    Generate an answer using Google Gemini.
    """
    # Hard failure: no context means we cannot answer
    if not context.strip():
        return NO_ANSWER_RESPONSE

    # Build the prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(context=context, question=question)}"
    
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
            )
        )
        
        answer = response.text.strip()
        
        if not answer:
             return NO_ANSWER_RESPONSE
             
        # Enforce "Not available" check if model hallucinates slightly different wording
        if "not available in the provided documents" in answer.lower():
            return NO_ANSWER_RESPONSE
            
        return answer

    except Exception as e:
        print(f"❌ Generation error: {e}")
        return "An error occurred during generation."
