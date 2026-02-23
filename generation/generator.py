"""
Ollama-powered answer generator.
Model: Configurable via LOCAL_LLM_MODEL
Strict RAG behavior.
"""

import requests
import json
from config import (
    LOCAL_LLM_MODEL,
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


def generate_answer(
    question: str,
    context: str,
) -> str:
    """
    Generate an answer using local Ollama instance.
    """
    # Hard failure: no context means we cannot answer
    if not context.strip():
        return NO_ANSWER_RESPONSE

    # Build the prompt
    full_prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT_TEMPLATE.format(context=context, question=question)}"
    
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": LOCAL_LLM_MODEL,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=60 # Reasonable timeout for local inference
        )
        response.raise_for_status()
        
        data = response.json()
        answer = data.get("response", "").strip()
        
        if not answer:
             return NO_ANSWER_RESPONSE
             
        # Enforce "Not available" check if model hallucinates slightly different wording
        if "not available in the provided documents" in answer.lower():
            return NO_ANSWER_RESPONSE
            
        return answer

    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama connection error: {e}")
        return "Error connecting to local LLM (Ollama). Ensure it is running."
    except Exception as e:
        print(f"❌ Generation error: {e}")
        return "An error occurred during generation."
