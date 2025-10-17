from typing import Optional, List, Dict, Any
import re


def format_response(
    response: str,
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    # Simple formatter: preserve the full AI response and paragraph breaks, do not modify content.
    resp_text = response.strip()
    # Normalize multiple blank lines to a single blank line between paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", resp_text) if p.strip()]
    if not paragraphs:
        return resp_text
    paragraph_block = "\n\n".join(paragraphs)
    return paragraph_block