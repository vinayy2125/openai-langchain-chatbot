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
    # Fix common markdown token splits that occur when LLM output is chunked.
    paragraph_block = re.sub(r"\*\s+\*", "**", paragraph_block)
    paragraph_block = re.sub(r"_\s+_", "__", paragraph_block)
    paragraph_block = re.sub(r"(`{1,3})\s+(`{1,3})", r"\1\2", paragraph_block)

    # Ensure the last bold Markdown follow-up or closing is on a new line
    # Find the last bold Markdown (e.g., **...**)
    bold_match = list(re.finditer(r"\*\*.*?\*\*", paragraph_block))
    if bold_match:
        last_bold = bold_match[-1]
        start, end = last_bold.span()
        before = paragraph_block[:start].rstrip()
        bold_text = paragraph_block[start:end]
        after = paragraph_block[end:].lstrip()
        # Always ensure a blank line before the last bold follow-up
        if not before.endswith("\n\n"):
            paragraph_block = before + "\n\n" + bold_text
        else:
            paragraph_block = before + bold_text
        # If there's trailing text after the bold, put it on a new line
        if after:
            paragraph_block += "\n" + after
    return paragraph_block