from typing import Optional, List, Dict, Any
import re


def format_response(
    response: str,
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    # Simple formatter: preserve the full AI response and paragraph breaks, do not modify content.
    resp_text = response.strip()
    # Improved normalization: avoid splitting markdown lists and headers
    # Only split paragraphs that are not part of a markdown list or header
    lines = resp_text.splitlines()
    new_paragraphs = []
    buffer = []
    for line in lines:
        if line.strip() == "":
            if buffer:
                new_paragraphs.append("\n".join(buffer))
                buffer = []
        else:
            buffer.append(line)
    if buffer:
        new_paragraphs.append("\n".join(buffer))
    # Use single newlines instead of double to prevent UI formatting issues
    paragraph_block = "\n".join(new_paragraphs)
    # Fix common markdown token splits that occur when LLM output is chunked.
    paragraph_block = re.sub(r"\*\s+\*", "**", paragraph_block)
    paragraph_block = re.sub(r"_\s+_", "__", paragraph_block)
    paragraph_block = re.sub(r"(`{1,3})\s+(`{1,3})", r"\1\2", paragraph_block)

    # Ensure proper spacing around bold text without creating excessive line breaks
    # Find the last bold Markdown (e.g., **...**)
    bold_match = list(re.finditer(r"\*\*.*?\*\*", paragraph_block))
    if bold_match:
        last_bold = bold_match[-1]
        start, end = last_bold.span()
        before = paragraph_block[:start].rstrip()
        bold_text = paragraph_block[start:end]
        after = paragraph_block[end:].lstrip()
        # Only add a single newline before the last bold if it's truly a separate paragraph
        if before and not before.endswith("\n"):
            paragraph_block = before + " " + bold_text
        else:
            paragraph_block = before + bold_text
        # If there's trailing text after the bold, add appropriate spacing
        if after:
            paragraph_block += " " + after
    return paragraph_block