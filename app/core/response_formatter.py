from typing import Optional, List, Dict, Any
import re


def format_response(
    response: str,
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    # Simple formatter: preserve the full AI response and paragraph breaks, do not modify content.
    resp_text = response.strip()
    # If markdown detected, preserve all newlines (do not collapse to single newlines)
    markdown_markers = ["- ", "* ", "#", "**", "__", "`", "[", "]("]
    if any(m in resp_text for m in markdown_markers):
        # Only fix common token splits, but preserve all newlines
        paragraph_block = re.sub(r"\*\s+\*", "**", resp_text)
        paragraph_block = re.sub(r"_\s+_", "__", paragraph_block)
        paragraph_block = re.sub(r"(`{1,3})\s+(`{1,3})", r"\1\2", paragraph_block)
        # Ensure all double-escaped newlines are real newlines (for safety)
        paragraph_block = paragraph_block.replace('\\n', '\n')
        return paragraph_block
    # Otherwise, apply legacy formatting (for plain text)
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
    paragraph_block = "\n".join(new_paragraphs)
    paragraph_block = re.sub(r"\*\s+\*", "**", paragraph_block)
    paragraph_block = re.sub(r"_\s+_", "__", paragraph_block)
    paragraph_block = re.sub(r"(`{1,3})\s+(`{1,3})", r"\1\2", paragraph_block)
    return paragraph_block