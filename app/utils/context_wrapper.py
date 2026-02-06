"""
Context Wrapper Service.

This module implements STRICT structural separation between System Instructions
and Retrieved Context. It treats all retrieved content as UNTRUSTED DATA.

Philosophy:
- Context is wrapped in explicit XML tags <trusted_context>
- No heuristic stripping or sanitization of content (avoids data loss)
- The LLM is instructed to treat this block purely as data
"""

from typing import List, Dict, Any
import html

def wrap_context_as_xml(chunks: List[Dict[str, Any]]) -> str:
    """
    Wrap retrieved chunks in a strict XML structure.
    
    Structure:
    <trusted_context>
        <document id="1" source="URL">
            ... content ...
        </document>
    </trusted_context>
    
    Args:
        chunks: List of dicts with 'text', 'metadata', 'source'
        
    Returns:
        String containing the strictly formatted XML block
    """
    if not chunks:
        return "<trusted_context></trusted_context>"
    
    xml_parts = ["<trusted_context>"]
    
    for idx, chunk in enumerate(chunks, 1):
        content = chunk.get("text", "") or ""
        # Escape HTML characters to prevent tag injection
        safe_content = html.escape(content)
        
        metadata = chunk.get("metadata", {}) or {}
        source = chunk.get("source", "unknown")
        
        # Build document tag
        # doc_id is essential for citation enforcement
        xml_parts.append(f'  <document id="{idx}" source="{html.escape(str(source))}">')
        xml_parts.append(f'{safe_content}')
        xml_parts.append('  </document>')
        
    xml_parts.append("</trusted_context>")
    
    return "\n".join(xml_parts)
