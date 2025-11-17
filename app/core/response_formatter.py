from typing import Optional, List, Dict, Any
import re
import asyncio
from urllib.parse import urlparse
import httpx


def _normalize_url(url: str) -> str:
    """Normalize URL by removing spaces and fixing common issues."""
    # Remove spaces within URL (common issue: "real- estate" -> "real-estate")
    url = re.sub(r"\s+", "", url)
    # Remove trailing spaces/punctuation that might have been captured
    url = url.rstrip(".,;:!?)")
    return url


def _extract_urls(text: str) -> List[str]:
    """Extract all URLs from markdown text."""
    # Pattern to match markdown links [text](url) and plain URLs
    url_pattern = r"\[([^\]]+)\]\(([^)]+)\)|(https?://[^\s\)]+)"
    matches = re.finditer(url_pattern, text)
    urls = []
    for match in matches:
        # Group 2 is URL from markdown link, Group 3 is plain URL
        url = match.group(2) or match.group(3)
        if url:
            urls.append(_normalize_url(url))
    return urls


async def _validate_url(url: str, timeout: float = 2.0) -> bool:
    """Check if URL is accessible (not 404)."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.head(url)
            # Accept 2xx and 3xx status codes, reject 4xx and 5xx
            return 200 <= response.status_code < 400
    except Exception:
        return False


async def _remove_invalid_urls(text: str) -> str:
    """Normalize and validate URLs, removing invalid ones (404s)."""
    if not text:
        return text

    # First, normalize all URLs in the text (fix spaces, etc.)
    def normalize_url_in_text(match):
        """Normalize URL found in markdown link or plain text."""
        if match.group(2):  # Markdown link [text](url)
            normalized_url = _normalize_url(match.group(2))
            return f"[{match.group(1)}]({normalized_url})"
        elif match.group(3):  # Plain URL
            return _normalize_url(match.group(3))
        return match.group(0)

    # Normalize URLs in text
    url_pattern = r"\[([^\]]+)\]\(([^)]+)\)|(https?://[^\s\)]+)"
    text = re.sub(url_pattern, normalize_url_in_text, text)

    # Extract normalized URLs for validation
    urls = _extract_urls(text)
    if not urls:
        return text

    try:
        # Validate all URLs concurrently with timeout
        validation_tasks = [_validate_url(url) for url in urls]
        validation_results = await asyncio.wait_for(
            asyncio.gather(*validation_tasks, return_exceptions=True), timeout=3.0
        )

        # Build mapping of invalid URLs
        invalid_urls = set()
        for url, is_valid in zip(urls, validation_results):
            if isinstance(is_valid, Exception) or not is_valid:
                invalid_urls.add(url)

        if not invalid_urls:
            return text

        # Remove invalid URLs from text
        result = text
        for url in invalid_urls:
            # Remove markdown links with invalid URLs: [text](url)
            result = re.sub(rf"\[([^\]]+)\]\({re.escape(url)}\)", r"\1", result)
            # Remove plain invalid URLs
            result = re.sub(re.escape(url), "", result)

        # Clean up extra spaces but preserve newlines (critical for markdown)
        # Only collapse multiple spaces within a line, not newlines
        result = re.sub(r"[ \t]+", " ", result)  # Collapse spaces/tabs, not newlines
        result = re.sub(r" +([.,!?])", r"\1", result)  # Remove space before punctuation
        # Preserve markdown structure - don't strip newlines
        return result.strip()
    except (asyncio.TimeoutError, Exception) as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"URL validation failed or timed out: {e}")
        return text


def _detect_content_structure(text: str) -> Dict[str, Any]:
    """Analyze content to determine optimal formatting structure."""
    structure = {
        "has_lists": bool(re.search(r"^[-*]\s", text, re.MULTILINE)),
        "has_bold": bool(re.search(r"\*\*[^*]+\*\*", text)),
        "has_headings": bool(re.search(r"^#{1,6}\s", text, re.MULTILINE)),
        "has_links": bool(re.search(r"\[.*?\]\(.*?\)", text)),
        "word_count": len(text.split()),
        "paragraph_count": len([p for p in text.split("\n\n") if p.strip()]),
        "list_item_count": len(re.findall(r"^[-*]\s", text, re.MULTILINE)),
    }
    return structure


def _should_use_list_format(text: str, structure: Dict[str, Any]) -> bool:
    """Determine if content should be formatted as a list."""
    # If already has lists, keep them
    if structure["has_lists"]:
        return True

    # If content mentions multiple items (services, features, etc.), suggest list format
    list_indicators = [
        r"\d+\s+(services?|items?|features?|options?|ways?|steps?)",
        r"(multiple|several|various|different)\s+(services?|items?|features?|options?)",
        r"(include|offer|provide|have)\s+(services?|features?|options?)",
    ]

    text_lower = text.lower()
    for pattern in list_indicators:
        if re.search(pattern, text_lower):
            return True

    # If content has multiple sentences that could be list items
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", text) if s.strip()]
    if len(sentences) >= 3 and structure["word_count"] > 50:
        # Check if sentences start with similar patterns (indicating list items)
        first_words = [s.split()[0].lower() if s.split() else "" for s in sentences[:5]]
        if len(set(first_words)) <= 2:  # Similar starting words suggest list format
            return True

    return False


def _enhance_formatting(text: str, structure: Dict[str, Any]) -> str:
    """Enhance formatting based on content analysis - minimal changes to preserve markdown."""
    # Don't modify markdown - LLM already outputs proper markdown
    # Only return as-is to preserve markdown structure
    # structure parameter kept for compatibility but not used
    _ = structure  # Suppress unused warning
    return text


def format_response(
    response: str,
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Smart formatter that adapts based on content structure, length, and type.
    Preserves markdown while enhancing readability.
    """
    resp_text = response.strip()
    if not resp_text:
        return resp_text

    # Analyze content structure
    structure = _detect_content_structure(resp_text)

    # If markdown detected, preserve all newlines (do not collapse to single newlines)
    markdown_markers = ["- ", "* ", "#", "**", "__", "`", "[", "]("]
    has_markdown = any(m in resp_text for m in markdown_markers)

    if has_markdown:
        # Fix common token splits, but preserve all newlines
        paragraph_block = re.sub(r"\*\s+\*", "**", resp_text)
        paragraph_block = re.sub(r"_\s+_", "__", paragraph_block)
        paragraph_block = re.sub(r"(`{1,3})\s+(`{1,3})", r"\1\2", paragraph_block)

        # Fix markdown list formatting - ensure proper spacing and line breaks
        paragraph_block = re.sub(r"([-*])\s*\n\s*([-*])", r"\1\n\2", paragraph_block)
        paragraph_block = re.sub(r"([-*])([^\s\*\-\n])", r"\1 \2", paragraph_block)
        paragraph_block = re.sub(r"(#{1,6})([^\s#\n])", r"\1 \2", paragraph_block)

        # Ensure all double-escaped newlines are real newlines
        paragraph_block = paragraph_block.replace("\\n", "\n")

        # Ensure proper line breaks before lists
        paragraph_block = re.sub(r"([^\n])\n([-*]\s)", r"\1\n\n\2", paragraph_block)

        # Ensure proper line breaks after lists
        paragraph_block = re.sub(r"([-*].*)\n([^-*\n\s])", r"\1\n\n\2", paragraph_block)

        # Enhance formatting based on content analysis
        paragraph_block = _enhance_formatting(paragraph_block, structure)

        return paragraph_block
    else:
        # Legacy formatting for plain text
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


async def format_response_with_url_validation(
    response: str,
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Format response and remove invalid URLs (404s).
    This is the async version that should be used for URL validation.
    """
    formatted = format_response(response, query, conversation_history)
    # Remove invalid URLs
    formatted = await _remove_invalid_urls(formatted)
    return formatted
