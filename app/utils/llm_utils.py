from typing import Any, Dict, List, Optional, Union
import re

from app.utils.llm_client import llm
from app.logger import get_logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = get_logger(__name__)

# Configuration (move to top for SonarQube / maintainability)
DEFAULT_MODEL: str = "gpt-4o"
DEFAULT_TEMPERATURE: float = 0.7
FALLBACK_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"  # kept for reference if needed

# Token limits for OpenAI API
MAX_REQUEST_TOKENS: int = 7000  # Keep below context limit with buffer
MAX_RESPONSE_TOKENS: int = 1000  # Reserve tokens for response
CHARS_PER_TOKEN: float = 4.0  # Approximate: 1 token ≈ 4 characters


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for a string.
    Uses character-based estimation (avg 4 chars per token).
    This is a conservative estimate suitable for most models.
    """
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN)


def estimate_messages_tokens(messages: List[Any]) -> int:
    """
    Estimate total tokens for a list of messages.
    Includes overhead for message structure (~4 tokens per message).
    """
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "") or ""
        total += estimate_tokens(content) + 4  # +4 for role/structure overhead
    return total


def truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately fit within max_tokens.
    Tries to break at sentence or word boundaries.
    """
    if not text:
        return text
    
    estimated = estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    
    # Calculate target character count
    target_chars = int(max_tokens * CHARS_PER_TOKEN)
    
    # Try to find a good break point (sentence end, newline, or space)
    truncated = text[:target_chars]
    
    # Look for last sentence boundary
    last_period = max(truncated.rfind('. '), truncated.rfind('.\n'))
    if last_period > target_chars * 0.5:  # At least 50% of content preserved
        return truncated[:last_period + 1] + "\n[...truncated for length...]"
    
    # Fall back to word boundary
    last_space = truncated.rfind(' ')
    if last_space > target_chars * 0.7:
        return truncated[:last_space] + "\n[...truncated for length...]"
    
    return truncated + "\n[...truncated for length...]"


def _truncate_messages_to_fit(
    messages: List[Any], 
    max_tokens: int = MAX_REQUEST_TOKENS
) -> List[Any]:
    """
    Truncate messages to fit within token limit.
    Strategy:
    1. Keep system message intact (truncate if necessary)
    2. Keep most recent messages
    3. Summarize/truncate older messages
    """
    if not messages:
        return messages
    
    current_tokens = estimate_messages_tokens(messages)
    if current_tokens <= max_tokens:
        return messages
    
    logger.warning(
        f"Messages exceed token limit ({current_tokens} > {max_tokens}). Truncating..."
    )
    
    # Separate system message from others
    system_msg = None
    other_msgs = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_msg = msg
        else:
            other_msgs.append(msg)
    
    # Calculate available tokens for conversation
    system_tokens = 0
    if system_msg:
        system_tokens = estimate_tokens(system_msg.content) + 4
        # If system message is too large, truncate it
        if system_tokens > max_tokens * 0.4:  # System shouldn't exceed 40%
            target_sys_tokens = int(max_tokens * 0.35)
            system_msg = SystemMessage(
                content=truncate_text_to_tokens(system_msg.content, target_sys_tokens)
            )
            system_tokens = target_sys_tokens + 4
    
    available_tokens = max_tokens - system_tokens
    
    # Keep recent messages, drop older ones
    result_msgs = []
    tokens_used = 0
    
    # Process from most recent to oldest
    for msg in reversed(other_msgs):
        msg_tokens = estimate_tokens(getattr(msg, "content", "") or "") + 4
        if tokens_used + msg_tokens <= available_tokens:
            result_msgs.insert(0, msg)
            tokens_used += msg_tokens
        else:
            # Try to include a truncated version of this message
            remaining = available_tokens - tokens_used
            if remaining > 50:  # Only if meaningful content can fit
                truncated_content = truncate_text_to_tokens(
                    getattr(msg, "content", ""), 
                    remaining - 10
                )
                if isinstance(msg, HumanMessage):
                    result_msgs.insert(0, HumanMessage(content=truncated_content))
                elif isinstance(msg, AIMessage):
                    result_msgs.insert(0, AIMessage(content=truncated_content))
            break
    
    # Prepend system message
    if system_msg:
        result_msgs.insert(0, system_msg)
    
    final_tokens = estimate_messages_tokens(result_msgs)
    logger.info(
        f"Truncated messages: {len(messages)} -> {len(result_msgs)} messages, "
        f"{current_tokens} -> {final_tokens} tokens"
    )
    
    return result_msgs


def _validate_and_build_messages(
    prompt: Union[str, List[Dict[str, Any]]],
    system_prompt: Optional[str] = None,
) -> Optional[List[Any]]:
    """
    Validate incoming prompt(s) and convert to langchain_core message objects.

    Returns:
        list of message objects or None on validation failure.
    """
    if isinstance(prompt, list):
        messages: List[Any] = []
        has_system = False
        for idx, item in enumerate(prompt):
            if not isinstance(item, dict):
                logger.error("Message at index %d is not a dict: %s", idx, item)
                return None
            if "role" not in item or "content" not in item:
                logger.error("Message at index %d missing required keys: %s", idx, item)
                return None

            role = item["role"]
            content = item["content"]
            if content is None:
                content = ""
            content = str(content).strip()

            if role == "system":
                has_system = True
                messages.append(SystemMessage(content=content))
            elif role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                logger.error("Invalid role '%s' at index %d", role, idx)
                return None

        if not has_system and system_prompt:
            messages.insert(0, SystemMessage(content=system_prompt))
        return messages

    # single string prompt
    text = str(prompt).strip()
    if not text:
        logger.warning("Empty prompt string provided")
        return None
    
    messages = [HumanMessage(content=text)]
    if system_prompt:
        messages.insert(0, SystemMessage(content=system_prompt))
    return messages


def _invoke_with_local_wrapper(messages: List[Any]) -> Any:
    """
    Try to invoke a local ChatOpenAI wrapper if available and compatible.
    This isolates local-specific invocation attempts.
    """
    model_name = getattr(llm, "model", DEFAULT_MODEL)
    temperature = getattr(llm, "temperature", DEFAULT_TEMPERATURE)

    local_llm = ChatOpenAI(model=model_name, temperature=temperature, streaming=False)

    # support multiple call patterns to be resilient across versions
    if hasattr(local_llm, "invoke"):
        return local_llm.invoke(messages)
    if hasattr(local_llm, "generate"):
        return local_llm.generate(messages)
    # last resort: call object if it's callable
    if callable(local_llm):
        return local_llm(messages)

    raise RuntimeError("Local ChatOpenAI has no usable call method")


def _invoke_with_configured_llm(messages: List[Any]) -> Any:
    """
    Invoke the configured `llm` client. Keep fallback handling here so
    generate_llm_response stays focused and easier to test.
    """
    if hasattr(llm, "invoke"):
        return llm.invoke(messages)
    if callable(llm):
        return llm(messages)
    raise RuntimeError("Configured llm object is not callable")


def _extract_text_from_response(response: Any) -> Optional[str]:
    """
    Extract human-readable text from common response shapes.

    Returns a stripped string or None.
    """
    try:
        # 1) LangChain-like ChatResult: .generations -> nested lists -> generation -> message/text
        generations = getattr(response, "generations", None)
        if generations:
            first = None
            if isinstance(generations, list) and generations:
                candidate = generations[0]
                first = candidate[0] if isinstance(candidate, list) and candidate else candidate
            if first is not None:
                msg = getattr(first, "message", None)
                if msg is not None:
                    text = getattr(msg, "content", None) or getattr(msg, "text", None)
                    if text:
                        return str(text).strip()
                text = getattr(first, "text", None) or getattr(first, "content", None)
                if text:
                    return str(text).strip()

        # 2) direct attribute .content
        if hasattr(response, "content"):
            content = getattr(response, "content")
            if content:
                return str(content).strip()

        # 3) mapping-like objects
        get = getattr(response, "get", None)
        if callable(get):
            val = get("content", None)
            if val:
                return str(val).strip()
        if isinstance(response, dict) and "content" in response:
            return str(response["content"]).strip()

        # 4) fallback to stringifying the object
        textual = str(response)
        if textual:
            stripped = textual.strip()
            if stripped:
                return stripped

    except Exception as exc:  # limited scope and logged
        logger.exception("Failed to extract text from response: %s", exc)

    return None


def generate_llm_response(
    prompt: Union[str, List[Dict[str, Any]]],
    system_prompt: Optional[str] = None,
    max_retries: int = 2,
) -> Optional[str]:
    """
    Produce an LLM response text given either a single string prompt or a
    list of role/content message dicts.

    Handles token limits by:
    1. Truncating messages to fit within MAX_REQUEST_TOKENS before sending
    2. Retrying with reduced context if rate limit errors occur

    Args:
        prompt: Single string or list of role/content message dicts
        system_prompt: Optional system prompt to prepend
        max_retries: Number of retries on rate limit errors (default: 2)

    Returns:
        stripped response text or None on error/empty response.
    """

    messages = _validate_and_build_messages(prompt, system_prompt)
    if messages is None:
        return None

    # Pre-emptively truncate to avoid rate limit errors
    messages = _truncate_messages_to_fit(messages, MAX_REQUEST_TOKENS)
    
    # Log token estimate
    estimated_tokens = estimate_messages_tokens(messages)
    logger.info(f"Sending request with ~{estimated_tokens} estimated tokens")

    # Retry loop for handling rate limit errors
    current_max_tokens = MAX_REQUEST_TOKENS
    for attempt in range(max_retries + 1):
        try:
            response: Any
            try:
                response = _invoke_with_local_wrapper(messages)
            except Exception as local_exc:
                error_str = str(local_exc).lower()
                # Check if this is a rate limit error
                if "rate_limit" in error_str or "413" in error_str or "too large" in error_str or "tokens" in error_str:
                    raise local_exc  # Re-raise to trigger retry logic
                logger.debug("Local wrapper invocation failed: %s", local_exc)
                response = _invoke_with_configured_llm(messages)
            
            # Success - extract and return response
            try:
                preview = str(response)
            except Exception:
                preview = "<unserializable-response>"
            logger.info("LLM returned type=%s preview=%s", type(response).__name__, preview)

            result_text = _extract_text_from_response(response)
            if not result_text:
                logger.warning("LLM returned empty or whitespace-only content")
                return None

            return result_text

        except Exception as invoke_exc:
            error_str = str(invoke_exc).lower()
            
            # Check if it's a rate limit / token limit error
            is_rate_limit = any(x in error_str for x in [
                "rate_limit", "413", "too large", "request too large",
                "tokens per minute", "tpm", "reduce your message"
            ])
            
            if is_rate_limit and attempt < max_retries:
                # Reduce token limit further and retry
                current_max_tokens = int(current_max_tokens * 0.6)  # Reduce by 40%
                logger.warning(
                    f"Rate limit error on attempt {attempt + 1}. "
                    f"Reducing context to ~{current_max_tokens} tokens and retrying..."
                )
                messages = _truncate_messages_to_fit(messages, current_max_tokens)
                continue
            
            logger.exception("LLM invocation failed: %s", invoke_exc)
            return None
    
    return None

