from typing import Any, Dict, List, Optional, Union

from app.utils.llm_client import llm
from app.utils.prompts import SHARED_SYSTEM_PROMPT
from app.logger import get_logger
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

logger = get_logger(__name__)

# Configuration (move to top for SonarQube / maintainability)
DEFAULT_MODEL: str = "gpt-4o"
DEFAULT_TEMPERATURE: float = 0.7
FALLBACK_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"  # kept for reference if needed


def _validate_and_build_messages(
    prompt: Union[str, List[Dict[str, Any]]],
    system_prompt: str,
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

        if not has_system:
            messages.insert(0, SystemMessage(content=system_prompt))
        return messages

    # single string prompt
    text = str(prompt).strip()
    if not text:
        logger.warning("Empty prompt string provided")
        return None
    return [SystemMessage(content=system_prompt), HumanMessage(content=text)]


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
) -> Optional[str]:
    """
    Produce an LLM response text given either a single string prompt or a
    list of role/content message dicts.

    Returns:
        stripped response text or None on error/empty response.
    """
    system_prompt = system_prompt or SHARED_SYSTEM_PROMPT

    messages = _validate_and_build_messages(prompt, system_prompt)
    if messages is None:
        return None

    # Attempt local wrapper first, then fallback to configured llm
    response: Any
    try:
        try:
            response = _invoke_with_local_wrapper(messages)
        except Exception as local_exc:
            logger.debug("Local wrapper invocation failed: %s", local_exc)
            response = _invoke_with_configured_llm(messages)
    except Exception as invoke_exc:
        logger.exception("LLM invocation failed: %s", invoke_exc)
        return None

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
