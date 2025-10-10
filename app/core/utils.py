from app.core.llm_client import llm
from langchain_openai import ChatOpenAI
from app.core.prompts import SHARED_SYSTEM_PROMPT
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)


def generate_llm_response(prompt):
    """
    Handles both string prompts and list of messages.
    Converts inputs into proper LangChain message objects.

    Returns:
      - str: the LLM's text response on success
      - None: on failure or empty response
    """
    messages = None
    try:
        # Build LangChain message objects
        if isinstance(prompt, list):
            messages = []
            has_system_message = False
            for message in prompt:
                if not isinstance(message, dict):
                    logger.error("Invalid message format (not a dict): %s", message)
                    return None
                if "role" not in message or "content" not in message:
                    logger.error("Invalid message keys: %s", message)
                    return None

                role = message["role"]
                content = message["content"]
                if role == "system":
                    has_system_message = True

                if role == "system":
                    messages.append(SystemMessage(content=content))
                elif role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))
                else:
                    logger.error("Invalid role type: %s", role)
                    return None

            if not has_system_message:
                messages.insert(0, SystemMessage(content=SHARED_SYSTEM_PROMPT))
        else:
            # Single string prompt
            messages = [SystemMessage(content=SHARED_SYSTEM_PROMPT), HumanMessage(content=str(prompt).strip())]

        # Diagnostic logging (truncated previews)
        try:
            preview = " | ".join([getattr(m, "content", "")[:200] for m in messages])
        except Exception:
            preview = "<<unavailable preview>>"
        logger.info("Invoking LLM (generate_llm_response)")
        logger.debug("Messages preview: %s", preview)

        # Prefer a non-streaming local LLM for single-shot responses to avoid
        # streaming objects being returned by the global streaming llm.
        try:
            model_name = getattr(llm, 'model_name', None) or getattr(llm, 'model', 'gpt-4o')
            local_llm = ChatOpenAI(
                model=model_name,
                temperature=getattr(llm, 'temperature', 0.7),
                streaming=False,
            )
            response = local_llm.invoke(messages)
        except Exception:
            # Fallback to global llm (may be streaming)
            response = llm.invoke(messages)

        # Log a small preview of raw response object
        try:
            raw_preview = str(response)[:400]
        except Exception:
            raw_preview = "<<unserializable response>>"
        logger.info("LLM returned object type=%s; preview=%s", type(response).__name__, raw_preview)

        # We still keep an in-memory preview in the normal logs for debugging.
        logger.debug("LLM raw preview stored in memory (no on-disk diag)")

        # Extract text (duck-typed): prefer .content, then dict-like get, else str()
        result = None
        try:
            if hasattr(response, "content"):
                result = response.content
            else:
                # dict-like access
                get = getattr(response, "get", None)
                if callable(get):
                    result = get("content")
                else:
                    result = str(response)
        except Exception:
            result = str(response)

        result_text = None
        if result is None:
            result_text = None
        elif isinstance(result, (str, bytes)):
            result_text = str(result).strip()
        else:
            try:
                result_text = str(result).strip()
            except Exception:
                result_text = None
        if not result_text:
            logger.warning("LLM returned empty or whitespace-only content")
            return None

        logger.debug("LLM text preview: %s", result_text[:400])
        return result_text

    except Exception as exc:
        logger.exception("Error during LLM invocation: %s", exc)
        logger.error("Messages that caused the error: %s", (messages if messages is not None else "<none>"))
        return None



