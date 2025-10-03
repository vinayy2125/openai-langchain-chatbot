from app.core.llm_client import llm
from langchain_openai import ChatOpenAI
from app.core.prompts import SHARED_SYSTEM_PROMPT
import logging
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from datetime import datetime

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

        # Best-effort server-side diagnostic JSON write so we can inspect the exact
        # object shape seen inside the running server process. This is append-only
        # and should never raise (we swallow exceptions).
        try:
            import json
            diag = {
                "ts": datetime.utcnow().isoformat(),
                "type": type(response).__name__,
                "repr_preview": raw_preview
            }
            with open(r"d:/Chatbot/logs/llm_server_diag.json", "a", encoding="utf-8") as f:
                f.write(json.dumps(diag, ensure_ascii=False) + "\n")
        except Exception:
            logger.debug("Failed to write llm_server_diag.json; continuing without diagnostics")

        # If the response is empty or None, also write a small diagnostic file for analysis
        try:
            if response is None or (isinstance(response, (str, bytes)) and not str(response).strip()):
                with open("d:/Chatbot/logs/llm_diag.log", "a", encoding="utf-8") as diag:
                    diag.write(f"{datetime.utcnow().isoformat()} - EMPTY_RESPONSE - type={type(response).__name__} repr={repr(response)[:1000]}\n")
        except Exception:
            # Best-effort; don't break execution
            logger.debug("Failed to write llm_diag.log")

        # Extract text
        if hasattr(response, "content"):
            result = response.content
        elif isinstance(response, dict) and "content" in response:
            result = response.get("content")
        else:
            result = str(response)

        result_text = result.strip() if isinstance(result, str) else str(result)
        if not result_text:
            logger.warning("LLM returned empty or whitespace-only content")
            return None

        logger.debug("LLM text preview: %s", result_text[:400])
        return result_text

    except Exception as exc:
        logger.exception("Error during LLM invocation: %s", exc)
        logger.error("Messages that caused the error: %s", (messages if messages is not None else "<none>"))
        return None
