from app.core.llm_client import llm
from langchain_openai import ChatOpenAI
from app.core.prompts import SHARED_SYSTEM_PROMPT
from app.logger import get_logger
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = get_logger(__name__)


def generate_llm_response(prompt):
    messages = None
    try:
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
            messages = [SystemMessage(content=SHARED_SYSTEM_PROMPT), HumanMessage(content=str(prompt).strip())]

        try:
            model_name = getattr(llm, 'model', 'gpt-4o')
            local_llm = ChatOpenAI(
                model=model_name,
                temperature=getattr(llm, 'temperature', 0.7),
                streaming=False,
            )
            response = local_llm.invoke(messages)
        except Exception:
            response = llm.invoke(messages)

        try:
            raw_preview = str(response)
        except Exception:
            raw_preview = "<<unserializable response>>"
        logger.info("LLM returned object type=%s; preview=%s", type(response).__name__, raw_preview)

        result = None
        try:
            if hasattr(response, "content"):
                result = response.content
            else:
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

        return result_text

    except Exception as exc:
        logger.exception("Error during LLM invocation: %s", exc)
        return None



