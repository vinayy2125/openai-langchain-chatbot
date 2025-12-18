"""
Dynamic prompt assembly utilities with Redis integration and fallback to prompts.py.
"""

from typing import Optional, Dict
from app.logger import get_logger

logger = get_logger("dynamic_prompts")


def build_dynamic_prompt(
    prompt_context: str,
    conversation_summary: str,
    query: str,
    count: int,
    user_details_known: bool,
    user_details: dict,
    last_assistant_prompt: str = None,
    last_user_reply: str = None
) -> str:
    """
    Build LLM prompt using dynamic sections from Redis with fallback to prompts.py.
    
    This function:
    1. Tries to fetch structured prompt sections from Redis
    2. Assembles the prompt using Redis sections + dynamic context
    3. Falls back to final_response_prompt from prompts.py if Redis unavailable
    """
    from app.utils.prompts import _build_user_details_context, _greeting_instruction, final_response_prompt
    from app.utils.redis_prompt_loader import get_prompt_sections_from_redis
    
    try:
        # Try to fetch prompt sections from Redis
        sections = get_prompt_sections_from_redis()
        
        if sections:
            # Use Redis sections for core prompt components
            core = sections.get("core", "")
            behavior = sections.get("behavior", "")
            funnel_logic = sections.get("funnel_logic", "")
            output_schema = sections.get("output_schema", "")
            reminders = sections.get("reminders", "")
            
            # Build dynamic blocks using existing functions from prompts.py
            user_details_context = _build_user_details_context(user_details, user_details_known)
            greeting = _greeting_instruction(count)
            
            # Build user entities context
            user_entities = ""
            if last_user_reply:
                user_entities += f"\nLast User Reply: {last_user_reply}"
            if last_assistant_prompt:
                user_entities += f"\nLast Assistant Prompt: {last_assistant_prompt}"
            
            # Build context block
            context_block = (
                "### Context\n"
                f"- KB Context: {prompt_context}\n"
                f"- Summary: {conversation_summary}\n"
                f"- Query: {query}\n"
                f"- Details Known: {user_details_known}\n"
                f"- Count: {count}\n"
                f"{user_entities}\n"
                f"{user_details_context}\n"
            )
            
            # Assemble final prompt by combining Redis sections with dynamic context
            prompt_parts = []
            
            if core:
                try:
                    # Apply formatting if the section contains placeholders
                    formatted_core = core.format(user_details_known=user_details_known)
                    prompt_parts.append(formatted_core)
                except Exception:
                    # Fallback to raw string if formatting fails or no placeholder
                    prompt_parts.append(core)
            
            if greeting and behavior:
                prompt_parts.append(f"{greeting}\n{behavior}")
            elif greeting:
                prompt_parts.append(greeting)
            elif behavior:
                prompt_parts.append(behavior)
            
            if funnel_logic:
                prompt_parts.append(funnel_logic)
            
            if output_schema:
                prompt_parts.append(output_schema)
            
            prompt_parts.append(context_block)
            
            if reminders:
                prompt_parts.append(reminders)
            
            prompt = "\n\n".join(filter(None, prompt_parts))
            
            logger.info("Successfully assembled prompt using dynamic sections from Redis")
            return prompt
            
    except Exception as e:
        logger.error(f"Error building dynamic prompt from Redis: {e}")
    
    # Fallback to original prompts.py function
    logger.info("Falling back to prompts.py final_response_prompt")
    return final_response_prompt(
        prompt_context,
        conversation_summary,
        query,
        count,
        user_details_known,
        user_details,
        last_assistant_prompt,
        last_user_reply
    )