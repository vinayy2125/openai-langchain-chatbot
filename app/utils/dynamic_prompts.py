"""
Dynamic prompt assembly utilities with Redis integration and fallback to prompts.py.
"""

from typing import Optional, Dict
from app.logger import get_logger

logger = get_logger("dynamic_prompts")

# Token limits for Groq API - keep synchronized with llm_utils.py
MAX_PROMPT_TOKENS: int = 6000  # Leave room for response and overhead
CHARS_PER_TOKEN: float = 4.0  # Approximate: 1 token ≈ 4 characters


def _estimate_tokens(text: str) -> int:
    """Estimate token count for a string."""
    if not text:
        return 0
    return int(len(text) / CHARS_PER_TOKEN)


def _truncate_text(text: str, max_tokens: int, label: str = "content") -> str:
    """Truncate text to approximately fit within max_tokens."""
    if not text:
        return text
    
    estimated = _estimate_tokens(text)
    if estimated <= max_tokens:
        return text
    
    target_chars = int(max_tokens * CHARS_PER_TOKEN)
    truncated = text[:target_chars]
    
    # Try to find a good break point
    last_newline = truncated.rfind('\n')
    if last_newline > target_chars * 0.5:
        result = truncated[:last_newline] + f"\n[...{label} truncated for token limits...]"
    else:
        last_space = truncated.rfind(' ')
        if last_space > target_chars * 0.7:
            result = truncated[:last_space] + f"\n[...{label} truncated for token limits...]"
        else:
            result = truncated + f"\n[...{label} truncated for token limits...]"
    
    logger.info(f"[TokenLimit] Truncated {label}: {estimated} -> ~{_estimate_tokens(result)} tokens")
    return result


def _enforce_prompt_token_limit(prompt: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    """
    Enforce token limit on the final assembled prompt.
    
    Truncation strategy:
    1. First try to identify and truncate context sections
    2. If still too large, truncate from the middle (preserve start/end)
    """
    estimated = _estimate_tokens(prompt)
    if estimated <= max_tokens:
        return prompt
    
    logger.warning(f"[TokenLimit] Prompt exceeds limit ({estimated} > {max_tokens}). Truncating...")
    
    # Try to identify context section and truncate it first
    # Look for common context markers
    context_markers = [
        "## CONTEXT FROM KNOWLEDGE BASE",
        "## RELEVANT CONTEXT",
        "## CONVERSATION HISTORY",
        "---\n## CONTEXT",
    ]
    
    for marker in context_markers:
        if marker in prompt:
            parts = prompt.split(marker, 1)
            if len(parts) == 2:
                before_context = parts[0]
                after_marker = parts[1]
                
                # Find the end of context section
                next_section = None
                for end_marker in ["## USER'S CURRENT QUERY", "## CURRENT QUERY", "## OUTPUT FORMAT", "---\n##"]:
                    pos = after_marker.find(end_marker)
                    if pos > 0:
                        next_section = pos
                        break
                
                if next_section:
                    context_content = after_marker[:next_section]
                    rest_of_prompt = after_marker[next_section:]
                    
                    # Calculate how much we can keep in context
                    overhead = _estimate_tokens(before_context) + _estimate_tokens(rest_of_prompt) + 50
                    available_for_context = max_tokens - overhead
                    
                    if available_for_context > 200:
                        truncated_context = _truncate_text(context_content, available_for_context, "context")
                        new_prompt = before_context + marker + truncated_context + rest_of_prompt
                        new_estimated = _estimate_tokens(new_prompt)
                        
                        if new_estimated <= max_tokens:
                            logger.info(f"[TokenLimit] Reduced prompt from {estimated} to {new_estimated} tokens by truncating context")
                            return new_prompt
    
    # Fallback: simple truncation preserving start and end
    target_chars = int(max_tokens * CHARS_PER_TOKEN)
    keep_start = int(target_chars * 0.6)  # Keep 60% from start
    keep_end = int(target_chars * 0.35)   # Keep 35% from end
    
    result = prompt[:keep_start] + "\n\n[...content truncated for token limits...]\n\n" + prompt[-keep_end:]
    logger.info(f"[TokenLimit] Truncated prompt using fallback: {estimated} -> ~{_estimate_tokens(result)} tokens")
    return result


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
    4. Enforces token limits to prevent Groq API rate limit errors
    """
    from app.utils.prompts import (
        _build_user_details_context, 
        _greeting_instruction, 
        _build_context_block,
        final_response_prompt
    )
    from app.utils.redis_prompt_loader import get_prompt_sections_from_redis
    
    # Pre-truncate large inputs before assembly
    # Context and conversation summary are the main token consumers
    max_context_tokens = 2000
    max_summary_tokens = 1500
    
    if _estimate_tokens(prompt_context) > max_context_tokens:
        prompt_context = _truncate_text(prompt_context, max_context_tokens, "knowledge base context")
    
    if _estimate_tokens(conversation_summary) > max_summary_tokens:
        conversation_summary = _truncate_text(conversation_summary, max_summary_tokens, "conversation summary")
    
    try:
        # Try to fetch prompt sections from Redis
        sections = get_prompt_sections_from_redis()
        
        if sections:
            # Use Redis sections for core prompt components
            core = sections.get("core", "")
            behavior = sections.get("behavior", "")
            prospect_profiling = sections.get("prospect_profiling", "")
            funnel_logic = sections.get("funnel_logic", "")
            output_schema = sections.get("output_schema", "")
            reminders = sections.get("reminders", "")
            
            # Build dynamic blocks using shared functions from prompts.py
            user_details_context = _build_user_details_context(user_details, user_details_known)
            greeting = _greeting_instruction(count)
            
            # Build user entities context
            user_entities = ""
            if last_user_reply:
                user_entities += f"\nLast User Reply: {last_user_reply}"
            if last_assistant_prompt:
                user_entities += f"\nLast Assistant Prompt: {last_assistant_prompt}"
            
            # Use shared context block builder
            context_block = _build_context_block(
                prompt_context,
                conversation_summary,
                query,
                user_details_known,
                count,
                user_entities,
                user_details_context,
            )
            
            # Assemble final prompt
            prompt_parts = []
            
            if core:
                try:
                    formatted_core = core.format(user_details_known=user_details_known)
                    prompt_parts.append(formatted_core)
                except Exception:
                    prompt_parts.append(core)
            
            if greeting and behavior:
                prompt_parts.append(f"{greeting}\n{behavior}")
            elif greeting:
                prompt_parts.append(greeting)
            elif behavior:
                prompt_parts.append(behavior)
            
            if prospect_profiling:
                prompt_parts.append(prospect_profiling)
            
            if funnel_logic:
                prompt_parts.append(funnel_logic)
            
            if output_schema:
                prompt_parts.append(output_schema)
            
            prompt_parts.append(context_block)
            
            if reminders:
                prompt_parts.append(reminders)
            
            prompt = "\n\n".join(filter(None, prompt_parts))
            
            # Enforce final token limit
            prompt = _enforce_prompt_token_limit(prompt)
            
            estimated = _estimate_tokens(prompt)
            logger.info(f"Successfully assembled prompt using dynamic sections from Redis (~{estimated} tokens)")
            return prompt
            
    except Exception as e:
        logger.error(f"Error building dynamic prompt from Redis: {e}")
    
    # Fallback to original prompts.py function
    logger.info("Falling back to prompts.py final_response_prompt")
    prompt = final_response_prompt(
        prompt_context,
        conversation_summary,
        query,
        count,
        user_details_known,
        user_details,
        last_assistant_prompt,
        last_user_reply
    )
    
    # Enforce token limit on fallback prompt too
    prompt = _enforce_prompt_token_limit(prompt)
    
    return prompt
