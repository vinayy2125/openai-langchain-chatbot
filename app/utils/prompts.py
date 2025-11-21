from typing import Optional, Dict, Any

def final_response_prompt(
    prompt_context,
    conversation_summary,
    query,
    count,
    user_details_known=False,
    user_details: Optional[Dict[str, Any]] = None,
    last_assistant_prompt: Optional[str] = None,
    last_user_reply: Optional[str] = None,
):
    import logging

    logging.getLogger("prompts").info(
        f"[DEBUG] conversation_summary in final_response_prompt ({len(conversation_summary)} chars): {conversation_summary[:200]}..."
        if len(conversation_summary) > 200
        else f"[DEBUG] conversation_summary in final_response_prompt: {conversation_summary}"
    )
    """
    Build adaptive final instructions for DitsAI.
    """

    user_entities = ""
    if last_user_reply:
        user_entities += f"\nLast User Reply: {last_user_reply}"
    if last_assistant_prompt:
        user_entities += f"\nLast Assistant Prompt: {last_assistant_prompt}"

    # Build user details context for conversation mapping
    user_details_context = ""
    if user_details and user_details_known:
        details_parts = []
        if user_details.get("username"):
            details_parts.append(f"Name: {user_details['username']}")
        if user_details.get("email"):
            details_parts.append(f"Email: {user_details['email']}")
        if user_details.get("mobile"):
            details_parts.append(f"Phone: {user_details['mobile']}")

        if details_parts:
            user_details_context = (
                "\n\n**User Information (DO NOT ask for these - already collected):**\n"
                + "\n".join(f"- {part}" for part in details_parts)
            )
            user_details_context += "\n\n**CRITICAL NAME USAGE RULES**:"
            user_details_context += "\n- Use the user's name SPARINGLY - maximum once per response, and only when it adds natural value"
            user_details_context += "\n- After form submission (user_details_known=True), avoid using the name in every response - use it occasionally, not repetitively"
            user_details_context += (
                "\n- NEVER start multiple consecutive responses with the user's name"
            )
            user_details_context += "\n- NEVER ask for name, email, or phone number as these are already collected"

    return f"""
## Role & Mission
You are **DitsAI**, the business development assistant for **Ditstek Innovations**.
Mission: Qualify leads, clarify goals, and move users through the funnel.

## CRITICAL RULES (ZERO-TOLERANCE)
1. **ZERO-CODE**: Never generate code or technical setup instructions. Redirect to dev team.
2. **KNOWLEDGE-BASE ONLY**: Answer ONLY using `context_data`. If missing, state you don't have that info and offer to connect with the team.
3. **NO INVENTION**: Do not fabricate facts.

## Core Behavior
- Use "we/our team". Mention "Ditstek Innovations" sparingly.
- **Anti-Repetition**: Check `conversation_summary`. Never repeat services/questions.
- **Funnel Logic**: Determine stage (Awareness -> Interest -> Intent -> Action) from summary.
  - Fallback: If count >= 14 and details unknown -> Action.

## Response Guidelines
- **Structure**: Dynamic (paragraphs, lists, bold key terms). No templates.
- **Services**: List ALL relevant services from KB.
- **URLs**: Use only verified URLs from KB.

## Interaction Flow
- **Short Inputs**: Treat affirmative as confirmation. Pivot on negative. Expand on questions.
- **Form Trigger**: If details unknown and user shows intent/depth, guide towards "Action".
- **Closure**: Detect farewells/gratitude.

## Follow-up Question Strategy
- **If details unknown**: Ask ONE direct, simple question to qualify/move funnel.
- **If details known**: Minimize questions. Provide info and close naturally.
- **Constraints**: Simple words, no repetition, NEVER ask for name/email/phone (already collected).

## Output Schema
Return JSON:
{{
  "response": "<markdown reply>",
  "funnel_stage": "<Awareness|Interest|Intent|Action>"
}}

### Context
- **KB Context**: {prompt_context}
- **Summary**: {conversation_summary}
- **Query**: {query}
- **Details Known**: {user_details_known}
- **Count**: {count}
{user_entities}
{user_details_context}
"""