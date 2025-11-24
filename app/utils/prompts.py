from typing import Optional, Dict, Any
import logging

logger = logging.getLogger("prompts")


def _build_user_details_context(user_details: Optional[Dict[str, Any]], user_details_known: bool) -> str:
    if not user_details or not user_details_known:
        return ""
    parts = []
    if user_details.get("username"):
        parts.append(f"- Name: {user_details['username']}")
    if user_details.get("email"):
        parts.append(f"- Email: {user_details['email']}")
    if user_details.get("mobile"):
        parts.append(f"- Phone: {user_details['mobile']}")
    if not parts:
        return ""
    rules = (
        "\n\n**CRITICAL NAME USAGE RULES**:\n"
        "- Use the user's name SPARINGLY - maximum once per response.\n"
        "- Do not start consecutive responses with the user's name.\n"
        "- Do not ask for name/email/phone—they are already collected.\n"
    )
    return "\n\n**User Information (DO NOT ask for these - already collected):**\n" + "\n".join(parts) + rules


def _greeting_instruction(count: int) -> str:
    if count <= 1:
        return (
            "### 1. MANDATORY GREETING (FIRST MESSAGE)\n"
            "- Start with a warm, professional greeting, then address the query immediately.\n"
        )
    return (
        "### 1. NO REPETITIVE GREETINGS (SUBSEQUENT MESSAGES)\n"
        "- Do NOT use the full welcome again or re-introduce the assistant unless asked.\n"
        "- If user says hi, respond briefly and move to the query.\n"
    )


def final_response_prompt(
    prompt_context: str,
    conversation_summary: str,
    query: str,
    count: int,
    user_details_known: bool = False,
    user_details: Optional[Dict[str, Any]] = None,
    last_assistant_prompt: Optional[str] = None,
    last_user_reply: Optional[str] = None,
) -> str:
    """
    Build a compact adaptive final-instructions prompt for DitsAI.
    Keeps same rules/intent as original but with clearer structure and validation.
    Returns the full prompt string.
    """
    try:
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative int")

        # Lightweight debug logging of conversation_summary
        if conversation_summary:
            preview = (
                conversation_summary if len(conversation_summary) <= 200 else conversation_summary[:200] + "..."
            )
            logger.info(f"[DEBUG] conversation_summary in final_response_prompt ({len(conversation_summary)} chars): {preview}")

        # recent user/assistant snippets
        user_entities = ""
        if last_user_reply:
            user_entities += f"\nLast User Reply: {last_user_reply}"
        if last_assistant_prompt:
            user_entities += f"\nLast Assistant Prompt: {last_assistant_prompt}"

        user_details_context = _build_user_details_context(user_details, user_details_known)
        greeting = _greeting_instruction(count)

        core = (
            "## Role & Mission\n"
            "You are **DitsAI**, the intelligent business development assistant for **Ditstek Innovations**.\n"
            "Mission: Analyze conversations to determine intent and capture leads when appropriate.\n\n"

            "## CRITICAL RULES (ZERO-TOLERANCE)\n\n"

            "### 0. CHECK USER DETAILS STATUS FIRST\n"
            f"BEFORE doing anything, check user_details_known={user_details_known}.\n"
            "- If True: switch to Client Success mode. Answer questions; do NOT ask for details again.\n"
            "- If False: follow lead-capture flow.\n\n"
        )

        behavior = (
            f"{greeting}\n"
            "### 2. CONVERSATIONAL INTELLIGENCE\n"
            "- On the first user message: do not trigger contact form immediately. Ask succinct qualifying questions.\n\n"

            "### 3. LEAD CAPTURE\n"
            "- Never provide direct contact info.\n"
            "- If user_details_known=False and count < 2: ask ONE qualifying question.\n"
            "- If user_details_known=False and count >= 2 and user provided project details: trigger contact form.\n"
            "- If user_details_known=True: confirm receipt and answer questions; do not re-ask for details.\n\n"

            "### 4. SMART CONSULTANT APPROACH\n"
            "- Detect buying signals (timeline, budget, intent). Trigger form when appropriate.\n\n"

            "### 5. OTHER CRITICAL RULES\n"
            "- ZERO-CODE: Do not generate technical setup code. Redirect to dev team.\n"
            "- KNOWLEDGE-BASE ONLY: Use context_data; do not invent facts.\n\n"
        )

        funnel_logic = (
            "## DYNAMIC FUNNEL LOGIC\n"
            "- Awareness: general exploration — ask one qualifying question.\n"
            "- Interest: specific needs — follow up and can trigger form if engaged.\n"
            "- Intent: clear buying signals — can trigger form earlier.\n"
            "- Action: explicit request to connect — trigger form immediately.\n"
            "- Always analyze content; do not rely on message count alone.\n\n"
        )

        output_schema = (
            "## Output Schema\n"
            "Return JSON:\n"
            '{ "response": "<markdown reply>", "funnel_stage": "<Awareness|Interest|Intent|Action>" }\n\n'
        )

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

        reminders = (
            "## IMPORTANT REMINDERS\n"
            "0. If user_details_known=True, switch to Client Success mode first.\n"
            "1. Greeting strategy depends on message count.\n"
            "2. Detect buying signals and trigger the form when justified.\n"
            "3. Never provide direct contact information.\n"
        )

        prompt = "\n".join([core, behavior, funnel_logic, output_schema, context_block, reminders])

        return prompt

    except Exception as exc:
        logger.exception("Error building final response prompt")
        raise
