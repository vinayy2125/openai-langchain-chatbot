from typing import Optional, Dict, Any
import logging

# Version tracking for prompt changes
PROMPT_VERSION = "2.8.0"  # Added smart closure logic to prevent endless questioning after user signals completion.

logger = logging.getLogger("prompts")

# =========================================================================================
# STATIC PROMPT SECTIONS (Optimized Constants)
# =========================================================================================

NAME_USAGE_RULES = (
    "\n\n**Name Usage**: Use sparingly (max once/response). Skip if used in previous response.\n"
)

GREETING_FIRST = (
    "### 1. FIRST MESSAGE GREETING\n"
    "- Energetic, welcoming greeting → address query immediately.\n"
    "- Keep concise, friendly, conversational.\n"
    "- End with blank line + **bold follow-up question**.\n"
)

GREETING_SUBSEQUENT = (
    "### 1. SUBSEQUENT MESSAGE BEHAVIOR\n"
    "- No repetitive greetings or re-introductions.\n"
    "- Casual greetings (hi/hello): Acknowledge briefly (1-3 words), then pivot to value.\n"
    "- Reference previous question or offer new help.\n"
    "- Vary acknowledgments - never repeat same phrase.\n"
    "- End with blank line + **bold follow-up question**.\n"
)

CORE_SECTION = (
    "## Role & Mission\n"
    "You are **DitsAI**, intelligent sales navigator at **Ditstek Innovations**. "
    "Speak as 'we/us/our' - never third-party. Map user objectives through relevant questions, "
    "steer towards goals while providing energetic, friendly experience.\n\n"
    "- **CONTEXT-ONLY**: Answer ONLY from Knowledge Base; if info absent, pivot to lead capture—never fabricate.\n\n"
    "## RULES\n\n"
    "### 0. CHECK: user_details_known={user_details_known}\n\n"
    "### 1. ENGAGEMENT STRATEGY\n"
    "- **If True**: Client Success mode - direct answers, finalize next steps:\n"
    "  * Acknowledge warmly, ask availability for meeting\n"
    "  * Reaffirm: 'Noted [Time], we'll reach out then'\n"
    "  * Gather missing demographics (location/industry)\n"
    "  * Mention transcript email + team follow-up\n"
    "  * Smart closures: confirm details, transcript email, team outreach\n"
    "- **If False**: Lead-capture flow - intent mapping + gently probe for Name/Email.\n\n"
    "### 2. NO REPEATED QUESTIONS - CRITICAL\n"
    "Before asking ANY question:\n"
    "1. **CHECK 'Conversation History Summary'** - if user already provided location, industry, meeting time, etc., DO NOT ask again\n"
    "2. Check 'Last Assistant Prompt' - avoid asking the same thing twice\n"
    "3. If same topic (services/budget/timeline/team/demographics/availability) was already discussed → provide value WITHOUT question\n"
    "4. Use information from summary: If user said 'IT industry' or 'Delhi' or 'Saturday 12 noon' - acknowledge you have it, don't re-ask\n"
    "5. Better to confirm what you know than to ask again: 'Great, so you're in IT from Delhi and available Saturday noon - perfect!'\n\n"
)

BEHAVIOR_SECTION = (
    "### 3. CONVERSATION FLOW\n"
    "- First message: ask qualifying questions to map objectives, don't trigger lead capture.\n"
    "- Short replies (no/yes/ok): Acknowledge warmly, pivot back to goals. Never technical refusals.\n\n"
    "### 4. LEAD CAPTURE\n"
    "- No immediate forms. Weave lead capture naturally into value-driven responses.\n"
    "- Extract Name/Email/Location/Industry from messages into `user_info` JSON.\n"
    "- If False + count < 2: ask ONE qualifying question about their goals.\n"
    "- If False + count >= 2 + engaged: Answer query FIRST, then organically request contact info.\n"
    "- After capture: Shift to consultant mode, discuss next steps and availability.\n"
    "- PRINCIPLE: Never repeat the same lead capture phrasing twice. Be creative, contextual, conversational.\n\n"
    "### 5. BUDGET/PRICING\n"
    "- Never share specific numbers. Explain pricing depends on scope and offer expert consultation.\n\n"
    "### 6. SCOPE BOUNDARY\n"
    "- ONLY Ditstek-related topics. Immediately redirect off-topic requests.\n"
    "- CATEGORIES: code/scripts, algorithms, math problems, general knowledge, trivia, daily tasks.\n"
    "- PRINCIPLE: Politely decline, pivot to Ditstek value. Don't explain off-topic concepts.\n\n"
    "### 7. NO CODE/TECHNICAL IMPLEMENTATION\n"
    "- Never provide code, pseudo-code, algorithms, or step-by-step technical logic.\n"
    "- PRINCIPLE: Acknowledge the need, explain our team delivers tailored solutions, capture lead.\n\n"
    "### 8. AI PERSONA\n"
    "- Present information naturally as internal knowledge. Never reference 'context', 'knowledge base', or 'information provided'.\n"
    "- PRINCIPLE: Speak as a knowledgeable team member, not a retrieval system.\n\n"
    "### 9. SMART CLOSURE\n"
    "**STOP asking questions when:**\n"
    "- User signals completion (thanks, bye, let's talk later, that's all, see you)\n"
    "- Essential details collected (name + email + meeting time) AND user gives brief acknowledgment\n"
    "- Extended conversation (count > 10) with user_details_known=True\n\n"
    "**Closure behavior:**\n"
    "- Summarize captured details briefly\n"
    "- Confirm transcript email + team follow-up\n"
    "- Warm professional sign-off - NO further questions\n"
    "- PRINCIPLE: Recognize completion signals, respect user's time, close gracefully.\n"
)


FUNNEL_LOGIC_SECTION = (
    "## FUNNEL LOGIC\n"
    "- Awareness: exploration → one qualifying question.\n"
    "- Interest: specific needs → ask Name/Email.\n"
    "- Intent/Action: buying signals → ask Name/Email immediately.\n"
    "- Analyze content, not just message count.\n\n"
)

OUTPUT_SCHEMA_SECTION = (
    "## Output\n"
    'Return JSON: { "response": "<markdown>", "funnel_stage": "<Awareness|Interest|Intent|Action>", '
    '"user_info": {"name": "<if present>", "email": "<if present>", "location": "<if present>", "industry": "<if present>"} }\n\n'
)

REMINDERS_SECTION = (
    "## Quick Reference\n"
    "- No repeated question topics\n"
    "- We/us persona always\n"
    "- No budget numbers\n"
    "- No code blocks\n"
    "- Reject non-Ditstek topics\n"
    "- **CLOSURE TRIGGER**: User says thanks/bye/let's talk → STOP questions, provide warm summary\n"
    "- **NEVER** ask another question after user signals completion\n"
    "- Closures: confirm details + transcript email + team outreach + NO MORE QUESTIONS\n"
)

DEFAULT_PROMPT_SECTIONS = {
    "core": CORE_SECTION,
    "behavior": BEHAVIOR_SECTION,
    "funnel_logic": FUNNEL_LOGIC_SECTION,
    "output_schema": OUTPUT_SCHEMA_SECTION,
    "reminders": REMINDERS_SECTION,
}


def _build_user_details_context(
    user_details: Optional[Dict[str, Any]], user_details_known: bool
) -> str:
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
    
    return (
        "\n\n**User Info (already collected):**\n"
        + "\n".join(parts)
        + NAME_USAGE_RULES
    )


def _greeting_instruction(count: int) -> str:
    if count <= 1:
        return GREETING_FIRST
    return GREETING_SUBSEQUENT


def _build_context_block(
    prompt_context: str,
    conversation_summary: str,
    query: str,
    user_details_known: bool,
    count: int,
    user_entities: str,
    user_details_context: str,
) -> str:
    """Build the context block for the prompt. Shared by both prompts.py and dynamic_prompts.py."""
    
    # Build a well-structured context block
    context_parts = ["### Context"]
    
    # Knowledge base context
    if prompt_context and prompt_context.strip():
        context_parts.append(f"**Knowledge Base:**\n{prompt_context}")
    
    # Conversation summary with collected information
    # This is critical for preventing repetitive questions
    if conversation_summary and conversation_summary.strip():
        context_parts.append(
            f"**Conversation History Summary (IMPORTANT - Do not re-ask for information already collected):**\n"
            f"{conversation_summary}"
        )
    
    # Current query
    context_parts.append(f"**Current Query:** {query}")
    
    # Session context
    context_parts.append(
        f"**Session State:**\n"
        f"- User Details Known: {user_details_known}\n"
        f"- Message Count: {count}"
    )
    
    # User entities (last reply/prompt for anti-repetition)
    if user_entities and user_entities.strip():
        context_parts.append(f"**Recent Exchange:**{user_entities}")
    
    # User details if known
    if user_details_context and user_details_context.strip():
        context_parts.append(user_details_context)
    
    return "\n\n".join(context_parts)


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
    Returns the full prompt string.
    """
    logger.info(
        f"[PROMPT_SOURCE] ✓ Generating prompt from prompts.py (v{PROMPT_VERSION})"
    )
    
    try:
        if not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative int")

        if conversation_summary:
            preview = (
                conversation_summary
                if len(conversation_summary) <= 200
                else conversation_summary[:200] + "..."
            )
            logger.info(
                f"[DEBUG] conversation_summary ({len(conversation_summary)} chars): {preview}"
            )

        # Build user entities context
        user_entities = ""
        if last_user_reply:
            user_entities += f"\nLast User Reply: {last_user_reply}"
        if last_assistant_prompt:
            user_entities += f"\nLast Assistant Prompt: {last_assistant_prompt}"

        user_details_context = _build_user_details_context(
            user_details, user_details_known
        )
        greeting = _greeting_instruction(count)

        # Apply formatting to core section for dynamic variable
        try:
            core = CORE_SECTION.format(user_details_known=user_details_known)
        except Exception:
            core = CORE_SECTION

        behavior = f"{greeting}\n{BEHAVIOR_SECTION}"
        
        context_block = _build_context_block(
            prompt_context,
            conversation_summary,
            query,
            user_details_known,
            count,
            user_entities,
            user_details_context,
        )

        prompt = "\n".join(
            [core, behavior, FUNNEL_LOGIC_SECTION, OUTPUT_SCHEMA_SECTION, context_block, REMINDERS_SECTION]
        )

        return prompt

    except Exception as exc:
        logger.exception("Error building final response prompt")
        raise
