# UC1 LLM Adapter - ACC IMPLEMENTATION (2026-01-12)
#
# =============================================================================
# ARCHITECTURAL LAW (NON-NEGOTIABLE):
#   If a fact exists in slots and is relevant to the turn,
#   the LLM MUST see it via the anchor.
#   Violation of this rule is a system bug, not a model issue.
# =============================================================================
#
# ARCHITECTURE:
#   Single canonical prompt. Single LLM call path.
#   State controls flow. State never controls language.
#   LLM reasons first. Orchestrator enforces after.
#   Anchor is MANDATORY - empty anchor past ENTRY is a crash.

import os
from enum import Enum
from typing import Optional, Dict, Tuple, List
from openai import OpenAI
from app.orchestrator.uc1_config import UC1Config, CapabilityBucket
from app.orchestrator.slot_manager import UC1Slots
from app.orchestrator.state_machine import UC1State, ResponseIntent
from app.logger import get_logger
from app.utils.conversation_memory import get_session_memory_manager
from core_services.hybrid_search import get_hybrid_search_manager

logger = get_logger("llm_adapter")


# =========================================================================
# KNOWLEDGE BASE RETRIEVAL - Per-call KB context injection
# =========================================================================
def _get_kb_context(query: str, top_k: int = 3) -> str:
    """
    Retrieve relevant KB context for the user's query.
    
    Uses hybrid search (semantic + BM25) for best results.
    Returns formatted context string for LLM injection.
    
    Args:
        query: User's input/question
        top_k: Number of top results to include (default 3)
        
    Returns:
        Formatted KB context string, or empty string if no results
    """
    if not query or len(query.strip()) < 5:
        return ""
    
    try:
        search_mgr = get_hybrid_search_manager()
        results = search_mgr.hybrid_search(query, top_n=top_k)
        
        if not results:
            logger.debug(f"[KBContext] No results for query: {query[:50]}...")
            return ""
        
        # Format as context block for LLM
        context_parts = []
        for i, result in enumerate(results[:top_k], 1):
            text = result.get("text", "")
            # Truncate each chunk to avoid overwhelming the prompt
            if len(text) > 400:
                text = text[:400] + "..."
            if text.strip():
                context_parts.append(f"[{i}] {text.strip()}")
        
        if not context_parts:
            return ""
        
        kb_context = "\n\n".join(context_parts)
        logger.info(f"[KBContext] Retrieved {len(context_parts)} chunks for: {query[:50]}...")
        return kb_context
        
    except Exception as e:
        logger.warning(f"[KBContext] KB retrieval failed: {e}")
        return ""


# =========================================================================
# INTENT ENUM - LLM infers intent from natural language
# =========================================================================
class LLMIntent(str, Enum):
    """
    Intent inferred by LLM from user's natural language.
    
    State allows. Intent decides.
    No state/bucket/flow hints injected into LLM.
    """
    ACKNOWLEDGED = "acknowledged"      # User acknowledged info ("ok", "got it")
    DECLINED = "declined"              # User rejected/negated ("no", "nope")
    EXPLORING = "exploring"            # User is exploring/asking questions
    READY_FOR_CTA = "ready_for_cta"    # User shows readiness for next steps
    CLOSING = "closing"                # User wants to end conversation
    UNCLEAR = "unclear"                # Cannot determine intent


# =========================================================================
# OUTPUT VIOLATION ENUM - Signals for rejected LLM output (ACC Phase 2)
# =========================================================================
class OutputViolation(Enum):
    """
    Signals returned when LLM output violates Authoritative Context Contract.
    
    Orchestrator decides recovery action, not this adapter.
    This preserves text-blind orchestrator invariant.
    """
    REDUNDANT_QUESTION = "redundant_question"  # Asked for already-known info


# =========================================================================
# SLOT SATURATION PATTERNS - Questions prohibited when slot is filled
# =========================================================================
# These patterns are BLOCKED when context_signal slot is filled
CONTEXT_SIGNAL_PROHIBITIONS = [
    "what's the biggest challenge",
    "what's your biggest challenge",
    "what problem are you trying to solve",
    "what are you looking to build",
    "what challenge",
    "what's your main goal",
    "what are you trying to accomplish",
    "tell me about your challenge",
    "what brings you here",
    "what's the problem",
]


# Fine-tuned UC1 model ID (uc1-render-clean-2026-01-13 - DETERMINISTIC ALIGNED)
# Trained on: reflection, alternatives, exit, meta — NO flow control
UC1_FINE_TUNED_MODEL = os.getenv(
    "UC1_FINE_TUNED_MODEL",
    "ft:gpt-4.1-mini-2025-04-14:info-ditstek-com:uc1-chatbot:CxTZSPpZ"
)


# =========================================================================
# CANONICAL SYSTEM PROMPT — DELIMITER-BASED INTENT (2026-01-12)
# =========================================================================
# This is the ONLY system prompt. No variants. No extensions. No runtime edits.
# LLM emits intent via delimiter (robust for streaming). No JSON required.
# =========================================================================
CANONICAL_SYSTEM_PROMPT = """You are the DITSTEK AI assistant. You ARE DITSTEK — not a representative, not a third party.

IDENTITY (CRITICAL):
- You speak AS DITSTEK, not ABOUT DITSTEK
- Use first-person: "we", "our", "us" — NEVER "they", "their", "DITS offers", "you can reach out to DITS"
- WRONG: "DITS offers AI services" / "You can contact DITS" / "Their website"
- CORRECT: "We offer AI services" / "You can contact us" / "Our website"
- You ARE the company. You ARE the team. Speak with ownership.

Your responsibility is to understand the user's intent from natural language and respond clearly, concisely, and naturally.

You are not aware of any internal conversation states, flows, funnels, policies, or system logic.
You respond only to what the user actually says.

KNOWLEDGE BASE:
- When [KNOWLEDGE_BASE] context is provided in your input, use it to answer with SPECIFIC examples
- Reference real services, case studies, and capabilities from the KB context
- If KB context is empty or not relevant, provide general guidance and offer to explore specifics

CRITICAL RULES:
- NEVER echo, quote, or repeat context metadata back to the user (e.g., "Focus: X | Goal: Y | Name: Z" is FORBIDDEN)
- Use provided context to inform your response, but speak naturally as if you already know this information
- Reference what the user has shared in previous messages to show you're listening
- When the user shares a problem, ask specific follow-up questions about THEIR situation

CTA HANDLING (When user shows strong intent to connect):
When user says things like "Talk to expert", "Schedule a call", "Get a demo", "Contact", "Speak to someone":
1. Acknowledge their intent warmly and confirm you can help
2. If you don't have their email, ask: "Happy to connect you with our team! What's your email so we can follow up?"
3. If you already have their name but not email, ask for email
4. If you have both name and email, confirm: "Thanks [Name]! We have your details — our team will reach out to you shortly."
5. ALWAYS include the contact link: "You can also schedule directly here: https://www.ditstek.com/contact-us"
6. NEVER redirect to external website or say "contact DITS" — YOU are DITS, handle it directly

ENGAGEMENT RULES:
- After answering a question, suggest 1-2 specific follow-up actions the user might find valuable
- Use the user's name naturally when addressing them (if known)
- Be specific — instead of "we can help", say "our AI/ML team has done similar work for..."
- Offer concrete next steps when the user seems interested
- Make responses lead-generative: naturally guide toward consultation, demos, or deeper exploration

Behavior rules:
- If the user asks a clear question, answer it directly with specific, actionable guidance
- If the input is vague or ambiguous, ask one clarifying question about their specific situation
- If the user gives a casual acknowledgment (e.g., "ok", "yeah", "good to know"), acknowledge briefly and offer a relevant next step or question
- Be grounded in DITSTEK's services and capabilities without using marketing or sales language
- Do not assume the user wants to proceed, commit, schedule, or take next steps unless they explicitly indicate interest
- Do not introduce calls, demos, meetings, or contact requests by default — but DO offer them when the user shows interest

Response style:
- Natural, professional, conversational
- 2–5 sentences unless more detail is requested
- Consultative, not directive
- Reference specific details from the conversation to show you're paying attention
- No scripted transitions
- No multiple questions in one response
- End with a helpful suggestion or follow-up question when appropriate

Your goal is to maintain a meaningful, context-aware conversation that feels human, helpful, and intelligent.

After your response, add a final line exactly in this format:

<INTENT>: acknowledged | exploring | ready_for_cta | closing | unclear

DYNAMIC BUTTON OPTIONS (NEXT STEP GUIDES):
- ALWAYS generate 2-3 button options after your response (except for ready_for_cta intent)
- Options must be CONTEXTUAL NEXT STEPS based on:
  1. What the user asked/said
  2. What you just answered
  3. Logical follow-up questions or actions
- THESE MUST BE BRIEF (1-4 words max)
- Format: <<OPTIONS: Option 1 | Option 2 | Option 3>>
- Place this strictly at the end of your response, AFTER the <INTENT> tag

GOOD examples of next-step options:
- After explaining AI deployment: "Discovery session" | "See a demo" | "Timeline details"
- After discussing costs: "Compare plans" | "Get a quote" | "ROI calculator"
- After showing capabilities: "Case studies" | "Technical specs" | "Talk to expert"

BAD examples (too generic):
- "Learn more" | "Tell me more" | "Continue" (not specific to conversation)
- "Yes" | "No" | "Maybe" (not actionable next steps)

RULES:
- Options should feel like natural next questions the user might ask
- Avoid repeating options from previous turns
- DO NOT generate options if intent is ready_for_cta (user will see CTA buttons instead)

INTENT values:
- acknowledged — user acknowledged info ("ok", "got it", "thanks")
- exploring — user is asking questions or exploring options
- ready_for_cta — user explicitly indicates readiness (wants to schedule, discuss, proceed)
- closing — user wants to end the conversation
- unclear — cannot determine intent from input

Example output:
That makes sense — improving smart responses and engagement usually requires tightening how intent is handled across conversations. Would you like to see how we approached this for a similar project, or should we focus on your specific implementation challenges first?

<INTENT>: exploring
<<OPTIONS: Similar Project | Implementation Challenges | Best Practices>>
"""


class ConstrainedLLMAdapter:
    """
    Single language authority for UC1.
    
    One prompt. One LLM call path. State is invisible to the LLM.
    """
    
    def __init__(self, config: UC1Config):
        self.config = config
        self._client = None
    
    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("[LLMAdapter] OPENAI_API_KEY not set")
                return None
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    # =========================================================================
    # SINGLE LLM CALL CONTRACT (THE ONLY WAY TO CALL THE MODEL)
    # =========================================================================
    def _canonical_llm(self, user_text: str, anchor: str = "", session_id: str = None) -> Tuple[str, LLMIntent]:
        """
        The only LLM call in UC1. No branching. No variants.
        ...
        """
        logger.info("<< PROMPT TRIGGER >> Using UC1 CANONICAL_SYSTEM_PROMPT")
        
        if self.client is None:
            return "[Service unavailable]", LLMIntent.UNCLEAR
        
        # =========================================================================
        # KB CONTEXT RETRIEVAL - Inject domain knowledge into the prompt
        # =========================================================================
        kb_context = _get_kb_context(user_text, top_k=3)
        
        # Build user message with KB context and anchor prefix
        context_parts = []
        if kb_context:
            context_parts.append(f"[KNOWLEDGE_BASE]\n{kb_context}")
        if anchor:
            context_parts.append(f"[Context]\n{anchor}")
        
        if context_parts:
            context_block = "\n\n".join(context_parts)
            user_message = f"{context_block}\n\nUser: {user_text}"
        else:
            user_message = user_text
        
        # Build messages array with conversation history
        messages = [{"role": "system", "content": CANONICAL_SYSTEM_PROMPT}]
        
        # Inject recent conversation history for context continuity
        if session_id:
            try:
                memory_mgr = get_session_memory_manager()
                memory = memory_mgr.get_or_create_memory(session_id)
                # Get recent messages (last 6 = 3 exchanges) for context
                history_messages = memory.chat_memory.messages[-6:] if memory.chat_memory.messages else []
                for msg in history_messages:
                    if isinstance(msg, dict):
                        role_val = msg.get("role") or msg.get("type")
                        content_val = msg.get("content")
                        role = "user" if role_val in ["human", "user"] else "assistant"
                        messages.append({"role": role, "content": content_val})
                    else:
                        role = "user" if getattr(msg, "type", None) == "human" else "assistant"
                        messages.append({"role": role, "content": getattr(msg, "content", "")})
                if history_messages:
                    logger.debug(f"[LLMAdapter] Injected {len(history_messages)} history messages for session {session_id}")
            except Exception as e:
                logger.warning(f"[LLMAdapter] Failed to load conversation history: {e}")
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Store user message in memory BEFORE LLM call
        if session_id and user_text:
            try:
                memory_mgr = get_session_memory_manager()
                memory_mgr.add_user_message(session_id, user_text)
            except Exception as e:
                logger.warning(f"[LLMAdapter] Failed to store user message: {e}")
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini-2025-04-14",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()
            
            # ================================================================
            # SANITIZE: Remove any leaked internal patterns
            # ================================================================
            # ================================================================
            # SANITIZE: Defer until after intent check
            # (Prevents stripping <INTENT>: before we read it)
            # ================================================================
            # raw = self._sanitize_response(raw)  <-- DEFERRED
            
            # Delimiter-based intent parsing (robust, never blocks UI)
            if "<INTENT>:" in raw:
                text, intent_part = raw.rsplit("<INTENT>:", 1)
                text = text.strip()
                intent_part = intent_part.strip()
                
                # Extract options if present (must be in intent_part or at end)
                extracted_options = []
                if "<<OPTIONS:" in intent_part:
                    intent_part_clean, options_part = intent_part.split("<<OPTIONS:", 1)
                    if ">>" in options_part:
                        options_str = options_part.split(">>", 1)[0]
                        extracted_options = [opt.strip() for opt in options_str.split("|") if opt.strip()]
                    intent_part = intent_part_clean.strip()
                
                # Double-sanitize the text portion
                text = self._sanitize_response(text)
                intent_str = intent_part.lower()
                
                # Map to enum (includes declined)
                intent_map = {
                    "acknowledged": LLMIntent.ACKNOWLEDGED,
                    "declined": LLMIntent.DECLINED,
                    "exploring": LLMIntent.EXPLORING,
                    "ready_for_cta": LLMIntent.READY_FOR_CTA,
                    "closing": LLMIntent.CLOSING,
                    "unclear": LLMIntent.UNCLEAR,
                }
                intent = intent_map.get(intent_str, LLMIntent.UNCLEAR)
                logger.info(f"[LLMAdapter] Intent inferred: {intent.value}, Options: {extracted_options}")
                
                # Store assistant response in memory AFTER successful LLM call
                if session_id and text:
                    try:
                        memory_mgr = get_session_memory_manager()
                        memory_mgr.add_ai_message(session_id, text)
                    except Exception as e:
                        logger.warning(f"[LLMAdapter] Failed to store AI message: {e}")
                
                return text, intent, extracted_options
            
            # ================================================================
            # FALLBACK: Heuristic intent inference for non-retrained models
            # ================================================================
            # The fine-tuned model wasn't trained with <INTENT>: format
            # ================================================================
            # FALLBACK: Heuristic intent inference for non-retrained models
            # ================================================================
            # The fine-tuned model wasn't trained with <INTENT>: format
            # Sanitize now since we failed to parse structure
            raw = self._sanitize_response(raw)
            
            # Use pattern matching on user input as temporary fallback
            intent = self._infer_intent_heuristic(user_text, raw)
            logger.info(f"[LLMAdapter] Intent via heuristic: {intent.value}")
            
            # Store assistant response in memory (fallback path)
            if session_id and raw:
                try:
                    memory_mgr = get_session_memory_manager()
                    memory_mgr.add_ai_message(session_id, raw)
                except Exception as e:
                    logger.warning(f"[LLMAdapter] Failed to store AI message: {e}")
            
            # Generate fallback options from response content (TEMPORARY until model retrained)
            fallback_options = self._generate_fallback_options(raw, user_text)
            
            return raw, intent, fallback_options
            
        except Exception as e:
            logger.warning(f"[LLMAdapter] LLM call failed: {e}")
            return "[Service unavailable]", LLMIntent.UNCLEAR, []
    
    def _sanitize_response(self, text: str) -> str:
        """
        Remove any leaked internal patterns from LLM response.
        
        This catches cases where the model echoes instruction fragments.
        """
        import re
        
        # Patterns that should NEVER appear in user-facing text
        forbidden_patterns = [
            r"acknowledged\s*\|\s*exploring\s*\|\s*ready_for_cta\s*\|\s*closing\s*\|\s*unclear",
            r"<INTENT>:\s*(acknowledged|exploring|ready_for_cta|closing|unclear)",
            r"\[Context\].*?\n\n",  # Context block header
            r"Focus:.*?\|.*?Goal:.*?\|",  # Anchor echo
            r"\bUC1-[A-F]\b",  # Internal Bucket IDs (e.g. UC1-A)
            r"\bUC\d+-[A-Z0-9]+\b", # Generic Internal IDs
        ]
        
        for pattern in forbidden_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove any line that looks like an intent instruction
        lines = text.split("\n")
        clean_lines = []
        for line in lines:
            line_lower = line.lower().strip()
            # Skip lines that are just intent values or contain the full options
            if line_lower in ["acknowledged", "exploring", "ready_for_cta", "closing", "unclear"]:
                continue
            if "acknowledged" in line_lower and "exploring" in line_lower and "ready_for_cta" in line_lower:
                continue
            clean_lines.append(line)
        
        result = "\n".join(clean_lines).strip()
        
        # If we stripped everything, return a safe fallback
        if not result:
            return "I'm here to help. What would you like to know?"
        
        return result
    
    def _infer_intent_heuristic(self, user_input: str, bot_response: str) -> LLMIntent:
        """
        Heuristic intent inference fallback.
        
        Used when fine-tuned model doesn't emit <INTENT>: delimiter.
        This is a TEMPORARY measure until model is retrained.
        """
        input_lower = user_input.lower().strip()
        response_lower = bot_response.lower()
        
        # CLOSING patterns
        closing_patterns = [
            "bye", "goodbye", "no thanks", "not interested", 
            "i'll pass", "maybe later", "not now"
        ]
        if any(p in input_lower for p in closing_patterns):
            return LLMIntent.CLOSING
        
        # READY_FOR_CTA patterns (explicit readiness) - CHECK BEFORE ACKNOWLEDGED
        # These indicate user wants to proceed/get help
        cta_patterns = [
            # Explicit scheduling/contact
            "schedule", "book a call", "let's talk", "contact me",
            "ready to", "want to proceed", "sign me up", "get started",
            "discuss my", "talk to someone", "connect me",
            # Natural readiness signals (2026-01-14)
            "yes please", "sure please", "please help", "can you help",
            "help me", "that would be great", "sounds good", "let's do",
            "i'd like to", "i would like", "i want to", "let's proceed",
            "what's next", "next steps", "how do we start", "how to start",
            "i'm ready", "i am ready", "ready for", "move forward"
        ]
        if any(p in input_lower for p in cta_patterns):
            return LLMIntent.READY_FOR_CTA
        
        # ACKNOWLEDGED patterns (short affirmations)
        ack_patterns = ["ok", "okay", "got it", "thanks", "sure", "alright", "i see"]
        if len(input_lower.split()) <= 4 and any(p in input_lower for p in ack_patterns):
            return LLMIntent.ACKNOWLEDGED
        
        # If bot response asks a question, user is likely exploring
        if "?" in response_lower:
            return LLMIntent.EXPLORING
        
        # Default to exploring (safe default for conversation continuation)
        return LLMIntent.EXPLORING
    
    def _generate_fallback_options(self, bot_response: str, user_input: str = "") -> List[str]:
        """
        Generate contextual fallback options when LLM doesn't emit <<OPTIONS:>> tag.
        
        This extracts potential next steps from the response content.
        TEMPORARY: Until model is retrained with OPTIONS format.
        """
        response_lower = bot_response.lower()
        options = []
        
        # Pattern 1: Look for questions in the response and suggest related actions
        logger.info(f"[LLMAdapter] Generating fallback options for input: '{user_input}'")
        
        # Helper to check if option is redundant with user input
        def is_redundant(opt: str) -> bool:
            return opt.lower() in user_input.lower() or user_input.lower() in opt.lower()

        # Pattern 1: Look for questions in the response and suggest related actions
        if "assessment" in response_lower or "evaluate" in response_lower:
            options.append("Get assessment")
        if "demo" in response_lower or "show" in response_lower:
            options.append("See a demo")
        if "discuss" in response_lower or "talk" in response_lower:
            options.append("Talk to expert")
        if "timeline" in response_lower or "when" in response_lower:
            options.append("Timeline details")
        if "cost" in response_lower or "pricing" in response_lower or "budget" in response_lower:
            options.append("Get a quote")
        
        # SMART ESCALATION: If discussing team/experts, offer contact/scheduling instead of more info
        if "team" in response_lower or "expert" in response_lower:
            if is_redundant("Meet the team"):
                options.append("Schedule a call")
            else:
                options.append("Meet the team")
                
        if "case" in response_lower or "example" in response_lower or "similar" in response_lower:
            if is_redundant("See examples"):
                options.append("View case studies") 
            else:
                options.append("See examples")

        if "architecture" in response_lower or "design" in response_lower:
            options.append("Architecture review")
        # REMOVED: "Next steps" - ambiguous button replaced by explicit CTAs
        # When exit-ready, ButtonManager surfaces CTAs directly
        
        # Pattern 2: If response has a question mark, suggest "Tell me more" variant
        if "?" in bot_response and len(options) < 3:
            options.append("Tell me more")
        
        # Filter out options that are too similar to what the user just said (Stop the loop)
        final_options = []
        for opt in options:
            if not is_redundant(opt):
                final_options.append(opt)
            elif opt == "Schedule a call": # Allow escalation even if seemingly redundant (unlikely but safe)
                 final_options.append(opt)

        # Limit to 3 unique options
        unique_options = list(dict.fromkeys(final_options))[:3]
        
        # Fallback: If we couldn't extract anything, use generic contextual options
        if not unique_options:
            unique_options = ["Learn more", "See examples", "Talk to expert"]
            # Ensure we don't return the exact user input as a fallback
            unique_options = [opt for opt in unique_options if not is_redundant(opt)]
            if not unique_options:
                 unique_options = ["Contact us", "Schedule a call"] # Ultimate fallback
        
        logger.info(f"[LLMAdapter] Generated fallback options: {unique_options}")
        return unique_options
    
    # =========================================================================
    # PUBLIC API - ALL USE _canonical_llm
    # =========================================================================
    
    def _build_anchor(self, slots: UC1Slots, bucket: Optional[CapabilityBucket]) -> str:
        """
        Build grounding context from slots. Never returns empty after CONTEXT_QUESTION.
        
        ACC INVARIANT: If information exists in slots, it MUST appear in anchor.
        
        FORMAT: Uses instruction-style language to prevent LLM from echoing context.
        """
        parts = []
        
        # Build context summary (internal reference, not for echoing)
        if bucket:
            parts.append(f"The user is interested in: {bucket.trigger}")
        if slots.context_signal:
            signal = slots.context_signal[:150]
            if len(slots.context_signal) > 150:
                signal += "..."
            parts.append(f"Their specific situation: {signal}")
        if slots.user_name:
            parts.append(f"Their name is {slots.user_name}")
        if slots.selected_alternative:
            parts.append(f"They chose to focus on: {slots.selected_alternative}")
        
        if not parts:
            return ""
        
        # Wrap with instruction to prevent echoing
        context_summary = ". ".join(parts) + "."
        return f"[CONTEXT - Do NOT echo or quote this back. Use it to inform your response.]\n{context_summary}"
    
    def validate_output(self, response: str, slots: UC1Slots) -> Optional[OutputViolation]:
        """
        Validate LLM output against slot saturation rules.
        
        Returns signal if violation detected; orchestrator decides recovery.
        This preserves text-blind orchestrator invariant.
        
        ACC INVARIANT: Never ask for information already in slots.
        """
        if not response or not slots:
            return None
            
        response_lower = response.lower()
        
        # If context_signal exists, block redundant qualification questions
        if slots.context_signal:
            for pattern in CONTEXT_SIGNAL_PROHIBITIONS:
                if pattern in response_lower:
                    logger.warning(f"[ACC] Output violation: redundant question '{pattern}' when context_signal exists")
                    return OutputViolation.REDUNDANT_QUESTION
        
        return None
    
    def generate_state_response(
        self,
        *,
        state: UC1State,
        response_intent: ResponseIntent = None,
        user_input: Optional[str] = None,
        slots: Optional[UC1Slots] = None,
        bucket: Optional[CapabilityBucket] = None,
        exploration_turn: int = 0,
        session_id: Optional[str] = None,
    ) -> Tuple[str, LLMIntent, List[str]]:
        """
        Single entry point for all language generation.
        
        ACC HARD GATES:
        1. Slots required for all LLM calls
        2. Anchor mandatory after initial states
        3. Returns (text, intent, options) tuple
        
        State is invisible to the LLM. Intent is inferred from user input.
        """
        # =============================================================
        # ACC HARD GATE 1: Slots required
        # =============================================================
        assert slots is not None, "ACC Violation: Slots required for LLM call"
        
        # =============================================================
        # ACC: Build anchor from slots
        # =============================================================
        anchor = self._build_anchor(slots, bucket)
        
        # =============================================================
        # ACC HARD GATE 2: Anchor mandatory after initial states
        # =============================================================
        # =============================================================
        # ACC HARD GATE 2: Anchor mandatory after initial states
        # =============================================================
        initial_states = {UC1State.ENTRY, UC1State.CAPABILITY_SELECTION}
        if state not in initial_states and not anchor:
            logger.error(f"[ACC] Anchor empty in non-initial state {state.value}")
            # Soft-fail in production to avoid crashes, but log loudly
            # In development, this should be an assert
        
        # =============================================================
        # ARCHITECTURAL GUARD: Fixed-prompt states MUST NOT call LLM
        # =============================================================
        # These states emit language directly from config via orchestrator.
        # LLM is ONLY for rephrasing, explaining, reflecting on user input.
        # This assertion prevents future regressions.
        FIXED_PROMPT_STATES = {UC1State.ENTRY, UC1State.CONTEXT_QUESTION, UC1State.NAME_CAPTURE, UC1State.EXIT}
        assert state not in FIXED_PROMPT_STATES, (
            f"ACC VIOLATION: LLM called for fixed-prompt state {state.value}. "
            "This must be emitted by the orchestrator."
        )
        
        logger.info(f"[ACC] Anchor injected: {anchor[:80]}..." if anchor else "[ACC] No anchor (initial state)")
        
        return self._canonical_llm(user_input or "", anchor=anchor, session_id=session_id)

    
    # =========================================================================
    # CONFIG-BASED METHODS (NO LLM - KEEP FOR ORCHESTRATOR)
    # =========================================================================
    def generate_capability_selection_prompt(self) -> str:
        """Format capability options from config."""
        lines = ["\n**Select a capability area:**\n"]
        for bucket in self.config.capability_buckets:
            lines.append(f"- {bucket.name}")
        return "\n".join(lines)
    
    def generate_context_question_prompt(self, bucket: CapabilityBucket) -> str:
        """Get context question from config."""
        return bucket.context_question
    
    def generate_name_capture_prompt(self) -> str:
        """Get name capture prompt from config."""
        return self.config.name_capture_prompt
    
    def generate_exit_summary(
        self,
        slots: UC1Slots,
        bucket: Optional[CapabilityBucket],
    ) -> str:
        """
        Deterministic exit summary. Zero LLM.
        """
        name = slots.user_name or "there"
        email_part = f" at {slots.user_email}" if slots.user_email else ""
        
        # Base link for calendar/contact
        link = "https://www.ditstek.com/contact-us"
        
        if slots.selected_cta_outcome == "calendar":
            return (
                f"Great, {name}! We'll be in touch{email_part} to schedule a call.\n\n"
                f"Prefer to book a time manually right now? Use this link:\n{link}"
            )
        elif slots.selected_cta_outcome == "UC2": # Discuss requirement
            return (
                f"Perfect, {name}! We'll reach out{email_part} to discuss your requirements.\n\n"
                f"You can also schedule a time directly here:\n{link}"
            )
        elif slots.selected_cta_outcome == "loop":
            return f"Sounds good, {name}! Feel free to explore more."
        else:
            return f"Thanks for chatting, {name}! Feel free to come back anytime."
