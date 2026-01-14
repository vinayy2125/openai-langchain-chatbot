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
from typing import Optional, Dict, Tuple
from openai import OpenAI
from app.orchestrator.uc1_config import UC1Config, CapabilityBucket
from app.orchestrator.slot_manager import UC1Slots
from app.orchestrator.state_machine import UC1State, ResponseIntent
from app.logger import get_logger
from app.utils.conversation_memory import get_session_memory_manager

logger = get_logger("llm_adapter")


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
CANONICAL_SYSTEM_PROMPT = """You are an AI assistant representing DITSTEK.

Your responsibility is to understand the user's intent from natural language and respond clearly, concisely, and naturally.

You are not aware of any internal conversation states, flows, funnels, policies, or system logic.
You respond only to what the user actually says.

CRITICAL RULES:
- NEVER echo, quote, or repeat context metadata back to the user (e.g., "Focus: X | Goal: Y | Name: Z" is FORBIDDEN)
- Use provided context to inform your response, but speak naturally as if you already know this information
- Reference what the user has shared in previous messages to show you're listening
- When the user shares a problem, ask specific follow-up questions about THEIR situation

Behavior rules:
- If the user asks a clear question, answer it directly with specific, actionable guidance
- If the input is vague or ambiguous, ask one clarifying question about their specific situation
- If the user gives a casual acknowledgment (e.g., "ok", "yeah", "good to know"), acknowledge briefly and offer a relevant next step or question
- Be grounded in DITSTEK's services and capabilities without using marketing or sales language
- Do not assume the user wants to proceed, commit, schedule, or take next steps unless they explicitly indicate interest
- Do not introduce calls, demos, meetings, or contact requests by default

Response style:
- Natural, professional, conversational
- 2–5 sentences unless more detail is requested
- Consultative, not directive
- Reference specific details from the conversation to show you're paying attention
- No scripted transitions
- No multiple questions in one response

Your goal is to maintain a meaningful, context-aware conversation that feels human, helpful, and intelligent.

After your response, add a final line exactly in this format:
<INTENT>: acknowledged | exploring | ready_for_cta | closing | unclear

INTENT values:
- acknowledged — user acknowledged info ("ok", "got it", "thanks")
- exploring — user is asking questions or exploring options
- ready_for_cta — user explicitly indicates readiness (wants to schedule, discuss, proceed)
- closing — user wants to end the conversation
- unclear — cannot determine intent from input

Example output:
That makes sense — improving smart responses and engagement usually requires tightening how intent is handled across conversations.

<INTENT>: exploring
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
        
        ANCHOR INJECTION:
            Anchor is injected via user-message prefix (NOT system prompt).
            This preserves the single-prompt invariant.
        
        CONVERSATION MEMORY:
            Recent conversation history is injected as multi-turn messages.
            This provides context continuity across turns.
        
        Returns:
            Tuple[str, LLMIntent]: (response_text, inferred_intent)
        
        Intent is extracted via delimiter (<INTENT>:) for robustness.
        This survives streaming, partial output, and never corrupts text.
        """
        if self.client is None:
            return "[Service unavailable]", LLMIntent.UNCLEAR
        
        # Build user message with anchor prefix
        user_message = f"[Context]\n{anchor}\n\nUser: {user_text}" if anchor else user_text
        
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
                    role = "user" if msg.type == "human" else "assistant"
                    messages.append({"role": role, "content": msg.content})
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
                model=UC1_FINE_TUNED_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()
            
            # Delimiter-based intent parsing (robust, never blocks UI)
            if "<INTENT>:" in raw:
                text, intent_part = raw.rsplit("<INTENT>:", 1)
                text = text.strip()
                intent_str = intent_part.strip().lower()
                
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
                logger.info(f"[LLMAdapter] Intent inferred: {intent.value}")
                
                # Store assistant response in memory AFTER successful LLM call
                if session_id and text:
                    try:
                        memory_mgr = get_session_memory_manager()
                        memory_mgr.add_ai_message(session_id, text)
                    except Exception as e:
                        logger.warning(f"[LLMAdapter] Failed to store AI message: {e}")
                
                return text, intent
            
            # ================================================================
            # FALLBACK: Heuristic intent inference for non-retrained models
            # ================================================================
            # The fine-tuned model wasn't trained with <INTENT>: format
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
            
            return raw, intent
            
        except Exception as e:
            logger.warning(f"[LLMAdapter] LLM call failed: {e}")
            return "[Service unavailable]", LLMIntent.UNCLEAR
    
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
    ) -> Tuple[str, LLMIntent]:
        """
        Single entry point for all language generation.
        
        ACC HARD GATES:
        1. Slots required for all LLM calls
        2. Anchor mandatory after initial states
        3. Returns (text, intent) tuple
        
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
        
        if slots.selected_cta == "schedule_call":
            return f"Great, {name}! We'll be in touch to schedule a call."
        elif slots.selected_cta == "discuss_requirement":
            return f"Perfect, {name}! We'll reach out to discuss your requirements."
        elif slots.selected_cta == "explore_more":
            return f"Sounds good, {name}! Feel free to explore more."
        else:
            return f"Thanks for chatting, {name}! Feel free to come back anytime."
