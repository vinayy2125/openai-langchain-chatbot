# UC1 LLM Adapter - SOLE LANGUAGE AUTHORITY
#
# ARCHITECTURE INVARIANT:
#   This adapter generates 100% of user-visible language.
#   The orchestrator generates 0%.
#
# The fine-tuned model generates natural language based on:
# - Current state (UC1State)
# - Response intent (ResponseIntent) 
# - Conversation context (slots, bucket)
#
# Fallback templates are CIRCUIT BREAKERS only - minimal, neutral text
# used when LLM fails, never for normal operation.

import os
from typing import Optional, Dict, List
from openai import OpenAI
from app.orchestrator.uc1_config import UC1Config, CapabilityBucket
from app.orchestrator.slot_manager import UC1Slots
from app.orchestrator.state_machine import UC1State, ResponseIntent
from app.logger import get_logger

logger = get_logger("llm_adapter")

# Fine-tuned UC1 model ID
UC1_FINE_TUNED_MODEL = os.getenv(
    "UC1_FINE_TUNED_MODEL",
    "ft:gpt-4.1-mini-2025-04-14:info-ditstek-com:uc1-chatbot:CvNCcniL"
)

# Fallback model
FALLBACK_MODEL = "gpt-4.1-mini-2025-04-14"


# Intent-to-rules mapping for fine-tuned model
INTENT_RULES: Dict[ResponseIntent, str] = {
    ResponseIntent.PROMPT: "Ask the appropriate question for this state. Be concise and consultative.",
    ResponseIntent.RETRY: "User input was unclear or empty. Ask again politely without repeating yourself.",
    ResponseIntent.TRANSITION: "Acknowledge what user shared and smoothly transition to the next topic.",
    ResponseIntent.ACKNOWLEDGE: "Acknowledge user by name warmly. Show you're listening.",
    ResponseIntent.REFLECT: "Reflect on what user shared, showing understanding. Then ask a follow-up.",
    ResponseIntent.PRESENT: "Present the options clearly. Be consultative, not pushy.",
    ResponseIntent.EXIT: "Provide appropriate closure. Thank the user.",
}


# Minimal fallback templates (circuit breakers only, not dialogue)
FALLBACK_TEMPLATES: Dict[UC1State, Dict[ResponseIntent, str]] = {
    UC1State.ENTRY: {
        ResponseIntent.PROMPT: "How would you like to start?",
    },
    UC1State.CAPABILITY_SELECTION: {
        ResponseIntent.RETRY: "Please select an option above.",
    },
    UC1State.CONTEXT_QUESTION: {
        ResponseIntent.PROMPT: "Could you tell me more?",
        ResponseIntent.RETRY: "I'd love to hear more about that.",
    },
    UC1State.NAME_CAPTURE: {
        ResponseIntent.PROMPT: "What should I call you?",
        ResponseIntent.RETRY: "Could you share your name?",
    },
    UC1State.EXPLORATION_LAYER: {
        ResponseIntent.PROMPT: "What's the biggest challenge you're facing?",
        ResponseIntent.REFLECT: "That makes sense. Many teams face similar challenges.",
    },
    UC1State.AI_SYNTHESIS: {
        ResponseIntent.PRESENT: "Here are a few ways we typically approach this:",
    },
    UC1State.CONSULTATIVE_ALTERNATIVES: {
        ResponseIntent.PRESENT: "Which of these resonates most with your situation?",
    },
    UC1State.RECOMMENDATION: {
        ResponseIntent.PRESENT: "How would you like to move forward?",
        ResponseIntent.RETRY: "What would you like to do next?",
    },
    UC1State.EXIT: {
        ResponseIntent.EXIT: "Thanks for chatting! Feel free to come back anytime.",
    },
}


# Mapping from UC1State enum to training data state names
# Training format: UC1_S[N]_[NAME]
STATE_TRAINING_NAMES: Dict[UC1State, str] = {
    UC1State.ENTRY: "UC1_S0_ENTRY",
    UC1State.CAPABILITY_SELECTION: "UC1_S1_CAPABILITY_PICK",
    UC1State.CONTEXT_QUESTION: "UC1_S2_CONTEXT_CLARIFIER",
    UC1State.NAME_CAPTURE: "UC1_S3_NAME_CAPTURE",
    UC1State.EXPLORATION_LAYER: "UC1_S5_EXPLORATION_LAYER",
    UC1State.AI_SYNTHESIS: "UC1_S6_ALTERNATIVES",
    UC1State.CONSULTATIVE_ALTERNATIVES: "UC1_S6_ALTERNATIVES",
    UC1State.RECOMMENDATION: "UC1_S7_EARNED_CTA",
    UC1State.EXIT: "UC1_S8_CLOSE",
}


class ConstrainedLLMAdapter:
    """
    SOLE AUTHORITY for user-visible language in UC1 flow.
    
    The fine-tuned model generates ALL responses based on:
    - State (where we are)
    - Intent (why we're speaking)
    - Context (slots, bucket, user input)
    
    Orchestrator is text-blind; this adapter is the only voice.
    """
    
    def __init__(self, config: UC1Config):
        """
        Initialize the adapter with UC1 config and OpenAI client.
        
        Args:
            config: The validated UC1Config
        """
        self.config = config
        self.forbidden_topics = set(t.lower() for t in config.forbidden_topics)
        
        # Initialize OpenAI client
        self._client = None
        self._model = UC1_FINE_TUNED_MODEL
        
    @property
    def client(self) -> OpenAI:
        """Lazy-load OpenAI client."""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("[LLMAdapter] OPENAI_API_KEY not set, using fallback templates")
                return None
            self._client = OpenAI(api_key=api_key)
        return self._client
    
    def generate_state_response(
        self,
        *,
        state: UC1State,
        response_intent: ResponseIntent,
        user_input: Optional[str] = None,
        slots: Optional[UC1Slots] = None,
        bucket: Optional[CapabilityBucket] = None,
        exploration_turn: int = 0,
    ) -> str:
        """
        SOLE AUTHORITY for user-visible language.
        
        Generates response using fine-tuned model based on state + intent.
        Falls back to minimal template ONLY if LLM fails.
        
        Args:
            state: Current conversation state
            response_intent: WHY we're speaking (not what)
            user_input: User's last message (if any)
            slots: Current conversation slots
            bucket: Selected capability bucket (if any)
            exploration_turn: Current exploration turn (for S5)
        
        Returns:
            Natural language response from fine-tuned model
        """
        # Use training-compatible state name (critical for fine-tuned model)
        state_name = STATE_TRAINING_NAMES.get(state, f"UC1_{state.value.upper()}")
        
        # Get intent-specific rules
        rules = INTENT_RULES.get(response_intent, "Respond appropriately.")
        
        # Build context from slots
        capability = bucket.trigger if bucket else None
        context = None
        user_name = None
        
        if slots:
            context = slots.context_signal
            user_name = slots.user_name
        
        # Add user name to rules if acknowledging
        if response_intent == ResponseIntent.ACKNOWLEDGE and user_name:
            rules = f"User's name is {user_name}. {rules}"
        
        # Add exploration context
        if state == UC1State.EXPLORATION_LAYER and exploration_turn > 0:
            rules = f"Exploration turn {exploration_turn}. {rules}"
        
        # Try fine-tuned model
        response = self._generate_with_state(
            state=state_name,
            user_message=user_input or "",
            capability=capability,
            context=context,
            rules=rules,
        )
        
        if response:
            logger.info(f"[LLMAdapter] Generated response for {state.value}/{response_intent.value}")
            return response
        
        # Fallback to minimal template (circuit breaker)
        fallback = FALLBACK_TEMPLATES.get(state, {}).get(
            response_intent, 
            "Please continue."
        )
        logger.warning(f"[LLMAdapter] Using fallback for {state.value}/{response_intent.value}")
        return fallback
    
    def _generate_with_state(
        self,
        state: str,
        user_message: str,
        capability: Optional[str] = None,
        context: Optional[str] = None,
        rules: str = "",
    ) -> Optional[str]:
        """
        Generate response using fine-tuned model with state context.
        
        Args:
            state: Current UC1 state (e.g., "UC1_S5_EXPLORATION_LAYER")
            user_message: The user's input
            capability: Optional capability name
            context: Optional context label
            rules: State-specific rules for the model
            
        Returns:
            Generated response or None if fallback needed
        """
        if self.client is None:
            return None
            
        # Build system message matching training format
        system_lines = [f"State: {state}"]
        if capability:
            system_lines.append(f"Capability: {capability}")
        if context:
            system_lines.append(f"Context: {context}")
        if rules:
            system_lines.append(f"Rules: {rules}")
        
        system_message = "\n".join(system_lines)
        
        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            content = response.choices[0].message.content
            
            # Validate LLM output
            if not content or not content.strip():
                logger.warning("[LLMAdapter] Empty response from fine-tuned model")
                return None
            
            if len(content.strip()) < 10:
                logger.warning(f"[LLMAdapter] Response too short ({len(content)} chars): {content[:50]}")
                return None
            
            return content.strip()
        except Exception as e:
            logger.warning(f"[LLMAdapter] Fine-tuned model failed: {e}, using fallback")
            return None
    
    def paraphrase_synthesis(
        self,
        bucket: CapabilityBucket,
        slots: UC1Slots,
    ) -> str:
        """
        Generate AI synthesis using fine-tuned model or template fallback.
        
        Args:
            bucket: The selected capability bucket
            slots: Current conversation slots
        
        Returns:
            str: The AI synthesis paragraph
        """
        user_name = slots.user_name or "there"
        context = slots.context_signal or "your goals"
        context_label = context.replace("_", " ").title() if context else "This Area"
        
        # Try fine-tuned model first
        generated = self._generate_with_state(
            state="UC1_S3_NAME_CAPTURE",
            user_message=user_name,
            capability=bucket.trigger,
            context=context_label,
            rules="Thank user by name. Synthesize understanding so far. No CTA."
        )
        
        if generated:
            logger.info(f"[LLMAdapter] Generated synthesis using fine-tuned model for {bucket.id}")
            return generated
        
        # Template fallback
        goal = bucket.goal
        synthesis = (
            f"Thanks for sharing. You're focused on {goal.lower()}, "
            f"and based on what you've shared about {context[:50]}{'...' if len(context) > 50 else ''}, "
            f"here's how we typically approach this.\n\n"
            f"We see a few possible directions:"
        )
        
        logger.info(f"[LLMAdapter] Generated synthesis using template for {bucket.id}")
        return synthesis
    
    def format_alternatives(
        self,
        bucket: CapabilityBucket,
    ) -> str:
        """
        Format the 3 alternatives for display.
        
        The alternatives are FIXED by the config - LLM only formats text.
        System has already selected which alternatives to show (all 3 from bucket).
        
        Args:
            bucket: The capability bucket containing alternatives
        
        Returns:
            str: Formatted alternatives text
        """
        # Deterministic formatting - no LLM creativity here
        alternatives = bucket.alternatives
        
        formatted_lines = []
        for i, alt in enumerate(alternatives, 1):
            formatted_lines.append(f"{i}. **{alt}**")
        
        result = "\n".join(formatted_lines)
        logger.info(f"[LLMAdapter] Formatted {len(alternatives)} alternatives for {bucket.id}")
        return result
    
    def format_exit_ctas(self) -> str:
        """
        Format the 4 exit CTAs for display.
        
        CTAs are FIXED by config - this just formats them for display.
        
        Returns:
            str: Formatted CTAs text
        """
        lines = ["\n**What would you like to do next?**\n"]
        for cta in self.config.exit_ctas:
            lines.append(f"- {cta.choice}")
        
        return "\n".join(lines)
    
    def generate_exit_summary(
        self,
        slots: UC1Slots,
        bucket: Optional[CapabilityBucket],
    ) -> str:
        """
        Generate a personalized exit summary.
        
        This summarizes the conversation and provides closure.
        
        Args:
            slots: Current conversation slots
            bucket: The capability bucket (if selected)
        
        Returns:
            str: Exit summary text
        """
        user_name = slots.user_name or "there"
        
        if slots.selected_cta_outcome == "UC2":
            summary = (
                f"Great, {user_name}! I'll connect you with our team to discuss "
                f"your requirements in detail. They'll follow up shortly."
            )
        elif slots.selected_cta_outcome == "calendar":
            summary = (
                f"Perfect, {user_name}! I'll help you schedule a quick call. "
                f"Check your email for the calendar invite."
            )
        elif slots.selected_cta_outcome == "loop":
            # This shouldn't generate exit summary - it loops back
            summary = (
                f"No problem, {user_name}! Let's explore more options. "
                f"What else would you like to learn about?"
            )
        else:  # exit
            summary = (
                f"Thanks for chatting, {user_name}! Feel free to come back "
                f"anytime. You can also explore our website for more details."
            )
        
        logger.info(f"[LLMAdapter] Generated exit summary for outcome: {slots.selected_cta_outcome}")
        return summary
    
    def generate_capability_selection_prompt(self) -> str:
        """
        Generate the capability selection prompt showing all 6 options.
        
        Returns:
            str: Formatted capability options
        """
        lines = []
        for bucket in self.config.capability_buckets:
            lines.append(f"- {bucket.trigger}")
        
        return "\n".join(lines)
    
    def generate_context_question_prompt(self, bucket: CapabilityBucket) -> str:
        """
        Get the context question for a capability bucket.
        
        Note: This is NOT LLM-generated - it comes directly from config.
        The question is FIXED per bucket.
        
        Args:
            bucket: The selected capability bucket
        
        Returns:
            str: The context question for this bucket
        """
        # Direct from config - no LLM involvement
        return bucket.context_question
    
    def generate_name_capture_prompt(self) -> str:
        """
        Get the name capture prompt.
        
        Returns:
            str: The name capture question
        """
        return self.config.name_capture_prompt
    
    # =========================================================================
    # FINE-TUNED MODEL METHODS (S5, S6, S7)
    # =========================================================================
    
    def generate_exploration_question(
        self,
        bucket: CapabilityBucket,
        slots: UC1Slots,
        question_number: int = 1,
    ) -> str:
        """
        Generate an exploration question using the fine-tuned model.
        
        Args:
            bucket: The selected capability bucket
            slots: Current conversation slots
            question_number: 1 or 2 (first or second exploration question)
        
        Returns:
            str: The exploration question
        """
        context_label = (slots.context_signal or "this area").replace("_", " ").title()
        user_input = slots.last_user_message or "That makes sense"
        
        rules = f"Ask exploration question {question_number}. Open-ended but bounded. No CTA."
        
        generated = self._generate_with_state(
            state="UC1_S5_EXPLORATION_LAYER",
            user_message=user_input,
            capability=bucket.trigger,
            context=context_label,
            rules=rules,
        )
        
        if generated:
            logger.info(f"[LLMAdapter] Generated exploration Q{question_number} using fine-tuned model")
            return generated
        
        # Template fallback
        fallback_questions = [
            "What's the biggest challenge you're facing right now in this area?",
            "What prompted you to look for help at this point?"
        ]
        question = fallback_questions[min(question_number - 1, len(fallback_questions) - 1)]
        logger.info(f"[LLMAdapter] Using template fallback for exploration Q{question_number}")
        return f"To understand this better — {question.lower()}"
    
    def generate_exploration_reflection(
        self,
        bucket: CapabilityBucket,
        slots: UC1Slots,
        user_response: str,
    ) -> str:
        """
        Generate a reflection on user's exploration answer.
        
        Args:
            bucket: The selected capability bucket
            slots: Current conversation slots
            user_response: What the user said
        
        Returns:
            str: The reflection response
        """
        context_label = (slots.context_signal or "this area").replace("_", " ").title()
        
        generated = self._generate_with_state(
            state="UC1_S5_EXPLORATION_LAYER",
            user_message=user_response,
            capability=bucket.trigger,
            context=context_label,
            rules="Reflect on user response. Show understanding. No CTA.",
        )
        
        if generated:
            logger.info(f"[LLMAdapter] Generated reflection using fine-tuned model")
            return generated
        
        # Template fallback
        logger.info(f"[LLMAdapter] Using template fallback for reflection")
        return "That makes sense. Many teams face similar challenges in this area."
    
    def generate_alternatives_framing(
        self,
        bucket: CapabilityBucket,
        slots: UC1Slots,
    ) -> str:
        """
        Generate the framing text for presenting alternatives.
        
        Note: The 3 alternatives themselves come from config.
        This generates the intro/framing text.
        
        Args:
            bucket: The selected capability bucket
            slots: Current conversation slots
        
        Returns:
            str: The alternatives framing text
        """
        context_label = (slots.context_signal or "this area").replace("_", " ").title()
        user_input = slots.last_user_message or "Continue"
        
        generated = self._generate_with_state(
            state="UC1_S6_CONSULTATIVE_ALTERNATIVES",
            user_message=user_input,
            capability=bucket.trigger,
            context=context_label,
            rules="Present exactly 3 consultative alternatives. Give recommendation. No CTA yet.",
        )
        
        if generated:
            logger.info(f"[LLMAdapter] Generated alternatives framing using fine-tuned model")
            return generated
        
        # Template fallback
        logger.info(f"[LLMAdapter] Using template fallback for alternatives framing")
        return "At this stage, teams in your situation usually consider a few paths:"
    
    def generate_cta_presentation(
        self,
        bucket: Optional[CapabilityBucket],
        slots: UC1Slots,
    ) -> str:
        """
        Generate the CTA presentation text.
        
        Note: The 4 CTAs themselves come from config.
        This generates the intro text.
        
        Args:
            bucket: The selected capability bucket (if available)
            slots: Current conversation slots
        
        Returns:
            str: The CTA presentation text
        """
        context_label = (slots.context_signal or "this area").replace("_", " ").title()
        capability = bucket.trigger if bucket else "your requirements"
        user_input = slots.last_user_message or "That makes sense"
        
        generated = self._generate_with_state(
            state="UC1_S7_EARNED_CTA",
            user_message=user_input,
            capability=capability,
            context=context_label,
            rules="Present 4 CTA options. CTA is now earned.",
        )
        
        if generated:
            logger.info(f"[LLMAdapter] Generated CTA presentation using fine-tuned model")
            return generated
        
        # Template fallback
        logger.info(f"[LLMAdapter] Using template fallback for CTA presentation")
        return "How would you like to move forward?"
