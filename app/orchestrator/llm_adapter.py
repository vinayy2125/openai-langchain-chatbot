# UC1 LLM Adapter - Paraphrasing Only
#
# ARCHITECTURE RULE: The LLM may ONLY paraphrase/format text.
# It NEVER decides:
# - Which alternatives to show (system decides)
# - What questions to ask (system decides)
# - Which CTAs to present (system decides)
#
# The adapter enforces these constraints by providing templates
# that the LLM can only humanize, not change in meaning.

from typing import Optional, List
from app.orchestrator.uc1_config import UC1Config, CapabilityBucket
from app.orchestrator.slot_manager import UC1Slots
from app.logger import get_logger

logger = get_logger("llm_adapter")


class ConstrainedLLMAdapter:
    """
    Constrained LLM adapter for UC1 conversation flow.
    
    This adapter ensures the LLM is used ONLY for paraphrasing,
    never for making decisions about conversation flow.
    
    ALLOWED:
    - Rephrase AI synthesis (template + slots → natural language)
    - Format alternative labels into conversational text
    - Generate personalized exit summary from slots
    
    NEVER:
    - Choose which alternatives to show (system decides)
    - Add questions beyond context_question
    - Introduce CTAs not in config
    - Mention forbidden topics
    """
    
    def __init__(self, config: UC1Config):
        """
        Initialize the adapter with UC1 config.
        
        Args:
            config: The validated UC1Config
        """
        self.config = config
        self.forbidden_topics = set(t.lower() for t in config.forbidden_topics)
    
    def paraphrase_synthesis(
        self,
        bucket: CapabilityBucket,
        slots: UC1Slots,
    ) -> str:
        """
        Generate AI synthesis paragraph by filling template with slots.
        
        This is primarily template-based with minimal LLM involvement.
        The template structure is fixed; only the slot values vary.
        
        Args:
            bucket: The selected capability bucket
            slots: Current conversation slots
        
        Returns:
            str: The AI synthesis paragraph
        """
        # For now, use a deterministic template
        # In production, could use LLM for more natural phrasing
        # but the structure and content must remain fixed
        
        user_name = slots.user_name or "there"
        context = slots.context_signal or "your goals"
        goal = bucket.goal
        
        synthesis = (
            f"Thanks, {user_name}. You're focused on helping to {goal.lower()}, "
            f"and based on what you've shared about {context[:50]}{'...' if len(context) > 50 else ''}, "
            f"here's how we typically approach this.\n\n"
            f"We see a few possible directions:"
        )
        
        logger.info(f"[LLMAdapter] Generated synthesis for {bucket.id}, user: {user_name}")
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
