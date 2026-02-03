# Button Manager - Single Source of Truth for UC1 Button Logic
#
# ARCHITECTURE:
#   All button decisions flow through ButtonManager.
#   Orchestrator and LLM adapter do NOT decide buttons.
#   This module is the ONLY place where button lists are generated.
#
# DESIGN PRINCIPLES:
#   1. State-driven: Buttons are determined by UC1 state
#   2. Slot-aware: Buttons change based on filled slots
#   3. Deterministic: Same state + slots = same buttons
#   4. Button click = Deterministic action (no KB fallback)

from typing import List, Optional, Tuple
from app.orchestrator.state_machine import UC1State
from app.orchestrator.slot_manager import UC1Slots
from app.orchestrator.uc1_config import UC1Config, CapabilityBucket, get_bucket_by_id
from app.logger import get_logger

logger = get_logger("button_manager")


class ButtonManager:
    """
    Single source of truth for UC1 button logic.
    
    BUTTON STRATEGY:
    - CTA states (RECOMMENDATION, AI_SYNTHESIS, EXIT) → STATIC from config
    - Exploration states → DYNAMIC from LLM (next-step guides)
    - Entry/fixed states → STATIC or none
    
    STATE → BUTTONS mapping:
    | State                     | Buttons                    | Type    |
    |---------------------------|----------------------------|---------|  
    | ENTRY                     | None                       | -       |
    | CAPABILITY_SELECTION      | 6 bucket triggers          | STATIC  |
    | CONTEXT_QUESTION          | None (text input)          | -       |
    | NAME_CAPTURE              | None (text input)          | -       |
    | EXPLORATION_LAYER         | Dynamic OR alternatives    | DYNAMIC |
    | CONSULTATIVE_ALTERNATIVES | Dynamic OR alternatives    | DYNAMIC |
    | RECOMMENDATION            | CTA buttons only           | STATIC  |
    | FREE_EXPLORATION          | Dynamic or topic-specific  | DYNAMIC |
    """
    
    def __init__(self, config: UC1Config):
        self.config = config
    
    def get_buttons_for_state(
        self,
        state: UC1State,
        slots: UC1Slots,
        bucket: CapabilityBucket = None,
        dynamic_options: List[str] = None
    ) -> List[str]:
        """
        Deterministic button generation based on state + slots.
        
        Args:
            state: Current UC1 state
            slots: Current slot values
            bucket: Current capability bucket (if any)
            dynamic_options: LLM-generated options (for FREE_EXPLORATION)
            
        Returns:
            List[str]: Buttons to display, may be empty
        """
        # ==========================================================
        # OPTION COMMITMENT CHECK - Prevents infinite button loops
        # ==========================================================
        if slots.selected_alternative:
            logger.info(f"[ButtonManager] Alternative selected: '{slots.selected_alternative}' → showing CTAs")
            return self._get_cta_buttons()
        
        if slots.alternatives_consumed:
            logger.info("[ButtonManager] Alternatives consumed → showing CTAs")
            return self._get_cta_buttons()
        
        # ==========================================================
        # STATE-BASED BUTTON SELECTION
        # ==========================================================
        
        if state == UC1State.ENTRY:
            # Entry state: No buttons, just welcome message
            return []
        
        if state == UC1State.CAPABILITY_SELECTION:
            # Show all bucket trigger buttons
            return [b.trigger for b in self.config.capability_buckets]
        
        if state == UC1State.CONTEXT_QUESTION:
            # No buttons - user needs to type their context
            return []
        
        if state == UC1State.NAME_CAPTURE:
            # No buttons - user needs to type their name
            return []
        
        if state == UC1State.EXPLORATION_LAYER:
            # DYNAMIC FIRST: Use LLM-generated next-step options if available
            # Note: dynamic_options=[] means "intentionally no options", None means "not generated"
            if dynamic_options is not None:
                if dynamic_options:  # Has actual options
                    valid_opts = [opt[:30] for opt in dynamic_options if opt.strip()][:3]
                    if valid_opts:
                        logger.info(f"[ButtonManager] EXPLORATION_LAYER: Using dynamic options: {valid_opts}")
                        return valid_opts
                # Empty list = intentionally no options
                logger.info(f"[ButtonManager] EXPLORATION_LAYER: Dynamic options empty (intentional), showing no buttons")
                return []
            # Fallback: Only when dynamic_options is None (not generated at all)
            logger.warning(f"[ButtonManager] EXPLORATION_LAYER: No dynamic options generated, falling back to static")
            if bucket and bucket.alternatives:
                return list(bucket.alternatives)
            if slots.capability_bucket:
                slot_bucket = get_bucket_by_id(self.config, slots.capability_bucket)
                if slot_bucket and slot_bucket.alternatives:
                    return list(slot_bucket.alternatives)
            return []
        
        if state == UC1State.CONSULTATIVE_ALTERNATIVES:
            # DYNAMIC FIRST: Use LLM-generated next-step options if available
            # Note: dynamic_options=[] means "intentionally no options", None means "not generated"
            if dynamic_options is not None:
                if dynamic_options:  # Has actual options
                    valid_opts = [opt[:30] for opt in dynamic_options if opt.strip()][:3]
                    if valid_opts:
                        logger.info(f"[ButtonManager] CONSULTATIVE_ALTERNATIVES: Using dynamic options: {valid_opts}")
                        return valid_opts
                # Empty list = intentionally no options
                logger.info(f"[ButtonManager] CONSULTATIVE_ALTERNATIVES: Dynamic options empty (intentional), showing no buttons")
                return []
            # Fallback: Only when dynamic_options is None (not generated at all)
            logger.warning(f"[ButtonManager] CONSULTATIVE_ALTERNATIVES: No dynamic options generated, falling back to static")
            if bucket and bucket.alternatives:
                return list(bucket.alternatives)
            return []
        
        if state == UC1State.RECOMMENDATION:
            # Show CTA buttons only
            return self._get_cta_buttons()
        
        if state == UC1State.AI_SYNTHESIS:
            # Show CTA buttons (synthesis is pre-CTA)
            return self._get_cta_buttons()
        
        if state == UC1State.FREE_EXPLORATION:
            # ============================================================
            # EXIT-READY CHECK: Show CTAs directly when conversation mature
            # This collapses the state machine: Exploration → CTA
            # ============================================================
            if slots and (slots.exploration_turn >= 2 or slots.alternatives_consumed or 
                          slots.selected_alternative or slots.exploration_complete):
                logger.info("[ButtonManager] FREE_EXPLORATION exit-ready: showing CTAs")
                return self._get_cta_buttons()
            
            # Use dynamic options if provided (LLM-generated next-step guides)
            if dynamic_options:
                valid_opts = [opt[:30] for opt in dynamic_options if opt.strip()][:3]
                if valid_opts:
                    logger.info(f"[ButtonManager] Using dynamic options: {valid_opts}")
                    return valid_opts
            return self._get_exploration_buttons(bucket, slots)
        
        if state == UC1State.EXIT:
            # Show post-CTA options
            return ["Restart Conversation", "Close Chat"]
        
        # Fallback: No buttons
        logger.warning(f"[ButtonManager] Unknown state: {state}")
        return []
    
    def is_button_click(
        self,
        user_input: str,
        bucket: CapabilityBucket = None
    ) -> Optional[Tuple[str, str]]:
        """
        Check if input matches a known button.
        
        Args:
            user_input: User's input text
            bucket: Current capability bucket (if any)
            
        Returns:
            Tuple[button_type, button_value] or None
            button_type: "bucket_trigger", "alternative", "cta"
        """
        if not user_input:
            return None
        
        input_lower = user_input.strip().lower()
        
        # Check bucket triggers
        for b in self.config.capability_buckets:
            if b.trigger.lower() == input_lower:
                logger.info(f"[ButtonManager] Bucket trigger click: {b.trigger}")
                return ("bucket_trigger", b.id)
        
        # Check alternatives (if bucket provided)
        if bucket and bucket.alternatives:
            for alt in bucket.alternatives:
                if alt.lower() == input_lower:
                    logger.info(f"[ButtonManager] Alternative click: {alt}")
                    return ("alternative", alt)
        
        # Check CTA buttons
        for cta in self.config.exit_ctas:
            if cta.choice.lower() == input_lower:
                logger.info(f"[ButtonManager] CTA click: {cta.choice}")
                return ("cta", cta.action)
        
        # Check Post-CTA flow buttons
        if input_lower == "restart conversation":
            return ("meta", "restart")
        if input_lower == "close chat":
            return ("meta", "close_chat")
        
        return None
    
    def _get_cta_buttons(self) -> List[str]:
        """Get CTA buttons from config."""
        return [cta.choice for cta in self.config.exit_ctas]
    
    def _get_exploration_buttons(
        self,
        bucket: CapabilityBucket = None,
        slots: UC1Slots = None
    ) -> List[str]:
        """
        Get topic-specific buttons for FREE_EXPLORATION mode.
        
        Uses capability bucket to determine relevant actions.
        Falls back to default buttons if bucket not set.
        """
        exploration_buttons = self.config.exploration_buttons
        
        # Try bucket-specific buttons first
        if bucket and bucket.id in exploration_buttons:
            buttons = exploration_buttons[bucket.id]
            logger.info(f"[ButtonManager] Using bucket-specific buttons for {bucket.id}")
            return buttons[:3]
        
        # Try slot-based bucket ID
        if slots and slots.capability_bucket and slots.capability_bucket in exploration_buttons:
            buttons = exploration_buttons[slots.capability_bucket]
            logger.info(f"[ButtonManager] Using slot-based buttons for {slots.capability_bucket}")
            return buttons[:3]
        
        # Fallback to default
        default_buttons = exploration_buttons.get("default", ["Get a demo", "See case studies", "Talk to an expert"])
        logger.info("[ButtonManager] Using default exploration buttons")
        return default_buttons[:3]
