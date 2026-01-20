# UC1 Configuration - Pure Data, No Logic
#
# ARCHITECTURE RULE: This file contains ONLY dataclasses and a loader function.
# No methods, no computed behavior, no branching logic.
# All logic lives in state_machine.py and orchestrator.py.
#
# Config is loaded from YAML at startup and validated by PolicyValidator.

from dataclasses import dataclass
from typing import Tuple, Literal, Optional, Dict, List
import yaml
import os
from app.logger import get_logger

logger = get_logger("uc1_config")

# Type definitions for strict validation
CapabilityBucketId = Literal["UC1-A", "UC1-B", "UC1-C", "UC1-D", "UC1-E", "UC1-F"]


@dataclass(frozen=True)
class CapabilityBucket:
    """
    Represents one of the 6 capability buckets (sub-use cases).
    
    INVARIANTS (enforced by PolicyValidator):
    - Exactly 6 buckets must exist
    - Each bucket has exactly 1 context_question
    - Each bucket has exactly 3 alternatives
    """
    id: CapabilityBucketId
    trigger: str  # User-visible selection text
    goal: str  # Internal goal description
    context_question: str  # Exactly 1 question for this bucket
    alternatives: Tuple[str, str, str]  # Exactly 3 consultative alternatives


@dataclass(frozen=True)
class ExitCTA:
    """
    Represents one of the 4 exit CTAs.
    
    INVARIANTS (enforced by PolicyValidator):
    - Exactly 4 CTAs must exist
    - Each CTA has a choice and outcome
    """
    choice: str  # User-visible text
    outcome: str  # Internal action identifier


@dataclass(frozen=True)
class EmailCaptureConfig:
    """Configuration for email capture flow."""
    min_turns_before_ask: int
    max_turns_before_ask: int
    prompt: str
    soft_prompt: str
    skip_phrases: Tuple[str, ...]


@dataclass(frozen=True)
class UC1Config:
    """
    Complete UC1 configuration - frozen/immutable after loading.
    
    This is the single source of truth for UC1 behavior.
    LLM prompts, state machine, and orchestrator all reference this.
    
    ARCHITECTURE RULE: No methods allowed. Pure data only.
    """
    intent_id: str  # "explore_services_capabilities"
    lock_version: str  # Semantic version for tracking changes
    entry_message: str  # Fixed welcome message
    capability_buckets: Tuple[CapabilityBucket, ...]  # Exactly 6
    exit_ctas: Tuple[ExitCTA, ...]  # Exactly 4
    forbidden_topics: Tuple[str, ...]  # Words/phrases LLM must never mention
    name_capture_prompt: str  # System prompt for name capture state
    synthesis_template: str  # Template for AI synthesis (slots filled by orchestrator)
    # NEW (2026-01-15): Dynamic exploration buttons
    exploration_buttons: Dict[str, List[str]]  # bucket_id -> button list ("default" for fallback)
    # NEW (2026-01-15): Email capture configuration
    email_capture: Optional[EmailCaptureConfig] = None


def load_uc1_config(path: Optional[str] = None) -> UC1Config:
    """
    Load UC1 configuration from YAML file.
    
    Args:
        path: Path to YAML config file. Defaults to uc1_config.yaml in this directory.
    
    Returns:
        UC1Config: Frozen, immutable configuration object.
    
    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config structure is invalid.
    """
    if path is None:
        # Default to uc1_config.yaml in the same directory as this file
        path = os.path.join(os.path.dirname(__file__), "uc1_config.yaml")
    
    logger.info(f"[UC1Config] Loading configuration from: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    
    # Parse capability buckets
    buckets = []
    for b in raw.get("capability_buckets", []):
        # Ensure alternatives is exactly 3 items
        alts = tuple(b.get("alternatives", []))
        if len(alts) != 3:
            raise ValueError(f"Bucket {b.get('id')} must have exactly 3 alternatives, got {len(alts)}")
        
        bucket = CapabilityBucket(
            id=b["id"],
            trigger=b["trigger"],
            goal=b["goal"],
            context_question=b["context_question"],
            alternatives=alts,
        )
        buckets.append(bucket)
    
    # Parse exit CTAs
    ctas = []
    for c in raw.get("exit_ctas", []):
        cta = ExitCTA(
            choice=c["choice"],
            outcome=c["outcome"],
        )
        ctas.append(cta)
    
    # Parse exploration buttons (NEW)
    exploration_buttons = raw.get("exploration_buttons", {})
    if not exploration_buttons:
        exploration_buttons = {"default": ["Get a demo", "See case studies", "Talk to an expert"]}
    
    # Parse email capture config (NEW)
    email_raw = raw.get("email_capture", {})
    email_capture = None
    if email_raw:
        email_capture = EmailCaptureConfig(
            min_turns_before_ask=email_raw.get("min_turns_before_ask", 4),
            max_turns_before_ask=email_raw.get("max_turns_before_ask", 6),
            prompt=email_raw.get("prompt", "What's the best email to reach you at?"),
            soft_prompt=email_raw.get("soft_prompt", "If you'd like, what's your email?"),
            skip_phrases=tuple(email_raw.get("skip_phrases", ["skip", "no", "later"])),
        )
    
    # Build config object
    config = UC1Config(
        intent_id=raw.get("intent_id", "explore_services_capabilities"),
        lock_version=raw.get("lock_version", "1.0.0"),
        entry_message=raw.get("entry_message", ""),
        capability_buckets=tuple(buckets),
        exit_ctas=tuple(ctas),
        forbidden_topics=tuple(raw.get("forbidden_topics", [])),
        name_capture_prompt=raw.get("name_capture_prompt", "What should I call you?"),
        synthesis_template=raw.get("synthesis_template", ""),
        exploration_buttons=exploration_buttons,
        email_capture=email_capture,
    )
    
    logger.info(f"[UC1Config] Loaded config v{config.lock_version}: {len(config.capability_buckets)} buckets, {len(config.exit_ctas)} CTAs")
    
    return config


def get_bucket_by_id(config: UC1Config, bucket_id: str) -> Optional[CapabilityBucket]:
    """
    Helper to find a capability bucket by ID.
    
    Note: This is a pure function, not a method on UC1Config (to keep config pure data).
    """
    for bucket in config.capability_buckets:
        if bucket.id == bucket_id:
            return bucket
    return None


def get_bucket_by_trigger(config: UC1Config, trigger: str) -> Optional[CapabilityBucket]:
    """
    Helper to find a capability bucket by trigger text (case-insensitive match).
    
    Note: This is a pure function, not a method on UC1Config (to keep config pure data).
    """
    trigger_lower = trigger.strip().lower()
    for bucket in config.capability_buckets:
        if bucket.trigger.strip().lower() == trigger_lower:
            return bucket
    return None
