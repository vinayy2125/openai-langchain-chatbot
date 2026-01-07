# UC1 Policy Validator - Startup Invariant Checking
#
# ARCHITECTURE RULE: This validator runs at startup to ensure
# the loaded configuration satisfies all freeze document invariants.
# If validation fails, the application should NOT start.
#
# This prevents silent regression when config is modified.

from typing import List
from app.orchestrator.uc1_config import UC1Config
from app.logger import get_logger

logger = get_logger("policy_validator")


class UC1PolicyViolation(Exception):
    """
    Raised when UC1 configuration violates freeze document invariants.
    
    This is a fatal error - the application should not start.
    """
    def __init__(self, violations: List[str]):
        self.violations = violations
        message = f"UC1 Policy Violations ({len(violations)}):\n" + "\n".join(f"  - {v}" for v in violations)
        super().__init__(message)


class UC1PolicyValidator:
    """
    Validates UC1 configuration against freeze document invariants.
    
    INVARIANTS (from freeze document):
    1. Exactly 6 capability buckets (UC1-A to UC1-F)
    2. Exactly 4 exit CTAs
    3. Exactly 3 alternatives per bucket
    4. Exactly 1 context question per bucket (non-empty)
    5. All bucket IDs must be in valid set
    6. No duplicate bucket IDs or CTA choices
    7. Entry message must be non-empty
    8. Forbidden topics list must be non-empty
    
    Run this at startup. If it fails, the application should not start.
    """
    
    VALID_BUCKET_IDS = {"UC1-A", "UC1-B", "UC1-C", "UC1-D", "UC1-E", "UC1-F"}
    REQUIRED_BUCKET_COUNT = 6
    REQUIRED_CTA_COUNT = 4
    REQUIRED_ALTERNATIVES_COUNT = 3
    
    def validate(self, config: UC1Config) -> None:
        """
        Validate the configuration against all invariants.
        
        Args:
            config: The UC1Config to validate.
        
        Raises:
            UC1PolicyViolation: If any invariants are violated.
        """
        errors: List[str] = []
        
        # 1. Check bucket count
        if len(config.capability_buckets) != self.REQUIRED_BUCKET_COUNT:
            errors.append(
                f"Expected exactly {self.REQUIRED_BUCKET_COUNT} capability buckets, "
                f"got {len(config.capability_buckets)}"
            )
        
        # 2. Check CTA count
        if len(config.exit_ctas) != self.REQUIRED_CTA_COUNT:
            errors.append(
                f"Expected exactly {self.REQUIRED_CTA_COUNT} exit CTAs, "
                f"got {len(config.exit_ctas)}"
            )
        
        # 3. Check each bucket's invariants
        seen_bucket_ids = set()
        for bucket in config.capability_buckets:
            # Check ID validity
            if bucket.id not in self.VALID_BUCKET_IDS:
                errors.append(f"Invalid bucket ID: {bucket.id}")
            
            # Check for duplicate IDs
            if bucket.id in seen_bucket_ids:
                errors.append(f"Duplicate bucket ID: {bucket.id}")
            seen_bucket_ids.add(bucket.id)
            
            # Check alternatives count
            if len(bucket.alternatives) != self.REQUIRED_ALTERNATIVES_COUNT:
                errors.append(
                    f"{bucket.id}: Expected exactly {self.REQUIRED_ALTERNATIVES_COUNT} alternatives, "
                    f"got {len(bucket.alternatives)}"
                )
            
            # Check context question is non-empty
            if not bucket.context_question or not bucket.context_question.strip():
                errors.append(f"{bucket.id}: Context question is empty")
            
            # Check trigger is non-empty
            if not bucket.trigger or not bucket.trigger.strip():
                errors.append(f"{bucket.id}: Trigger text is empty")
        
        # 4. Check that all required bucket IDs are present
        missing_ids = self.VALID_BUCKET_IDS - seen_bucket_ids
        if missing_ids:
            errors.append(f"Missing bucket IDs: {', '.join(sorted(missing_ids))}")
        
        # 5. Check CTA uniqueness
        seen_cta_choices = set()
        for cta in config.exit_ctas:
            if not cta.choice or not cta.choice.strip():
                errors.append("Exit CTA has empty choice text")
            elif cta.choice in seen_cta_choices:
                errors.append(f"Duplicate CTA choice: {cta.choice}")
            seen_cta_choices.add(cta.choice)
            
            if not cta.outcome or not cta.outcome.strip():
                errors.append(f"CTA '{cta.choice}' has empty outcome")
        
        # 6. Check entry message
        if not config.entry_message or not config.entry_message.strip():
            errors.append("Entry message is empty")
        
        # 7. Check forbidden topics
        if not config.forbidden_topics or len(config.forbidden_topics) == 0:
            errors.append("Forbidden topics list is empty")
        
        # 8. Check lock version format (semantic versioning)
        if not config.lock_version or not config.lock_version.strip():
            errors.append("Lock version is empty")
        
        # Raise if any errors found
        if errors:
            logger.error(f"[PolicyValidator] {len(errors)} violations found")
            for err in errors:
                logger.error(f"[PolicyValidator]   - {err}")
            raise UC1PolicyViolation(errors)
        
        logger.info(f"[PolicyValidator] ✓ Config v{config.lock_version} passed all invariant checks")


def validate_uc1_config(config: UC1Config) -> None:
    """
    Convenience function to validate UC1 config.
    
    Call this at startup after loading config.
    """
    validator = UC1PolicyValidator()
    validator.validate(config)
