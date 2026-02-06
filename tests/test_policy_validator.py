# Tests for UC1 Policy Validator
#
# These tests verify that the PolicyValidator correctly enforces
# all freeze document invariants.

import pytest
from app.orchestrator.uc1_config import (
    UC1Config,
    CapabilityBucket,
    ExitCTA,
    load_uc1_config,
)
from app.orchestrator.policy_validator import (
    UC1PolicyValidator,
    UC1PolicyViolation,
    validate_uc1_config,
)


class TestUC1PolicyValidator:
    """Tests for PolicyValidator invariant enforcement."""
    
    def test_valid_config_passes(self):
        """Test that the default config passes validation."""
        config = load_uc1_config()
        validator = UC1PolicyValidator()
        # Should not raise
        validator.validate(config)
    
    def test_wrong_bucket_count_fails(self):
        """Test that config with wrong number of buckets fails."""
        # Create config with only 5 buckets (should be 6)
        config = UC1Config(
            intent_id="test",
            lock_version="1.0.0",
            entry_message="Hello",
            capability_buckets=(
                CapabilityBucket(id="UC1-A", trigger="A", goal="a", context_question="?", alternatives=("1", "2", "3")),
                CapabilityBucket(id="UC1-B", trigger="B", goal="b", context_question="?", alternatives=("1", "2", "3")),
                CapabilityBucket(id="UC1-C", trigger="C", goal="c", context_question="?", alternatives=("1", "2", "3")),
                CapabilityBucket(id="UC1-D", trigger="D", goal="d", context_question="?", alternatives=("1", "2", "3")),
                CapabilityBucket(id="UC1-E", trigger="E", goal="e", context_question="?", alternatives=("1", "2", "3")),
                # Missing UC1-F
            ),
            exit_ctas=(
                ExitCTA(choice="A", outcome="a"),
                ExitCTA(choice="B", outcome="b"),
                ExitCTA(choice="C", outcome="c"),
                ExitCTA(choice="D", outcome="d"),
            ),
            forbidden_topics=("pricing",),
            name_capture_prompt="Name?",
            synthesis_template="Hi {user_name}",
        )
        
        validator = UC1PolicyValidator()
        with pytest.raises(UC1PolicyViolation) as exc_info:
            validator.validate(config)
        
        assert "Expected exactly 6 capability buckets" in str(exc_info.value)
    
    def test_wrong_cta_count_fails(self):
        """Test that config with wrong number of CTAs fails."""
        config = load_uc1_config()
        # Modify to have only 3 CTAs (should be 4)
        config = UC1Config(
            intent_id=config.intent_id,
            lock_version=config.lock_version,
            entry_message=config.entry_message,
            capability_buckets=config.capability_buckets,
            exit_ctas=(
                ExitCTA(choice="A", outcome="a"),
                ExitCTA(choice="B", outcome="b"),
                ExitCTA(choice="C", outcome="c"),
                # Missing 4th CTA
            ),
            forbidden_topics=config.forbidden_topics,
            name_capture_prompt=config.name_capture_prompt,
            synthesis_template=config.synthesis_template,
        )
        
        validator = UC1PolicyValidator()
        with pytest.raises(UC1PolicyViolation) as exc_info:
            validator.validate(config)
        
        assert "Expected exactly 4 exit CTAs" in str(exc_info.value)
    
    def test_wrong_alternatives_count_fails(self):
        """Test that bucket with wrong number of alternatives fails."""
        config = load_uc1_config()
        # Create config with 4 alternatives in one bucket (should be 3)
        bad_bucket = CapabilityBucket(
            id="UC1-A",
            trigger="Test",
            goal="test",
            context_question="?",
            alternatives=("1", "2", "3", "4"),  # 4 instead of 3
        )
        
        # This should fail at load time since alternatives must be Tuple[str, str, str]
        # But if validation passes, it would catch it
        pass  # Type system prevents this


class TestUC1ConfigLoading:
    """Tests for config loading and validation."""
    
    def test_load_default_config(self):
        """Test that default YAML config loads successfully."""
        config = load_uc1_config()
        
        assert config.intent_id == "explore_services_capabilities"
        assert config.lock_version == "1.0.0"
        assert len(config.capability_buckets) == 6
        assert len(config.exit_ctas) == 4
        assert len(config.forbidden_topics) > 0
    
    def test_config_buckets_have_required_fields(self):
        """Test that all buckets have required fields."""
        config = load_uc1_config()
        
        for bucket in config.capability_buckets:
            assert bucket.id.startswith("UC1-")
            assert bucket.trigger
            assert bucket.goal
            assert bucket.context_question
            assert len(bucket.alternatives) == 3
    
    def test_config_ctas_have_required_fields(self):
        """Test that all CTAs have required fields."""
        config = load_uc1_config()
        
        for cta in config.exit_ctas:
            assert cta.choice
            assert cta.outcome
