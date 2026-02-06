# Tests for UC1 Slot Manager
#
# These tests verify slot management and engagement scoring.

import pytest
from app.orchestrator.slot_manager import (
    UC1Slots,
    SlotManager,
    EngagementEvent,
    ENGAGEMENT_DELTAS,
)


class TestUC1Slots:
    """Tests for UC1Slots dataclass."""
    
    def test_default_slots_are_empty(self):
        """Test that new slots have None values."""
        slots = UC1Slots()
        assert slots.capability_bucket is None
        assert slots.user_name is None
        assert slots.context_signal is None
        assert slots.engagement_score == 0.0
    
    def test_get_filled_slots_empty(self):
        """Test get_filled_slots returns empty set for new slots."""
        slots = UC1Slots()
        filled = slots.get_filled_slots()
        assert len(filled) == 0
    
    def test_get_filled_slots_with_values(self):
        """Test get_filled_slots returns filled slot names."""
        slots = UC1Slots(
            capability_bucket="UC1-A",
            user_name="John",
            context_signal=None  # Not filled
        )
        filled = slots.get_filled_slots()
        assert "capability_bucket" in filled
        assert "user_name" in filled
        assert "context_signal" not in filled
    
    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        original = UC1Slots(
            capability_bucket="UC1-B",
            user_name="Alice",
            context_signal="Building a new platform",
            engagement_score=0.5,
        )
        
        data = original.to_dict()
        restored = UC1Slots.from_dict(data)
        
        assert restored.capability_bucket == original.capability_bucket
        assert restored.user_name == original.user_name
        assert restored.context_signal == original.context_signal
        assert restored.engagement_score == original.engagement_score


class TestSlotManager:
    """Tests for SlotManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.test_session_id = "test-session-123"
        # Clear any existing test session
        SlotManager.clear_session(self.test_session_id)
    
    def teardown_method(self):
        """Cleanup test data."""
        SlotManager.clear_session(self.test_session_id)
    
    def test_create_new_session(self):
        """Test that new session gets empty slots."""
        manager = SlotManager(self.test_session_id)
        assert manager.slots.capability_bucket is None
        assert manager.slots.engagement_score == 0.0
    
    def test_set_capability_bucket(self):
        """Test setting capability bucket."""
        manager = SlotManager(self.test_session_id)
        manager.set_capability_bucket("UC1-C")
        assert manager.slots.capability_bucket == "UC1-C"
    
    def test_set_user_name(self):
        """Test setting user name with trimming."""
        manager = SlotManager(self.test_session_id)
        manager.set_user_name("  Alice  ")
        assert manager.slots.user_name == "Alice"
    
    def test_engagement_button_click(self):
        """Test engagement score increases on button click."""
        manager = SlotManager(self.test_session_id)
        initial = manager.slots.engagement_score
        
        manager.increment_engagement(EngagementEvent.BUTTON_CLICK)
        
        expected = initial + ENGAGEMENT_DELTAS[EngagementEvent.BUTTON_CLICK]
        assert manager.slots.engagement_score == expected
    
    def test_engagement_text_provided(self):
        """Test engagement score increases on text input."""
        manager = SlotManager(self.test_session_id)
        
        manager.increment_engagement(EngagementEvent.TEXT_PROVIDED)
        
        assert manager.slots.engagement_score == ENGAGEMENT_DELTAS[EngagementEvent.TEXT_PROVIDED]
    
    def test_engagement_retry_decreases(self):
        """Test engagement score decreases on retry."""
        manager = SlotManager(self.test_session_id)
        manager.slots.engagement_score = 0.5  # Start with some score
        
        manager.increment_engagement(EngagementEvent.RETRY)
        
        expected = 0.5 + ENGAGEMENT_DELTAS[EngagementEvent.RETRY]  # Negative delta
        assert manager.slots.engagement_score == expected
        assert manager.slots.retry_count == 1
    
    def test_engagement_never_goes_negative(self):
        """Test engagement score can't go below 0."""
        manager = SlotManager(self.test_session_id)
        
        # Multiple retries should not go negative
        for _ in range(10):
            manager.increment_engagement(EngagementEvent.RETRY)
        
        assert manager.slots.engagement_score >= 0.0
    
    def test_session_persistence(self):
        """Test that slots persist across SlotManager instances."""
        # Set up first instance
        manager1 = SlotManager(self.test_session_id)
        manager1.set_capability_bucket("UC1-D")
        manager1.set_user_name("Bob")
        
        # Create second instance for same session
        manager2 = SlotManager(self.test_session_id)
        
        # Should see same values
        assert manager2.slots.capability_bucket == "UC1-D"
        assert manager2.slots.user_name == "Bob"
    
    def test_validate_required_slots(self):
        """Test slot validation returns missing slots."""
        manager = SlotManager(self.test_session_id)
        manager.set_capability_bucket("UC1-A")
        # user_name and context_signal not set
        
        missing = manager.validate_required_slots(("capability_bucket", "user_name"))
        
        assert "user_name" in missing
        assert "capability_bucket" not in missing
