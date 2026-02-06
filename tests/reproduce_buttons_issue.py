
import sys
import os
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.getcwd())

from app.orchestrator.orchestrator import ConversationOrchestrator, UC1State
from app.orchestrator.slot_manager import SlotManager

def reproduce_buttons_issue():
    print("=== Reproducing Button Issue ===")
    session_id = "test_buttons_repro"
    
    # Clear previous session
    ConversationOrchestrator.clear_session(session_id)
    orch = ConversationOrchestrator(session_id)
    
    # 1. Force state to EXPLORATION_LAYER with a valid bucket
    print("\n1. Setting up state: EXPLORATION_LAYER, turn 2")
    orch._current_state = UC1State.EXPLORATION_LAYER
    orch.slot_manager.set_capability_bucket("UC1-A") # Product dev
    orch.slot_manager.set_user_name("User")
    orch.slot_manager.set_context_signal("Building new app")
    orch.slot_manager.set_exploration_turn(2)
    
    # Check bucket exists
    bucket = orch.slots.capability_bucket
    print(f"   Bucket set to: {bucket}")
    
    # 2. Simulate user input to complete exploration
    print("\n2. Sending input to complete exploration...")
    # This should trigger transition to CONSULTATIVE_ALTERNATIVES
    response = orch.process_input("I want to know about timelines")
    
    print(f"   New State: {response.state}")
    print(f"   Input Type: {response.input_type}")
    print(f"   Options: {response.options}")
    
    # 3. Verify options
    if response.state == UC1State.CONSULTATIVE_ALTERNATIVES:
        if response.options and len(response.options) == 3:
            print("✅ SUCCESS: 3 options returned")
            print(f"   Buttons: {response.options}")
            return True
        else:
            print("❌ FAILURE: Options missing or incorrect count")
            print(f"   Count: {len(response.options) if response.options else 0}")
            return False
    else:
        print(f"❌ FAILURE: Did not transition to CONSULTATIVE_ALTERNATIVES. State: {response.state}")
        return False

if __name__ == "__main__":
    reproduce_buttons_issue()
