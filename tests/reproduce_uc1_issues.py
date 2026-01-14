import sys
import os
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

from app.orchestrator.orchestrator import ConversationOrchestrator, AdapterCallSpec, OrchestratorResponse
from app.orchestrator.state_machine import UC1State, ResponseIntent
from app.orchestrator.slot_manager import UC1Slots
from app.orchestrator.uc1_config import load_uc1_config

def test_exit_message_determinism():
    print("\n=== Testing ZERO-LLM Exit Architecture ===")
    
    # Setup mocks
    mock_config = load_uc1_config()
    orch = ConversationOrchestrator("test_session_exit")
    orch.config = mock_config
    
    # Test Cases
    test_cases = [
        ("Discuss my requirement now", "UC2", "connect you with our team"),
        ("Schedule a quick call", "calendar", "calendar invite"),
        ("I'll explore on my own", "exit", "Thanks for chatting"),
    ]
    
    all_passed = True
    
    for cta_text, expected_outcome, expected_snippet in test_cases:
        print(f"\nTesting Outcome: {expected_outcome} (Trigger: '{cta_text}')")
        
        # Manually set state to RECOMMENDATION and set bucket
        orch._current_state = UC1State.RECOMMENDATION
        orch.slots.capability_bucket = "UC1-F"
        
        # Process input to trigger exit
        # Note: In real flow, user clicks CTA. Orchestrator calls _handle_recommendation -> _handle_exit
        response = orch._handle_recommendation(cta_text)
        
        # Verify call spec intent
        if response.call_spec.response_intent != ResponseIntent.GRACEFUL_EXIT:
            print(f"❌ Intent Mismatch: Expected GRACEFUL_EXIT, got {response.call_spec.response_intent}")
            all_passed = False
        else:
            print(f"✅ Intent Correct: GRACEFUL_EXIT")
            
        # Verify message content (Deterministic)
        msg = response.message
        if expected_snippet.lower() in msg.lower():
            print(f"✅ Message Content Verified: Found '{expected_snippet}'")
        else:
            print(f"❌ Message Content Mismatch: '{expected_snippet}' NOT found in:\n'{msg}'")
            all_passed = False
            
    return all_passed

def test_duplication_constraints():
    print("\n=== Testing Duplication Constraints (S7) ===")
    
    # Setup
    orch = ConversationOrchestrator("test_session_dupe")
    slots = UC1Slots()
    slots.context_signal = "need_ai_help"
    slots.selected_alternative = "Explore options together" # The S6 selection
    
    # Mock the LLM client to avoid real calls, but we want to verify the PROMPT construction mostly
    # However, since we can't easily intercept the prompt inside the class without heavy patching,
    # we will rely on validating the `rules` passed to `_generate_with_state` if we can mock it,
    # OR we just inspect the `generate_cta_presentation` method logic.
    
    with patch.object(orch.llm_adapter, '_generate_with_state', return_value="Mocked Response") as mock_gen:
        print("\nCalling generate_cta_presentation...")
        orch.llm_adapter.generate_cta_presentation(bucket=None, slots=slots)
        
        # Check arguments passed to _generate_with_state
        call_args = mock_gen.call_args[1] # kwargs
        
        # 1. Verify Context (Should rely on selected_alternative)
        context_arg = call_args.get('context', '')
        print(f"Context passed: '{context_arg}'")
        if "Selected Alternative: Explore options together" in context_arg:
             print("✅ Context Correct: Passed selected token properly")
        else:
             print("❌ Context Error: Did not pass selected alternative token properly")

        # 2. Verify Rules (Strict constraints)
        rules_arg = call_args.get('rules', '')
        print(f"Rules passed: '{rules_arg}'")
        
        required_constraints = [
            "reference only the selected option",
            "one-sentence justification max",
            "transition immediately to cta",
            "max 2 sentences total",
            "do not list other alternatives"
        ]
        
        constraints_met = True
        for const in required_constraints:
            if const.lower() not in rules_arg.lower():
                print(f"❌ Missing constraint: '{const}'")
                constraints_met = False
        
        if constraints_met:
            print("✅ All Strict Constraints Present in Prompt")

if __name__ == "__main__":
    exit_passed = test_exit_message_determinism()
    dupe_passed = test_duplication_constraints() # This runs the mock check
    
    if exit_passed:
        print("\n[SUCCESS] Exit Logic Verified")
    else:
        print("\n[FAILURE] Exit Logic Failed")
