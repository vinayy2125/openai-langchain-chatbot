
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
import json

# UC1 E2E Test Suite (Sync version using TestClient)

@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as client:
        yield client

class TestUC1Flow:
    def test_full_happy_path(self, client):
        """
        Verify the complete Happy Path:
        1. Entry -> Greeting
        2. S2 Capability Selection (Click Button)
        3. S3 Context Question (Provide Answer)
        4. S5 Exploration (Ask 2 questions)
        5. S6 Consultative Alternatives (Verify Buttons appear!)
        6. Selection -> S7 Recommendation (Verify CTAs)
        7. Exit -> Continue exploring (Loop back)
        """
        session_id = "e2e_test_session_001b" # Unique ID
        
        # Helper to send message and get full response text + events
        def send(msg, session=session_id):
            response = ""
            events = []
            payload = {
                "query": msg,
                "session_id": session
            }
            # TestClient.stream returns a context manager
            with client.stream("POST", "/api/chat/send-stream", json=payload) as response_stream:
                for line in response_stream.iter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            events.append(data)
                            if data.get("status") == "chunk":
                                        response += str(data.get("chunk", ""))
                        except json.JSONDecodeError:
                            pass
            return response, events

        # 1. Entry (Implicit or Explicit "Hi")
        resp, events = send("Hi")
        
        # 2. Select Capability (e.g., "Find a service") 
        # Using a likely trigger from config or reproduction attempt
        resp, events = send("I want to explore options") 
        # Or "Find a service" depending on config.
        # Assuming "I want to explore options" triggers the default flow or general inquiry.
        
        # 3. Provide Context Answer
        resp, events = send("I need help with cloud migration")
        
        # 3b. Name Capture (S4) - Standard flow usually asks for name here
        # Assuming the system prompts for name. Even if it doesn't (if skipped), providing it is safe?
        # Better: check if response asks for name, or just send it 'My name is John'.
        # If we are in S5 already, 'My name is John' might be treated as exploration usage.
        # But if 'user_name' is blank, buttons are hidden.
        # If checking response is hard, I'll assume S4 exists.
        
        # Providing name to satisfy slot requirement
        resp, events = send("My name is QA Bot")
        
        # 4. Exploration Loop (Turn 1)
        resp, events = send("What are the security risks?")
        
        # Turn 2
        resp, events = send("How much does it cost?")
        
        # 5. Transition to S6 (Trigger it)
        # Using clear completion signal
        resp, events = send("I have no more questions.") 
        
        # Verify S6 (Consultative Alternatives)
        options = None
        for e in events:
            if e.get("status") == "meta":
                chunk = e.get("chunk", {})
                if isinstance(chunk, dict) and "uc1_options" in chunk:
                    options = chunk["uc1_options"]
        
        # KEY VERIFICATION:
        assert options is not None, "Alternatives buttons missing in S6!"
        assert isinstance(options, list), "Options should be a list"
        assert len(options) > 0, "No alternatives returned"
        
        # 6. Select Alternative
        selected_alt = options[0]
        resp, events = send(selected_alt)
        
        # 7. Verify Recommendation (S7)
        cta_options = None
        for e in events:
            if e.get("status") == "meta":
                chunk = e.get("chunk", {})
                if isinstance(chunk, dict) and "uc1_options" in chunk:
                    cta_options = chunk["uc1_options"]
        
        assert cta_options is not None, "CTA buttons missing in S7!"
        
        # 8. Loop Back
        loop_cta = "Continue exploring" 
        resp, events = send(loop_cta)
        
        # Should go back to S2 (Capability Selection)
        cap_options = None
        for e in events:
            if e.get("status") == "meta":
                chunk = e.get("chunk", {})
                if isinstance(chunk, dict) and "uc1_options" in chunk:
                    cap_options = chunk["uc1_options"]
                
        assert cap_options is not None
        assert len(cap_options) >= 2


class TestUC1RegressionFixes:
    """
    Regression tests for fixes implemented on 2026-02-04:
    1. "Talk to expert" CTA interception in Exploration Layer
    2. Smart email skip when email is already known
    3. EXIT state buttons consistency across all UC-1 sub-cases
    """
    
    def _send_message(self, client, msg, session_id):
        """Helper to send message and get full response text + events."""
        response = ""
        events = []
        payload = {
            "query": msg,
            "session_id": session_id
        }
        with client.stream("POST", "/api/chat/send-stream", json=payload) as response_stream:
            for line in response_stream.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        events.append(data)
                        if data.get("status") == "chunk":
                            response += str(data.get("chunk", ""))
                    except json.JSONDecodeError:
                        pass
        return response, events
    
    def _get_options_from_events(self, events):
        """Extract uc1_options from meta events."""
        for e in events:
            if e.get("status") == "meta":
                chunk = e.get("chunk", {})
                if isinstance(chunk, dict) and "uc1_options" in chunk:
                    return chunk["uc1_options"]
        return None
    
    def _get_state_from_events(self, events):
        """Extract uc1_state from meta events."""
        for e in events:
            if e.get("status") == "meta":
                chunk = e.get("chunk", {})
                if isinstance(chunk, dict) and "uc1_state" in chunk:
                    return chunk["uc1_state"]
        return None

    def test_talk_to_expert_triggers_exit_flow(self, client):
        """
        REGRESSION: "Talk to expert" should trigger EMAIL_CAPTURE -> EXIT flow.
        Previously, it was treated as chat text by the Agent/LLM.
        
        FIX: Added CTA interception in _handle_exploration_layer()
        """
        session_id = "regression_talk_to_expert_001"
        
        # 1. Start UC1 flow
        self._send_message(client, "Hi", session_id)
        
        # 2. Select AI/ML capability
        self._send_message(client, "AI/ML & Automation", session_id)
        
        # 3. Provide context
        self._send_message(client, "I want to build an AI chatbot", session_id)
        
        # 4. Provide name
        self._send_message(client, "My name is TestUser", session_id)
        
        # 5. Type "Talk to expert" (HIGH INTENT PHRASE)
        resp, events = self._send_message(client, "Talk to expert", session_id)
        
        # VERIFICATION: Should transition to EMAIL_CAPTURE or EXIT
        state = self._get_state_from_events(events)
        
        # Should be in EMAIL_CAPTURE (asking for email) or EXIT (if email already known)
        assert state in ["EMAIL_CAPTURE", "EXIT"], \
            f"Expected EMAIL_CAPTURE or EXIT state, got: {state}"
        
        # If in EMAIL_CAPTURE, verify it asks for email (system prompt)
        if state == "EMAIL_CAPTURE":
            assert "email" in resp.lower(), \
                "EMAIL_CAPTURE state should ask for email"

    def test_email_skip_when_already_known(self, client):
        """
        REGRESSION: When email is already captured, subsequent CTAs should 
        skip EMAIL_CAPTURE and go directly to EXIT.
        
        FIX: Added email check before transitioning to EMAIL_CAPTURE
        """
        session_id = "regression_email_skip_001"
        
        # 1. Start UC1 flow
        self._send_message(client, "Hi", session_id)
        
        # 2. Select capability
        self._send_message(client, "Product development & engineering", session_id)
        
        # 3. Provide context
        self._send_message(client, "I need a mobile app", session_id)
        
        # 4. Provide name
        self._send_message(client, "My name is TestUser", session_id)
        
        # 5. Provide email (first time)
        self._send_message(client, "my email is test@example.com", session_id)
        
        # 6. Trigger HIGH INTENT CTA
        resp, events = self._send_message(client, "Schedule a quick call", session_id)
        
        # VERIFICATION: Should skip asking for email and go to EXIT
        state = self._get_state_from_events(events)
        
        # Should be EXIT (email already known, skip capture)
        assert state == "EXIT", \
            f"Expected EXIT state (email skip), got: {state}"
        
        # Verify EXIT buttons present
        options = self._get_options_from_events(events)
        assert options is not None, "EXIT state should have options"
        assert "Restart Conversation" in options, "Missing 'Restart Conversation' button"
        assert "Close Chat" in options, "Missing 'Close Chat' button"

    def test_exit_buttons_consistency_uc1_a(self, client):
        """
        REGRESSION: UC1-A (Product development) should show Restart/Close buttons.
        """
        session_id = "regression_exit_uc1a_001"
        
        # Navigate to EXIT via full flow
        self._send_message(client, "Hi", session_id)
        self._send_message(client, "Product development & engineering", session_id)
        self._send_message(client, "I need a web app", session_id)
        self._send_message(client, "My name is Test", session_id)
        self._send_message(client, "test@example.com", session_id)
        resp, events = self._send_message(client, "Schedule a quick call", session_id)
        
        # VERIFICATION: EXIT buttons
        options = self._get_options_from_events(events)
        assert options == ["Restart Conversation", "Close Chat"], \
            f"UC1-A EXIT buttons incorrect: {options}"

    def test_exit_buttons_consistency_uc1_d(self, client):
        """
        REGRESSION: UC1-D (AI/ML) should show Restart/Close buttons.
        This was the failing case reported by user.
        """
        session_id = "regression_exit_uc1d_001"
        
        # Navigate to EXIT via full flow
        self._send_message(client, "Hi", session_id)
        self._send_message(client, "AI/ML & Automation", session_id)
        self._send_message(client, "I want to build a smart chatbot", session_id)
        self._send_message(client, "My name is Vinay", session_id)
        self._send_message(client, "vinay@example.com", session_id)
        resp, events = self._send_message(client, "Talk to expert", session_id)
        
        # VERIFICATION: EXIT buttons
        options = self._get_options_from_events(events)
        assert options == ["Restart Conversation", "Close Chat"], \
            f"UC1-D EXIT buttons incorrect: {options}"

    def test_high_intent_phrases_all_variations(self, client):
        """
        REGRESSION: All HIGH_INTENT_PHRASES should trigger proper Exit flow.
        """
        phrases = [
            "talk to expert",
            "talk to an expert", 
            "schedule a call",
            "book a call",
            "discuss my requirement",
        ]
        
        for idx, phrase in enumerate(phrases):
            session_id = f"regression_phrase_{idx}"
            
            # Quick setup
            self._send_message(client, "Hi", session_id)
            self._send_message(client, "AI/ML & Automation", session_id)
            self._send_message(client, "I need AI help", session_id)
            self._send_message(client, "TestUser", session_id)
            
            # Test the phrase
            resp, events = self._send_message(client, phrase, session_id)
            state = self._get_state_from_events(events)
            
            # Should be EMAIL_CAPTURE or EXIT
            assert state in ["EMAIL_CAPTURE", "EXIT"], \
                f"Phrase '{phrase}' should trigger Exit flow, got state: {state}"
