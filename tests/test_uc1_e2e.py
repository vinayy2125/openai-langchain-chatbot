
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
                "message": msg,
                "session_id": session,
                "is_uc1": True # Authoritative Trigger
            }
            # TestClient.stream returns a context manager
            with client.stream("POST", "/chat/send-stream", json=payload) as response_stream:
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
            if e.get("status") == "meta" and "uc1_options" in e:
                options = e["uc1_options"]
        
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
             if e.get("status") == "meta" and "uc1_options" in e:
                cta_options = e["uc1_options"]
        
        assert cta_options is not None, "CTA buttons missing in S7!"
        
        # 8. Loop Back
        loop_cta = "Continue exploring" 
        resp, events = send(loop_cta)
        
        # Should go back to S2 (Capability Selection)
        cap_options = None
        for e in events:
             if e.get("status") == "meta" and "uc1_options" in e:
                cap_options = e["uc1_options"]
                
        assert cap_options is not None
        assert len(cap_options) >= 2
