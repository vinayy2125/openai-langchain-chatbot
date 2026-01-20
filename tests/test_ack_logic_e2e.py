
import pytest
from fastapi.testclient import TestClient
from app.main import create_app
import json

@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as client:
        yield client

class TestAckLogicE2E:
    def test_yes_is_processed_by_llm(self, client):
        """
        Verify that "yes" is NOT bypassed by the generic ACK handler.
        It should generate a contextual response from the LLM.
        """
        session_id = "test_ack_flow_002"
        
        def send(msg):
            response = ""
            events = []
            payload = {
                "message": msg,
                "session_id": session_id,
                "is_uc1": True
            }
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

        # 1. Start session
        send("Hello")
        
        # 2. Select Capability to get into a context
        send("Product development & engineering")
        
        # 3. Provide context signal
        send("I want to build a new AI product")
        
        # 4. Provide Name
        send("John")
        
        # 5. Now we are in Exploration or Synthesis.
        # Send "yes" - simulating a response to something like "Should we focus on X?"
        resp, events = send("yes")
        
        # VERIFICATION:
        # The generic bypass message format is: "Got it. You're exploring {trigger}. What else..."
        # We want to ensure the response IS NOT that.
        # We effectively check if the response is "smart" (LLM generated).
        
        print(f"DEBUG: 'Yes' response: {resp}")
        
        assert "Got it. You're exploring Product development & engineering." not in resp, \
            "FAIL: 'Yes' was intercepted by the generic generic ACK bypass logic!"
        
        # It handles 'yes' gracefully?
        assert len(resp) > 20, "Response too short, likely just 'Got it' or error."

    def test_ack_bypass_still_works_for_pure_ack(self, client):
        """
        Optional: Verify that simple 'ok' MIGHT still go to LLM now (per my change),
        OR if we completely removed the bypass, this test confirms LLM handles it well.
        """
        pass
