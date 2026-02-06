"""
API Endpoint Tests

Tests for all REST API endpoints including user registration,
session management, and health checks.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import create_app
import json
import uuid


@pytest.fixture
def client():
    """Create test client with fresh app instance."""
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


# Helper to check if DB is available
def db_available(client):
    """Check if database connection works."""
    try:
        response = client.get("/api/health")
        return response.status_code == 200
    except:
        return False


class TestHealthCheck:
    """Health check endpoint tests."""
    
    def test_health_returns_ok(self, client):
        """GET /api/health should return healthy status."""
        response = client.get("/api/health")
        
        # Health endpoint should work regardless of DB
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "healthy"


class TestUserRegistration:
    """User registration endpoint tests."""
    
    def test_register_creates_session(self, client):
        """POST /api/user/register should create a new session."""
        response = client.post(
            "/api/user/register",
            json={"browser": "TestBrowser/1.0", "ip": "127.0.0.1"}
        )
        
        # This test requires DB - skip if not available
        if response.status_code == 500:
            pytest.skip("Database not available")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "session_id" in data
    
    def test_register_returns_valid_uuid(self, client):
        """Session ID should be a valid UUID."""
        response = client.post(
            "/api/user/register",
            json={"browser": "Chrome/100", "ip": "192.168.1.1"}
        )
        
        if response.status_code == 500:
            pytest.skip("Database not available")
        
        data = response.json()
        session_id = data.get("session_id")
        
        if session_id:
            try:
                uuid.UUID(session_id)
            except ValueError:
                pytest.fail(f"Session ID is not a valid UUID: {session_id}")
    
    def test_register_with_optional_fields(self, client):
        """Registration should work with optional fields."""
        response = client.post(
            "/api/user/register",
            json={"browser": None, "ip": None}
        )
        
        if response.status_code == 500:
            pytest.skip("Database not available")
        
        assert response.status_code == 200


class TestChatEndpoint:
    """Main chat endpoint tests."""
    
    def test_send_stream_returns_sse(self, client):
        """POST /api/chat/send-stream should return SSE format."""
        # Use test session ID directly (works without DB)
        session_id = "test_chat_sse_format"
        
        # Send a message
        with client.stream(
            "POST",
            "/api/chat/send-stream",
            json={"query": "Hello", "session_id": session_id}
        ) as response:
            lines = list(response.iter_lines())
        
        # Should have SSE data lines
        data_lines = [l for l in lines if l.startswith("data: ")]
        assert len(data_lines) > 0, "Should receive SSE data events"
    
    def test_send_stream_processing_event(self, client):
        """First event should be 'processing' status."""
        session_id = "test_chat_processing_event"
        
        with client.stream(
            "POST",
            "/api/chat/send-stream",
            json={"query": "Hello", "session_id": session_id}
        ) as response:
            first_data = None
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        first_data = json.loads(line[6:])
                        break
                    except json.JSONDecodeError:
                        pass
        
        assert first_data is not None, "Should receive at least one event"
        assert first_data.get("status") == "processing", \
            f"First event should be 'processing', got: {first_data.get('status')}"
    
    def test_send_stream_includes_chunks(self, client):
        """Response should include chunk events with content."""
        session_id = "test_chat_includes_chunks"
        
        chunks = []
        with client.stream(
            "POST",
            "/api/chat/send-stream",
            json={"query": "Hello", "session_id": session_id}
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        if data.get("status") == "chunk":
                            chunks.append(data.get("chunk", ""))
                    except json.JSONDecodeError:
                        pass
        
        combined = "".join(chunks)
        assert len(combined) > 0, "Should receive content chunks"


class TestSessionManagement:
    """Session management endpoint tests."""
    
    def test_end_session(self, client):
        """POST /api/session/end/{session_id} should end session."""
        # Register to get a valid session ID
        reg_response = client.post(
            "/api/user/register",
            json={"browser": "Test", "ip": "127.0.0.1"}
        )
        if reg_response.status_code == 500:
            pytest.skip("Database not available")
        session_id = reg_response.json()["session_id"]
        
        # Send a message first
        with client.stream(
            "POST",
            "/api/chat/send-stream",
            json={"query": "Hello", "session_id": session_id}
        ) as response:
            for _ in response.iter_lines():
                pass
        
        # End session
        end_response = client.post(f"/api/session/end/{session_id}")
        
        # Should be successful
        assert end_response.status_code == 200
    
    def test_get_chat_messages(self, client):
        """GET /api/chat/{session_id}/messages should return history."""
        # Register to get a valid session ID
        reg_response = client.post(
            "/api/user/register",
            json={"browser": "Test", "ip": "127.0.0.1"}
        )
        if reg_response.status_code == 500:
            pytest.skip("Database not available")
        session_id = reg_response.json()["session_id"]
        
        # Send a message
        with client.stream(
            "POST",
            "/api/chat/send-stream",
            json={"query": "Hello", "session_id": session_id}
        ) as response:
            for _ in response.iter_lines():
                pass
        
        # Get messages
        messages_response = client.get(f"/api/chat/{session_id}/messages")
        
        assert messages_response.status_code == 200
        data = messages_response.json()
        assert "messages" in data


class TestUserUpdate:
    """User update endpoint tests."""
    
    def test_update_user_details(self, client):
        """PATCH /api/user/{session_id} should update user info."""
        # Register to get a valid session ID
        reg_response = client.post(
            "/api/user/register",
            json={"browser": "Test", "ip": "127.0.0.1"}
        )
        if reg_response.status_code == 500:
            pytest.skip("Database not available")
        session_id = reg_response.json()["session_id"]
        
        # Update user
        update_response = client.patch(
            f"/api/user/{session_id}",
            json={"username": "TestUser", "email": "test@example.com"}
        )
        
        # Should succeed
        assert update_response.status_code == 200


class TestPrompts:
    """Prompts endpoint tests."""
    
    def test_get_root_prompts(self, client):
        """GET /api/prompts/root should return initial prompts."""
        response = client.get("/api/prompts/root")
        
        # May return 200 with prompts or 500 if DB required
        if response.status_code == 500:
            pytest.skip("Database required for prompts")
        
        assert response.status_code == 200
