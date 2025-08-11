import requests
from typing import Optional
from datetime import datetime


API_BASE_URL = "http://localhost:8000"  # Adjust this to your backend URL


def start_new_chat(query: str, context: list = None) -> dict:
    """
    Start a new chat session with the backend.
    """
    payload = {
        "query": query,
        "context": context or [],  # Include context if provided
        "timestamp": datetime.now().isoformat(),  # Only here!

    }
    try:
        response = requests.post(f"{API_BASE_URL}/chat/new", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 422:
            raise ValueError("The backend rejected the request. Please check the input format.")
        raise e
    except requests.RequestException as e:
        raise ConnectionError("Failed to connect to the backend server.") from e


def continue_chat(query: str, session_id: str, context: list = None) -> dict:
    """
    Continue an existing chat session.
    Handle errors gracefully and validate session ID.
    """
    payload = {
        "query": query,
        "session_id": session_id,
        "context": context or []  # Include context if provided
    }
    try:
        response = requests.post(f"{API_BASE_URL}/chat/continue", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise ValueError(f"Session ID {session_id} not found.")
        raise e
    except requests.RequestException as e:
        raise ConnectionError("Failed to connect to the backend server.") from e


def get_chat_history(session_id: str) -> dict:
    """
    Retrieve the chat history for a given session.
    Handle 404 errors gracefully by returning an empty history.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/chat/history/{session_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            return {"history": []}  # Return empty history if session not found
        raise e


def ping_server() -> bool:
    """
    Check if the backend server is running.
    """
    try:
        response = requests.get(f"{API_BASE_URL}/ping")
        return response.status_code == 200
    except requests.RequestException:
        return False