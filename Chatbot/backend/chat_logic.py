# streamlit_app/chat_logic.py
from .retriever import retriever
from .llm_client import call_llm_with_context

def build_chatbot_response(query: str, chat_history: list):
    """
    Retrieves context for the query, calls LLM, and returns chatbot response.
    
    Args:
        query (str): User's latest message.
        chat_history (list): List of tuples [(role, message), ...].
    
    Returns:
        tuple: (answer: str, success: bool)
    """
    # Step 1: Retrieve relevant docs
    docs = retriever.get_relevant_documents(query)

    # Debug: Log retrieved documents
    print(f"[DEBUG] Retrieved documents: {docs}")

    context_text = "\n\n".join(doc.page_content for doc in docs)

    # Debug: Log constructed context
    print(f"[DEBUG] Constructed context: {context_text}")

    # Step 2: If no context found, block LLM call
    if not context_text.strip():
        return (
            "No relevant content found. Please visit the website directly"
            "[DITS](https://www.ditstek.com).",
            True
        )

    # Step 3: Format history for the prompt
    history_text = "\n".join(
        f"{'User' if role == 'user' else 'Assistant'}: {msg}"
        for role, msg in chat_history
    )

    # Step 4: Call LLM
    answer = call_llm_with_context(
        context=context_text,
        history=history_text,
        question=query
    )

    # Step 5: Handle "no content" case gracefully
    if "No relevant content found" in answer:
        return (
            "No relevant content found. Please submit your query via our "
            "[Contact Form](https://www.ditstek.com/contact).",
            True
        )

    return answer, True
