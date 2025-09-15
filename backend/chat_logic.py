from typing import AsyncGenerator, List, Dict, Any, Optional
import json
import logging
from backend.llm_client import llm
from backend.services.chatbot_optimizer import OptimizedChatbot
from backend.nested_follow_up_manager import FollowUpManager

logger = logging.getLogger(__name__)

# def build_chatbot_response(
#     session_id: str,
#     follow_up_manager: FollowUpManager,
#     conversation_history: Optional[List[Dict[str, Any]]] = None,
#     prompt_context: Optional[str] = None,
#     with_followup: bool = True
# ) -> Generator[str, None, None]:
#     """
#     De-duplicate retrieved documents and keep metadata for traceability.

#     Args:
#         docs: List of documents, each with `page_content` and `metadata`.

#     Returns:
#         List of tuples: (text, metadata)
#     """
#     seen = set()
#     unique = []
    
#     for d in docs:
#         # Extract text and metadata
#         text = d.page_content.strip() if hasattr(d, "page_content") else str(d)
#         metadata = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}

#         # Skip empty or duplicate texts
#         if not text or text in seen:
#             continue

#         seen.add(text)
#         unique.append((text, metadata))
    
#     return unique


# def _maybe_expand_queries(query: str) -> List[str]:
#     # Lightweight RAG fusion: expand the query to reduce “same answer” effect
#     return list(dict.fromkeys([
#         query,
#         f"Details about {query}",
#         f"In-depth explanation of {query}",
#     ]))

# # Initialize the optimized chatbot (do this once at application startup)
# # Initialize FollowUpManager with the LLM instance
# follow_up_manager = FollowUpManager(llm=llm)

from backend.llm_client import llm
import json
import logging
from typing import AsyncGenerator, List, Dict, Any, Optional

logger = logging.getLogger(__name__)

async def build_chatbot_response(session_id: str, follow_up_manager, conversation_history: Optional[List[Dict[str, Any]]] = None, prompt_context: Optional[str] = None, mode: str = "complete") -> AsyncGenerator[str, None]:
    """
    Build a streaming response from the chatbot that handles both follow-up generation and complete responses.
    Uses OptimizedChatbot for context optimization and token management.
    
    Args:
        session_id: The session identifier
        follow_up_manager: Instance of FollowUpManager
        conversation_history: List of conversation messages
        prompt_context: Original prompt context
        mode: Either "follow_up" or "complete" to determine response type
    
    Yields:
        Formatted SSE messages for streaming response
    """
    try:
        # Get session data and validate
        session_data = follow_up_manager.get_session_data(session_id)
        if not session_data:
            yield f"data: {{\"error\": \"Session not found\"}}\n\n"
            return

        # Get the latest query from conversation history
        latest_query = ""
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    latest_query = msg.get("content", "")
                    break
        
        if not latest_query and mode != "follow_up":
            yield f"data: {{\"error\": \"No query found in conversation\"}}\n\n"
            return

        # Use OptimizedChatbot for streaming responses
        chat_history = [(msg["role"], msg["content"]) for msg in (conversation_history or [])]
        response_stream = follow_up_manager.chatbot.get_detailed_response(
            query=latest_query,
            chat_history=chat_history,
            stream=True
        )

        yield f"data: {json.dumps({'status': 'processing', 'message': 'Generating response...'})}\n\n"

        if mode == "follow_up":
            # For follow-ups, accumulate the response to get a complete question
            follow_up_response = ""
            async for chunk in response_stream:
                if chunk:
                    follow_up_response += chunk.strip()
            
            if follow_up_response:
                # Format as a question if needed
                if not follow_up_response.endswith("?"):
                    follow_up_response += "?"
                
                follow_up_manager.add_to_conversation_history(session_id, "assistant", follow_up_response)
                yield f"data: {json.dumps({'status': 'follow_up', 'content': follow_up_response})}\n\n"
            
        else:  # mode == "complete"
            # Stream complete response chunks immediately
            complete_response = ""
            async for chunk in response_stream:
                if chunk:
                    complete_response += chunk
                    yield f"data: {json.dumps({'status': 'complete', 'content': chunk})}\n\n"
            
            if complete_response:
                # Save to conversation history
                follow_up_manager.add_to_conversation_history(session_id, "assistant", complete_response)
                # Send final message
                yield f"data: {json.dumps({
                    'status': 'complete',
                    'content': complete_response,
                    'final': True
                })}\n\n"
            
            if complete_response:
                follow_up_manager.add_to_conversation_history(session_id, "assistant", complete_response)
                yield f"data: {json.dumps({'status': 'complete', 'content': complete_response, 'final': True})}\n\n"

    except Exception as e:
        error_msg = f"Error in build_chatbot_response: {str(e)}"
        logger.error(error_msg)
        yield f"data: {json.dumps({'error': error_msg})}\n\n"

# Clean up the rest of the file to remove unrelated code

        # Prepare context from conversation history
        context = f"Context from previous conversation:\n{follow_up_manager.format_conversation_history(conversation_history)}\n" if conversation_history else ""
        if prompt_context:
            context += f"\nOriginal prompt context:\n{prompt_context}\n"

        # Generate main response
        messages = [
            {
                "role": "system",
                "content": f"You are a helpful AI assistant. {context}"
            },
            {
                "role": "user",
                "content": latest_query
            }
        ]

        # Get streaming response
        for chunk in llm.stream(messages):
            if chunk:
                yield f"data: {json.dumps({'content': chunk})}\n\n"

        # Generate follow-ups if enabled
        if with_followup:
            follow_ups = follow_up_manager.generate_next_follow_up(session_id)
            if follow_ups:
                yield f"data: {json.dumps({'follow_ups': follow_ups})}\n\n"

    except Exception as e:
        error_msg = f"Error in build_chatbot_response: {str(e)}"
        yield f"data: {json.dumps({'error': error_msg})}\n\n"
    """
    Enhanced chatbot response function with follow-up management.
    """
    optimized_chatbot = OptimizedChatbot(llm, model="gpt-4o-mini")  # Ensure model matches llm

    try:
        response, success = optimized_chatbot.get_detailed_response(
            query=query,
            chat_history=chat_history,
            site=site,
            detailed=detailed  # ✅ fixed comma + pass flag
        )

        if success:
            # Add follow-ups dynamically
            follow_ups = optimized_chatbot.generate_followups(query)
            for follow_up in follow_ups:
                follow_up_manager.add_follow_up("session_id_placeholder", follow_up)

            return response, True
        else:
            return _fallback_to_original(query, chat_history, site)

    except Exception as e:
        print(f"[ERROR] Optimized chatbot failed: {e}")
        return _fallback_to_original(query, chat_history, site)


# ✅ ADD - Fallback function (simplified version of your original logic)
def _fallback_to_original(query: str, chat_history: list, site: str):
    """Fallback to simplified original behavior"""
    try:
        # Simplified version of your original logic
        variant_queries = _maybe_expand_queries(query)
        pooled_docs = []
        for q in variant_queries:
            pooled_docs.extend(retriever.get_relevant_documents(q))
        
        unique_texts = _dedupe_chunks(pooled_docs)
        context_text = "\n\n---\n\n".join(unique_texts[:8])  # Reduced chunks for fallback
        
        if not context_text.strip():
            search_results = search_site(query, site)
            scraped_texts = []
            for res in search_results:
                url = res.get("url")
                title = res.get("title") or url
                if url:
                    text = scrape_url(url)
                    if text:
                        scraped_texts.append(f"[{title}]({url}): {text}")
            context_text = "\n\n".join(scraped_texts[:5])
            if not context_text.strip():
                return (
                    "No relevant content found. Please visit the website directly "
                    f"[{site}](https://{site}).",
                    True
                )
        
        history_text = "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {msg}"
            for role, msg in chat_history
        )
        
        # Simple fallback prompt
        fallback_prompt = f"""
You are a helpful assistant.
Conversation: {history_text}
Context: {context_text}
Question: {query}
Please provide a helpful answer.
"""
        
        raw_answer = llm.invoke(fallback_prompt)
        answer = raw_answer.content if hasattr(raw_answer, 'content') else str(raw_answer)
        
        return answer, True if answer.strip() else ("No response generated.", False)
        
    except Exception as e:
        return f"I apologize, but I'm experiencing technical difficulties: {str(e)}", False

def get_prompt_response(session_id: str, selected_prompt_id: int):
    """
    Return nested follow-ups for a given prompt ID.
    Include detailed answer if requested.
    """
    conn = _get_conn()
    cursor = conn.cursor()

    # Fetch the selected prompt
    cursor.execute("""
        SELECT prompt_text, response_text
        FROM prompts
        WHERE id = %s
    """, (selected_prompt_id,))
    prompt = cursor.fetchone()

    if not prompt:
        cursor.close()
        conn.close()
        log_event("Prompt not found", prompt_id=selected_prompt_id, session_id=session_id)
        return None, "Prompt not found", False

    prompt_text, response_text = prompt

    # Fetch child prompts
    cursor.execute("""
        SELECT id, prompt_text, response_text, display_order
        FROM prompts
        WHERE parent_id = %s
        ORDER BY display_order ASC
    """, (selected_prompt_id,))
    child_prompts = cursor.fetchall()

    cursor.close()
    conn.close()

    log_event("Follow-up prompts generated", parent_prompt_id=selected_prompt_id, follow_up_ids=[row[0] for row in child_prompts], session_id=session_id)

    # Format child prompts
    follow_ups = [
        {
            "id": row[0],
            "prompt_text": row[1],
            "response_text": row[2],
            "display_order": row[3]
        }
        for row in child_prompts
    ]

    return follow_ups, response_text, True

# Modify build_chatbot_response to enforce website-only responses
def build_chatbot_response(query, history):
    # Initialize meta
    meta = {}

    # Enforce website-only responses
    if not meta.get("used_web"):
        return "I don’t have this detail.", False, meta

    # ...existing logic...
    return "Response generated.", True, meta
