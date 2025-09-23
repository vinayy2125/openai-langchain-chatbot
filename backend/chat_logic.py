from backend.llm_client import llm
from backend.services.chatbot_optimizer import OptimizedChatbot
from backend.nested_follow_up_manager import FollowUpManager

import json
import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, AsyncGenerator

# Initialize logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Utility helpers (restored minimal versions for optimizer imports)
def _maybe_expand_queries(query: str) -> list:
    """Return lightweight expanded query variants (deduplicated)."""
    variants = [
        query,
        f"Details about {query}",
        f"In-depth explanation of {query}"
    ]
    # Preserve order while deduping
    seen = set()
    deduped = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            deduped.append(v)
    return deduped

def _dedupe_chunks(docs) -> list:
    """Simplify dedupe: accepts list of doc objects or strings, returns list of (text, meta)."""
    seen = set()
    result = []
    for d in docs:
        text = getattr(d, 'page_content', str(d)).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        meta = getattr(d, 'metadata', {}) if hasattr(d, 'metadata') else {}
        result.append((text, meta))
    return result

async def build_chatbot_response(
    session_id: str,
    follow_up_manager,
    conversation_history: Optional[List[Dict[str, Any]]] = None,
    prompt_context: Optional[str] = None,
    mode: str = "complete"
) -> AsyncGenerator[str, None]:
    """
    Build a streaming response from the chatbot that handles both direct responses and follow-up suggestions.

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
            yield f"{json.dumps({'error': 'Session not found'})}\n\n"
            return

        # Get the latest user query from conversation history
        latest_query = ""
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    latest_query = msg.get("content", "")
                    break
        if not latest_query and prompt_context:
            latest_query = prompt_context.split("\n")[0][:140]

        # Check if requirements are complete and decide response strategy
        requirements_complete = follow_up_manager.check_requirements(session_id)
        
        if requirements_complete:
            # Generate comprehensive final response
            logger.info(f"Requirements complete for session {session_id}, generating comprehensive response")
            comprehensive_response = follow_up_manager.generate_comprehensive_response(session_id)
            
            # Stream the comprehensive response preserving markdown formatting
            # Split by double line breaks to preserve markdown structure and fix any header issues
            sections = re.split(r'\n\n', comprehensive_response)
            for section in sections:
                if section.strip():
                    # Convert any H1-H2 headers to H3 headers for better readability
                    section = re.sub(r'^#{1,2}\s', '### ', section.strip(), flags=re.MULTILINE)
                    
                    # Check if this is a heading or regular content
                    if section.startswith('###'):
                        # Send heading with proper line breaks
                        yield {'status': 'complete_chunk', 'chunk': '\n\n' + section + '\n\n'}
                    else:
                        # Send content with word-level streaming for readability
                        words = section.split(' ')
                        word_buffer = ""
                        for word in words:
                            if word_buffer:
                                word_buffer += ' ' + word  # Space before word (not after)
                            else:
                                word_buffer = word  # First word, no space before
                            # Send in chunks of ~5 words to maintain proper spacing
                            if len(word_buffer.split()) >= 5:
                                yield {'status': 'complete_chunk', 'chunk': word_buffer}
                                word_buffer = ""
                        # Send remaining words without extra trailing space
                        if word_buffer.strip():
                            yield {'status': 'complete_chunk', 'chunk': word_buffer}
            
            # Add spacing and final suggestions
            yield {'status': 'separator', 'chunk': '\n\n' }  # Extra spacing after main content
            
            # Add single practical suggestion using the suggestion system (no header/label)
            try:
                practical_suggestions = follow_up_manager.generate_suggestions(session_id, context="comprehensive response completed")
                if practical_suggestions:
                    yield {'status': 'suggestion', 'chunk': practical_suggestions[0]}
                else:
                    yield {'status': 'suggestion', 'chunk': "Consider implementing a proof of concept to validate the approach"}
            except Exception as e:
                logger.error(f"Error generating practical suggestions: {e}")
                yield {'status': 'suggestion', 'chunk': "Consider implementing a proof of concept to validate the approach"}
            
            # Add follow-up section with proper spacing
            yield {'status': 'separator', 'chunk': '\n\n'}
            
            # Generate single dynamic follow-up based on context
            try:
                follow_ups = follow_up_manager.generate_follow_ups(session_id, latest_query, context="comprehensive response")
                if follow_ups and len(follow_ups) > 0:
                    yield {'status': 'followup_question', 'chunk': follow_ups[0]}
                else:
                    yield {'status': 'followup_question', 'chunk': "Would you like me to dive deeper into any specific aspect?"}
            except Exception as e:
                logger.error(f"Error generating dynamic follow-up: {e}")
                yield {'status': 'followup_question', 'chunk': "Would you like me to dive deeper into any specific aspect?"}
                
        else:
            # Generate concise response + specific follow-ups to gather more requirements
            logger.info(f"Requirements incomplete for session {session_id}, generating concise response with follow-ups")
            
            # FIX: Smart context detection instead of hardcoded patterns
            assistant_messages = [msg for msg in (conversation_history or []) if msg.get("role") == "assistant"]
            is_prompt_selection = prompt_context is not None and len(assistant_messages) == 0
            is_manual_query = prompt_context is None and len(assistant_messages) == 0
            is_followup_response = len(assistant_messages) > 0
            
            # FIX: Unified prompt instruction with smart context awareness
            if is_prompt_selection or is_manual_query:
                # Retrieve context text from the chatbot service for initial queries
                context_text = follow_up_manager.chatbot._retrieve_context(latest_query, "ditstek.com")
                
                enhanced_query = f"""
You are "Ditstek Assistant", answering on behalf of the Ditstek team. 
Your response must primarily use the provided Knowledge Base context (≈80%) and may include a short supplemental note (≈20%) if needed.

SMART RESPONSE GUIDELINES:

1. **DITSTEK VOICE:**
   - Always respond as "we at Ditstek" or "our team".
   - Showcase solutions as if implemented in Ditstek projects.

2. **CONTENT SOURCING:**
   - Use the KB context provided to form the majority of your answer (≈80%).
   - If additional helpful detail is needed, add a short section labeled:
     **Ditstek note — supplemental:** (max ≈20% of reply).
   - If the KB lacks the answer, respond: 
     "We couldn’t find this in our knowledge base. Please check the source directly."

3. **FORMATTING REQUIREMENTS:**
   - Use **bold text** for key technologies, concepts, and terms
   - Use ONLY ###### headings for sections:
     * ###### Key Points
     * ###### Technical Details
     * ###### Recommendations
   - Use bullet points and numbered lists naturally
   - Include clear line breaks for readability

4. **STRUCTURE:**
   - Start with a short overview paragraph
   - Provide well-structured sections (with headings)
   - Include relevant specifics (examples, tools, approaches)
   - End with **highlighted next steps**

5. **PROFESSIONAL POLISH:**
   - Tone: engaging, professional, and confident
   - Explicitly mention Ditstek’s role in applying the approach
   - Cite sources from KB at the end in format:
     *Sources: [Doc Title — url — score]*

CONTEXT (use this for main part of the answer):
{context_text}

USER QUERY:
{latest_query}
"""
            else:
                # For follow-up responses, continue conversation naturally with proper formatting
                # Retrieve context text from the chatbot service
                context_text = follow_up_manager.chatbot._retrieve_context(latest_query, "ditstek.com")

                enhanced_query = f"""
Continue this conversation as "Ditstek Assistant", always answering on behalf of the Ditstek team. 
Base your answer primarily on the Knowledge Base context (≈80%), with an optional short supplement (≈20%) if needed.

CRITICAL FORMATTING REQUIREMENTS:
1. Use **bold text** for key terms and technologies
2. Use ONLY ###### headings when sections are needed:
   - ###### Key Points
   - ###### Technical Details
   - ###### Recommendations
3. Use bullet points for clarity
4. Maintain natural paragraph breaks
5. End with actionable takeaways

CONTENT GUIDELINES:
- Ground answers in KB context whenever possible
- Keep supplement minimal and clearly labeled: 
  **Ditstek note — supplemental:**
- Always speak as Ditstek team
- Provide sources for KB material at the end

Previous conversation context: {conversation_history[-2:] if len(conversation_history) >= 2 else 'None'}

CONTEXT:
{context_text}

USER QUERY:
{latest_query}
"""

            # Generate the main response with word-level streaming
            main_response = ""
            response_stream = follow_up_manager.chatbot.get_detailed_response(
                query=enhanced_query,
                chat_history=[(msg["role"], msg["content"]) for msg in (conversation_history or [])],
                stream=True
            )
            
            # Ensure structured formatting for manual queries
            for chunk in response_stream:
                if not chunk:
                    continue
                text_chunk = str(chunk)
                main_response += text_chunk

                # Apply word-level streaming with proper spacing
                # Split into words and yield each word individually with proper spacing
                words = text_chunk.split()
                for word in words:
                    if word.strip():  # Only yield non-empty words
                        # Yield each word with a trailing space for proper separation
                        yield {'status': 'word', 'chunk': word + ' '}
                
                # Handle paragraph breaks and line breaks properly
                if '\n\n' in text_chunk:
                    yield {'status': 'paragraph_break', 'chunk': '\n\n'}
                elif '\n' in text_chunk and not text_chunk.strip().startswith('#'):
                    yield {'status': 'line_break', 'chunk': '\n'}
            
            # Add spacing after main response (no separator lines)
            yield {'status': 'separator', 'chunk': '\n\n'}  # Just spacing, no lines

            # Save the main response to the conversation history
            if main_response:
                follow_up_manager.add_to_conversation_history(session_id, "assistant", main_response)

            # Generate intelligent follow-up questions based on conversation state
            conversation_history_data = follow_up_manager.get_session_data(session_id).get('conversation_history', [])
            user_messages = [msg for msg in conversation_history_data if msg.get("role") == "user"]
            if user_messages:
                # Access the latest user message for further processing or context
                latest_user_message = user_messages[-1].get("content", "")
                logger.info(f"Latest user message: {latest_user_message}")
                # Use the latest user message for generating follow-ups or suggestions
                follow_ups = follow_up_manager.generate_follow_ups(session_id, latest_user_message, context=main_response[:150])
                if follow_ups and len(follow_ups) > 0:
                    # Clean and send the first follow-up
                    clean_followup = re.sub(r'#{1,6}\s*', '', follow_ups[0])
                    followup_text = ' '.join(clean_followup.split())
                    yield {'status': 'followup_question', 'chunk': followup_text}
                else:
                    yield {'status': 'followup_question', 'chunk': "What would you like to focus on next?"}
            
            # Use the existing chatbot optimizer's generate_followups method
            try:
                # Generate suggestions first (before follow-ups)
                suggestions = follow_up_manager.generate_suggestions(session_id, context=main_response[:100])
                
                # Send single suggestion with word-level streaming
                if suggestions:
                    best_suggestion = suggestions[0]
                    # Remove any existing numbering and clean up
                    clean_suggestion = re.sub(r'^\d+\.\s*', '', best_suggestion).strip()
                    # Keep markdown formatting for ChatGPT-like display
                    # Only remove headers, preserve **bold** and *italic*
                    clean_suggestion = re.sub(r'#{1,6}\s*', '', clean_suggestion)  # Remove headers only
                    
                    # Stream suggestion word by word with proper spacing
                    suggestion_words = clean_suggestion.split()
                    suggestion_text = ' '.join(suggestion_words)  # Rebuild without extra spaces
                    yield {'status': 'suggestion', 'chunk': suggestion_text}
                else:
                    # Fallback suggestion without word streaming
                    fallback_suggestion = "Consider starting with a proof of concept to validate your approach"
                    yield {'status': 'suggestion', 'chunk': fallback_suggestion}
                
                # Add spacing between suggestion and follow-ups (no separator lines)
                yield {'status': 'separator', 'chunk': '\n\n'}  # Just spacing, no lines
                
                # No separator lines as requested by user
                # Just extra spacing before follow-ups
                yield {'status': 'separator', 'chunk': '\n'}
                
                # The follow-up generation is now handled by the dynamic system below
                pass
                
                # Send single dynamic follow-up
                yield {'status': 'separator', 'chunk': '\n\n'}
                
                # Generate single contextual follow-up based on conversation state
                try:
                    follow_ups = follow_up_manager.generate_follow_ups(session_id, latest_query, context=main_response[:150])
                    if follow_ups and len(follow_ups) > 0:
                        # Clean and send the first follow-up
                        clean_followup = re.sub(r'#{1,6}\s*', '', follow_ups[0])
                        followup_text = ' '.join(clean_followup.split())
                        yield {'status': 'followup_question', 'chunk': followup_text}
                    else:
                        # Use keyword-based fallback for single follow-up
                        context_keywords = latest_query.lower()
                        if any(word in context_keywords for word in ['app', 'mobile', 'web', 'application']):
                            yield {'status': 'followup_question', 'chunk': "What specific features or functionalities are most important for your app?"}
                        elif any(word in context_keywords for word in ['ai', 'machine learning', 'ml', 'chatbot']):
                            yield {'status': 'followup_question', 'chunk': "What type of AI functionality do you envision for your project?"}
                        elif any(word in context_keywords for word in ['website', 'site', 'web', 'portal']):
                            yield {'status': 'followup_question', 'chunk': "What's the main purpose of your website - e-commerce, corporate, or service-based?"}
                        else:
                            yield {'status': 'followup_question', 'chunk': "What's the primary goal you want to achieve with this project?"}
                except Exception as inner_e:
                    logger.error(f"Error in dynamic follow-up generation: {inner_e}")
                    yield {'status': 'followup_question', 'chunk': "What would you like to focus on next?"}
                    
            except Exception as e:
                logger.error(f"Error generating follow-ups: {e}")
                # Fallback single follow-up for errors
                yield {'status': 'suggestion', 'chunk': "Consider starting with a basic prototype to test core functionality"}
                yield {'status': 'separator', 'chunk': '\n\n'}
                yield {'status': 'followup_question', 'chunk': "What specific features are most important to you?"}

    except Exception as e:
        yield {'status': 'error', 'message': str(e)}



# ✅ ADD - Fallback function (simplified version of your original logic)
def _fallback_to_original(query: str, chat_history: list, site: str):
    """Fallback disabled (legacy dependencies removed)."""
    return "Fallback not available.", False

def get_prompt_response(session_id: str, selected_prompt_id: int):
    """
    Return nested follow-ups for a given prompt ID.
    Include detailed answer if requested.
    """
    # NOTE: Legacy DB prompt retrieval below is currently disabled (dependencies missing).
    return None, "Legacy prompt retrieval disabled", False

    # conn = _get_conn()
    # cursor = conn.cursor()

    # Fetch the selected prompt
    # (Disabled legacy implementation)

# Legacy synchronous variant (renamed to avoid clashing with async build_chatbot_response used by FollowUpManager)
def legacy_build_chatbot_response(query, history):
    # Initialize meta
    meta = {}

    # Enforce website-only responses
    if not meta.get("used_web"):
        return "I don’t have this detail.", False, meta

    # ...existing logic...
    return "Response generated.", True, meta
