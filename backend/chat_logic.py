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
            
            # Format and stream the response in a concise, structured way
            # Process the response to ensure proper markdown formatting and structure
            lines = comprehensive_response.split('\n')
            current_section = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Convert headers to consistent format
                if line.startswith('#'):
                    # If we have accumulated content, send it
                    if current_section:
                        yield {'status': 'complete_chunk', 'chunk': '\n'.join(current_section)}
                        current_section = []
                    
                    # Format header and send
                    header = re.sub(r'^#{1,6}\s*', '###### ', line)
                    yield {'status': 'complete_chunk', 'chunk': '\n\n' + header + '\n'}
                else:
                    # Process regular content
                    # Ensure proper bold formatting
                    line = re.sub(r'\*\*([^*]+)\*\*', r'**\1**', line)
                    # Ensure bullet points are properly formatted
                    if line.lstrip().startswith('- '):
                        line = line.strip()
                    current_section.append(line)
            
            # Send any remaining content
            if current_section:
                yield {'status': 'complete_chunk', 'chunk': '\n'.join(current_section)}
            
            # Add spacing and final suggestions
            yield {'status': 'separator', 'chunk': '\n\n' }  # Extra spacing after main content
            
            # Add clear separation before suggestions and follow-ups
            yield {'status': 'separator', 'chunk': '\n\n'}

            try:
                # Generate both suggestions and follow-ups
                suggestions = follow_up_manager.generate_suggestions(session_id, context="comprehensive response completed")
                follow_ups = follow_up_manager.generate_follow_ups(session_id, latest_query, context="comprehensive response")

                # Format suggestions with bullet points and bold highlights
                if suggestions:
                    formatted_suggestions = []
                    for s in suggestions[:2]:  # Limit to 2 suggestions
                        # Ensure suggestion starts with bullet point
                        s = s.strip()
                        if not s.startswith('- '):
                            s = '- ' + s
                        # Ensure key terms are bold
                        if '**' not in s:
                            words = s.split()
                            for i, word in enumerate(words):
                                if word.lower() in ['implement', 'create', 'use', 'develop', 'build', 'integrate']:
                                    words[i] = f'**{word}**'
                            s = ' '.join(words)
                        formatted_suggestions.append(s)
                    yield {'status': 'suggestions', 'chunk': '\n'.join(formatted_suggestions)}
                else:
                    yield {'status': 'suggestions', 'chunk': "- Consider implementing a **proof of concept** to validate the approach"}

                # Add spacing between sections
                yield {'status': 'separator', 'chunk': '\n\n'}

                # Format follow-ups with clear structure
                if follow_ups and len(follow_ups) > 0:
                    formatted_followups = []
                    for f in follow_ups[:2]:  # Limit to 2 follow-ups
                        f = f.strip()
                        if not f.startswith('- '):
                            f = '- ' + f
                        formatted_followups.append(f)
                    yield {'status': 'followups', 'chunk': '\n'.join(formatted_followups)}
                else:
                    yield {'status': 'followups', 'chunk': "- What specific aspects would you like to explore further?"}

            except Exception as e:
                logger.error(f"Error generating suggestions and follow-ups: {e}")
                # Fallback with generic grouped suggestions and follow-ups
                yield {'status': 'suggestions', 'chunk': "Consider implementing a proof of concept to validate the approach"}
                yield {'status': 'separator', 'chunk': '\n\n'}
                yield {'status': 'followups', 'chunk': "Would you like to explore any specific aspect of the solution?"}
                
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
Your response must primarily use the provided Knowledge Base context (≈80%) 
and may include a short supplemental note (≈20%) if needed.

SMART RESPONSE GUIDELINES:

FORMATTING INSTRUCTIONS:  
1. Limit the response to **around 200 words** — concise, informative, and focused.  
2. Use **bold text** to highlight important terms, technologies, and key concepts.  
3. Maintain a natural, conversational flow similar to ChatGPT responses.  
4. Use Markdown formatting naturally (bold, lists, headings) for readability. 

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

            # Generate and format the main response
            main_response = ""
            response_stream = follow_up_manager.chatbot.get_detailed_response(
                query=enhanced_query,
                chat_history=[(msg["role"], msg["content"]) for msg in (conversation_history or [])],
                stream=True
            )
            
            # Process and format the response
            current_section = []
            for chunk in response_stream:
                if not chunk:
                    continue
                
                text_chunk = str(chunk).strip()
                main_response += text_chunk
                
                # Split into lines to process sections
                lines = text_chunk.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        if current_section:
                            # Join and send accumulated section
                            section_text = ' '.join(current_section)
                            yield {'status': 'complete_chunk', 'chunk': section_text}
                            current_section = []
                        continue
                    
                    # Handle headers
                    if line.startswith('#'):
                        if current_section:
                            section_text = ' '.join(current_section)
                            yield {'status': 'complete_chunk', 'chunk': section_text}
                            current_section = []
                        
                        # Format header consistently
                        header = re.sub(r'^#{1,6}\s*', '###### ', line)
                        yield {'status': 'complete_chunk', 'chunk': '\n\n' + header + '\n'}
                    else:
                        # Process regular content
                        # Ensure bullet points are properly formatted
                        if line.lstrip().startswith('- '):
                            if current_section:
                                section_text = ' '.join(current_section)
                                yield {'status': 'complete_chunk', 'chunk': section_text}
                                current_section = []
                            # Send bullet points as complete chunks
                            yield {'status': 'complete_chunk', 'chunk': line}
                        else:
                            current_section.append(line)
            
            # Send any remaining content
            if current_section:
                section_text = ' '.join(current_section)
                yield {'status': 'complete_chunk', 'chunk': section_text}
            
            # Add spacing after main response (no separator lines)
            yield {'status': 'separator', 'chunk': '\n\n'}  # Just spacing, no lines

            # Save the main response to the conversation history
            if main_response:
                follow_up_manager.add_to_conversation_history(session_id, "assistant", main_response)

            # Add spacing after main response
            yield {'status': 'separator', 'chunk': '\n\n'}

            try:
                # Generate single suggestion and follow-up
                suggestion = follow_up_manager.generate_suggestions(session_id, context=main_response[:200])[0] if follow_up_manager.generate_suggestions(session_id, context=main_response[:200]) else None
                follow_up = follow_up_manager.generate_follow_ups(session_id, latest_query, context=main_response[:200])[0] if follow_up_manager.generate_follow_ups(session_id, latest_query, context=main_response[:200]) else None
                
                # Format suggestion
                if suggestion:
                    # Clean up and format suggestion
                    clean_suggestion = re.sub(r'#{1,6}\s*', '', suggestion.strip())
                    clean_suggestion = re.sub(r'^\d+\.\s*|-\s*', '', clean_suggestion)
                    # Ensure proper bold formatting for key terms
                    for term in ['implement', 'create', 'use', 'integrate', 'develop', 'leverage']:
                        pattern = f'(?i)\\b{term}\\b'
                        clean_suggestion = re.sub(pattern, f'**{term}**', clean_suggestion)
                    yield {'status': 'suggestions', 'chunk': f'- {clean_suggestion.strip()}'}
                else:
                    yield {'status': 'suggestions', 'chunk': '- Consider **implementing** a proof of concept to validate your approach'}

                # Single separator
                yield {'status': 'separator', 'chunk': '\n\n'}

                # Format follow-up
                # Format and send a single follow-up
                follow_up = follow_up_manager.generate_follow_ups(session_id, latest_query, context=main_response[:150])[0] if follow_up_manager.generate_follow_ups(session_id, latest_query, context=main_response[:150]) else None
                
                if follow_up:
                    # Clean up and format follow-up
                    clean_followup = re.sub(r'#{1,6}\s*', '', follow_up.strip())
                    clean_followup = re.sub(r'^\d+\.\s*|-\s*', '', clean_followup)
                    
                    # Make it more engaging if it's not already a question
                    if not any(clean_followup.lower().startswith(q) for q in ['what', 'how', 'could', 'would', 'can', 'which']):
                        clean_followup = f"Could you tell us more about {clean_followup.lower()}"
                    
                    yield {'status': 'followup', 'chunk': f'- {clean_followup.strip()}?'}
                else:
                    # Context-aware fallback follow-up
                    context_keywords = latest_query.lower() if latest_query else ""
                    if any(word in context_keywords for word in ['app', 'mobile', 'web', 'application']):
                        yield {'status': 'followup', 'chunk': "- What specific features or functionalities are most important for your app?"}
                    elif any(word in context_keywords for word in ['ai', 'machine learning', 'ml', 'chatbot']):
                        yield {'status': 'followup', 'chunk': "- What type of AI functionality do you envision for your project?"}
                    elif any(word in context_keywords for word in ['website', 'site', 'web', 'portal']):
                        yield {'status': 'followup', 'chunk': "- What's the main purpose of your website - e-commerce, corporate, or service-based?"}
                    else:
                        yield {'status': 'followup', 'chunk': "- What specific aspects would you like to explore further?"}
            except Exception as e:
                logger.error(f"Error generating follow-ups: {e}")
                # Fallback with generic suggestion and follow-up
                yield {'status': 'suggestion', 'chunk': "- Consider **implementing** a basic prototype to test core functionality"}
                yield {'status': 'separator', 'chunk': '\n\n'}
                yield {'status': 'followup', 'chunk': "- What specific features are most important to you?"}

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
