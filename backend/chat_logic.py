from backend.llm_client import llm
from backend.services.chatbot_optimizer_new import OptimizedChatbot
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
            
            # Add follow-up section with proper spacing (no separation line)
            yield {'status': 'separator', 'chunk': '\n\n'}
            
            # Generate exploration follow-ups with proper formatting (no numbering)
            yield {'status': 'followup_question', 'chunk': "Would you like me to dive deeper into any specific aspect?"}
            yield {'status': 'separator', 'chunk': '\n\n'}
            yield {'status': 'followup_question', 'chunk': "Are there any particular implementation details you'd like to explore?"}
                
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
                enhanced_query = f"""Analyze this user query and provide an intelligent, engaging response like ChatGPT: "{latest_query}"

SMART RESPONSE GUIDELINES:

1. **CONVERSATIONAL TONE:**
   - Start naturally: "Got it ✅" or "Absolutely!" when appropriate
   - Be friendly, professional, and engaging
   - Show understanding: "Here's what I can help you with..."

2. **DYNAMIC FORMATTING:**
   - Use contextual headers: "Quick Overview", "Here's the breakdown", "Key Points"
   - Include relevant emojis: ✅ 🚀 💡 ⚡ 📊 (sparingly)
   - Ensure proper spacing: "Yes, we can definitely help you with that!"
   - Use proper headers with clear spacing

3. **SMART STRUCTURE:**
   - **Bold** key technologies, concepts, and important terms
   - Bullet points for features/benefits
   - Numbered lists for step-by-step processes  
   - Clear line breaks and proper formatting
   - Use meaningful headers (like "Quick Overview", "Key Benefits", "Implementation")

4. **CONTEXT DETECTION:**
   - If about services we provide (web dev, mobile apps, AI, cloud), start confidently
   - For technical questions, provide direct helpful information
   - For vague queries, ask for clarification friendly

5. **PROFESSIONAL POLISH:**
   - Keep responses concise but comprehensive (80-150 words)
   - End with practical next steps or takeaways
   - Structure like ChatGPT with natural flow

Query: {latest_query}"""
            else:
                # For follow-up responses, continue conversation naturally
                enhanced_query = f"""Continue this conversation naturally based on the context: "{latest_query}"

FORMATTING REQUIREMENTS:
- Use **bold** for key technologies, concepts, and important terms
- Use proper headers if needed (not multiple hash symbols)
- Keep response focused and helpful
- Structure with clear, actionable information

Previous conversation context: {conversation_history[-2:] if len(conversation_history) >= 2 else 'None'}

Query: {latest_query}"""

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

                # Apply structured formatting logic
                sections = re.split(r'\n\n', text_chunk)
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
            
            # Add spacing after main response (no separator lines)
            yield {'status': 'separator', 'chunk': '\n\n'}  # Just spacing, no lines

            # Save the main response to the conversation history
            if main_response:
                follow_up_manager.add_to_conversation_history(session_id, "assistant", main_response)

            # Generate intelligent follow-up questions based on conversation state
            conversation_history_data = follow_up_manager.get_session_data(session_id).get('conversation_history', [])
            user_messages = [msg for msg in conversation_history_data if msg.get("role") == "user"]
            
            if len(user_messages) <= 1:
                # For initial queries, ask broad requirement-gathering questions
                follow_up_prompt = f"""Based on this initial user query about: {latest_query}

Generate the SINGLE BEST follow-up question to understand their requirements better.

Instructions:
- Generate only ONE specific, actionable follow-up question
- Focus on gathering the most important information needed
- Keep the question conversational and brief
- Prioritize the most critical detail that would help provide a better response
- Do NOT number the question, provide it as plain text
- Make it the most valuable question to ask

Format:
[The single best follow-up question]"""
            else:
                # For ongoing conversations, ask targeted questions to fill gaps
                follow_up_prompt = f"""Analyze this ongoing conversation and generate the SINGLE BEST targeted follow-up question.

Recent Response: {main_response[:200]}...

Conversation History: {len(conversation_history_data)} exchanges

Instructions:
- Generate only ONE targeted follow-up question
- Identify the most important gap in understanding
- Ask about the detail that would most improve the final recommendation
- Keep the question focused and actionable
- If no important gaps exist, respond with: COMPLETE
- Do NOT number the question, provide it as plain text

Format:
[The single best targeted follow-up question]
OR
COMPLETE"""

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
                
                # Generate 2 contextual follow-ups based on the conversation state and context
                if len(user_messages) <= 1:
                    # For initial queries, generate context-specific follow-ups based on the query topic
                    context_keywords = latest_query.lower()
                    
                    if any(word in context_keywords for word in ['app', 'mobile', 'web', 'application']):
                        followups = [
                            "What specific features or functionalities are most important for your app?",
                            "What's your target platform - mobile, web, or both?"
                        ]
                    elif any(word in context_keywords for word in ['ai', 'machine learning', 'ml', 'chatbot', 'automation']):
                        followups = [
                            "What type of AI functionality do you envision for your project?",
                            "Do you have existing data that can be used for training?"
                        ]
                    elif any(word in context_keywords for word in ['website', 'site', 'web', 'portal']):
                        followups = [
                            "What's the main purpose of your website - e-commerce, corporate, or service-based?",
                            "Do you need any specific integrations or third-party services?"
                        ]
                    elif any(word in context_keywords for word in ['ecommerce', 'e-commerce', 'shop', 'store', 'payment']):
                        followups = [
                            "What payment methods and gateways do you want to integrate?",
                            "How many products do you expect to manage initially?"
                        ]
                    elif any(word in context_keywords for word in ['api', 'integration', 'system', 'backend']):
                        followups = [
                            "What systems or platforms do you need to integrate with?",
                            "What's your expected volume of API requests?"
                        ]
                    elif any(word in context_keywords for word in ['cloud', 'aws', 'azure', 'deployment']):
                        followups = [
                            "Do you have a preferred cloud provider or hosting environment?",
                            "What are your scalability and performance requirements?"
                        ]
                    else:
                        # Generic but relevant follow-ups for unclear queries
                        followups = [
                            "What's the primary goal you want to achieve with this project?",
                            "What's your preferred timeline for implementation?"
                        ]
                else:
                    # For ongoing conversations, generate follow-ups based on conversation context
                    recent_context = main_response[:200] + " " + latest_query
                    context_lower = recent_context.lower()
                    
                    if any(word in context_lower for word in ['implement', 'development', 'build', 'create']):
                        followups = [
                            "Would you like to discuss the technical implementation approach?",
                            "What's your preferred development timeline and budget range?"
                        ]
                    elif any(word in context_lower for word in ['feature', 'functionality', 'capability']):
                        followups = [
                            "Are there any specific features you'd like to prioritize first?",
                            "Do you need any advanced or custom functionalities?"
                        ]
                    elif any(word in context_lower for word in ['technology', 'platform', 'framework']):
                        followups = [
                            "Do you have any technology preferences or constraints?",
                            "Are there existing systems this needs to integrate with?"
                        ]
                    else:
                        # Default ongoing conversation follow-ups
                        followups = [
                            "Are there any specific aspects you'd like me to elaborate on?",
                            "What would you like to focus on next?"
                        ]
                
                # Send follow-ups with word-level streaming (no numbering)
                for followup in followups:
                    # Add proper spacing before each follow-up
                    yield {'status': 'separator', 'chunk': '\n\n'}
                    
                    # Keep markdown formatting for ChatGPT-like display
                    # Only remove headers, preserve **bold** and *italic*
                    clean_followup = re.sub(r'#{1,6}\s*', '', followup)  # Remove headers only
                    
                    # Stream follow-up word by word with proper spacing
                    # Send follow-up question without word-by-word streaming
                    followup_text = ' '.join(clean_followup.split())  # Clean and rebuild text
                    yield {'status': 'followup_question', 'chunk': followup_text}
                
                # Add final spacing after follow-ups
                yield {'status': 'separator', 'chunk': '\n'}
                    
            except Exception as e:
                logger.error(f"Error generating follow-ups: {e}")
                # Fallback suggestions and follow-ups for initial queries only
                user_messages = [msg for msg in conversation_history_data if msg.get("role") == "user"]
                if len(user_messages) <= 1:
                    yield {'status': 'suggestion', 'chunk': "Consider starting with a basic prototype to test core functionality"}
                    yield {'status': 'separator', 'chunk': '\n\n'}
                    yield {'status': 'followup_question', 'chunk': "What specific features are most important to you?"}
                    yield {'status': 'separator', 'chunk': '\n\n'}
                    yield {'status': 'followup_question', 'chunk': "What's your target platform preference?"}

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
