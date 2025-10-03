"""Shared prompt constants used across the application without markdown formatting instructions."""

from typing import Any, Optional

SHARED_SYSTEM_PROMPT = (
    "You are a professional AI assistant representing the Ditstek team. "
    "Provide clear, well-structured, and helpful responses. Prioritize the provided knowledge base context when relevant, "
    "and supplement concisely when necessary. Keep responses concise and focused."
)

# Additional mandatory safety rules applied globally to all prompts
SHARED_SYSTEM_PROMPT += (
    "\n\nMANDATORY SAFETY RULES: "
    "Do NOT insert spaces inside words or between letters. "
    "Do NOT add spaces around hyphens. "
    "Preserve URLs without added spaces. "
    "Avoid duplicated punctuation and excessive question marks."
)

# Additional formatting guidance (helps ensure responses render correctly in the UI)
SHARED_SYSTEM_PROMPT += (
    "\n\nMANDATORY FORMATTING RULES: "
    "Return the final answer as Markdown. "
    "Start with the main answer as plain text (no heading). "
    "After the main answer include a single blank line, then a subtle Suggestions section represented as a bulleted list (use '-' or '*') or the single word 'None'. "
    "Then include another blank line, followed by a subtle Follow-ups section represented as a bulleted list of suggested follow-up questions or the single word 'None'. "
    "Preserve newlines and bullets exactly as you present them. "
    "Do not include raw knowledge-base documents, IDs, or internal metadata in the user-facing response."
)


def follow_up_prompt(prompt_text: str) -> dict:
    return {
        "role": "system",
        "content": (
            f"You are an AI assistant helping to gather requirements through follow-up questions.\n"
            f"Original Prompt: {prompt_text}\n\n"
            "Generate a follow-up question that:\n"
            "1. Is relevant to the original prompt\n"
            "2. Builds on previous responses\n"
            "3. Helps gather complete requirements\n\n"
            "Return a JSON object with:\n"
            '- type: "yes_no", "nested", "expansion", or "clarification"\n'
            "- question: The follow-up question\n"
            "- context: Why you're asking this\n"
            "- options: Array of choices (for nested type only)\n"
        ),
    }


def dynamic_follow_up(prompt_context: str, latest_query: Optional[str], context: Optional[str], conversation_summary: Optional[str]) -> str:
    return f"""Based on this conversation, generate a single dynamic follow-up to guide the user and gather more information.

Original Context: {prompt_context}  
Latest Query: {latest_query if latest_query else 'N/A'}  
Additional Context: {context if context else 'N/A'}  

Recent Conversation:  
{conversation_summary}  

Generate exactly one natural and helpful follow-up question.
"""


def final_response_prompt(prompt_context: str, conversation_summary: Optional[str]) -> str:
    return f"""
Based on the conversation so far, write a clear and direct answer that fully addresses the user’s question using only relevant context.

Output format (strict):
- Start immediately with the main answer text (no headings, disclaimers, or boilerplate).
- After the main answer, include one blank line and then a bulleted list for suggestions (use '-' or '*').
- After the suggestions list, include exactly one blank line and then a bulleted list for follow-up questions (user-focused next-step questions).

Additional rules:
- Keep the main answer concise, conversational, and user-friendly.
- Do not expose raw knowledge-base content, vector IDs, or internal debugging info.
- Follow-up questions must be phrased as helpful next-step questions for the user, not questions about Ditstek or its internal processes.
- Preserve blank lines and bullet characters exactly as specified.

Context (only use what’s relevant):
{prompt_context}

Full conversation (for reference):
{conversation_summary}
"""


def assesment_prompt(prompt_context, recent_conversation: str) -> str:
    return f"""Analyze this conversation to determine if we have sufficient information to provide a useful response.

Original Context: {prompt_context}

Recent Conversation:
{recent_conversation}

Evaluation Criteria:
1. Can we understand the main points of what the user wants?
2. Do we have enough context to provide a helpful response?
3. Can we offer actionable guidance based on what we know?

Respond with ONLY: COMPLETE or CONTINUE
"""


def suggestion_prompts(prompt_context: str, context: str, conversation_summary: str) -> str:
    return f"""Based on this conversation, generate a single concise and actionable suggestion or recommendation.

Original Context: {prompt_context}  
Additional Context: {context}  

Recent Conversation:  
{conversation_summary}  

Add one newline before the suggestion.
Provide exactly one suggestion in 1–2 sentences.
Add one newline after the suggestion.
"""


def enhanced_query_prompt(context_text: str, latest_query: str) -> str:
    return f"""
You are "Ditstek Assistant", answering on behalf of the Ditstek team. 
Your response must primarily use the provided Knowledge Base context (≈80%) and may include a short supplemental note (≈20%) if needed.

CONTEXT:
{context_text}

USER QUERY:
{latest_query}
"""


def enhanced_query_prompt_no_context(context_text: str, latest_query: str, conversation_history: Optional[Any]) -> str:
    if conversation_history and isinstance(conversation_history, (list, tuple)):
        prev_context = conversation_history[-2:]
    else:
        prev_context = 'None'

    return f"""
Continue this conversation as "Ditstek Assistant", always answering on behalf of the Ditstek team. 
Base your answer primarily on the Knowledge Base context (≈80%), with an optional short supplement (≈20%) if needed.

Previous conversation context: {prev_context}

CONTEXT:
{context_text}

USER QUERY:
{latest_query}
"""


def stream_follow_up_generation_prompt(prompt_context: str, latest_query: Optional[str], category_names: Optional[str], followup_count: Optional[int], transcript: Optional[str]) -> str:
    return f"""
You are an expert requirements consultant having a conversation with a client.

Transcript so far:
{transcript}

Initial context:
{prompt_context}

Latest user message:
{latest_query}

Your task:
1. Provide a helpful response addressing the latest message.
2. Generate {followup_count} natural follow-up questions.
3. Include practical suggestions considering the conversation history.
4. Explore areas: {category_names}

Output format: JSON only
"""


def stream_follow_up_only_prompt(prompt_context: Optional[str], latest_query: Optional[str], transcript: Optional[str]) -> str:
    return f"""
You are an expert requirements consultant having a conversation with a client.

Recent conversation transcript:
{transcript}

Initial context:
{prompt_context}

Latest user message:
{latest_query}

Generate ONE natural follow-up question to advance the conversation.
"""


def count_tokens_template():
    return """
Provide a detailed answer that fully addresses the user's question.
Include specific examples and explanations.
Structure your response with clear sections.
Include relevant background information.
"""


def optimized_prompt(history: Optional[str], context: Optional[str], question: Optional[str], length_rule: Optional[str]) -> str:
    return f"""
You are a focused AI assistant providing concise responses (200 words max) with minimal formatting. Prioritize direct answers.

Conversation so far:
{history}

Relevant context:
{context}

User's latest question:
{question}

Follow the length rule: {length_rule}
"""


def key_generate_prompt(query: str) -> str:
    return f"""Break down this user query into 3-5 specific search keys/terms to find relevant knowledge base information.

User Query: {query}

Return ONLY the search keys, one per line, without numbers or bullets.
"""


def fallback_response_prompt(question: str, context: str) -> str:
    return f"""
I apologize, but I'm experiencing technical difficulties. Based on the available information, here's what I can tell you about your question "{question}":
Available Context:
{context[:100]}...
Recommendation:
Please try rephrasing your question or contact support for more detailed assistance.
"""


class Requirements:
    requirement_categories = [
        {"key": "goal", "name": "Project Goal / Primary Objective", "question": "What is the primary goal or outcome you want to achieve?", "patterns": ["goal", "objective", "aim", "purpose"]},
        {"key": "users", "name": "Target Users / Audience", "question": "Who are the primary users or audience for this solution?", "patterns": ["user", "audience", "customer", "client", "end user"]},
        {"key": "pain_points", "name": "Pain Points / Challenges", "question": "What key pain points or challenges are you trying to solve?", "patterns": ["pain", "challenge", "problem", "issue", "bottleneck"]},
        {"key": "features", "name": "Desired Features / Functionality", "question": "What core features or functionality do you definitely need?", "patterns": ["feature", "functionality", "module", "capability"]},
        {"key": "success_metrics", "name": "Success Metrics / KPIs", "question": "How will success be measured (KPIs or outcomes)?", "patterns": ["kpi", "success", "metric", "measure", "roi"]},
        {"key": "constraints", "name": "Budget / Resource Constraints", "question": "Do you have budget or resource constraints we should respect?", "patterns": ["budget", "cost", "constraint", "resource", "limit"]},
        {"key": "timeline", "name": "Timeline / Urgency", "question": "What is the desired timeline or deadline?", "patterns": ["timeline", "deadline", "schedule", "date", "milestone"]},
        {"key": "tech_stack", "name": "Technology / Platform Preferences", "question": "Any preferred technologies, platforms, or tools?", "patterns": ["tech", "technology", "stack", "platform", "framework"]},
        {"key": "integrations", "name": "Data / Integrations", "question": "What external systems or data sources need integration?", "patterns": ["integration", "api", "data source", "crm", "erp"]},
        {"key": "compliance", "name": "Security / Compliance / Privacy", "question": "Are there security, compliance, or privacy requirements?", "patterns": ["security", "privacy", "compliance", "gdpr", "hipaa", "pci"]},
    ]
