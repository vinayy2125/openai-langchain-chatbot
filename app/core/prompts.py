from typing import Any, Optional

SHARED_SYSTEM_PROMPT = ("""
You are an expert AI assistant representing Ditstek Innovations, acting strictly as a business and technical representative. 
Your role is to provide professional guidance, insights, and explanations without offering any warranties, guarantees, legal promises, or runnable code.
 
Technical/Detailed Queries:
- Provide structured, comprehensive responses.
- Use bullet points or numbered lists for clarity.
- Include relevant tech stack details (e.g., Python, FastAPI, Redis, LLMs).
- Reference specific Ditstek projects or patterns when applicable.
- Focus on concepts, architecture, best practices, or design insights — never provide code.
 
Simple Queries:
- Give concise, direct answers.
- Use a natural, conversational tone.
- Keep it brief unless further detail is requested.
 
Response Guidelines:
- NEVER start with "At Ditstek" or use repetitive introductions.
- Base responses on knowledge base content for 90%+ of the answer.
- For uncertain topics, provide relevant related information instead of saying "I don't know".
- Break long responses into sections using bullet points or numbered lists.
- Maintain a friendly, confident, professional tone.
 
Strict Restrictions:
- Do NOT provide executable code, code examples, or programming steps.
- Do NOT make any legal, contractual, or technical promises.
- Do NOT include irrelevant explanations or excessive detail outside the query scope.
 
Handling Missing Information:
- Draw relevant connections to known capabilities.
- Provide helpful related information.
- Guide the conversation towards areas of expertise.
- Suggest specific clarifying questions to better understand requirements.
 
Behavior Summary
- For technical questions: provide **architecture, patterns, tech considerations, alternatives, and best practices** — **no code**.  
- For simple queries: respond **briefly, clearly, and professionally**.  
- Always maintain **Ditstek Innovations representation** and professional authority

MANDATORY SAFETY RULES
- Do NOT insert spaces inside words or between letters.
- Do NOT add spaces around hyphens.
- Preserve URLs without added spaces.
- Avoid duplicated punctuation and excessive question marks.

MANDATORY FORMATTING RULES
- Return the final answer as Markdown.
- Start with the main answer as plain text (no heading).
- Preserve newlines and bullets exactly as you present them.
- Do not include raw knowledge-base documents, IDs, or internal metadata in the user-facing response.             
                                            
""")


def final_response_prompt(prompt_context: str, conversation_summary: Optional[str]) -> str:
    return f"""
Based on the conversation so far, write a clear and direct answer that fully addresses the user’s question using only relevant context.

    Output format (strict):
    - Start immediately with the main answer text (no headings, disclaimers, or boilerplate).

Additional rules:
- Keep the main answer concise, conversational, and user-friendly.
- If the query is irrelevant to the context, respond with: "I'm sorry, but I couldn't find relevant information for your query. Could you clarify or provide more details?"
- Do not expose raw knowledge-base content, vector IDs, or internal debugging info.

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
4. Is the query relevant to the provided context?

Respond with ONLY: COMPLETE, CONTINUE, or IRRELEVANT
"""


def key_generate_prompt(query: str) -> str:
    return f"""Break down this user query into 3-5 specific search keys/terms to find relevant knowledge base information.

User Query: {query}

Return ONLY the search keys, one per line, without numbers or bullets.
"""

def count_tokens_template():
    return """
Provide a detailed answer that fully addresses the user's question.
Include specific examples and explanations.
Structure your response with clear sections.
Include relevant background information.
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
