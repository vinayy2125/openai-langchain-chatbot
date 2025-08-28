# streamlit_app/backend/llm_client.py
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage

# Define LLM instance here so it always exists
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2, max_tokens=None)

prompt_template = PromptTemplate.from_template("""
You are a knowledgeable and thorough assistant providing comprehensive information.
Your goal is to give detailed, well-structured answers that fully address the user's question.

Conversation so far:
{history}

Relevant context from the knowledge base:
{context}

User's latest question:
{question}

Instructions for responding:
1. Provide a comprehensive answer that thoroughly addresses all aspects of the question
2. Include specific details, examples, and explanations where appropriate
3. Structure your response with clear sections using markdown formatting
4. Use **bold** for key terms and concepts
5. When relevant, include bullet points or numbered lists to organize information
6. If the context contains multiple relevant pieces of information, synthesize them into a cohesive response
7. If context is limited or partial, still provide the most complete answer possible using your general knowledge
8. Do NOT be overly brief - aim for thoroughness and completeness
9. Include relevant background information that helps understand the topic

Format your response with:
- A clear introductory paragraph
- Well-organized body sections with appropriate headings
- A brief conclusion when appropriate
""")

def call_llm_with_context(context: str, history: str, question: str, detail_level: str = "high") -> str:
    """
    Calls the LLM with history, context, and user question.
    
    Args:
        context: The context information from knowledge base
        history: Conversation history
        question: User's question
        detail_level: Controls response detail ("low", "medium", "high")
    """
    try:
        # Add detail instruction based on level
        detail_instruction = {
            "low": "Provide a brief but complete answer to the question.",
            "medium": "Provide a moderately detailed answer with key points and explanations.",
            "high": "Provide a comprehensive, thorough answer with detailed explanations, examples, and multiple perspectives where relevant."
        }.get(detail_level, "Provide a detailed answer.")
        
        # Create a temporary prompt template with detail instruction
        temp_prompt_template = PromptTemplate.from_template(f"""
You are a knowledgeable and thorough assistant providing comprehensive information.
Your goal is to give detailed, well-structured answers that fully address the user's question.

Conversation so far:
{{history}}

Relevant context from the knowledge base:
{{context}}

User's latest question:
{{question}}

Detail instruction: {detail_instruction}

Instructions for responding:
1. Provide a comprehensive answer that thoroughly addresses all aspects of the question
2. Include specific details, examples, and explanations where appropriate
3. Structure your response with clear sections using markdown formatting
4. Use **bold** for key terms and concepts
5. When relevant, include bullet points or numbered lists to organize information
6. If the context contains multiple relevant pieces of information, synthesize them into a cohesive response
7. If context is limited or partial, still provide the most complete answer possible using your general knowledge
8. Do NOT be overly brief - aim for thoroughness and completeness
9. Include relevant background information that helps understand the topic

Format your response with:
- A clear introductory paragraph
- Well-organized body sections with appropriate headings
- A brief conclusion when appropriate
""")
        
        prompt = temp_prompt_template.format(history=history, context=context, question=question)
        
        # 🔎 Debug: show constructed prompt
        print("[DEBUG] Full LLM prompt (first 1200 chars):\n"
              f"{prompt[:1200]}{'...' if len(prompt) > 1200 else ''}\n[DEBUG] End prompt\n")
        
        raw_answer = llm.invoke(prompt)
        return raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)
    except Exception as e:
        return f"[Error invoking LLM: {e}]"
