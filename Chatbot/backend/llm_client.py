# streamlit_app/backend/llm_client.py
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage

# Define LLM instance here so it always exists
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

prompt_template = PromptTemplate.from_template("""
You are a helpful assistant.

Conversation so far:
{history}

Relevant context from the knowledge base:
{context}

User's latest question:
{question}

Rules:
- Prefer answers grounded in the context, but if context is partial or incomplete,
  still provide the best possible response using your general knowledge.
- If context is empty, fall back to your general knowledge (do NOT refuse).
- Be clear and concise.

Format your final answer with **bold** for key points and structured formatting where helpful.
""")

def call_llm_with_context(context: str, history: str, question: str) -> str:
    """
    Calls the LLM with history, context, and user question.
    """
    try:
        prompt = prompt_template.format(history=history, context=context, question=question)

        # 🔎 Debug: show constructed prompt
        print("[DEBUG] Full LLM prompt (first 1200 chars):\n"
              f"{prompt[:1200]}{'...' if len(prompt) > 1200 else ''}\n[DEBUG] End prompt\n")

        raw_answer = llm.invoke(prompt)
        return raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)

    except Exception as e:
        return f"[Error invoking LLM: {e}]"
