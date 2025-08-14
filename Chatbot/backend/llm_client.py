# streamlit_app/llm_client.py
from langchain.chat_models.openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import AIMessage
from dotenv import load_dotenv

load_dotenv()

# LLM setup
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# Prompt template
prompt_template = PromptTemplate.from_template("""
You are a helpful assistant.

Conversation so far:
{history}

Relevant context from the knowledge base:
{context}

User's latest question:
{question}

Rules:
- Only use the provided context and conversation history.
- If the answer is not in the context, respond exactly:
  "No relevant content found. Please visit the website directly."
- Do not use outside knowledge.

Format your final answer with **bold** for key points and use structured formatting where helpful.
""")

def call_llm_with_context(context: str, history: str, question: str) -> str:
    """
    Calls the LLM with formatted prompt and returns the response content.
    """
    prompt = prompt_template.format(
        history=history,
        context=context,
        question=question
    )

    raw_answer = llm.invoke(prompt)
    return raw_answer.content if isinstance(raw_answer, AIMessage) else str(raw_answer)
