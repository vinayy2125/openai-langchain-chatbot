# streamlit_app/chat_logic.py
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

load_dotenv()

# --- Embeddings and Vector Store ---
embedding_model = OpenAIEmbeddings()
vector_store = FAISS.load_local(
    "vectorstore/faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
retriever = vector_store.as_retriever(search_kwargs={"k": 4})

# --- LLM Setup ---
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# --- Prompt Template with history ---
prompt_template = PromptTemplate.from_template(
    """
    You are a helpful assistant.

    Here is the conversation so far:
    {history}

    Here is the relevant context from the knowledge base:
    {context}

    User's latest question:
    {question}

    Please answer based on the context and the conversation history.
    If the answer is not present in the context, respond with:
    "No relevant content found. Please visit the website directly."
    """
)

def get_answer_from_context(query: str, chat_history: list):
    """
    Get an answer using FAISS context + chat history.
    chat_history: list of (role, message) tuples, e.g. [("user", "..."), ("bot", "...")]
    """
    # 1. Retrieve context from FAISS
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join(doc.page_content for doc in docs)

    # 2. Format chat history for the prompt
    history_text = "\n".join(f"{'User' if role == 'user' else 'Assistant'}: {msg}" for role, msg in chat_history)

    # 3. Create the final prompt
    prompt = prompt_template.format(
        history=history_text,
        context=context_text,
        question=query
    )

    # 4. Get the model's answer
    answer = llm.predict(prompt)

    if "No relevant content found" in answer:
        return answer, False
    return answer, True
