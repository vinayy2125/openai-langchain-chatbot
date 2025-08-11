from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

# --- Load environment variables ---
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

# --- LLM Setup (OpenAI) ---
llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)

# --- Prompt Template ---
prompt_template = PromptTemplate.from_template(
    """
    You are a helpful assistant. Use the below context to answer the question:

    Context:
    {context}

    Question:
    {question}

    If the answer is not present in the context, respond with:
    "No relevant content found. Please visit the website directly."
    """
)

# --- QA Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt_template}
)

# --- Core Function ---
def get_answer_from_context(query: str, source_url: str = None):
    result = qa_chain.invoke({"query": query})
    answer = result.get("result")
    
    if "No relevant content found" in answer:
        return answer, False
    return answer, True
