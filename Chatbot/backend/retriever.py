# streamlit_app/retriever.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

def get_retriever(k: int = 4):
    """
    Loads the FAISS vector store and returns a retriever.
    """
    embedding_model = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "vectorstore/faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={"k": k})

# Create a default retriever for app-wide use
retriever = get_retriever()
