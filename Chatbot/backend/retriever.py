import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

INDEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore", "faiss_index"))
embedding_model = OpenAIEmbeddings()

# Load FAISS index
vectorstore = FAISS.load_local(
    INDEX_DIR,
    embedding_model,
    allow_dangerous_deserialization=True  # required for some LC versions
)

# Use MMR to reduce duplicate-y chunks, fetch wider, return top-k diverse
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,         # final docs to return
        "fetch_k": 24,  # pool to choose diverse results from
        "lambda_mult": 0.5  # 0=diversity, 1=similarity — 0.5 is a good balance
    }
)
