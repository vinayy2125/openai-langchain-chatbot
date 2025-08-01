import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List

# Initialize the embedding model (you can use 'all-MiniLM-L6-v2' for fast, general-purpose embeddings)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
_embedding_model = None


def get_embedding_model():
    """
    Load and cache the sentence transformer embedding model.
    Returns a SentenceTransformer instance.
    """
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


# Initialize Chroma client and collection
_chroma_client = None
_collection = None


def get_chroma_collection(collection_name="doc_chunks"):
    """
    Initialize and return a Chroma collection for storing embeddings.
    """
    global _chroma_client, _collection
    if _chroma_client is None:
        _chroma_client = chromadb.Client(Settings(persist_directory=".chroma_store"))
    if _collection is None:
        _collection = _chroma_client.get_or_create_collection(collection_name)
    return _collection


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using the sentence transformer model.
    Returns a list of embedding vectors.
    """
    model = get_embedding_model()
    return model.encode(texts, show_progress_bar=False).tolist()


def add_chunks_to_chroma(
    chunks: List[str], metadatas: List[dict] = None, collection_name="doc_chunks"
):
    """
    Add text chunks and their embeddings to the Chroma collection.
    Optionally, attach metadata to each chunk.
    """
    collection = get_chroma_collection(collection_name)
    embeddings = embed_texts(chunks)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas
    )


def search_similar_chunks(
    query: str, top_k=3, collection_name="doc_chunks"
) -> List[str]:
    """
    Search for the top_k most similar chunks to the query using Chroma.
    Returns a list of the most relevant chunk texts.
    """
    collection = get_chroma_collection(collection_name)
    query_embedding = embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results["documents"] else []
