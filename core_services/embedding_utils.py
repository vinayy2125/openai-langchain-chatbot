# core_services/embedding_utils.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("google/embeddinggemma-300m")

def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a given text using the loaded model."""
    emb = model.encode_query(text)
    return emb.tolist()
