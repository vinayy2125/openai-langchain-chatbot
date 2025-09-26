from sentence_transformers import SentenceTransformer
from app.db import redis_operations as redis_crud

# Download from the 🤗 Hub
model = SentenceTransformer("google/embeddinggemma-300m")

# Run inference with queries and documents
query = "Which planet is known as the Red Planet?"
documents = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet."
]
query_embeddings = model.encode_query(query)
document_embeddings = model.encode_document(documents)
print(query_embeddings.shape, document_embeddings.shape)
# (768,) (4, 768)

# Compute similarities to determine a ranking
similarities = model.similarity(query_embeddings, document_embeddings)
print(similarities)
# tensor([[0.3011, 0.6359, 0.4930, 0.4889]])


def get_embedding(text: str) -> list[float]:
    """Get embedding vector for a given text using the loaded model."""
    emb = model.encode_query(text)
    return emb.tolist()

def generate_and_store_embedding(r, text: str, metadata: dict = None) -> str:
    """Generate embedding for the given text and store it in Redis with optional metadata.

    Args:
        r: Redis client instance.
        text: The text to generate an embedding for.
        metadata: Optional dictionary of metadata to store alongside the embedding.

    Returns:
        The Redis key under which the document is stored.
    """
    embedding = get_embedding(text)
    import uuid
    doc_id = str(uuid.uuid4())
    key = f"{redis_crud.PREFIX}{doc_id}"
    doc = {
        "text": text,
        "embedding": embedding,
        "metadata": metadata or {}
    }
    r.json().set(key, '$', doc)
    return doc_id