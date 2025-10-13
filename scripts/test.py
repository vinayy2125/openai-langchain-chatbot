# scripts/test_retrieval.py
import numpy as np
from redis.commands.search.query import Query
from app.config import get_redis
from core_services.embedding_utils import get_embedding

r = get_redis

# 1. Embed a real query
query = "Why hire UI/UX Developers From Dits We are a team."
emb = get_embedding(query)
vec_bytes = np.array(emb, dtype=np.float32).tobytes()

# 2. Search in universal KB
q = Query('(@session_id:universal_session_id)=>[KNN 3 @embedding $vec AS score]') \
    .return_fields("response", "chunk_id") \
    .dialect(2)

res = r.ft("kb_index").search(q, query_params={"vec": vec_bytes})

# 3. Print results
print(f"\n✅ Found {len(res.docs)} relevant chunks for: '{query}'\n")
for i, doc in enumerate(res.docs, 1):
    print(f"{i}. Chunk: {doc.chunk_id}")
    print(f"   Text: {doc.response[:250]}...\n")