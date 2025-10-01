import numpy as np
from app.config import get_redis

r = get_redis
key = "session:universal_session_id"

print("🔍 Fetching universal session...")
session = r.json().get(key)
if not session:
    print("❌ No session found.")
    exit(1)

chunks = session.get("queries", [])
print(f"📦 Migrating {len(chunks)} chunks to individual hashes...")

for i, chunk in enumerate(chunks):
    chunk_id = chunk.get("query", f"chunk_{i:06d}")
    text = chunk.get("response", "")
    emb_list = chunk.get("query_embedding", [])

    if not emb_list:
        print(f"⚠️ Skipping chunk {chunk_id}: no embedding")
        continue

    try:
        # Convert list of floats → float32 bytes
        emb_array = np.array(emb_list, dtype=np.float32)
        emb_bytes = emb_array.tobytes()
    except Exception as e:
        print(f"❌ Failed to convert embedding for {chunk_id}: {e}")
        continue

    # Store as individual hash
    doc_key = f"kb:{chunk_id}"
    r.hset(doc_key, mapping={
        "session_id": "universal_session_id",
        "chunk_id": chunk_id,
        "response": text,
        "embedding": emb_bytes  # ✅ Now it's valid bytes
    })

    if i % 100 == 0:
        print(f"✅ Processed {i}/{len(chunks)}")

print("🎉 Migration complete!")