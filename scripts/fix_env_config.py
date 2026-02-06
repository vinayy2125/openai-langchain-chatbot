import os
from pathlib import Path

def fix_env():
    env_path = Path(".env")
    if not env_path.exists():
        print("❌ .env file not found!")
        return

    print(f"Reading {env_path}...")
    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        print(f"❌ Failed to read .env: {e}")
        return

    new_lines = []
    keys_updated = set()
    
    # Config to enforce
    config = {
        "CHROMA_SERVER_URL": "http://localhost:8001",
        "CHROMA_USE_HTTP_CLIENT": "true",
        "CHROMA_COLLECTION_NAME": "website_embeddings"
    }

    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            new_lines.append(line)
            continue
        
        updated = False
        for key, value in config.items():
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}")
                keys_updated.add(key)
                print(f"✅ Updated {key}")
                updated = True
                break
        
        if not updated:
            new_lines.append(line)

    # Add missing keys
    for key, value in config.items():
        if key not in keys_updated:
            new_lines.append(f"{key}={value}")
            print(f"➕ Added {key}")

    # Write back
    try:
        env_path.write_text("\n".join(new_lines), encoding="utf-8")
        print("✅ .env configuration saved successfully.")
    except Exception as e:
        print(f"❌ Failed to write .env: {e}")

if __name__ == "__main__":
    fix_env()
