"""
Start fine-tuning job for website data.
Target: fine_tuning_data/website_finetune.jsonl
Model: gpt-4o-mini-2024-07-18
"""

import os
import sys
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
TRAINING_FILE = r'fine_tuning_data/website_finetune.jsonl'
BASE_MODEL = 'gpt-4o-mini-2024-07-18'
SUFFIX = 'ditstek-website-v1'

def main():
    print(f"[*] Initializing Fine-Tuning Job for: {TRAINING_FILE}")
    
    # Check API Key
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not found in environment.")
        return

    client = OpenAI()

    # 1. Validation Check
    if not os.path.exists(TRAINING_FILE):
        print(f"[ERROR] File not found: {TRAINING_FILE}")
        return

    # 2. Upload File
    print(f"[*] Uploading file to OpenAI...")
    try:
        with open(TRAINING_FILE, 'rb') as f:
            file_response = client.files.create(
                file=f,
                purpose='fine-tune'
            )
        file_id = file_response.id
        print(f"    - File ID: {file_id}")
    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        return

    # 3. Create Job
    print(f"[*] Creating fine-tuning job (Model: {BASE_MODEL})...")
    try:
        job = client.fine_tuning.jobs.create(
            training_file=file_id,
            model=BASE_MODEL,
            suffix=SUFFIX
        )
        print(f"    - Job ID: {job.id}")
        print(f"    - Status: {job.status}")
        print(f"\n[SUCCESS] Job started successfully!")
        print(f"Monitor at: https://platform.openai.com/finetune/{job.id}")
        
    except Exception as e:
        print(f"[ERROR] Job creation failed: {e}")

if __name__ == "__main__":
    main()
