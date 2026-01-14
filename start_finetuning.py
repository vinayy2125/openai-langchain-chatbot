"""
Start fine-tuning job with OpenAI.

This script uploads training data and initiates a fine-tuning job.
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Configuration
TRAINING_FILE = 'train_final.jsonl'
BASE_MODEL = 'gpt-4o-mini-2024-07-18'  # Latest GPT-4o-mini for fine-tuning
SUFFIX = 'ditstek-conversational-v2'   # Model suffix for identification

def main():
    print("=" * 70)
    print("OPENAI FINE-TUNING JOB CREATOR")
    print("=" * 70)
    
    # Initialize client
    client = OpenAI()
    
    # Verify file exists
    if not os.path.exists(TRAINING_FILE):
        print(f"\n❌ Training file not found: {TRAINING_FILE}")
        return
    
    # Count examples
    with open(TRAINING_FILE, 'r', encoding='utf-8') as f:
        example_count = sum(1 for line in f if line.strip())
    
    print(f"\n[CONFIGURATION]")
    print(f"   Training file: {TRAINING_FILE}")
    print(f"   Examples: {example_count}")
    print(f"   Base model: {BASE_MODEL}")
    print(f"   Suffix: {SUFFIX}")
    
    # Upload file
    print(f"\n[1. UPLOADING FILE]")
    with open(TRAINING_FILE, 'rb') as f:
        file_response = client.files.create(
            file=f,
            purpose='fine-tune'
        )
    
    file_id = file_response.id
    print(f"   ✓ File uploaded: {file_id}")
    
    # Create fine-tuning job
    print(f"\n[2. CREATING FINE-TUNING JOB]")
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=BASE_MODEL,
        suffix=SUFFIX,
        hyperparameters={
            "n_epochs": 3
        }
    )
    
    job_id = job.id
    print(f"   ✓ Job created: {job_id}")
    print(f"   Status: {job.status}")
    
    # Save job ID for tracking
    with open('finetuning_job_id.txt', 'w') as f:
        f.write(job_id)
    
    print(f"\n[3. JOB DETAILS]")
    print(f"   Job ID: {job_id}")
    print(f"   Model: {BASE_MODEL}")
    print(f"   Status: {job.status}")
    print(f"   Created: {job.created_at}")
    
    print(f"\n" + "=" * 70)
    print(f"FINE-TUNING JOB STARTED")
    print(f"=" * 70)
    
    print(f"""
Job ID saved to: finetuning_job_id.txt

To check status, run:
  python check_finetuning_status.py

Or use OpenAI dashboard:
  https://platform.openai.com/finetune

Expected completion time: 15-30 minutes
""")


if __name__ == "__main__":
    main()
