"""
UC1 Fine-Tuning Job Manager

Upload training data and start fine-tuning job with OpenAI API.
Target model: gpt-4.1-mini
"""

import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("❌ OpenAI library not installed. Run: pip install openai")
    sys.exit(1)


def main():
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        print("\nSet it with:")
        print('  $env:OPENAI_API_KEY="sk-your-key-here"')
        sys.exit(1)
    
    print("✓ OpenAI API key found")
    
    client = OpenAI(api_key=api_key)
    
    # File paths
    train_file = Path("fine_tuning_data/train.jsonl")
    val_file = Path("fine_tuning_data/validation.jsonl")
    
    if not train_file.exists():
        print(f"❌ Training file not found: {train_file}")
        sys.exit(1)
    
    if not val_file.exists():
        print(f"❌ Validation file not found: {val_file}")
        sys.exit(1)
    
    print(f"\n📁 Files to upload:")
    print(f"  Training: {train_file} ({train_file.stat().st_size:,} bytes)")
    print(f"  Validation: {val_file} ({val_file.stat().st_size:,} bytes)")
    
    # Upload training file
    print("\n⬆️  Uploading training file...")
    with open(train_file, "rb") as f:
        train_response = client.files.create(file=f, purpose="fine-tune")
    train_file_id = train_response.id
    print(f"  ✓ Training file uploaded: {train_file_id}")
    
    # Upload validation file
    print("\n⬆️  Uploading validation file...")
    with open(val_file, "rb") as f:
        val_response = client.files.create(file=f, purpose="fine-tune")
    val_file_id = val_response.id
    print(f"  ✓ Validation file uploaded: {val_file_id}")
    
    # Start fine-tuning job
    print("\n🚀 Starting fine-tuning job...")
    print("  Model: gpt-4.1-mini")
    print("  Suffix: uc1-chatbot")
    
    job = client.fine_tuning.jobs.create(
        training_file=train_file_id,
        validation_file=val_file_id,
        model="gpt-4.1-mini-2025-04-14",
        suffix="uc1-chatbot",
        hyperparameters={
            "n_epochs": 3
        }
    )
    
    print(f"\n✅ Fine-tuning job started!")
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    print(f"\n📊 Monitor progress at:")
    print(f"  https://platform.openai.com/finetune/{job.id}")
    print(f"\nOr run: python fine_tuning_data/check_job_status.py {job.id}")
    
    # Save job ID
    with open("fine_tuning_data/job_id.txt", "w") as f:
        f.write(job.id)
    print(f"\n💾 Job ID saved to: fine_tuning_data/job_id.txt")


if __name__ == "__main__":
    main()
