"""
Check status of fine-tuning job.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def main():
    print("=" * 70)
    print("FINE-TUNING JOB STATUS")
    print("=" * 70)
    
    # Load job ID
    job_id_file = 'finetuning_job_id.txt'
    if not os.path.exists(job_id_file):
        print(f"\n❌ No job ID file found: {job_id_file}")
        print("Run start_finetuning.py first.")
        return
    
    with open(job_id_file, 'r') as f:
        job_id = f.read().strip()
    
    print(f"\n[JOB ID]: {job_id}")
    
    # Initialize client
    client = OpenAI()
    
    # Get job status
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    print(f"\n[STATUS]")
    print(f"   Status: {job.status}")
    print(f"   Model: {job.model}")
    print(f"   Created: {job.created_at}")
    
    if job.finished_at:
        print(f"   Finished: {job.finished_at}")
    
    if job.fine_tuned_model:
        print(f"\n[FINE-TUNED MODEL READY]")
        print(f"   Model name: {job.fine_tuned_model}")
        print(f"\n   Use this model ID in your application!")
        
        # Save model name
        with open('finetuned_model_name.txt', 'w') as f:
            f.write(job.fine_tuned_model)
        print(f"   Saved to: finetuned_model_name.txt")
    
    if job.error:
        print(f"\n[ERROR]")
        print(f"   {job.error}")
    
    # Get events
    print(f"\n[RECENT EVENTS]")
    events = client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=5)
    for event in events.data:
        print(f"   {event.created_at}: {event.message}")
    
    print("\n" + "=" * 70)
    
    if job.status == 'running':
        print("Job is still running. Check again in a few minutes.")
    elif job.status == 'succeeded':
        print(f"✓ Fine-tuning complete!")
        print(f"Model: {job.fine_tuned_model}")
    elif job.status == 'failed':
        print("❌ Fine-tuning failed. Check errors above.")


if __name__ == "__main__":
    main()
