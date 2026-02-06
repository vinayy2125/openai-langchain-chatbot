"""Check status of a fine-tuning job."""

import os
import sys
from openai import OpenAI


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    
    # Get job ID from argument or file
    job_id = None
    if len(sys.argv) > 1:
        job_id = sys.argv[1]
    else:
        job_file = "fine_tuning_data/job_id.txt"
        if os.path.exists(job_file):
            with open(job_file) as f:
                job_id = f.read().strip()
    
    if not job_id:
        print("❌ No job ID provided. Usage: python check_job_status.py <job_id>")
        sys.exit(1)
    
    # Get job status
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    print(f"\n📊 Fine-Tuning Job Status")
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")
    print(f"  Model: {job.model}")
    
    if job.fine_tuned_model:
        print(f"\n✅ Fine-tuned model ready!")
        print(f"  Model ID: {job.fine_tuned_model}")
        print(f"\n  Use in your code:")
        print(f'    model="{job.fine_tuned_model}"')
    elif job.status == "failed":
        print(f"\n❌ Job failed")
        if job.error:
            print(f"  Error: {job.error.message}")
    else:
        print(f"\n⏳ Job still running...")
        print(f"  Check again in a few minutes.")
    
    # Show events
    print(f"\n📜 Recent Events:")
    events = client.fine_tuning.jobs.list_events(job_id, limit=5)
    for event in events.data:
        print(f"  - {event.message}")


if __name__ == "__main__":
    main()
