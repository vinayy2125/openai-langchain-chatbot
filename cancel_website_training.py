import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# The specific Job ID to cancel
JOB_ID = 'ftjob-rjkCqoV4dAuhPNGL3IReUHSn'

def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set.")
        return

    client = OpenAI()
    
    print(f"[*] Attempting to cancel Job: {JOB_ID}...")
    
    try:
        # Check status first
        job = client.fine_tuning.jobs.retrieve(JOB_ID)
        print(f"    Current Status: {job.status.upper()}")
        
        if job.status in ['succeeded', 'failed', 'cancelled']:
            print(f"[INFO] Job is already {job.status}. Cannot cancel.")
            return

        # Cancel
        client.fine_tuning.jobs.cancel(JOB_ID)
        print(f"[SUCCESS] Cancellation request sent.")
        
        # Verify
        job = client.fine_tuning.jobs.retrieve(JOB_ID)
        print(f"    New Status: {job.status.upper()}")

    except Exception as e:
        print(f"[ERROR] Cancellation failed: {e}")

if __name__ == "__main__":
    main()
