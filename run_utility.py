import subprocess
import os
import sys

def run_utility():
    # Path to the utility script
    script_path = os.path.join("llm_instructions_utility", "util_app.py")
    
    if not os.path.exists(script_path):
        print(f"Error: Utility script not found at {script_path}")
        return

    # Set PYTHONPATH to include current directory
    env = os.environ.copy()
    current_root = os.getcwd()
    env["PYTHONPATH"] = current_root + os.pathsep + env.get("PYTHONPATH", "")

    print(f"Starting LLM Instructions Utility from {script_path}...")
    
    # Run streamlit as a module
    cmd = [sys.executable, "-m", "streamlit", "run", script_path]
    
    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        print("\nStopping Utility...")

if __name__ == "__main__":
    run_utility()
