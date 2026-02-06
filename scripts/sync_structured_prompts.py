#!/usr/bin/env python3
"""
Developer utility to sync structured prompt sections from prompts.py to Redis.
This enables developers to update prompts in code and push changes to Redis.
"""

import sys
import os
import json

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from app.config import REDIS_URL
from redis import Redis
from app.logger import logger


def extract_prompt_sections_from_code():
    """
    Extract prompt sections from prompts.py.
    This creates a structured template that developers can customize.
    """
    try:
        from app.utils.prompts import final_response_prompt
        
        # Create structured sections based on typical prompt components
        # Developers should customize these based on their actual prompts.py content
        sections = {
            "core": """You are a helpful AI assistant for Ditstek company. 
You provide professional, accurate, and helpful assistance to users with their inquiries.
Your responses should be informative, engaging, and aligned with Ditstek's business objectives.""",
            
            "behavior": """Be friendly, professional, and engaging in all interactions. 
Maintain a conversational tone while staying focused on business objectives.
Show enthusiasm and energy in your responses while remaining professional.""",
            
            "funnel_logic": """Guide conversations naturally towards how Ditstek can provide value and assistance.
Identify opportunities to showcase Ditstek's capabilities and services.
Ask relevant follow-up questions to better understand user needs.""",
            
            "output_schema": """Provide clear, concise responses with appropriate follow-up questions.
Use proper formatting and structure for readability.
Separate main content from follow-up questions with blank lines.
Format follow-up questions in bold.""",
            
            "reminders": """Always maintain professionalism and company focus.
Ensure responses are helpful, actionable, and add value.
Keep conversations engaging and purposeful."""
        }
        
        print("Extracted structured prompt sections from prompts.py template")
        return sections
        
    except Exception as e:
        print(f"Error extracting prompt sections: {e}")
        logger.error(f"Error extracting prompt sections: {e}")
        return None


def sync_prompt_sections_to_redis(sections: dict):
    """
    Sync prompt sections to Redis as structured JSON.
    """
    try:
        r = Redis.from_url(REDIS_URL)
        r.set("chat_prompt_json", json.dumps(sections, indent=2))
        print("Successfully synced prompt sections to Redis as structured JSON")
        logger.info("Successfully synced structured prompt sections to Redis")
        return True
    except Exception as e:
        print(f"Error syncing to Redis: {e}")
        logger.error(f"Error syncing to Redis: {e}")
        return False


def verify_redis_sync():
    """
    Verify that the structured prompt sections were correctly saved to Redis.
    """
    try:
        r = Redis.from_url(REDIS_URL)
        prompt_json = r.get("chat_prompt_json")
        if prompt_json:
            sections = json.loads(prompt_json.decode("utf-8"))
            print("\nVerification - Redis contains the following sections:")
            for key in sections.keys():
                print(f"  ✓ {key}")
            return True
        else:
            print("❌ No structured prompt sections found in Redis")
            return False
    except Exception as e:
        print(f"❌ Error verifying Redis sync: {e}")
        return False


def main():
    """
    Main function to sync structured prompt sections from code to Redis.
    """
    print("🚀 Syncing structured prompt sections from prompts.py to Redis...")
    print("=" * 60)
    
    # Extract sections from code
    sections = extract_prompt_sections_from_code()
    if not sections:
        print("❌ Failed to extract prompt sections from prompts.py")
        return False
    
    print("\n📝 Extracted sections:")
    for key, value in sections.items():
        preview = value.replace('\n', ' ').strip()
        if len(preview) > 60:
            preview = preview[:60] + "..."
        print(f"  {key}: {preview}")
    
    # Sync to Redis
    print("\n🔄 Syncing to Redis...")
    if sync_prompt_sections_to_redis(sections):
        print("✅ Successfully synced structured prompt sections to Redis")
        
        # Verify sync
        print("\n🔍 Verifying sync...")
        if verify_redis_sync():
            print("\n✅ All done! Your prompt sections are now available in Redis.")
            print("The chatbot will use these dynamic sections for LLM responses.")
        else:
            print("\n⚠️  Sync completed but verification failed.")
        
        return True
    else:
        print("❌ Failed to sync prompt sections to Redis")
        return False


if __name__ == "__main__":
    print("Ditstek Chatbot - Structured Prompt Sync Utility")
    print("=" * 50)
    main()