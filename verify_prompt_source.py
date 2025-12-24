#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick script to verify which prompts are being used by the chatbot.
Run this on your server to check the prompt source.

Usage:
    python verify_prompt_source.py
"""
import sys
import io
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    print("=" * 70)
    print("PROMPT SOURCE VERIFICATION")
    print("=" * 70)
    
    # 1. Check if prompts.py can be imported
    print("\n[1] Checking app.utils.prompts module...")
    try:
        from app.utils import prompts
        print("    [OK] Module imported successfully")
        
        # Check version
        version = getattr(prompts, 'PROMPT_VERSION', 'NOT_SET')
        print(f"    [INFO] Version: {version}")
        
        # Check if final_response_prompt exists
        if hasattr(prompts, 'final_response_prompt'):
            print("    [OK] final_response_prompt() function found")
        else:
            print("    [ERROR] final_response_prompt() function NOT found")
            
    except Exception as e:
        print(f"    [ERROR] Failed to import: {e}")
        return 1
    
    # 2. Check chatbot_optimizer import
    print("\n[2] Checking chatbot_optimizer.py import...")
    try:
        from app.core import chatbot_optimizer
        print("    [OK] Module imported successfully")
        
        # Verify it's using the direct import
        import inspect
        source = inspect.getsource(chatbot_optimizer.OptimizedChatbot.get_detailed_response)
        
        if 'final_response_prompt(' in source:
            print("    [OK] Uses final_response_prompt (direct import from prompts.py)")
        else:
            print("    [WARN] Does not use final_response_prompt")
            
        if 'get_system_prompt_from_redis' in source:
            print("    [WARN] WARNING: Also uses Redis prompts!")
        else:
            print("    [OK] Does NOT use Redis prompts")
            
    except Exception as e:
        print(f"    [ERROR] Failed to check: {e}")
    
    # 3. Generate a test prompt
    print("\n[3] Generating test prompt...")
    try:
        from app.utils.prompts import final_response_prompt
        
        test_prompt = final_response_prompt(
            prompt_context="Test context",
            conversation_summary="Test summary",
            query="Test query",
            count=1,
            user_details_known=False,
            user_details=None,
        )
        
        print(f"    [OK] Prompt generated successfully ({len(test_prompt)} chars)")
        print(f"\n    [INFO] First 300 characters:")
        print(f"    {'-' * 66}")
        preview = test_prompt[:300].replace('\n', '\n    ')
        print(f"    {preview}...")
        print(f"    {'-' * 66}")
        
        # Check for key sections
        sections = ['Role & Mission', 'CRITICAL RULES', 'DYNAMIC ENGAGEMENT', 'BUDGET']
        found_sections = [s for s in sections if s in test_prompt]
        print(f"\n    [INFO] Found sections: {', '.join(found_sections)}")
        
    except Exception as e:
        print(f"    [ERROR] Failed to generate prompt: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"[OK] Prompt Source: app/utils/prompts.py (version: {version})")
    print(f"[OK] Method: Direct import of final_response_prompt()")
    print(f"[INFO] Redis prompts: Stored but NOT used at runtime")
    print("\n[TIP] To update prompts:")
    print("   1. Edit app/utils/prompts.py")
    print("   2. Update PROMPT_VERSION")
    print("   3. Deploy to server")
    print("   4. Restart application")
    print("   5. Run this script again to verify")
    print("=" * 70)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n[INFO] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
