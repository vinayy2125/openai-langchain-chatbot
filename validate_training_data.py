"""
Step 2 Dataset Validation Script

Run: python validate_training_data.py train_clean.jsonl

This script MUST pass before retraining. No exceptions.
"""

import json
import re
import sys

# Patterns that indicate contaminated training data
# These should NEVER appear in user or assistant messages
FORBIDDEN_PATTERNS_USER_ASSISTANT = [
    r"schedule a call",
    r"book a call",
    r"let's connect",
    r"how would you like to proceed",
    r"where would you like to go from here",
    r"what feels like the right next step",
    r"discuss my requirement",
    r"continue exploring",
    r"explore on my own",
    r"present exactly",
    r"best-in-class",
    r"cutting-edge",
    r"world-class",
    r"• discuss",
    r"• schedule",
    r"• continue",
]

# Patterns that indicate old state/rule contamination in system messages
# (should NOT be in system prompt - but canonical prompt has "Behavior rules:" which is OK)
FORBIDDEN_PATTERNS_SYSTEM = [
    r"^State:",  # State label at start of line
    r"^Rules:",  # Rules label at start of line  
    r"^Capability:",  # Capability label at start of line
    r"^Context:",  # Context label at start of line
    r"\bNo CTA\b",
    r"\bCTA is now earned\b",
    r"ask one guided",
    r"present exactly \d",
]

CANONICAL_SYSTEM_PREFIX = "You are an AI assistant representing DITSTEK"

def validate_file(path):
    errors = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON - {e}")
                continue
            
            messages = obj.get("messages", [])
            
            # Check system message
            if not messages or messages[0]["role"] != "system":
                errors.append(f"Line {i}: Missing system message")
                continue
            
            system_content = messages[0]["content"]
            if not system_content.startswith(CANONICAL_SYSTEM_PREFIX):
                errors.append(f"Line {i}: System message doesn't match canonical prompt")
            
            # Check for forbidden patterns in system messages
            if messages[0]["role"] == "system":
                sys_text = messages[0]["content"]
                for p in FORBIDDEN_PATTERNS_SYSTEM:
                    if re.search(p, sys_text, re.MULTILINE | re.IGNORECASE):
                        errors.append(f"Line {i}: Forbidden system pattern '{p}'")
            
            # Check for forbidden patterns in user/assistant messages
            for msg in messages:
                if msg["role"] in ("user", "assistant"):
                    text = msg["content"]
                    for p in FORBIDDEN_PATTERNS_USER_ASSISTANT:
                        if re.search(p, text, re.IGNORECASE):
                            errors.append(f"Line {i}: Forbidden pattern '{p}' in {msg['role']} message")
            
            # Check assistant response length
            for msg in messages:
                if msg["role"] == "assistant":
                    sentences = len([s for s in msg["content"].split('.') if s.strip()])
                    if sentences > 5:
                        errors.append(f"Line {i}: Assistant response too long ({sentences} sentences)")
    
    if errors:
        print(f"VALIDATION FAILED - {len(errors)} errors:\n")
        for e in errors[:20]:  # Show first 20
            print(f"  ❌ {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        sys.exit(1)
    else:
        print(f"✅ Dataset clean. Ready for retraining.")
        sys.exit(0)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_training_data.py <path_to_jsonl>")
        sys.exit(1)
    validate_file(sys.argv[1])
