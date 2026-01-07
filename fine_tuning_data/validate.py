"""Enhanced validation for UC1 fine-tuning data."""
import json
import re

def validate():
    errors = []
    warnings = []
    
    for filename in ['fine_tuning_data/train.jsonl', 'fine_tuning_data/validation.jsonl']:
        with open(filename, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip()]
        
        print(f"\n=== {filename} ===")
        print(f"Total examples: {len(lines)}")
        
        state_counts = {}
        
        for i, line in enumerate(lines, 1):
            ex = json.loads(line)
            msgs = ex['messages']
            sys_msg = msgs[0]['content']
            user_msg = msgs[1]['content']
            asst_msg = msgs[2]['content']
            
            # Extract state
            state_match = re.search(r'State: (UC1_S\d+_\w+)', sys_msg)
            state = state_match.group(1) if state_match else "UNKNOWN"
            state_counts[state] = state_counts.get(state, 0) + 1
            
            # Check 1: State naming (must be UC1_S*)
            if not sys_msg.startswith('State: UC1_'):
                errors.append(f"Line {i}: State not normalized to UC1_S* format")
            
            # Check 2: No names in assistant output (except in thank patterns)
            if re.search(r'Thanks, \w+\.', asst_msg) and 'Thanks, ' not in asst_msg[:10]:
                errors.append(f"Line {i}: Name leakage in assistant output")
            
            # Check 3: No synthetic placeholders
            if '[synthesis complete]' in user_msg or '[exploration complete]' in user_msg or '[alternatives presented]' in user_msg:
                errors.append(f"Line {i}: Synthetic placeholder in user input")
            
            # Check 4: CTA only in S7+
            if 'Discuss my requirement now' in asst_msg or 'Schedule a quick call' in asst_msg:
                if 'UC1_S7' not in state and 'UC1_S8' not in state:
                    errors.append(f"Line {i}: CTA found before UC1_S7")
            
            # Check 5: S5 single intent (no R+Q combined)
            if 'UC1_S5' in state:
                question_marks = asst_msg.count('?')
                if question_marks > 1:
                    warnings.append(f"Line {i}: Multiple questions in S5 ({question_marks} '?')")
            
            # Check 6: Alternatives count == 3
            if 'UC1_S6' in state:
                bullet_count = asst_msg.count('•') + len(re.findall(r'^\d+\.', asst_msg, re.MULTILINE))
                if bullet_count < 3:
                    warnings.append(f"Line {i}: S6 has {bullet_count} alternatives (expected 3)")
        
        print("\nState distribution:")
        for state in sorted(state_counts.keys()):
            print(f"  {state}: {state_counts[state]}")
    
    print(f"\n=== Summary ===")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    if errors:
        print("\nErrors:")
        for e in errors[:5]:
            print(f"  ❌ {e}")
        if len(errors) > 5:
            print(f"  ... and {len(errors)-5} more")
    
    if warnings:
        print("\nWarnings:")
        for w in warnings[:5]:
            print(f"  ⚠ {w}")
        if len(warnings) > 5:
            print(f"  ... and {len(warnings)-5} more")
    
    if not errors and not warnings:
        print("\n✓ All validation checks passed!")
    elif not errors:
        print("\n✓ No critical errors! Review warnings if needed.")

if __name__ == "__main__":
    validate()
