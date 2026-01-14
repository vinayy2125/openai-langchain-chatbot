"""
Validate and prepare train_final.jsonl for OpenAI fine-tuning.

This script:
1. Validates JSONL format
2. Checks for required message structure
3. Counts tokens and estimates cost
4. Prepares for fine-tuning upload
"""

import json
import tiktoken
from collections import Counter

INPUT_FILE = 'train_final.jsonl'

def load_jsonl(filepath):
    """Load JSONL file with error handling."""
    examples = []
    errors = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    example = json.loads(line)
                    examples.append(example)
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON - {str(e)}")
    
    return examples, errors

def validate_message_structure(example, index):
    """Validate message structure for OpenAI fine-tuning."""
    issues = []
    
    # Check messages array exists
    if 'messages' not in example:
        issues.append(f"Example {index}: Missing 'messages' key")
        return issues
    
    messages = example['messages']
    
    # Check message count (minimum 2: system/user and assistant)
    if len(messages) < 2:
        issues.append(f"Example {index}: Too few messages ({len(messages)})")
        return issues
    
    # Check roles
    roles = [m.get('role') for m in messages]
    
    # Must have system (optional) + user + assistant pattern
    if len(messages) == 3:
        if roles != ['system', 'user', 'assistant']:
            issues.append(f"Example {index}: Invalid role sequence {roles}")
    elif len(messages) == 2:
        if roles != ['user', 'assistant']:
            issues.append(f"Example {index}: Invalid role sequence {roles}")
    
    # Check content exists for all messages
    for j, msg in enumerate(messages):
        if 'content' not in msg or not msg['content']:
            issues.append(f"Example {index}: Empty content in message {j}")
    
    return issues

def count_tokens(text, encoding):
    """Count tokens in text."""
    return len(encoding.encode(text))

def estimate_cost(total_tokens, model="gpt-4o-mini-2024-07-18"):
    """Estimate fine-tuning cost based on tokens."""
    # GPT-4o-mini fine-tuning pricing (as of 2024)
    # Training: $0.003 per 1K tokens
    # Default epochs: 3
    epochs = 3
    cost_per_1k = 0.003
    
    training_cost = (total_tokens * epochs / 1000) * cost_per_1k
    
    return training_cost, epochs

def main():
    print("=" * 70)
    print("FINE-TUNING DATA VALIDATION")
    print("=" * 70)
    
    # Load and parse
    print(f"\n[1. LOADING FILE]: {INPUT_FILE}")
    examples, parse_errors = load_jsonl(INPUT_FILE)
    print(f"    Loaded: {len(examples)} examples")
    print(f"    Parse errors: {len(parse_errors)}")
    
    if parse_errors:
        print("\n    ❌ Parse errors found:")
        for err in parse_errors[:5]:
            print(f"       {err}")
        if len(parse_errors) > 5:
            print(f"       ... and {len(parse_errors) - 5} more")
        return
    
    # Validate structure
    print("\n[2. VALIDATING STRUCTURE]")
    structure_issues = []
    for i, example in enumerate(examples, 1):
        issues = validate_message_structure(example, i)
        structure_issues.extend(issues)
    
    if structure_issues:
        print(f"    ❌ Found {len(structure_issues)} issues:")
        for issue in structure_issues[:5]:
            print(f"       {issue}")
        if len(structure_issues) > 5:
            print(f"       ... and {len(structure_issues) - 5} more")
        # HARD GATE: Exit non-zero on validation failure (Phase 6.2)
        sys.exit(1)
    else:
        print("    ✓ All examples have valid structure")
    
    # Token counting
    print("\n[3. TOKEN ANALYSIS]")
    try:
        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    except:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    example_tokens = []
    
    for example in examples:
        example_text = ""
        for msg in example['messages']:
            example_text += msg.get('content', '')
        tokens = count_tokens(example_text, encoding)
        example_tokens.append(tokens)
        total_tokens += tokens
    
    avg_tokens = total_tokens / len(examples) if examples else 0
    max_tokens = max(example_tokens) if example_tokens else 0
    min_tokens = min(example_tokens) if example_tokens else 0
    
    print(f"    Total tokens: {total_tokens:,}")
    print(f"    Average per example: {avg_tokens:.1f}")
    print(f"    Range: {min_tokens} - {max_tokens}")
    
    # Cost estimation
    print("\n[4. COST ESTIMATION]")
    cost, epochs = estimate_cost(total_tokens)
    print(f"    Model: gpt-4o-mini")
    print(f"    Estimated epochs: {epochs}")
    print(f"    Estimated cost: ~${cost:.2f}")
    
    # Analysis
    print("\n[5. CONTENT ANALYSIS]")
    
    # State distribution
    state_counts = Counter()
    for example in examples:
        if example['messages']:
            system_msg = example['messages'][0].get('content', '')
            if 'State:' in system_msg:
                state_line = [l for l in system_msg.split('\n') if l.startswith('State:')]
                if state_line:
                    state = state_line[0].replace('State:', '').strip()
                    state_counts[state] += 1
    
    print(f"    Unique states/categories: {len(state_counts)}")
    
    # Final status
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = len(parse_errors) == 0 and len(structure_issues) == 0
    
    print(f"""
File: {INPUT_FILE}
Examples: {len(examples)}
Total tokens: {total_tokens:,}
Estimated cost: ~${cost:.2f}

Validation Status: {'✓ PASSED' if all_passed else '❌ FAILED'}
""")
    
    if all_passed:
        print("Ready for fine-tuning!")
        print("\nTo start fine-tuning, run:")
        print("  python start_finetuning.py")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
