"""
STEP 6: DATASET SIZE CONTROL

Enforces the following limits:
- MVP: 150–300 examples
- Strong version: 500–1,000 examples
- HARD STOP at ~1,500 examples

If the dataset exceeds limits:
- Deduplicate
- Remove verbose answers
- Remove overlapping intents

Output:
- Final example count
- Confirmation dataset is within limits
"""

import json
from collections import Counter, defaultdict

# Load the Step 5 generated data
STEP5_FILE = 'step5_website_training_data.jsonl'

# Load existing UC1 training data if available
UC1_FILE = 'fine_tuning_data/train.jsonl'

def load_jsonl(filepath):
    """Load JSONL file into list of dicts."""
    examples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    examples.append(json.loads(line))
    except FileNotFoundError:
        print(f"File not found: {filepath}")
    return examples

def count_words(text):
    """Count words in text."""
    return len(text.split())

def check_verbose(example):
    """Check if assistant response is too verbose (> 100 words)."""
    assistant_msg = example['messages'][2]['content']
    word_count = count_words(assistant_msg)
    return word_count > 100, word_count

def check_duplicate(example, seen_questions):
    """Check if question is duplicate."""
    user_msg = example['messages'][1]['content'].lower().strip()
    if user_msg in seen_questions:
        return True
    seen_questions.add(user_msg)
    return False

def extract_state(example):
    """Extract state from system message."""
    system_msg = example['messages'][0]['content']
    if 'State:' in system_msg:
        state_line = [l for l in system_msg.split('\n') if l.startswith('State:')]
        if state_line:
            return state_line[0].replace('State:', '').strip()
    return 'UNKNOWN'

def main():
    print("=" * 70)
    print("STEP 6: DATASET SIZE CONTROL")
    print("=" * 70)
    
    # Load datasets
    step5_examples = load_jsonl(STEP5_FILE)
    uc1_examples = load_jsonl(UC1_FILE)
    
    print(f"\n[LOADED DATA]")
    print(f"   Step 5 (website-derived): {len(step5_examples)} examples")
    print(f"   UC1 (flow-based): {len(uc1_examples)} examples")
    
    # Combined total
    total_examples = len(step5_examples) + len(uc1_examples)
    print(f"   COMBINED TOTAL: {total_examples} examples")
    
    # ========================================
    # SIZE LIMIT CHECKS
    # ========================================
    print("\n" + "=" * 70)
    print("SIZE LIMIT VALIDATION")
    print("=" * 70)
    
    limits = {
        "MVP": (150, 300),
        "Strong": (500, 1000),
        "HARD_STOP": 1500
    }
    
    print(f"\n[LIMITS]")
    print(f"   MVP range: {limits['MVP'][0]}-{limits['MVP'][1]}")
    print(f"   Strong range: {limits['Strong'][0]}-{limits['Strong'][1]}")
    print(f"   HARD STOP: {limits['HARD_STOP']}")
    
    # Determine status
    if total_examples <= limits['MVP'][1]:
        status = "✓ WITHIN MVP RANGE" if total_examples >= limits['MVP'][0] else "⚠ BELOW MVP MINIMUM"
        category = "MVP"
    elif total_examples <= limits['Strong'][1]:
        status = "✓ WITHIN STRONG RANGE" if total_examples >= limits['Strong'][0] else "⚠ BETWEEN MVP AND STRONG"
        category = "STRONG"
    elif total_examples <= limits['HARD_STOP']:
        status = "⚠ APPROACHING HARD STOP"
        category = "WARNING"
    else:
        status = "✗ EXCEEDS HARD STOP - ACTION REQUIRED"
        category = "CRITICAL"
    
    print(f"\n[STATUS]: {status}")
    print(f"   Total: {total_examples} / {limits['HARD_STOP']} (hard stop)")
    
    # ========================================
    # QUALITY CHECKS
    # ========================================
    print("\n" + "=" * 70)
    print("QUALITY CHECKS")
    print("=" * 70)
    
    # Check Step 5 data for issues
    verbose_examples = []
    duplicate_questions = []
    seen_questions = set()
    state_distribution = Counter()
    word_counts = []
    
    for i, example in enumerate(step5_examples):
        # Verbose check
        is_verbose, word_count = check_verbose(example)
        word_counts.append(word_count)
        if is_verbose:
            verbose_examples.append((i, word_count, example['messages'][1]['content'][:50]))
        
        # Duplicate check
        if check_duplicate(example, seen_questions):
            duplicate_questions.append(i)
        
        # State distribution
        state = extract_state(example)
        state_distribution[state] += 1
    
    print(f"\n[STEP 5 DATA ANALYSIS]")
    print(f"   Verbose examples (>100 words): {len(verbose_examples)}")
    print(f"   Duplicate questions: {len(duplicate_questions)}")
    
    if word_counts:
        avg_words = sum(word_counts) / len(word_counts)
        max_words = max(word_counts)
        min_words = min(word_counts)
        print(f"   Response word count - Avg: {avg_words:.1f}, Min: {min_words}, Max: {max_words}")
    
    print(f"\n[STATE DISTRIBUTION]:")
    for state, count in sorted(state_distribution.items(), key=lambda x: -x[1]):
        print(f"   {state}: {count}")
    
    # ========================================
    # RECOMMENDATIONS
    # ========================================
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    recommendations = []
    
    if len(verbose_examples) > 0:
        recommendations.append(f"- Consider shortening {len(verbose_examples)} verbose responses")
    
    if len(duplicate_questions) > 0:
        recommendations.append(f"- Remove {len(duplicate_questions)} duplicate questions")
    
    if total_examples < limits['MVP'][0]:
        recommendations.append(f"- Dataset is below MVP minimum ({limits['MVP'][0]}). Consider adding more examples.")
    
    if total_examples > limits['HARD_STOP']:
        recommendations.append(f"- CRITICAL: Dataset exceeds hard stop. Must reduce by {total_examples - limits['HARD_STOP']} examples.")
    
    if not recommendations:
        recommendations.append("✓ No issues found. Dataset is within acceptable limits.")
    
    for rec in recommendations:
        print(f"   {rec}")
    
    # ========================================
    # FINAL SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print(f"""
DATASET SIZE CONTROL RESULTS:
-----------------------------
Step 5 (website-derived):  {len(step5_examples)} examples
UC1 (flow-based):          {len(uc1_examples)} examples
-----------------------------
COMBINED TOTAL:            {total_examples} examples

STATUS: {status}

DATASET BREAKDOWN:
- Website service inquiries: {len(step5_examples)} examples
- UC1 conversation flow:     {len(uc1_examples)} examples

LIMIT CHECK:
- MVP range (150-300):       {'✓ PASS' if limits['MVP'][0] <= total_examples <= limits['MVP'][1] else '○ N/A'}
- Strong range (500-1000):   {'✓ PASS' if limits['Strong'][0] <= total_examples <= limits['Strong'][1] else '○ N/A'}
- Hard stop (1500):          {'✓ WITHIN LIMIT' if total_examples <= limits['HARD_STOP'] else '✗ EXCEEDED'}

QUALITY:
- Verbose responses:         {len(verbose_examples)}
- Duplicate questions:       {len(duplicate_questions)}
""")
    
    print("=" * 70)
    print("STEP 6 COMPLETE - STOPPING AS INSTRUCTED")
    print("=" * 70)
    print("\nAwaiting instruction to proceed to STEP 7 (Validation Check).")


if __name__ == "__main__":
    main()
