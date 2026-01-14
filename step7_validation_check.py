"""
STEP 7: VALIDATION CHECK

Review examples and verify:
- Responses are concise
- Many examples handle vague inputs ("yeah", "not sure")
- No example sounds like copied website text
- Tone feels consultative, not marketing

If any rule is violated:
- Flag examples
- Do NOT auto-fix unless instructed

Output:
- Validation summary
- List of flagged issues (if any)
"""

import json
import re
from collections import Counter, defaultdict

# Files to validate
STEP5_FILE = 'step5_website_training_data.jsonl'
UC1_FILE = 'fine_tuning_data/train.jsonl'

# Validation thresholds
MAX_RESPONSE_WORDS = 80
MAX_SENTENCES = 5
MIN_RESPONSE_LENGTH = 10  # characters

# Marketing/brochure-like patterns to flag
MARKETING_PATTERNS = [
    r'\bworld.?class\b',
    r'\bcutting.?edge\b',
    r'\bmarket.?leading\b',
    r'\bindustry.?leading\b',
    r'\brobust solution\b',
    r'\bcomprehensive solution\b',
    r'\bstate.?of.?the.?art\b',
    r'\bunparalleled\b',
    r'\bseamless\b',
    r'\bleverage\b',
    r'\bsynergy\b',
    r'\bholistic\b',
    r'\bend.?to.?end\b',
    r'\bfull.?suite\b',
    r'\bturnkey\b',
    r'\bscalable and secure\b',
    r'\bworld.?renowned\b',
    r'\btruly unique\b',
    r'\bexceptional quality\b',
]

# Website-like patterns (copy-paste indicators)
WEBSITE_PATTERNS = [
    r'contact us today',
    r'get in touch',
    r'request a quote',
    r'learn more about our',
    r'our team of experts',
    r'we are committed to',
    r'we strive to',
    r'our mission is to',
    r'dedicated to providing',
    r'delivering excellence',
]

# Vague input patterns (should have good handling)
VAGUE_INPUTS = [
    'yeah', 'yes', 'yep', 'yup',
    'okay', 'ok', 'sure',
    'not sure', 'maybe', 'possibly',
    'hmm', 'hm', 'uh',
    'interesting', 'I see',
    'go on', 'continue',
    'tell me more', 'what else',
]

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

def count_sentences(text):
    """Count approximate sentences in text."""
    sentence_enders = re.findall(r'[.!?]', text)
    return len(sentence_enders)

def check_marketing_tone(text):
    """Check for marketing/brochure-like language."""
    text_lower = text.lower()
    found = []
    for pattern in MARKETING_PATTERNS:
        if re.search(pattern, text_lower):
            found.append(pattern)
    return found

def check_website_copy(text):
    """Check for website copy-paste patterns."""
    text_lower = text.lower()
    found = []
    for pattern in WEBSITE_PATTERNS:
        if pattern in text_lower:
            found.append(pattern)
    return found

def check_concise(example):
    """Check if response is concise (2-5 sentences, reasonable word count)."""
    assistant_msg = example['messages'][2]['content']
    word_count = count_words(assistant_msg)
    sentence_count = count_sentences(assistant_msg)
    
    issues = []
    if word_count > MAX_RESPONSE_WORDS:
        issues.append(f"Too many words: {word_count} (max: {MAX_RESPONSE_WORDS})")
    if sentence_count > MAX_SENTENCES:
        issues.append(f"Too many sentences: {sentence_count} (max: {MAX_SENTENCES})")
    if len(assistant_msg) < MIN_RESPONSE_LENGTH:
        issues.append(f"Too short: {len(assistant_msg)} chars")
    
    return issues

def check_consultative_tone(example):
    """Check if response has consultative tone (includes questions, not pushy)."""
    assistant_msg = example['messages'][2]['content']
    
    # Good indicators
    has_question = '?' in assistant_msg
    uses_soft_language = any(phrase in assistant_msg.lower() for phrase in [
        'could you', 'would you', 'what about', 'tell me', 'help me understand',
        'depends', 'typically', 'usually', 'often', 'it varies'
    ])
    
    return has_question or uses_soft_language

def is_vague_input(user_msg):
    """Check if user message is vague/exploratory."""
    user_lower = user_msg.lower().strip()
    return any(vague in user_lower for vague in VAGUE_INPUTS) or len(user_lower) < 15

def main():
    print("=" * 70)
    print("STEP 7: VALIDATION CHECK")
    print("=" * 70)
    
    # Load datasets
    step5_examples = load_jsonl(STEP5_FILE)
    uc1_examples = load_jsonl(UC1_FILE)
    
    print(f"\n[LOADED DATA]")
    print(f"   Step 5 (website-derived): {len(step5_examples)} examples")
    print(f"   UC1 (flow-based): {len(uc1_examples)} examples")
    
    all_examples = step5_examples + uc1_examples
    
    # ========================================
    # VALIDATION CHECKS
    # ========================================
    
    flagged_issues = defaultdict(list)
    stats = {
        'concise_pass': 0,
        'concise_fail': 0,
        'marketing_found': 0,
        'website_copy_found': 0,
        'consultative_pass': 0,
        'consultative_fail': 0,
        'vague_input_handled': 0,
        'follow_up_questions': 0,
    }
    
    print("\n" + "=" * 70)
    print("RUNNING VALIDATION CHECKS")
    print("=" * 70)
    
    for i, example in enumerate(all_examples):
        user_msg = example['messages'][1]['content']
        assistant_msg = example['messages'][2]['content']
        source = "Step5" if i < len(step5_examples) else "UC1"
        
        # 1. Check conciseness
        concise_issues = check_concise(example)
        if concise_issues:
            stats['concise_fail'] += 1
            flagged_issues['NOT_CONCISE'].append({
                'index': i,
                'source': source,
                'user': user_msg[:50],
                'issues': concise_issues
            })
        else:
            stats['concise_pass'] += 1
        
        # 2. Check for marketing language
        marketing_matches = check_marketing_tone(assistant_msg)
        if marketing_matches:
            stats['marketing_found'] += 1
            flagged_issues['MARKETING_TONE'].append({
                'index': i,
                'source': source,
                'user': user_msg[:50],
                'patterns': marketing_matches
            })
        
        # 3. Check for website copy
        website_matches = check_website_copy(assistant_msg)
        if website_matches:
            stats['website_copy_found'] += 1
            flagged_issues['WEBSITE_COPY'].append({
                'index': i,
                'source': source,
                'user': user_msg[:50],
                'patterns': website_matches
            })
        
        # 4. Check consultative tone
        if check_consultative_tone(example):
            stats['consultative_pass'] += 1
        else:
            stats['consultative_fail'] += 1
        
        # 5. Check vague input handling
        if is_vague_input(user_msg):
            stats['vague_input_handled'] += 1
        
        # 6. Count follow-up questions
        if '?' in assistant_msg:
            stats['follow_up_questions'] += 1
    
    # ========================================
    # VALIDATION RESULTS
    # ========================================
    
    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)
    
    total = len(all_examples)
    
    print(f"\n[1. CONCISENESS CHECK]")
    print(f"   Pass: {stats['concise_pass']} / {total} ({100*stats['concise_pass']/total:.1f}%)")
    print(f"   Fail: {stats['concise_fail']} / {total}")
    
    print(f"\n[2. MARKETING TONE CHECK]")
    print(f"   Clean: {total - stats['marketing_found']} / {total} ({100*(total-stats['marketing_found'])/total:.1f}%)")
    print(f"   Flagged: {stats['marketing_found']} / {total}")
    
    print(f"\n[3. WEBSITE COPY CHECK]")
    print(f"   Clean: {total - stats['website_copy_found']} / {total} ({100*(total-stats['website_copy_found'])/total:.1f}%)")
    print(f"   Flagged: {stats['website_copy_found']} / {total}")
    
    print(f"\n[4. CONSULTATIVE TONE CHECK]")
    print(f"   Consultative: {stats['consultative_pass']} / {total} ({100*stats['consultative_pass']/total:.1f}%)")
    print(f"   Non-consultative: {stats['consultative_fail']} / {total}")
    
    print(f"\n[5. VAGUE INPUT HANDLING]")
    print(f"   Vague inputs handled: {stats['vague_input_handled']} examples")
    print(f"   Examples with follow-up questions: {stats['follow_up_questions']} ({100*stats['follow_up_questions']/total:.1f}%)")
    
    # ========================================
    # FLAGGED ISSUES DETAIL
    # ========================================
    
    print("\n" + "=" * 70)
    print("FLAGGED ISSUES (if any)")
    print("=" * 70)
    
    if not any(flagged_issues.values()):
        print("\n   ✓ No critical issues found!")
    else:
        for issue_type, issues in flagged_issues.items():
            if issues:
                print(f"\n[{issue_type}] - {len(issues)} issues")
                print("-" * 50)
                for issue in issues[:5]:  # Show first 5
                    print(f"   Example {issue['index']} ({issue['source']})")
                    print(f"   User: \"{issue['user']}...\"")
                    if 'issues' in issue:
                        print(f"   Problems: {issue['issues']}")
                    if 'patterns' in issue:
                        print(f"   Patterns: {issue['patterns']}")
                if len(issues) > 5:
                    print(f"   ... and {len(issues) - 5} more")
    
    # ========================================
    # FINAL VALIDATION SUMMARY
    # ========================================
    
    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    
    # Determine overall status
    critical_issues = (
        stats['marketing_found'] + 
        stats['website_copy_found']
    )
    
    has_good_vague_handling = stats['vague_input_handled'] >= 15
    has_good_followup_ratio = stats['follow_up_questions'] / total > 0.3
    has_good_consultative = stats['consultative_pass'] / total > 0.7
    
    print(f"""
VALIDATION SUMMARY:
-------------------
Total examples validated: {total}

RULE 1: Responses are concise
   Result: {stats['concise_pass']}/{total} pass ({100*stats['concise_pass']/total:.1f}%)
   Status: {'✓ PASS' if stats['concise_pass'] == total else '⚠ NEEDS REVIEW'}

RULE 2: Handles vague inputs
   Result: {stats['vague_input_handled']} vague inputs in dataset
   Status: {'✓ PASS' if has_good_vague_handling else '⚠ CONSIDER ADDING MORE'}

RULE 3: No copied website text
   Result: {stats['website_copy_found']} flagged examples
   Status: {'✓ PASS' if stats['website_copy_found'] == 0 else '⚠ NEEDS REVIEW'}

RULE 4: Consultative tone (not marketing)
   Result: {stats['consultative_pass']}/{total} consultative ({100*stats['consultative_pass']/total:.1f}%)
   Marketing flags: {stats['marketing_found']}
   Status: {'✓ PASS' if has_good_consultative and stats['marketing_found'] == 0 else '⚠ NEEDS REVIEW'}

OVERALL STATUS: {'✓ VALIDATION PASSED' if critical_issues == 0 else '⚠ ISSUES FOUND - REVIEW REQUIRED'}

ACTION REQUIRED: {'None - proceed to Step 8' if critical_issues == 0 else 'Review flagged examples before proceeding'}
""")
    
    print("=" * 70)
    print("STEP 7 COMPLETE - STOPPING AS INSTRUCTED")
    print("=" * 70)
    print("\nAwaiting instruction to proceed to STEP 8 (Final Deliverable).")


if __name__ == "__main__":
    main()
