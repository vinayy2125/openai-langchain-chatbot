import pytest
import json
from pathlib import Path

# Config
DATA_DIR = Path(__file__).parent.parent / "fine_tuning_data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "validation.jsonl"

def load_examples(file_path):
    examples = []
    if not file_path.exists():
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

@pytest.fixture(scope="module")
def training_data():
    train = load_examples(TRAIN_FILE)
    val = load_examples(VAL_FILE)
    return train + val

def test_files_exist():
    assert TRAIN_FILE.exists(), "train.jsonl missing"
    assert VAL_FILE.exists(), "validation.jsonl missing"

def test_no_questions_in_assistant_output(training_data):
    """
    Assistant should NEVER ask questions in generation-controlled states.
    All flow exploration is done via Orchestrator (fixed prompts), not LLM.
    LLM output should be declarative/imperative.
    """
    for ex in training_data:
        # Check standard states (S0, S5, S6, S8)
        # S8 might have rhetorical questions? "What else?" -> We removed them.
        state = ex["messages"][0]["content"] # Extract state from system message usually? 
        # Actually our jsonl format: messages=[{role:system, content:...}, {role:user...}, {role:assistant...}]
        # The system message content starts with "State: UC1_S..."
        
        sys_msg = ex["messages"][0]["content"]
        asst_msg = ex["messages"][2]["content"]
        
        # Exception: URL (http://...?) - unlikely in our data
        # Exception: We allow NO questions.
        
        if "?" in asst_msg:
             pytest.fail(f"Forbidden '?' found in assistant output (State: {sys_msg[:20]}...): '{asst_msg}'")

def test_ctas_only_in_authorized_states(training_data):
    """
    CTAs (Schedule, Book, Call) should only appear in UC1_S8_CLOSE or META/Edge contexts.
    They should NOT appear in S0, S5, S6.
    """
    # Allowed states for CTA language
    ALLOWED_CTA_STATES = ["UC1_S8_CLOSE", "EDGE", "META"] # S8 is close/conversion.
    # Note: S7 is CTA state, but we don't train it (orchestrator handles).
    
    forbidden_terms = ["schedule a call", "book a time", "book a meeting", "talk to an expert"]
    
    for ex in training_data:
        sys_msg = ex["messages"][0]["content"]
        asst_msg = ex["messages"][2]["content"].lower()
        
        is_safe_state = any(s in sys_msg for s in ALLOWED_CTA_STATES)
        
        if not is_safe_state:
            for term in forbidden_terms:
                if term in asst_msg:
                    pytest.fail(f"Forbidden CTA term '{term}' found in non-CTA state ({sys_msg[:20]}...): '{asst_msg}'")

def test_alternatives_are_formatted_correctly(training_data):
    """
    S6 (Consultative Alternatives) must present options, usually formatted or listed.
    They should NOT invent new options (generative hallucination). 
    We just check structure here.
    """
    for ex in training_data:
        sys_msg = ex["messages"][0]["content"]
        if "UC1_S6_CONSULTATIVE_ALTERNATIVES" in sys_msg:
            asst_msg = ex["messages"][2]["content"]
            # Should have multiple lines or options
            assert len(asst_msg) > 50, f"S6 response too short: {asst_msg}"
            
def test_no_state_leakage(training_data):
    """
    Assistant should not output internal state names like 'UC1_S5'.
    """
    forbidden = ["UC1_S", "State:", "CAPABILITY_PICK", "EXPLORATION_LAYER"]
    for ex in training_data:
        asst_msg = ex["messages"][2]["content"]
        for term in forbidden:
            assert term not in asst_msg, f"Internal term '{term}' leaked in assistant output: '{asst_msg}'"
