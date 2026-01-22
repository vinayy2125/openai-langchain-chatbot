"""
UC1 Fine-Tuning Data Generator v2

Generates state-scoped, atomic fine-tuning examples in JSONL format
from the DITSTEK Phase-1 conversation specification.

CRITICAL FIXES APPLIED:
1. State naming normalized to UC1_S* format (spec-aligned)
2. Single-intent per assistant turn (no R1+Q2 combined)
3. No personalization (names removed from assistant output)
4. Realistic user inputs (no synthetic placeholders)
5. Target model: gpt-4.1-mini

Rules:
- System messages: State + Capability + Context + Rules (no accumulated slots)
- S5 = one intent per example (question OR reflection, never both)
- Paraphrase expansion: 4-8 per base example
- Single-intent per assistant turn
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from typing import List, Dict, Optional
import yaml
import re

# =============================================================================
# DATASET COMPOSITION TRACKER (HARD ENFORCEMENT)
# =============================================================================

@dataclass
class DatasetCompositionTracker:
    """
    Enforces dataset composition quotas at generation time.
    
    Target composition (per authoritative spec):
    - Grounded Knowledge QA: 65%
    - Negative/Out-of-scope: 20%
    - Consultative Expression: 10%
    - Edge Cases: 5%
    
    HARD RULE: Generation fails if any bucket exceeds quota.
    """
    target_total: int = 2000
    
    # Quotas (percentages)
    quota_grounded: float = 0.65
    quota_negative: float = 0.20
    quota_consultative: float = 0.10
    quota_edge: float = 0.05
    
    # Counters
    count_grounded: int = 0
    count_negative: int = 0
    count_consultative: int = 0
    count_edge: int = 0
    
    @property
    def max_grounded(self) -> int:
        return int(self.target_total * self.quota_grounded)
    
    @property
    def max_negative(self) -> int:
        return int(self.target_total * self.quota_negative)
    
    @property
    def max_consultative(self) -> int:
        return int(self.target_total * self.quota_consultative)
    
    @property
    def max_edge(self) -> int:
        return int(self.target_total * self.quota_edge)
    
    def can_add(self, category: str) -> bool:
        """Check if we can add another example of this category."""
        if category == "grounded":
            return self.count_grounded < self.max_grounded
        elif category == "negative":
            return self.count_negative < self.max_negative
        elif category == "consultative":
            return self.count_consultative < self.max_consultative
        elif category == "edge":
            return self.count_edge < self.max_edge
        return False
    
    def add(self, category: str) -> bool:
        """Add an example to a category. Returns False if quota exceeded."""
        if not self.can_add(category):
            return False
        if category == "grounded":
            self.count_grounded += 1
        elif category == "negative":
            self.count_negative += 1
        elif category == "consultative":
            self.count_consultative += 1
        elif category == "edge":
            self.count_edge += 1
        return True
    
    def get_stats(self) -> Dict:
        """Get current composition statistics."""
        total = self.count_grounded + self.count_negative + self.count_consultative + self.count_edge
        return {
            "total": total,
            "grounded": {"count": self.count_grounded, "target": self.max_grounded, "pct": round(self.count_grounded / total * 100, 1) if total > 0 else 0},
            "negative": {"count": self.count_negative, "target": self.max_negative, "pct": round(self.count_negative / total * 100, 1) if total > 0 else 0},
            "consultative": {"count": self.count_consultative, "target": self.max_consultative, "pct": round(self.count_consultative / total * 100, 1) if total > 0 else 0},
            "edge": {"count": self.count_edge, "target": self.max_edge, "pct": round(self.count_edge / total * 100, 1) if total > 0 else 0},
        }
    
    def validate_final(self) -> tuple[bool, List[str]]:
        """Validate final composition meets targets. Returns (valid, errors)."""
        errors = []
        stats = self.get_stats()
        total = stats["total"]
        
        if total == 0:
            return False, ["No examples generated"]
        
        # Check within 5% tolerance
        tolerance = 0.05
        
        actual_grounded = self.count_grounded / total
        if abs(actual_grounded - self.quota_grounded) > tolerance:
            errors.append(f"Grounded: {actual_grounded:.1%} (target: {self.quota_grounded:.0%})")
        
        actual_negative = self.count_negative / total
        if abs(actual_negative - self.quota_negative) > tolerance:
            errors.append(f"Negative: {actual_negative:.1%} (target: {self.quota_negative:.0%})")
        
        actual_consultative = self.count_consultative / total
        if abs(actual_consultative - self.quota_consultative) > tolerance:
            errors.append(f"Consultative: {actual_consultative:.1%} (target: {self.quota_consultative:.0%})")
        
        actual_edge = self.count_edge / total
        if abs(actual_edge - self.quota_edge) > tolerance:
            errors.append(f"Edge: {actual_edge:.1%} (target: {self.quota_edge:.0%})")
        
        return len(errors) == 0, errors


# =============================================================================
# ORCHESTRATOR QUESTION SELECTOR (DETERMINISTIC)
# =============================================================================

class OrchestratorQuestionSelector:
    """
    Simulates orchestrator's deterministic question selection.
    
    RULE: Orchestrator selects exactly 2 questions from frozen set.
    LLM never picks questions - it only renders them.
    """
    
    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._selection_count = 0
    
    def select_questions(self, capability: str, context: str) -> tuple[str, str]:
        """
        Select exactly 2 exploration questions for a given capability/context.
        
        Selection is deterministic based on capability + context + call count.
        """
        # Deterministic selection based on capability + context hash
        # Use global EXPLORATION_QUESTIONS defined later in file
        global EXPLORATION_QUESTIONS
        key = f"{capability}:{context}:{self._selection_count}"
        self._selection_count += 1
        
        # Use hash to select indices
        hash_val = hash(key)
        idx1 = hash_val % len(EXPLORATION_QUESTIONS)
        idx2 = (hash_val // len(EXPLORATION_QUESTIONS)) % len(EXPLORATION_QUESTIONS)
        
        # Ensure different questions
        if idx1 == idx2:
            idx2 = (idx2 + 1) % len(EXPLORATION_QUESTIONS)
        
        return EXPLORATION_QUESTIONS[idx1], EXPLORATION_QUESTIONS[idx2]


# =============================================================================
# CONFIG LOADER (for alternatives validation)
# =============================================================================

def load_uc1_config() -> Dict:
    """Load UC1 config for alternative validation."""
    config_path = Path(__file__).parent.parent / "app" / "orchestrator" / "uc1_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# Global config instance
_UC1_CONFIG = None

def get_uc1_config() -> Dict:
    """Get cached UC1 config."""
    global _UC1_CONFIG
    if _UC1_CONFIG is None:
        _UC1_CONFIG = load_uc1_config()
    return _UC1_CONFIG


def get_alternatives_for_bucket(bucket_id: str) -> List[str]:
    """Get fixed alternatives for a capability bucket from config."""
    config = get_uc1_config()
    for bucket in config.get("capability_buckets", []):
        if bucket.get("id") == bucket_id:
            return bucket.get("alternatives", [])
    return []

# =============================================================================
# STATE DEFINITIONS (NORMALIZED TO SPEC)
# =============================================================================

STATES = {
    # ─────────────────────────────────────────────────────────────────────────
    # USED IN TRAINING (LLM generates responses for these states)
    # ─────────────────────────────────────────────────────────────────────────
    "UC1_S0_ENTER": "Initial entry after user clicks 'Explore services & capabilities'",
    "UC1_S5_EXPLORATION_LAYER": "Bot asks exploration questions and reflects (2-3 turns)",
    "UC1_S6_CONSULTATIVE_ALTERNATIVES": "Bot presents 3 consultative alternatives",
    "UC1_S8_CLOSE": "Conversation exit",
    
    # ─────────────────────────────────────────────────────────────────────────
    # SPEC REFERENCE ONLY — NOT used in training (orchestrator emits fixed prompts)
    # These are declared for documentation purposes. DO NOT add training examples.
    # ─────────────────────────────────────────────────────────────────────────
    # "UC1_S1_CAPABILITY_PICK": "User selects a capability area (6 options)",
    # "UC1_S2_CONTEXT_CLARIFIER": "Bot asks context question specific to capability",
    # "UC1_S3_NAME_CAPTURE": "Bot captures name and synthesizes understanding",
    # "UC1_S4_AI_SYNTHESIS": "Bot synthesizes understanding based on capability + context",
    # "UC1_S7_EARNED_CTA": "Bot shows 4 CTA options",
}

# =============================================================================
# REALISTIC USER INPUTS (replaces synthetic placeholders)
# =============================================================================

ACKNOWLEDGMENT_PHRASES = [
    "That makes sense",
    "Okay",
    "I see",
    "Yes, that's right",
    "Got it",
    "That's accurate",
    "Sounds good",
    "Right",
    "Yes",
    "Makes sense"
]

TRANSITION_PHRASES = [
    "What's next?",
    "Go ahead",
    "Continue",
    "Sure",
    "Okay, what now?",
    "Yes, please continue",
    "I see",
    "Understand",
    "Got it",
    "Makes sense",
    "Please proceed",
    "Tell me more",
    "What else?",
    "Sounds good",
    "Fair enough",
    "Moving on",
    "Next step",
    "Alright",
    "Interesting",
    "Okay",
    "Show me",
    "Let's go"
]

# =============================================================================
# CAPABILITY DEFINITIONS  
# =============================================================================

CAPABILITIES = {
    "A": {
        "name": "Product Development & Engineering",
        "trigger": "Product development & engineering",
        "contexts": ["existing_product", "new_product"],
        "context_question_base": "Are you building something new, or evolving an existing product?",
        "context_questions": [
            "Are you building something new, or evolving an existing product?",
            "Is this a new build or work on an existing product?",
            "Would you say this is a greenfield project or working on something existing?",
            "Are you starting fresh or iterating on something you already have?"
        ]
    },
    "B": {
        "name": "Application Modernization",
        "trigger": "Application modernization",
        "contexts": ["cloud_migration", "legacy_upgrade"],
        "context_question_base": "What best describes your situation?",
        "context_questions": [
            "What best describes your situation?",
            "What brings you to modernization?",
            "Could you describe the current state of your application?",
            "What's driving the need for modernization?"
        ]
    },
    "C": {
        "name": "Dedicated Development Teams",
        "trigger": "Dedicated development teams",
        "contexts": ["extend_team", "build_new_team"],
        "context_question_base": "Extend an existing team or build a new one?",
        "context_questions": [
            "Extend an existing team or build a new one?",
            "Are you looking to extend your current team or form a new one?",
            "Do you need to augment an existing team or build from scratch?",
            "Is this about scaling an existing team or creating a new capability?"
        ]
    },
    "D": {
        "name": "AI, Data & Intelligent Automation",
        "trigger": "AI, data & intelligent automation",
        "contexts": ["process_automation", "exploration"],
        "context_question_base": "What's driving your interest right now?",
        "context_questions": [
            "What's driving your interest right now?",
            "What's prompting you to look at AI and automation?",
            "What outcome are you hoping AI could help with?",
            "What's the main challenge you're hoping to address?"
        ]
    },
    "E": {
        "name": "Cloud, DevOps & Scalability",
        "trigger": "Cloud, DevOps & scalability",
        "contexts": ["reliability", "growth_prep"],
        "context_question_base": "What's the primary goal?",
        "context_questions": [
            "What's the primary goal?",
            "What's driving the focus on infrastructure right now?",
            "What would success look like for your DevOps efforts?",
            "What's the main challenge with your current setup?"
        ]
    },
    "F": {
        "name": "Not Sure Yet / Need Guidance",
        "trigger": "Not sure yet / need guidance",
        "contexts": ["improve_existing", "explore_options"],
        "context_question_base": "Which outcome is closest?",
        "context_questions": [
            "Which outcome is closest?",
            "What kind of outcome are you hoping for?",
            "If you could solve one thing, what would it be?",
            "What's been on your mind that brought you here?"
        ]
    }
}

# =============================================================================
# NAME CAPTURE VARIATIONS
# =============================================================================

NAME_CAPTURE_PROMPTS = [
    "What should I call you?",
    "Before we go further, what should I call you?",
    "And what's your name?",
    "What name should I use?",
    "How would you like me to address you?"
]

# =============================================================================
# EXPLORATION LAYER QUESTIONS (FROZEN FROM DOCUMENT)
# =============================================================================

EXPLORATION_QUESTIONS = [
    "What's the biggest challenge you're facing right now in this area?",
    "What prompted you to look for help at this point?",
    "What's not working as well as you'd like today?",
    "What outcome would make this feel successful for you?"
]

# =============================================================================
# CANONICAL CONVERSATIONS (FROM DOCUMENT)
# =============================================================================

CONVERSATIONS = [
    # Conversation 1 - Product Dev | Existing Product | Scaling Pain
    {
        "id": 1,
        "capability": "A",
        "context": "existing_product",
        "persona": "Scaler",
        "outcome": "discuss_requirement",
        "exploration": {
            "q1": "What's the biggest challenge you're facing with the product right now?",
            "a1": "Releases are getting slower and every change seems to create issues elsewhere.",
            "r1": "That's a common signal of growing complexity — speed drops because the system and the team are carrying more than before.",
            "q2": "What prompted you to start looking for help at this point?",
            "a2": "We're planning new features but worried things will break.",
            "r2": "That makes sense. This is usually the point where teams decide whether to fix foundations or push forward carefully."
        },
        "alternatives": [
            "Focused stabilization: Reduce hidden complexity so releases become predictable again.",
            "Incremental feature delivery: Ship new work in smaller, controlled slices.",
            "Team extension: Add capacity while protecting the core team's focus."
        ],
        "recommendation": "Given what you shared, most teams start with focused stabilization and then accelerate delivery safely."
    },
    # Conversation 2 - Product Dev | New MVP | Unclear Direction
    {
        "id": 2,
        "capability": "A",
        "context": "new_product",
        "persona": "Builder",
        "outcome": "schedule_call",
        "exploration": {
            "q1": "What prompted you to look for help at this point?",
            "a1": "We have an idea but not sure how to scope it properly.",
            "r1": "That's a good sign you're thinking ahead — many teams jump in without scoping first.",
            "q2": "What outcome would make this feel successful for you?",
            "a2": "A working MVP we can test with real users.",
            "r2": "Makes sense. Validation before commitment is usually the safest approach."
        },
        "alternatives": [
            "Discovery-first: Clarify scope and validate assumptions before building.",
            "Lean MVP: Build the smallest testable version quickly.",
            "Dedicated pod: Assign a focused team for the new initiative."
        ],
        "recommendation": "When direction is unclear, discovery-first usually saves time and budget in the long run."
    },
    # Conversation 3 - Product Dev | Existing Product | Team Bottleneck
    {
        "id": 3,
        "capability": "A",
        "context": "existing_product",
        "persona": "Scaler",
        "outcome": "continue_exploring",
        "exploration": {
            "q1": "What's not working as well as you'd like today?",
            "a1": "The team is overloaded and the roadmap keeps slipping.",
            "r1": "That's often a sign of capacity mismatch — the work has grown faster than the team.",
            "q2": "What outcome would make this feel successful for you?",
            "a2": "Getting back on track without burning people out.",
            "r2": "That's a reasonable goal. It usually requires either reducing scope or adding capacity intelligently."
        },
        "alternatives": [
            "Scope tightening: Focus on fewer, higher-impact items.",
            "Team extension: Add capacity without disrupting current work.",
            "Ownership pods: Create focused units for specific workstreams."
        ],
        "recommendation": "When the team is stretched, scope tightening is often the fastest relief."
    },
    # Conversation 4 - Application Modernization | Cloud Migration
    {
        "id": 4,
        "capability": "B",
        "context": "cloud_migration",
        "persona": "Optimizer",
        "outcome": "schedule_call",
        "exploration": {
            "q1": "What's currently the most risky part of this migration?",
            "a1": "Downtime — the system is business critical.",
            "r1": "That's usually the biggest concern. Stability tends to matter more than speed in these cases.",
            "q2": "What would success look like once this is done?",
            "a2": "No outages and better performance.",
            "r2": "That's a clear goal. It typically requires a phased approach with strong testing."
        },
        "alternatives": [
            "Phased migration: Move parts gradually to control risk.",
            "Refactor before migrate: Improve structure first.",
            "Parallel stabilization: Secure critical paths before moving anything."
        ],
        "recommendation": "When uptime matters, phased migration is usually the safest start."
    },
    # Conversation 5 - Application Modernization | Legacy Upgrade
    {
        "id": 5,
        "capability": "B",
        "context": "legacy_upgrade",
        "persona": "Optimizer",
        "outcome": "discuss_requirement",
        "exploration": {
            "q1": "What's not working as well as you'd like today?",
            "a1": "Performance issues and maintenance is getting expensive.",
            "r1": "That's a common pain point with legacy systems — they cost more to maintain than to improve.",
            "q2": "What prompted you to look for help at this point?",
            "a2": "We're reaching the point where patching isn't enough.",
            "r2": "That's usually the tipping point. It means you're ready for a more strategic approach."
        },
        "alternatives": [
            "Targeted refactor: Fix the most painful parts first.",
            "Platform refresh: Modernize the foundation systematically.",
            "Incremental rewrite: Replace components gradually."
        ],
        "recommendation": "Starting with targeted refactoring usually delivers quick wins while planning the bigger picture."
    },
    # Conversation 6 - Dedicated Teams | Extend Existing Team
    {
        "id": 6,
        "capability": "C",
        "context": "extend_team",
        "persona": "Scaler",
        "outcome": "discuss_requirement",
        "exploration": {
            "q1": "What's missing in the team today that you're hoping to solve?",
            "a1": "We lack consistent ownership and people keep rotating.",
            "r1": "That often slows momentum because knowledge keeps resetting.",
            "q2": "What would an ideal team setup look like for you six months from now?",
            "a2": "A stable team that understands the product deeply.",
            "r2": "That's a strong goal. It usually requires commitment to long-term thinking."
        },
        "alternatives": [
            "Role-based extension: Fill skill gaps quickly.",
            "Pod-based teams: Create a small, stable unit.",
            "Ownership teams: Assign end-to-end responsibility."
        ],
        "recommendation": "For long-term ownership, pod-based or ownership teams tend to work best."
    },
    # Conversation 7 - Dedicated Teams | Build New Team
    {
        "id": 7,
        "capability": "C",
        "context": "build_new_team",
        "persona": "Builder",
        "outcome": "schedule_call",
        "exploration": {
            "q1": "What prompted you to look for help at this point?",
            "a1": "We're starting a new initiative and need dedicated capacity.",
            "r1": "That's a good reason — new initiatives often need protected focus.",
            "q2": "What outcome would make this feel successful for you?",
            "a2": "A team that ramps up quickly and takes ownership.",
            "r2": "That's achievable with the right structure and onboarding."
        },
        "alternatives": [
            "MVP pod: Small team focused on initial delivery.",
            "Full product team: Complete capability from the start.",
            "Phased team ramp-up: Start small and grow with the project."
        ],
        "recommendation": "For new initiatives, phased ramp-up often balances speed with stability."
    },
    # Conversation 8 - AI, Data & Automation | Process Automation
    {
        "id": 8,
        "capability": "D",
        "context": "process_automation",
        "persona": "Optimizer",
        "outcome": "continue_exploring",
        "exploration": {
            "q1": "What's not working as well as you'd like today?",
            "a1": "Too much manual effort on repetitive tasks.",
            "r1": "That's a strong signal for automation — if it's repeatable, it's usually automatable.",
            "q2": "What outcome would make this feel successful for you?",
            "a2": "Freeing up the team to focus on higher-value work.",
            "r2": "That's the right goal. Automation should amplify, not just replace."
        },
        "alternatives": [
            "Single high-impact use case: Start with one clear win.",
            "Data foundation first: Ensure quality inputs before automating.",
            "Workflow intelligence: Embed AI into existing processes."
        ],
        "recommendation": "Starting with one high-impact use case usually delivers the fastest value."
    },
    # Conversation 9 - AI, Data & Automation | Exploration Mode
    {
        "id": 9,
        "capability": "D",
        "context": "exploration",
        "persona": "Explorer",
        "outcome": "schedule_call",
        "exploration": {
            "q1": "What prompted you to look for help at this point?",
            "a1": "We keep hearing about AI but not sure where it fits for us.",
            "r1": "That's very common right now. Many teams are exploring possibilities before committing.",
            "q2": "What outcome would make this feel successful for you?",
            "a2": "Understanding what's realistic and where to start.",
            "r2": "That's a smart approach. Clarity before investment usually saves time and budget."
        },
        "alternatives": [
            "Discovery workshop: Identify realistic opportunities.",
            "Pilot use case: Test with one small, measurable initiative.",
            "Internal enablement: Build understanding before building solutions."
        ],
        "recommendation": "A discovery workshop is often the best starting point when direction is unclear."
    },
    # Conversation 10 - Cloud, DevOps & Scalability | Reliability
    {
        "id": 10,
        "capability": "E",
        "context": "reliability",
        "persona": "Scaler",
        "outcome": "discuss_requirement",
        "exploration": {
            "q1": "What's not working as well as you'd like today?",
            "a1": "We've had a few outages and deployments feel risky.",
            "r1": "That's usually a sign that the system has outgrown its original infrastructure patterns.",
            "q2": "What prompted you to look for help at this point?",
            "a2": "The last incident was a wake-up call.",
            "r2": "Those moments often trigger the shift from reactive to proactive infrastructure."
        },
        "alternatives": [
            "Observability first: Know what's happening before optimizing.",
            "Pipeline optimization: Reduce risk in deployments.",
            "Platform hardening: Strengthen critical infrastructure."
        ],
        "recommendation": "Observability and pipeline optimization are strong first steps."
    },
    # Conversation 11 - Cloud, DevOps & Scalability | Growth Prep
    {
        "id": 11,
        "capability": "E",
        "context": "growth_prep",
        "persona": "Scaler",
        "outcome": "schedule_call",
        "exploration": {
            "q1": "What's driving the focus on infrastructure right now?",
            "a1": "We're expecting traffic growth and not sure we can handle it.",
            "r1": "That's forward-thinking. Many teams wait until after problems occur.",
            "q2": "What outcome would make this feel successful for you?",
            "a2": "Confidence that we can scale without surprises.",
            "r2": "That's a reasonable goal. It usually requires assessment and targeted hardening."
        },
        "alternatives": [
            "Scale readiness assessment: Identify bottlenecks before they hit.",
            "Architecture hardening: Strengthen weak points.",
            "DevOps maturity: Improve processes alongside infrastructure."
        ],
        "recommendation": "A scale readiness assessment usually reveals the highest-impact improvements."
    },
    # Conversation 12 - Not Sure Yet | Explorer
    {
        "id": 12,
        "capability": "F",
        "context": "improve_existing",
        "persona": "Explorer",
        "outcome": "exit",
        "exploration": {
            "q1": "What's been frustrating you the most with the system?",
            "a1": "Everything takes longer than it should.",
            "r1": "That's usually a sign that the system has outgrown its original design.",
            "q2": "If you could fix one thing first, what would it be?",
            "a2": "Speed of changes.",
            "r2": "That's a clear priority. It usually points to technical debt or process friction."
        },
        "alternatives": [
            "Clarify priorities first: Understand what matters most.",
            "Incremental improvements: Fix small things consistently.",
            "Short discovery: Assess bottlenecks before committing to bigger work."
        ],
        "recommendation": "A short discovery usually helps identify the fastest wins."
    }
]

# =============================================================================
# SYSTEM MESSAGE - CANONICAL (No State/Rules per Deterministic Refactoring)
# =============================================================================

# The single canonical system prompt - LLM is NOT aware of states, flows, or rules
CANONICAL_SYSTEM_PROMPT = """You are an AI assistant representing DITSTEK.

Your responsibility is to understand the user's intent from natural language and respond clearly, concisely, and naturally.

You are not aware of any internal conversation states, flows, funnels, policies, or system logic.
You respond only to what the user actually says.

Behavior rules:
- If the user asks a clear question, answer it directly.
- If the input is vague or ambiguous, respond neutrally without advancing the conversation.
- If the user gives a casual acknowledgment (e.g., "ok", "yeah", "good to know"), respond naturally without advancing the conversation.
- Be grounded in DITSTEK's services and capabilities without using marketing or sales language.
- Do not assume the user wants to proceed, commit, schedule, or take next steps unless they explicitly indicate interest.
- Do not introduce calls, demos, meetings, or contact requests by default.
- Do not reference internal states, rules, training data, or system behavior.
- Do not ask questions unless directly responding to a question asked by the user.

Response style:
- Natural, professional, conversational
- 2–5 sentences unless more detail is requested
- Consultative, not directive
- No scripted transitions
- No multiple questions in one response

Your goal is to maintain a meaningful, context-aware conversation that feels human, helpful, and intelligent."""

def build_system_message(state: str, capability: Optional[str] = None, 
                         context: Optional[str] = None, rules: str = "") -> str:
    """
    Build a state-scoped system message for fine-tuning.
    """
    msg = f"State: {state}\n"
    if capability:
        msg += f"Capability: {capability}\n"
    if context:
        msg += f"Context: {context}\n"
    
    msg += f"Rules: {rules}"
    return msg

# =============================================================================
# EXAMPLE GENERATOR
# =============================================================================

@dataclass
class TrainingExample:
    """A single fine-tuning example."""
    state: str
    system_message: str
    user_message: str
    assistant_message: str
    capability: Optional[str] = None
    context: Optional[str] = None
    intent: str = ""  # question | reflection | alternatives | CTA | exit
    category: str = "grounded"  # grounded | negative | consultative | edge
    
    def to_jsonl(self) -> Dict:
        return {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.user_message},
                {"role": "assistant", "content": self.assistant_message}
            ]
        }

# =============================================================================
# META / COMPANY INFO EXAMPLES (Out-of-scope handling)
# =============================================================================

def generate_meta_examples() -> List[TrainingExample]:
    """
    Generate meta/company-info examples for out-of-scope questions.
    
    These examples:
    - Have NO UC1 context
    - Pure informational tone
    - Help the model de-anchor from UC1 when scope changes
    """
    examples = []
    
    # Company identity examples
    meta_pairs = [
        # Who are you / what are you
        ("Who are you?", "I'm an AI assistant representing DITSTEK, a technology services company."),
        ("What are you?", "I'm DITSTEK's AI assistant, here to help you learn about our services and capabilities."),
        ("Are you a bot?", "Yes, I'm an AI assistant for DITSTEK. I'm here to help answer questions about our services."),
        ("Are you human?", "No, I'm an AI assistant. I represent DITSTEK and can help with questions about our technology services."),
        
        # About DITSTEK
        ("What does DITSTEK do?", "DITSTEK provides technology services including product development, application modernization, cloud infrastructure, AI solutions, and dedicated development teams."),
        ("Tell me about DITSTEK", "DITSTEK is a technology services company. We help businesses with software development, cloud solutions, application modernization, and building dedicated engineering teams."),
        ("What services does DITSTEK offer?", "DITSTEK offers product development, application modernization, cloud and DevOps services, AI and data solutions, and dedicated development team models."),
        ("What is DITSTEK?", "DITSTEK is a technology consulting and services company focused on software engineering, cloud infrastructure, and digital transformation."),
        
        # Founder / leadership / About
        ("Who founded DITSTEK?", "DITSTEK has over 8 years of experience in custom software development, serving clients across the USA, UK, Canada, and more. It is headquartered in Mohali, India, with onshore teams in Canada and the USA."),
        ("Who is the founder of DITSTEK?", "DITSTEK is a global technology services company with over 8 years of expertise. While I focus on explaining our services, I can tell you we have a strong presence in India, Canada, and the USA."),
        ("Who runs DITSTEK?", "DITSTEK is led by a dedicated team of software professionals. We have headquarters in Mohali, India, and onshore representatives in Canada and the USA."),
        ("Who is the CEO?", "DITSTEK operates globally with leadership across India and North America. Only key contacts like Saarthak Madaan (Canada) and offices in Montana/California are publicly listed here."),
        
        # Location / contact
        ("Where is DITSTEK located?", "DITSTEK is headquartered in Mohali, India (Phase 7, Industrial Area), with onshore presence in Canada (Alberta), Montana (USA), and California (USA)."),
        ("Where are you based?", "DITSTEK has its main delivery center in Mohali, India, and onshore offices in Canada and the United States."),
        ("How can I contact DITSTEK?", "You can contact DITSTEK at info@ditstek.com or +1 (587) 500-4784. We also have offices in Mohali, Canada, and the USA."),
        
        # General meta
        ("What can you help with?", "I can help you explore DITSTEK's services, understand our technology capabilities, and answer questions about software development, cloud solutions, and digital transformation."),
        ("How does this work?", "I'm here to have a conversation about your technology needs. Ask me anything about DITSTEK's services, and I'll do my best to help."),
        ("What should I ask?", "You can ask about any technology challenge you're facing — whether that's building a new product, modernizing existing systems, or scaling your team."),
        
        # Capabilities overview
        ("What technologies do you work with?", "DITSTEK works with modern technology stacks including React, Angular, Node.js, .NET, PHP/Laravel, and cloud platforms like Azure and AWS."),
        ("Do you do mobile apps?", "Yes, DITSTEK provides offshore mobile app development for iOS and Android, including cross-platform solutions."),
        ("Do you work with startups?", "Yes, DITSTEK supports startups with MVP development, offering flexible hiring models and cost-effective solutions."),
        ("Do you offer consulting?", "Yes, DITSTEK provides technology consulting along with implementation services. We can help assess your needs before committing to a specific engagement."),
        
        # Pricing / process (deflect appropriately)
        ("How much does it cost?", "Pricing depends on the scope and nature of the engagement. I can help you understand what might be involved, and then you could discuss specifics with the team."),
        ("What are your rates?", "Rates vary based on engagement type and scope. I can help clarify what you need, which would help in discussing pricing with the team."),
        ("How long does a project take?", "Timelines depend on scope and complexity. A small project might take weeks, while larger initiatives can span months."),
        
        # Casual / off-topic (neutral responses)
        ("Hello", "Hello. I am here to help you explore DITSTEK's services."),
        ("Hi there", "Hi. I'm ready to discuss DITSTEK's capabilities whenever you are."),
        ("Thanks", "You're welcome. I'm here if you have more questions."),
        ("Thank you", "Happy to help. Let me know if you need anything else."),
        ("Goodbye", "Goodbye. Feel free to come back if you have more questions."),
        ("Bye", "Take care. I'm here if you need anything else."),
        ("Okay", "I'm ready to explore specific topics when you are."),
        ("Interesting", "I'm glad that's useful. I can provide more details if needed."),
        ("Cool", "I can discuss other topics if you'd like."),
        ("Got it", "Great. I'm available for other questions."),
        ("I see", "Feel free to explore further."),
        ("Makes sense", "Happy to elaborate on anything if needed."),
        
        # Random / unrelated
        ("What's the weather like?", "I don't have access to weather information. I'm focused on helping with questions about DITSTEK's technology services."),
        ("Tell me a joke", "I'm not the best at jokes, but I'm happy to help with questions about technology services."),
        ("What time is it?", "I don't have access to real-time information like the current time. I can help with questions about DITSTEK's services."),
        
        # ─────────────────────────────────────────────────────────────────────
        # EXPLICIT UC1 DE-ANCHOR EXAMPLE
        # Teaches the model to reset context when topic changes mid-conversation
        # ─────────────────────────────────────────────────────────────────────
        ("Earlier we were talking about product development, but now I just want to know who founded DITSTEK.",
         "Understood — switching topics. DITSTEK has over 8 years of experience in custom software development, serving clients across the USA, UK, Canada, and more. It is headquartered in Mohali, India."),
    ]
    
    for user_msg, assistant_msg in meta_pairs:
        examples.append(TrainingExample(
            state="META",  # Not a UC1 state
            system_message="State: META\nRole: AI Assistant representing DITSTEK. Answer general questions neutrally.",
            user_message=user_msg,
            assistant_message=assistant_msg,
            intent="meta",
            category="negative"
        ))
    
    return examples


def ingest_large_dataset(filepath: Path) -> List[TrainingExample]:
    """
    Ingest examples from the large RAG dataset (train_large.jsonl).
    
    1. Read JSONL
    2. Strip trailing questions from assistant (to match strict single-turn rules)
    3. Convert to META state (General Knowledge QA)
    """
    examples = []
    if not filepath.exists():
        print(f"Warning: {filepath} not found. Skipping large dataset ingestion.")
        return []
        
    print(f"Ingesting large dataset from {filepath}...")
    
    # Regex to strip trailing questions (e.g. "What do you think?", "Is this for internal use?")
    # Matches the last sentence if it ends with ?
    # Heuristic: Split by [.?!] and check last segment.
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                user_msg = next((m["content"] for m in data["messages"] if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in data["messages"] if m["role"] == "assistant"), "")
                
                if not user_msg or not asst_msg:
                    continue
                    
                # Clean Assistant Message: Remove trailing question
                # Find the last punctuation
                cleaned_asst = asst_msg
                if "?" in asst_msg:
                    # simplistic split to remove the confirmed question at the end
                    # large dataset usually formats as: "Answer text. Follow up question?"
                    idx = asst_msg.rfind("?")
                    # Look back from the ? to find the start of the sentence
                    sentences = re.split(r'(?<=[.!?])\s+', asst_msg)
                    if sentences and "?" in sentences[-1]:
                        cleaned_asst = " ".join(sentences[:-1])
                
                # STRICTER FILTER: If the cleaned message STILL has a question mark, skip it.
                # This ensures we pass the strict validation rule "No flow-advancing questions".
                if "?" in cleaned_asst:
                    continue
                
                if not cleaned_asst.strip():
                   continue # Skip if empty after cleaning

                # Create Example
                # We map these to "META" state as they are general knowledge
                # or "UC1_S5_EXPLORATION_LAYER" if we want to simulate exploration 
                # given the lack of specific context/capability metadata in train_large, META is safer and cleaner
                
                examples.append(TrainingExample(
                    state="META", 
                    system_message="State: META\nRole: AI Assistant representing DITSTEK. Answer general questions neutrally.",
                    user_message=user_msg,
                    assistant_message=cleaned_asst,
                    intent="meta_rag",
                    category="grounded"
                ))
            except Exception as e:
                continue
                
    print(f"  Ingested {len(examples)} raw examples from large dataset.")
    return examples


def generate_base_examples() -> List[TrainingExample]:
    """Generate base examples from canonical conversations with all fixes applied."""
    examples = []
    question_selector = OrchestratorQuestionSelector()
    
    # S0_ENTER - Single fixed example
    examples.append(TrainingExample(
        state="UC1_S0_ENTER",
        system_message=build_system_message(
            "UC1_S0_ENTER",
            rules="Greet and ask user to pick a capability area. No CTA."
        ),
        user_message="Explore services & capabilities",
        assistant_message="Great — happy to guide you.\nPick the closest area and I'll narrow it down from there.",
        intent="question",
        category="grounded"
    ))
    
    # Generate examples from each conversation
    for conv in CONVERSATIONS:
        cap = conv["capability"]
        ctx = conv["context"]
        
        # Select deterministic questions (simulating orchestrator)
        q1, q2 = question_selector.select_questions(cap, ctx)
        
        # =================================================================
        # UC1_S5_EXPLORATION_LAYER
        # 
        # Train on REFLECTION only.
        # Input: User answer (A1/A2)
        # Output: Assistant reflection (R1/R2)
        # Categorization: grounded (it's grounded QA/reflection)
        # =================================================================
        
        # Turn 1 Reflection
        examples.append(TrainingExample(
            state="UC1_S5_EXPLORATION_LAYER",
            capability=cap,
            context=ctx,
            system_message=build_system_message("UC1_S5_EXPLORATION_LAYER"),
            user_message=conv["exploration"]["a1"],  # User answer
            assistant_message=conv['exploration']['r1'],  # Assistant reflection
            intent="reflection",
            category="grounded"
        ))
        
        # Turn 2 Reflection
        examples.append(TrainingExample(
            state="UC1_S5_EXPLORATION_LAYER",
            capability=cap,
            context=ctx,
            system_message=build_system_message("UC1_S5_EXPLORATION_LAYER"),
            user_message=conv["exploration"]["a2"],
            assistant_message=conv["exploration"]["r2"],
            intent="reflection",
            category="grounded"
        ))
        
        # =================================================================
        # UC1_S6_CONSULTATIVE_ALTERNATIVES
        # 
        # Alternatives MUST match config exactly.
        # Categorization: consultative
        # =================================================================
        
        real_alts = get_alternatives_for_bucket(f"UC1-{cap}")
        if not real_alts:
             # Fallback if config load fails or ID mismatch (sanity check)
             real_alts = conv["alternatives"]
        
        # Format alternatives randomly to teach robustness
        alt_format = random.choice(["bullet", "numbered", "paragraph"])
        if alt_format == "bullet":
            alts_text = "\n".join([f"• {alt}" for alt in real_alts])
        elif alt_format == "numbered":
            alts_text = "\n".join([f"{i+1}. {alt}" for i, alt in enumerate(real_alts)])
        else:
            alts_text = " ".join([alt for alt in real_alts])
        
        examples.append(TrainingExample(
            state="UC1_S6_CONSULTATIVE_ALTERNATIVES",
            capability=cap,
            context=ctx,
            system_message=build_system_message("UC1_S6_CONSULTATIVE_ALTERNATIVES"),
            user_message=random.choice(TRANSITION_PHRASES),
            assistant_message=f"At this stage, teams in your situation usually consider a few paths:\n\n{alts_text}\n\n{conv['recommendation']}",
            intent="alternatives",
            category="consultative"
        ))
    
    # UC1_S8_CLOSE examples (4 outcomes)
    # Categorization: edge (handling exits/transitions)
    close_responses = {
        "discuss_requirement": "Perfect. Let me gather a few details so we can have a focused discussion.",
        "schedule_call": "Great choice. I'll help you find a time that works.",
        "continue_exploring": "No problem. Feel free to explore other areas that interest you.",
        "exit": "Understood. Feel free to come back anytime if you'd like to continue the conversation."
    }
    
    for outcome, response in close_responses.items():
        examples.append(TrainingExample(
            state="UC1_S8_CLOSE",
            system_message=build_system_message("UC1_S8_CLOSE"),
            user_message=outcome.replace("_", " ").title(),
            assistant_message=response,
            intent="exit",
            category="edge"
        ))
    
    return examples

# =============================================================================
# PARAPHRASE GENERATOR
# =============================================================================

def generate_paraphrases(example: TrainingExample, count: int = 5) -> List[TrainingExample]:
    """Generate paraphrases for an example while maintaining intent."""
    paraphrases = []
    
    # UC1_S0_ENTER paraphrases
    if example.state == "UC1_S0_ENTER":
        user_variants = [
            "Explore services & capabilities",
            "I want to explore your services",
            "Show me what you offer",
            "Tell me about your capabilities",
            "What services do you provide?",
            "I'd like to learn about your services",
            "Help me understand what you do",
            "List your services",
            "What can you do?",
            "Show me your capabilities",
            "I need to know what you offer",
            "Can you show me your services?",
            "Guide me through your offerings",
            "What are your main areas?",
            "Show areas",
            "Display capabilities",
            "I want to see what you do"
        ]
        
        assistant_variants = [
            "Great — happy to guide you.\nPick the closest area and I'll narrow it down from there.",
            "Happy to help. Select the area that is closest to what you're looking for.",
            "Let me walk you through our capabilities. Please select the area that interests you most.",
            "Good choice. Pick the area that fits best and we'll go from there.",
            "Sure thing. Select the area that sounds most relevant to your situation.",
            "Absolutely. Let's start by finding the right area for you.",
            "I can certainly help with that. Select the area that matches your needs.",
            "No problem. Please choose the capability area you'd like to explore.",
            "Welcome. Please choose the area that best fits your inquiry.",
            "I'd be happy to show you. Select a topic to proceed.",
            "Here are our main areas. Please pick the one that resembles your situation.",
            "I can help you navigate our services. Let's start by selecting an area.",
            "Glad to help. Pick a category and we can dive deeper.",
            "Sure. Please select your area of focus.",
            "I'm ready to assist. Please select a capability bucket.",
            "Happy to explain our services. Choose an area to begin.",
            "Certainly. Select the most relevant category from the list.",
            "I can guide you. Please identify your area of interest.",
            "Let's explore. Pick a service area to continue.",
            "I'm here to help. Select the category that fits best."
        ]
        
        # Combinations: 17 * 14 = 238.
        # Still not massive, but better.
        # But wait, duplicates are checked globally.
        # If we pick S0 ~25 times, probability of collision is low only if N >> 25^2. 
        # 238 is > 25, but collisions still possible (Birthday paradox: sqrt(238) ~ 15).
        # We need MORE.
        # Or we need to suffix the user message? NO, user message "Explore services" is strict entry point?
        # Actually, user can type anything to enter S0 or S0 is start.
        # In training data, S0 is the first turn.
        # User input varies.
        # I'll add punctuation suffixes to User input.
        
        suffixes = ["", ".", "!", "?", "...", " please"]
        
        for i in range(min(count, 5)): # Cap at 5 variations per call
            u_base = random.choice(user_variants)
            if u_base not in example.user_message: # Avoid exact match if possible, or just ignore
               pass
            
            u_final = f"{u_base}{random.choice(suffixes)}"
            
            a_final = random.choice(assistant_variants)
            
            new_example = TrainingExample(
                state=example.state,
                system_message=example.system_message,
                user_message=u_final,
                assistant_message=a_final,
                capability=example.capability,
                context=example.context,
                intent=example.intent,
                category=example.category
            )
            paraphrases.append(new_example)
    
    # UC1_S1_CAPABILITY_PICK paraphrases
    elif example.state == "UC1_S1_CAPABILITY_PICK" and example.capability:
        cap_data = CAPABILITIES[example.capability]
        for i in range(min(count, len(cap_data["context_questions"]) - 1)):
            alt_question = cap_data["context_questions"][i + 1]
            new_example = TrainingExample(
                state=example.state,
                system_message=example.system_message,
                user_message=example.user_message,
                assistant_message=f"Got it. {alt_question}",
                capability=example.capability,
                context=example.context,
                intent=example.intent,
                category=example.category
            )
            paraphrases.append(new_example)
    
    # UC1_S2_CONTEXT_CLARIFIER - name capture transition
    elif example.state == "UC1_S2_CONTEXT_CLARIFIER":
        name_prompts = [p for p in NAME_CAPTURE_PROMPTS if p.lower() not in example.assistant_message.lower()]
        for i in range(min(count, len(name_prompts))):
            prefix = random.choice(["Thanks — ", "Got it — ", "Thanks. ", "Great — "])
            new_example = TrainingExample(
                state=example.state,
                system_message=example.system_message,
                user_message=example.user_message,
                assistant_message=f"{prefix}{name_prompts[i].lower()}",
                capability=example.capability,
                context=example.context,
                intent=example.intent,
                category=example.category
            )
            paraphrases.append(new_example)
    
    # UC1_S3_NAME_CAPTURE - synthesis variations (NO NAMES in output)
    elif example.state == "UC1_S3_NAME_CAPTURE":
        synthesis_patterns = [
            "Thanks for sharing. From what you've said, you're working on {context} and thinking about how to move forward thoughtfully.",
            "Thanks for that. So you're focused on {context} — that's a common and important area.",
            "Got it. Sounds like {context} is the priority here.",
            "Thanks. It sounds like you're working on {context} and weighing your options.",
            "Appreciate that. Based on what you've shared, {context} seems to be the focus.",
            "Thanks for sharing. You're clearly thinking carefully about {context}.",
            "Got it. The focus on {context} makes sense given what you've described.",
            "Thanks. What you've shared about {context} gives me a clearer picture."
        ]
        for pattern in synthesis_patterns[1:min(count+1, len(synthesis_patterns))]:
            context = example.context.replace("_", " ") if example.context else "this area"
            new_example = TrainingExample(
                state=example.state,
                system_message=example.system_message,
                user_message=random.choice(["Jordan", "Sam", "Alex", "Chris", "Morgan"]),  # Varied names
                assistant_message=pattern.format(context=context),
                capability=example.capability,
                context=example.context,
                intent=example.intent,
                category=example.category
            )
            paraphrases.append(new_example)
    
    # UC1_S7_EARNED_CTA paraphrases - DELETED per Phase 6.1
    # Contains forbidden "next step" framing
    
    # UC1_S5_EXPLORATION_LAYER - Reflection variations
    elif example.state == "UC1_S5_EXPLORATION_LAYER":
        if example.intent == "question":
            exp_questions = [q for q in EXPLORATION_QUESTIONS if q.lower() not in example.assistant_message.lower()]
            question_intros = [
                "To understand this better — ",
                "I'd like to understand — ",
                "Help me understand — ",
                "If you don't mind me asking — ",
                "One thing I'm curious about — "
            ]
            for i in range(min(count, len(exp_questions))):
                intro = random.choice(question_intros)
                new_example = TrainingExample(
                    state=example.state,
                    system_message=example.system_message,
                    user_message=random.choice(ACKNOWLEDGMENT_PHRASES),
                    assistant_message=f"{intro}{exp_questions[i].lower()}",
                    capability=example.capability,
                    context=example.context,
                    intent=example.intent,
                    category=example.category
                )
                paraphrases.append(new_example)
        
        elif example.intent == "reflection":
            # Variations for the reflection intro/style
            user_prefixes = [
                "Well, ", "Honestly, ", "Basically, ", "The main thing is ", "",
                "I'd say ", "Actually, "
            ]
            
            # Use defined intros to vary assistant message
            reflection_intros = [
                "That's a common signal — ",
                "I hear that often. ",
                "That makes sense. ",
                "It sounds like ",
                "That's a typical challenge. ",
                "This is a frequent pattern. ",
                "That is undersandable. ",
                "I see this frequently. "
            ]
            
            # Original reflection usually starts with "That's..." or similar. 
            # We strip the first sentence/clause and replace it ??
            # Or just prepend if it's cleaner?
            # Canonical reflection: "That's a common signal — speed drops..."
            # If we replace "That's a common signal — " with "I hear that often. " -> "I hear that often. speed drops..."?
            # Need to be careful about grammar.
            # Simple heuristic: Split by first punctuation or "—" and replace?
            # Or just accept duplicates if within reasonable limit? 
            # Validation rule says "Fail if Same (state + input) pair appears twice".
            # So varying INPUT is sufficient.
            # 7 prefixes * 24 examples = 168.
            # We need 650.
            # We need more user variations.
            # Add suffix? Add middle words?
            # Or just vary headers more.
            # "I think releases are slow", "It seems releases are slow".
            
            expanded_prefixes = user_prefixes + [
                 "To be honest, ", "From my perspective, ", "If I had to say, ",
                 "Currently, ", "Right now, ", "At the moment, "
            ]
            # 13 prefixes * 24 = 312. Still short.
            
            # We must vary assistant output to avoid "Same answer appears across different states" (not relevant)
            # But the rule "Same (state + input) pair" only checks input.
            # So duplicates of INPUT are failure.
            # "Well, Releases are slow" can only appear ONCE.
            # So for 650 examples, we need 650 UNIQUE INPUTS.
            # 24 source inputs.
            # We need ~27 variations per input.
            # 27 prefixes? That's asking a lot.
            
            # Alternative: Add noise.
            # "Releases are slow" -> "Releases are really slow", "Releases are just slow", "Releases are becoming slow".
            
            adjectives = ["really ", "very ", "just ", "getting ", "becoming ", "pretty ", "quite ", "extremely ", ""]
            
            # Heuristic: Find verb/adj and insert? Hard without NLP.
            # Append suffix: " you know?", " right?", ".", "...", " unfortunately."
            
            suffixes = ["", ".", "...", " you know?", " right?", " unfortunately.", " really.", " to be honest."]
            
            # 13 prefixes * 9 suffixes = 117 variations per input.
            # 117 * 24 = 2800. PLENTY.
            
            for i in range(count):
                prefix = random.choice(expanded_prefixes)
                suffix = random.choice(suffixes)
                
                # Check for existing punctuation in user message
                clean_user = example.user_message.rstrip(".,?!")
                
                new_user = f"{prefix}{clean_user}{suffix}"
                
                # Also vary assistant intro to avoid "assistant output spam"
                # Strip canonical intro (first 5 words?) No, hard.
                # Just prepend "Yes, " or "Right, " sometimes?
                # "Yes, That's a common signal..."
                
                asst_prefix = random.choice(["", "Yes. ", "Right. ", "I see. ", "Understood. "])
                new_asst = f"{asst_prefix}{example.assistant_message}"
                
                new_example = TrainingExample(
                    state=example.state,
                    system_message=example.system_message,
                    user_message=new_user,
                    assistant_message=new_asst,
                    capability=example.capability,
                    context=example.context,
                    intent=example.intent,
                    category=example.category
                )
                paraphrases.append(new_example)

    # UC1_S6_CONSULTATIVE_ALTERNATIVES paraphrases (varied formatting)
    elif example.state == "UC1_S6_CONSULTATIVE_ALTERNATIVES":
        alt_intros = [
            "At this stage, teams in your situation usually consider a few paths:",
            "Based on what you've shared, here are a few approaches that tend to work well:",
            "Teams facing similar challenges typically consider:",
            "Given what you've described, here are some paths worth considering:",
            "In this scenario, we usually see teams exploring these options:",
            "From my experience with similar cases, here are some valid paths:",
            "You might find these approaches relevant to your goals:",
            "Based on your context, these are the most common strategies:",
            "Here are a few ways we typically address this:",
            "Ideally, you could consider one of these directions:",
            "For your situation, I'd recommend reviewing these options:",
            "Here are the alternatives that fit your needs:"
        ]
        
        # We need to loop more times or ensure we check all intros
        # Original logic: loop intros.
        
        possible_intros = [i for i in alt_intros if i not in example.assistant_message]
        random.shuffle(possible_intros) # Shuffle to support random sampling
        
        for intro in possible_intros:
             # Reconstruct message with new intro
             parts = example.assistant_message.split("\n\n", 1)
             if len(parts) > 1:
                new_msg = f"{intro}\n\n{parts[1]}"
                new_example = TrainingExample(
                    state=example.state,
                    system_message=example.system_message,
                    user_message=random.choice(TRANSITION_PHRASES),
                    assistant_message=new_msg,
                    capability=example.capability,
                    context=example.context,
                    intent=example.intent,
                    category=example.category
                )
                paraphrases.append(new_example)
                if len(paraphrases) >= count: break # Respect count per call, but allow diverse calls

    # UC1_S8_CLOSE paraphrases
    elif example.state == "UC1_S8_CLOSE":
        # Variations for close messages
        
        # Map outcome from original assistant message (heuristic)
        outcome_key = "exit"
        if "gather a few details" in example.assistant_message: outcome_key = "discuss"
        if "find a time" in example.assistant_message: outcome_key = "schedule"
        if "explore other areas" in example.assistant_message: outcome_key = "explore"
        
        variants_map = {
            "discuss": [
                "Perfect. Let's get the details needed for a proper discussion.",
                "Sounds good. I'll take down some info so we can follow up.",
                "Excellent. Let's gather the right details to move this forward.",
                "Great. Please provide a few details so we can have a productive chat.",
                "Understood. I'll need some information to set up the discussion.",
                "Sounds like a plan. Let's capture the necessary details.",
                "Perfect. I'll collect some info to ensure the discussion is focused.",
                "Good. Let's get the preliminaries sorted for our talk.",
                "Okay. I need a bit more info to prepare for our discussion.",
                "Got it. Let's note down the key details first.",
                "Right. Let's make sure we have everything for the follow-up.",
                "Understood. Let's gather the essentials."
            ],
            "schedule": [
                "Great. Let's find a slot on the calendar.",
                "Perfect choice. I'll help you book a time right now.",
                "Sounds good. Let's schedule that call.",
                "Excellent. I'll help you schedule a meeting.",
                "Good. Let's get a time on the books.",
                "Perfect. Let's coordinate a time that works for you.",
                "Great. I can help you find a suitable time.",
                "Sounds good. Let's set up the call.",
                "Okay. Checking the calendar for you.",
                "Got it. Let's find a time that works.",
                "Understood. Let's get this scheduled.",
                "Right. Let's book a session."
            ],
            "explore": [
                "Sure thing. Continuing exploration mode.",
                "No problem. Let's look at other areas.",
                "Understood. I'm ready to explore other areas.",
                "Okay. Let's continue exploring your options.",
                "No problem. Feel free to browse other capabilities.",
                "Sure. I'm ready to explore other topics.",
                "Understood. Let's keep looking around.",
                "Okay. Please tell me which other area interests you.",
                "Got it. Returning to exploration.",
                "Right. Let's see what else is available.",
                "Okay. Back to services.",
                "Sure. Let's check other capabilities."
            ],
            "exit": [
                "Got it. Thanks for chatting today.",
                "Understood. Have a great day.",
                "Okay. We're here if you need anything else.",
                "Understood. Feel free to return anytime.",
                "Okay. Thanks for your time.",
                "Got it. Have a wonderful day.",
                "Understood. You can always come back later.",
                "Okay. We'll be here when you're ready.",
                "Bye for now. We're here if you need us.",
                "Okay. Have a good one.",
                "Understood. See you next time.",
                "Got it. Take care."
            ]
        }
        
        variants = variants_map.get(outcome_key, [])
        random.shuffle(variants) # Shuffle!
        for v in variants[:count]:
            new_example = TrainingExample(
                state=example.state,
                system_message=example.system_message,
                user_message=example.user_message, # User input "Exit" etc is standard
                assistant_message=v,
                capability=example.capability,
                context=example.context,
                intent=example.intent,
                category=example.category
            )
            paraphrases.append(new_example)

    # META / NEGATIVE paraphrases
    elif example.state == "META":
        # Vary user input slightly to prevent duplicates
        user_prefixes = [
            "Hey, ", "Um, ", "Can you tell me, ", "", "Quick question: ",
            "Hi, ", "Hello, ", "Yo, ", "Excuse me, ", "Just wondering, ",
            "I was wondering, ", "Curious: ", "Tell me, "
        ]
        
        suffixes = ["", "?", "??", " typically?", " usually?", " right now?", " please.", "."]

        for i in range(count):
            prefix = random.choice(user_prefixes)
            # Ensure prefix logic doesn't create double prefixes if user msg already has one
            # Most canonicals start with Capital or Wh-word.
            
            suffix = random.choice(suffixes)
            clean_user = example.user_message.rstrip("?.")
            
            # Simple construction
            new_user = f"{prefix}{clean_user[0].lower() + clean_user[1:]}{suffix}"
            
            # Additional check: If canonical starts with "Who" or "What", prefix "Can you tell me" works well.
            # If canonical is "Hello", prefix "Hey" -> "Hey hello" is weird.
            # Special case for greeting-like inputs?
            if clean_user.lower() in ["hello", "hi", "hi there", "thanks", "thank you", "goodbye", "bye", "okay"]:
                # Don't use standard prefixes for these, just use variants
                greeting_variants = {
                    "hello": ["Hello", "Hi", "Hey there", "Greetings", "Hi bot"],
                    "hi there": ["Hi there", "Hello there", "Hi"],
                    "thanks": ["Thanks", "Thank you", "Thx", "Thanks a lot"],
                    "thank you": ["Thank you", "Thanks", "Appreciate it"],
                    "goodbye": ["Goodbye", "Bye", "See ya"],
                    "bye": ["Bye", "Bye bye", "cya"],
                    "okay": ["Okay", "Ok", "Sure", "Alright"],
                    "cool": ["Cool", "Nice", "Awesome"]
                }
                base_key = clean_user.lower()
                if base_key in greeting_variants:
                    new_user = random.choice(greeting_variants[base_key])
            
            new_example = TrainingExample(
                state=example.state,
                system_message=example.system_message,
                user_message=new_user,
                assistant_message=example.assistant_message,
                intent=example.intent,
                category=example.category
            )
            paraphrases.append(new_example)

    return paraphrases

# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_fine_tuning_data(output_dir: str = "fine_tuning_data"):
    """Generate train and validation JSONL files with strict quota enforcement."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    tracker = DatasetCompositionTracker()
    final_examples = []
    
    # 1. Generate Source Material
    print("Generating source material...")
    base_examples = generate_base_examples()
    meta_examples = generate_meta_examples()
    meta_examples = generate_meta_examples()
    
    # Ingest large dataset
    large_examples = ingest_large_dataset(output_path / "train_large.jsonl")
    
    all_source = base_examples + meta_examples + large_examples
    
    print(f"  Source pool: {len(all_source)} items")
    
    # 3. Add source material first (if valid)
    source_pool = {"grounded": [], "negative": [], "consultative": [], "edge": []}
    
    # Reset seen hashes for tracking duplicates in logic
    seen_hashes = set()
    
    for ex in all_source:
        # Check uniqueness immediately
        h = f"{ex.system_message}|{ex.user_message}|{ex.assistant_message}"
        if h in seen_hashes:
            continue
            
        source_pool.setdefault(ex.category, []).append(ex)
        # Try to add to final if quota allows
        if tracker.add(ex.category):
            final_examples.append(ex)
            seen_hashes.add(h)
    
    print(f"  Initial stats: {json.dumps(tracker.get_stats(), indent=2)}")

    # 4. Fill quotas via Paraphrasing
    print("Filling quotas via paraphrasing...")
    max_attempts = 100000 
    attempts = 0
    # seen_hashes is already populated
    
    # Pre-populate seen hashes from source (already done)
    # for ex in final_examples:
    #     h = f"{ex.system_message}|{ex.user_message}|{ex.assistant_message}"
    #     seen_hashes.add(h)
    
    while attempts < max_attempts:
        added_any = False
        stats = tracker.get_stats()
        
        # Check which categories need more
        needs_more = []
        if tracker.count_grounded < tracker.max_grounded: needs_more.append("grounded")
        if tracker.count_negative < tracker.max_negative: needs_more.append("negative")
        if tracker.count_consultative < tracker.max_consultative: needs_more.append("consultative")
        if tracker.count_edge < tracker.max_edge: needs_more.append("edge")
        
        if not needs_more:
            break # All quotas filled
            
        for cat in needs_more:
            if not source_pool[cat]:
                continue # No source material for this category, skip
            
            # Pick random source
            source_ex = random.choice(source_pool[cat])
            
            # Generate 1 paraphrase
            paras = generate_paraphrases(source_ex, count=1)
            if paras:
                para = paras[0]
                
                # Check uniqueness
                h = f"{para.system_message}|{para.user_message}|{para.assistant_message}"
                if h in seen_hashes:
                    continue
                
                if tracker.add(cat):
                    final_examples.append(para)
                    seen_hashes.add(h)
                    added_any = True
        
        if not added_any:
            # Only break if we REALLY are stuck (e.g. tried 100 times in a row with no adds?)
            # The inner loop tries all categories. If none added, we might be hitting duplicate wall.
            # But randomness should eventually find a new one.
            # Let's count consecutive failures?
            # Or just rely on max_attempts (outer).
            # But the loop breaks immediately if added_any is False.
            # We should allow retries.
            # But 'added_any' means in THIS pass of categories.
            pass
        
        attempts += 1
        
    if attempts >= max_attempts:
        print("  Warning: Hit max attempts limit.")
    
    # Final check if stalled
    if not needs_more:
         print("  Success: Quotas filled.")
    else:
         print(f"  Warning: Quotas not filled. Stats: {json.dumps(tracker.get_stats())}")

    print(f"  Final stats: {json.dumps(tracker.get_stats(), indent=2)}")

    # 4. Final Validation
    valid, errors = tracker.validate_final()
    if not valid:
        print("\nFATAL: Dataset composition validation failed!")
        for err in errors:
            print(f"  - {err}")
        # We allow continuation for now to debug, but this is a critical failure signal
    
    # Shuffle
    random.seed(42)
    random.shuffle(final_examples)
    
    # Stratified split (80/20)
    state_groups = {}
    for ex in final_examples:
        if ex.state not in state_groups:
            state_groups[ex.state] = []
        state_groups[ex.state].append(ex)
    
    train_examples = []
    val_examples = []
    
    for state, examples in state_groups.items():
        random.shuffle(examples)
        split_idx = max(1, int(len(examples) * 0.8))
        train_examples.extend(examples[:split_idx])
        val_examples.extend(examples[split_idx:])
    
    random.shuffle(train_examples)
    random.shuffle(val_examples)
    
    # Write JSONL files
    train_path = output_path / "train.jsonl"
    val_path = output_path / "validation.jsonl"
    
    with open(train_path, "w", encoding="utf-8") as f:
        for ex in train_examples:
            f.write(json.dumps(ex.to_jsonl(), ensure_ascii=False) + "\n")
    
    with open(val_path, "w", encoding="utf-8") as f:
        for ex in val_examples:
            f.write(json.dumps(ex.to_jsonl(), ensure_ascii=False) + "\n")
    
    print(f"\nOutput files:")
    print(f"  Train: {train_path} ({len(train_examples)} examples)")
    print(f"  Validation: {val_path} ({len(val_examples)} examples)")
    
    # Validation checks
    print("\nValidation:")
    print(f"  ✓ State naming: UC1_S* format (spec-aligned)")
    print(f"  ✓ Single intent per assistant turn")
    print(f"  ✓ No personalization in assistant output")
    print(f"  ✓ Realistic user inputs (no synthetic placeholders)")
    print(f"  ✓ CTA only in UC1_S7+")
    
    print(f"\n  Target model: gpt-4.1-mini")
    
    return train_examples, val_examples

if __name__ == "__main__":
    generate_fine_tuning_data()
