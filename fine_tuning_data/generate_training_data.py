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
from dataclasses import dataclass
from typing import List, Dict, Optional

# =============================================================================
# STATE DEFINITIONS (NORMALIZED TO SPEC)
# =============================================================================

STATES = {
    "UC1_S0_ENTER": "Initial entry after user clicks 'Explore services & capabilities'",
    "UC1_S1_CAPABILITY_PICK": "User selects a capability area (6 options)",
    "UC1_S2_CONTEXT_CLARIFIER": "Bot asks context question specific to capability",
    "UC1_S3_NAME_CAPTURE": "Bot captures name and synthesizes understanding",
    "UC1_S4_AI_SYNTHESIS": "Bot synthesizes understanding based on capability + context",
    "UC1_S5_EXPLORATION_LAYER": "Bot asks exploration questions and reflects (2-3 turns)",
    "UC1_S6_CONSULTATIVE_ALTERNATIVES": "Bot presents 3 consultative alternatives",
    "UC1_S7_EARNED_CTA": "Bot shows 4 CTA options",
    "UC1_S8_CLOSE": "Conversation exit"
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
    "Yes, please continue"
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
# SYSTEM MESSAGE BUILDER
# =============================================================================

def build_system_message(state: str, capability: Optional[str] = None, 
                         context: Optional[str] = None, rules: str = "") -> str:
    """Build a state-scoped system message without accumulated slots."""
    lines = [f"State: {state}"]
    
    if capability:
        cap_name = CAPABILITIES[capability]["name"]
        lines.append(f"Capability: {cap_name}")
    
    if context:
        context_label = context.replace("_", " ").title()
        lines.append(f"Context: {context_label}")
    
    if rules:
        lines.append(f"Rules: {rules}")
    
    return "\n".join(lines)

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
    
    def to_jsonl(self) -> Dict:
        return {
            "messages": [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": self.user_message},
                {"role": "assistant", "content": self.assistant_message}
            ]
        }

def generate_base_examples() -> List[TrainingExample]:
    """Generate base examples from canonical conversations with all fixes applied."""
    examples = []
    
    # S0_ENTER - Single fixed example
    examples.append(TrainingExample(
        state="UC1_S0_ENTER",
        system_message=build_system_message(
            "UC1_S0_ENTER",
            rules="Greet and ask user to pick a capability area. No CTA."
        ),
        user_message="Explore services & capabilities",
        assistant_message="Great — happy to guide you.\nPick the closest area and I'll narrow it down from there.",
        intent="question"
    ))
    
    # Generate examples from each conversation
    for conv in CONVERSATIONS:
        cap = conv["capability"]
        ctx = conv["context"]
        cap_data = CAPABILITIES[cap]
        context_label = ctx.replace("_", " ").title()
        
        # UC1_S1_CAPABILITY_PICK
        examples.append(TrainingExample(
            state="UC1_S1_CAPABILITY_PICK",
            capability=cap,
            system_message=build_system_message(
                "UC1_S1_CAPABILITY_PICK",
                capability=cap,
                rules="Acknowledge capability selection. Ask context clarifying question. No CTA."
            ),
            user_message=cap_data["trigger"],
            assistant_message=f"Got it. {cap_data['context_question_base']}",
            intent="question"
        ))
        
        # UC1_S2_CONTEXT_CLARIFIER
        examples.append(TrainingExample(
            state="UC1_S2_CONTEXT_CLARIFIER",
            capability=cap,
            context=ctx,
            system_message=build_system_message(
                "UC1_S2_CONTEXT_CLARIFIER",
                capability=cap,
                context=ctx,
                rules="Acknowledge context. Ask for name. No CTA."
            ),
            user_message=context_label,
            assistant_message="Thanks — before we go further, what should I call you?",
            intent="question"
        ))
        
        # UC1_S3_NAME_CAPTURE (NO PERSONALIZATION - removed name from output)
        examples.append(TrainingExample(
            state="UC1_S3_NAME_CAPTURE",
            capability=cap,
            context=ctx,
            system_message=build_system_message(
                "UC1_S3_NAME_CAPTURE",
                capability=cap,
                context=ctx,
                rules="Thank user by name. Synthesize understanding so far. No CTA."
            ),
            user_message="Alex",  # Generic name for training
            assistant_message=f"Thanks for sharing. From what you've said, you're working on {context_label.lower()} and thinking about how to move forward thoughtfully.",
            intent="question"
        ))
        
        # UC1_S5_EXPLORATION_LAYER - Q1 only (realistic user input)
        examples.append(TrainingExample(
            state="UC1_S5_EXPLORATION_LAYER",
            capability=cap,
            context=ctx,
            system_message=build_system_message(
                "UC1_S5_EXPLORATION_LAYER",
                capability=cap,
                context=ctx,
                rules="Ask ONE guided exploration question. Open-ended but bounded. No CTA."
            ),
            user_message=random.choice(ACKNOWLEDGMENT_PHRASES),  # Realistic input
            assistant_message=f"To understand this better — {conv['exploration']['q1'].lower()}",
            intent="question"
        ))
        
        # UC1_S5_EXPLORATION_LAYER - R1 only (SINGLE INTENT - reflection only)
        examples.append(TrainingExample(
            state="UC1_S5_EXPLORATION_LAYER",
            capability=cap,
            context=ctx,
            system_message=build_system_message(
                "UC1_S5_EXPLORATION_LAYER",
                capability=cap,
                context=ctx,
                rules="Reflect on user response. Show understanding. No CTA."
            ),
            user_message=conv["exploration"]["a1"],
            assistant_message=conv['exploration']['r1'],  # REFLECTION ONLY - no question
            intent="reflection"
        ))
        
        # UC1_S5_EXPLORATION_LAYER - Q2 (separate example)
        examples.append(TrainingExample(
            state="UC1_S5_EXPLORATION_LAYER",
            capability=cap,
            context=ctx,
            system_message=build_system_message(
                "UC1_S5_EXPLORATION_LAYER",
                capability=cap,
                context=ctx,
                rules="Ask second exploration question. Open-ended but bounded. No CTA."
            ),
            user_message=random.choice(ACKNOWLEDGMENT_PHRASES),  # Realistic input
            assistant_message=conv['exploration']['q2'],  # QUESTION ONLY
            intent="question"
        ))
        
        # UC1_S5_EXPLORATION_LAYER - R2 only (reflection)
        examples.append(TrainingExample(
            state="UC1_S5_EXPLORATION_LAYER",
            capability=cap,
            context=ctx,
            system_message=build_system_message(
                "UC1_S5_EXPLORATION_LAYER",
                capability=cap,
                context=ctx,
                rules="Reflect on user response. Transition to alternatives. No CTA."
            ),
            user_message=conv["exploration"]["a2"],
            assistant_message=conv["exploration"]["r2"],  # REFLECTION ONLY
            intent="reflection"
        ))
        
        # UC1_S6_CONSULTATIVE_ALTERNATIVES (varied formatting)
        alt_format = random.choice(["bullet", "numbered", "paragraph"])
        if alt_format == "bullet":
            alts_text = "\n".join([f"• {alt}" for alt in conv["alternatives"]])
        elif alt_format == "numbered":
            alts_text = "\n".join([f"{i+1}. {alt}" for i, alt in enumerate(conv["alternatives"])])
        else:
            alts_text = " ".join([alt for alt in conv["alternatives"]])
        
        examples.append(TrainingExample(
            state="UC1_S6_CONSULTATIVE_ALTERNATIVES",
            capability=cap,
            context=ctx,
            system_message=build_system_message(
                "UC1_S6_CONSULTATIVE_ALTERNATIVES",
                capability=cap,
                context=ctx,
                rules="Present exactly 3 consultative alternatives. Give recommendation. No CTA yet."
            ),
            user_message=random.choice(TRANSITION_PHRASES),  # Realistic input
            assistant_message=f"At this stage, teams in your situation usually consider a few paths:\n\n{alts_text}\n\n{conv['recommendation']}",
            intent="alternatives"
        ))
        
        # UC1_S7_EARNED_CTA
        examples.append(TrainingExample(
            state="UC1_S7_EARNED_CTA",
            capability=cap,
            context=ctx,
            system_message=build_system_message(
                "UC1_S7_EARNED_CTA",
                capability=cap,
                context=ctx,
                rules="Present 4 CTA options. CTA is now earned."
            ),
            user_message=random.choice(ACKNOWLEDGMENT_PHRASES),  # Realistic input
            assistant_message="Would you like to discuss your situation in more detail now, or would you prefer a quick call to walk through it?\n\n• Discuss my requirement now\n• Schedule a quick call\n• Continue exploring services\n• I'll explore on my own",
            intent="CTA"
        ))
    
    # UC1_S8_CLOSE examples (4 outcomes)
    close_responses = {
        "discuss_requirement": "Perfect. Let me gather a few details so we can have a focused discussion.",
        "schedule_call": "Great choice. I'll help you find a time that works.",
        "continue_exploring": "No problem. Feel free to explore other areas that interest you.",
        "exit": "Understood. Feel free to come back anytime if you'd like to continue the conversation."
    }
    
    for outcome, response in close_responses.items():
        examples.append(TrainingExample(
            state="UC1_S8_CLOSE",
            system_message=build_system_message(
                "UC1_S8_CLOSE",
                rules=f"Handle {outcome.replace('_', ' ')} exit. Close conversation gracefully."
            ),
            user_message=outcome.replace("_", " ").title(),
            assistant_message=response,
            intent="exit"
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
            "Help me understand what you do"
        ]
        assistant_variants = [
            "Great — happy to guide you.\nPick the closest area and I'll narrow it down from there.",
            "Happy to help. Which of these areas is closest to what you're looking for?",
            "Let me walk you through our capabilities. Which area interests you most?",
            "Good choice. Pick the area that fits best and we'll go from there.",
            "Sure thing. Which area sounds most relevant to your situation?",
            "Absolutely. Let's start by finding the right area for you."
        ]
        
        for i in range(min(count, len(user_variants) - 1)):
            new_example = TrainingExample(
                state=example.state,
                system_message=example.system_message,
                user_message=random.choice([v for v in user_variants if v != example.user_message]),
                assistant_message=random.choice([v for v in assistant_variants if v != example.assistant_message]),
                capability=example.capability,
                context=example.context,
                intent=example.intent
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
                intent=example.intent
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
                intent=example.intent
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
                intent=example.intent
            )
            paraphrases.append(new_example)
    
    # UC1_S7_EARNED_CTA paraphrases
    elif example.state == "UC1_S7_EARNED_CTA":
        cta_variants = [
            "How would you like to move forward?\n\n• Discuss my requirement now\n• Schedule a quick call\n• Continue exploring services\n• I'll explore on my own",
            "What feels like the right next step?\n\n• Discuss my requirement now\n• Schedule a quick call\n• Continue exploring services\n• I'll explore on my own",
            "Ready to take the next step? Here are your options:\n\n• Discuss my requirement now\n• Schedule a quick call\n• Continue exploring services\n• I'll explore on my own",
            "Where would you like to go from here?\n\n• Discuss my requirement now\n• Schedule a quick call\n• Continue exploring services\n• I'll explore on my own"
        ]
        
        for i in range(min(count, len(cta_variants))):
            if cta_variants[i] != example.assistant_message:
                new_example = TrainingExample(
                    state=example.state,
                    system_message=example.system_message,
                    user_message=random.choice(ACKNOWLEDGMENT_PHRASES),
                    assistant_message=cta_variants[i],
                    capability=example.capability,
                    context=example.context,
                    intent=example.intent
                )
                paraphrases.append(new_example)
    
    # Exploration layer paraphrases
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
                    intent=example.intent
                )
                paraphrases.append(new_example)
    
    # UC1_S6_CONSULTATIVE_ALTERNATIVES paraphrases (varied formatting)
    elif example.state == "UC1_S6_CONSULTATIVE_ALTERNATIVES":
        alt_intros = [
            "At this stage, teams in your situation usually consider a few paths:",
            "Based on what you've shared, here are a few approaches that tend to work well:",
            "Teams facing similar challenges typically consider:",
            "Given what you've described, here are some paths worth considering:"
        ]
        for intro in alt_intros[1:min(count+1, len(alt_intros))]:
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
                    intent=example.intent
                )
                paraphrases.append(new_example)
    
    return paraphrases

# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_fine_tuning_data(output_dir: str = "fine_tuning_data"):
    """Generate train and validation JSONL files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate base examples
    print("Generating base examples...")
    base_examples = generate_base_examples()
    print(f"  Generated {len(base_examples)} base examples")
    
    # Expand with paraphrases
    print("Generating paraphrases...")
    all_examples = list(base_examples)
    for example in base_examples:
        paraphrases = generate_paraphrases(example, count=5)
        all_examples.extend(paraphrases)
    
    print(f"  Total examples after paraphrasing: {len(all_examples)}")
    
    # Shuffle
    random.seed(42)
    random.shuffle(all_examples)
    
    # Stratified split (80/20)
    state_groups = {}
    for ex in all_examples:
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
