"""
STEP 4: DECIDE RESPONSE STRATEGY PER QUESTION

For each question, choose ONE strategy:
1. BRIEF - Brief answer only (2-3 sentences, direct)
2. PARTIAL_FOLLOWUP - Partial answer + follow-up question (engage further)
3. DEFER - Defer details intentionally (redirect, ask clarifying question first)

Rules:
- Do NOT fully explain unless necessary
- Do NOT dump website content
- Favor engagement over completeness
"""
import json

# Load questions
with open('step3_intent_questions.json', 'r', encoding='utf-8') as f:
    INTENT_QUESTIONS = json.load(f)

# Strategy definitions
STRATEGIES = {
    "BRIEF": "Direct 2-3 sentence answer, no follow-up needed",
    "PARTIAL_FOLLOWUP": "Partial answer + ask follow-up to understand their needs better",
    "DEFER": "Don't answer directly yet - ask clarifying question first"
}

# ============================================================
# STRATEGY MAPPING
# ============================================================
QUESTION_STRATEGIES = []

# SERVICE_DISCOVERY - Mix of PARTIAL_FOLLOWUP and BRIEF
service_discovery_mapping = [
    ("What kind of software development services do you offer?", "PARTIAL_FOLLOWUP", "Broad question - give overview, ask what they're looking for"),
    ("Do you build custom software applications?", "PARTIAL_FOLLOWUP", "Yes, but ask about their specific needs"),
    ("Can you help us with mobile app development?", "PARTIAL_FOLLOWUP", "Yes, ask iOS/Android/cross-platform preference"),
    ("Do you do web application development?", "BRIEF", "Simple yes with brief capability mention"),
    ("What about cloud services?", "PARTIAL_FOLLOWUP", "Yes, ask about their current infrastructure"),
    ("Do you offer AI or chatbot development?", "PARTIAL_FOLLOWUP", "Yes, ask about their use case"),
    ("Can you modernize our old legacy systems?", "PARTIAL_FOLLOWUP", "Yes, ask what tech stack they're on"),
    ("Do you provide QA and testing services?", "BRIEF", "Yes, straightforward capability"),
    ("What is full-stack development and do you do it?", "BRIEF", "Brief explanation + confirmation"),
    ("Do you do backend development?", "BRIEF", "Simple yes"),
    ("Can you help with cross-platform mobile apps?", "PARTIAL_FOLLOWUP", "Yes, ask about target platforms"),
    ("Do you offer IT consulting?", "PARTIAL_FOLLOWUP", "Yes, understand their challenge first"),
    ("What is product engineering?", "BRIEF", "Brief definition + we do it"),
    ("Can you help with digital transformation?", "DEFER", "Broad term - ask what it means to them"),
    ("Do you build SaaS products?", "PARTIAL_FOLLOWUP", "Yes, ask about their product idea"),
    ("Can you help us build an MVP?", "PARTIAL_FOLLOWUP", "Yes, ask about their stage and idea"),
    ("Do you develop AI agents?", "PARTIAL_FOLLOWUP", "Yes, ask about their use case"),
    ("What about API development?", "BRIEF", "Yes, straightforward"),
]

# CAPABILITY_INQUIRY - Mostly BRIEF (binary answers)
capability_mapping = [
    ("Do you work with React?", "BRIEF", "Yes, it's in our core stack"),
    ("Can your team handle .NET development?", "BRIEF", "Yes, strong .NET expertise"),
    ("Do you have experience with Node.js?", "BRIEF", "Yes"),
    ("What about Python development?", "BRIEF", "Yes"),
    ("Do you work with AWS or Azure?", "BRIEF", "Yes, both"),
    ("Can you integrate with our existing systems?", "PARTIAL_FOLLOWUP", "Yes, ask what systems they have"),
    ("Do you have expertise in MongoDB?", "BRIEF", "Yes"),
    ("Can you work with PostgreSQL?", "BRIEF", "Yes"),
    ("Do you do PHP or Laravel development?", "BRIEF", "Yes"),
    ("What technologies do you specialize in?", "PARTIAL_FOLLOWUP", "Overview, then ask what they need"),
    ("Can you build HIPAA-compliant solutions?", "PARTIAL_FOLLOWUP", "Yes, ask about their healthcare context"),
    ("Do you have experience with HL7 integration?", "BRIEF", "Yes, healthcare experience"),
    ("Can you handle complex enterprise systems?", "PARTIAL_FOLLOWUP", "Yes, ask about their scale"),
    ("Do you build AI-powered features?", "PARTIAL_FOLLOWUP", "Yes, ask about their use case"),
    ("Can you handle real-time applications?", "BRIEF", "Yes"),
    ("Do you have IoT development experience?", "PARTIAL_FOLLOWUP", "Yes, ask about their IoT project"),
]

# INDUSTRY_FIT - Mostly PARTIAL_FOLLOWUP (understand their specific needs)
industry_mapping = [
    ("Do you have healthcare software experience?", "PARTIAL_FOLLOWUP", "Yes, ask about their specific need (RPM, EHR, etc.)"),
    ("Have you built HIPAA-compliant apps before?", "BRIEF", "Yes, mention compliance expertise"),
    ("Can you develop remote patient monitoring systems?", "PARTIAL_FOLLOWUP", "Yes, ask about their patient monitoring needs"),
    ("Do you understand healthcare regulations?", "BRIEF", "Yes, HIPAA/HL7 experience"),
    ("Do you work with fintech companies?", "PARTIAL_FOLLOWUP", "Yes, ask about their fintech challenge"),
    ("Have you built financial applications?", "BRIEF", "Yes, with examples"),
    ("Do you understand payment processing?", "BRIEF", "Yes"),
    ("Do you have real estate software experience?", "PARTIAL_FOLLOWUP", "Yes, ask property management/CRM/etc"),
    ("Can you build property management platforms?", "PARTIAL_FOLLOWUP", "Yes, ask about their requirements"),
    ("Have you worked on education platforms?", "PARTIAL_FOLLOWUP", "Yes, ask about their edtech vision"),
    ("Can you build e-learning solutions?", "PARTIAL_FOLLOWUP", "Yes, ask about target audience"),
    ("Do you develop retail software?", "PARTIAL_FOLLOWUP", "Yes, ask B2B/B2C/ecommerce"),
    ("Can you build e-commerce solutions?", "PARTIAL_FOLLOWUP", "Yes, ask about their store needs"),
    ("Do you have logistics software experience?", "PARTIAL_FOLLOWUP", "Yes, ask about their logistics challenge"),
    ("Can you build fleet management systems?", "BRIEF", "Yes"),
    ("Do you work with insurance companies?", "PARTIAL_FOLLOWUP", "Yes, ask about their insurance software needs"),
    ("Have you done automotive software?", "BRIEF", "Yes"),
    ("Do you have mining industry experience?", "BRIEF", "Yes"),
    ("What about agriculture software?", "BRIEF", "Yes"),
    ("Do you work with IoT projects?", "PARTIAL_FOLLOWUP", "Yes, ask about their IoT requirements"),
    ("Have you built workflow automation tools?", "PARTIAL_FOLLOWUP", "Yes, ask about their processes"),
]

# PRICING_COST - Mix of DEFER and PARTIAL_FOLLOWUP (need context first)
pricing_mapping = [
    ("How much does custom software development cost?", "DEFER", "Too broad - ask about their project first"),
    ("What's your pricing model?", "BRIEF", "Explain models, ask which interests them"),
    ("Can you give me a rough estimate?", "DEFER", "Need project details first"),
    ("How much would a mobile app cost?", "DEFER", "Depends on complexity - ask about app"),
    ("What's the cost of building an MVP?", "PARTIAL_FOLLOWUP", "Range depends - ask about their MVP"),
    ("How much cheaper is offshore development?", "BRIEF", "Typically 40-60% savings, brief explanation"),
    ("What are your hourly rates?", "PARTIAL_FOLLOWUP", "Ranges vary - ask about skills needed"),
    ("Do you offer fixed-price projects?", "BRIEF", "Yes, explain briefly"),
    ("How much does a dedicated team cost?", "PARTIAL_FOLLOWUP", "Depends on team size - ask about needs"),
    ("What's included in your pricing?", "BRIEF", "Overview of what's included"),
    ("Are there any hidden costs?", "BRIEF", "No, transparent pricing"),
    ("How do you handle budget overruns?", "BRIEF", "Explain approach briefly"),
    ("Can you work within our budget?", "DEFER", "Ask about their budget range first"),
    ("What's the minimum project size you take?", "BRIEF", "Answer directly"),
    ("How much would AI chatbot development cost?", "PARTIAL_FOLLOWUP", "Depends on complexity - ask about use case"),
    ("What factors affect the cost?", "BRIEF", "List main factors briefly"),
]

# ENGAGEMENT_MODEL - Mostly PARTIAL_FOLLOWUP (understand preferences)
engagement_mapping = [
    ("How does working with you actually work?", "PARTIAL_FOLLOWUP", "Brief overview, ask about their preference"),
    ("What engagement models do you offer?", "BRIEF", "List 3 models briefly"),
    ("What's a dedicated team model?", "BRIEF", "Brief explanation"),
    ("How does the fixed-price model work?", "BRIEF", "Brief explanation"),
    ("Can we hire developers on an hourly basis?", "BRIEF", "Yes, explain briefly"),
    ("Do you work as an extension of our team?", "BRIEF", "Yes, that's our approach"),
    ("How do we communicate during the project?", "BRIEF", "Explain communication approach"),
    ("What's your development process?", "PARTIAL_FOLLOWUP", "Brief overview, ask about their process"),
    ("How do you handle project management?", "BRIEF", "Explain approach"),
    ("Can we scale the team up or down?", "BRIEF", "Yes, flexibility explained"),
    ("What time zone do you work in?", "BRIEF", "Direct answer"),
    ("How often do we get updates?", "BRIEF", "Daily/weekly based on preference"),
    ("Who will be our point of contact?", "BRIEF", "Dedicated PM/lead"),
    ("Can we interview the developers?", "BRIEF", "Yes, absolutely"),
    ("How do you ensure code quality?", "BRIEF", "QA process overview"),
]

# COMPARISON_DIFFERENTIATION - Mix of BRIEF and PARTIAL_FOLLOWUP
comparison_mapping = [
    ("What makes you different from other development companies?", "PARTIAL_FOLLOWUP", "Key differentiators, ask what matters to them"),
    ("Why should we choose offshore over local developers?", "PARTIAL_FOLLOWUP", "Benefits, ask about their concerns"),
    ("How are you different from freelancers?", "BRIEF", "Key differences"),
    ("What's the advantage of working with you?", "PARTIAL_FOLLOWUP", "Top advantages, ask what they value"),
    ("Why hire a dedicated team instead of in-house?", "BRIEF", "Cost/speed/flexibility benefits"),
    ("How do you compare to other offshore companies?", "PARTIAL_FOLLOWUP", "Our strengths, ask about their experience"),
    ("What's your competitive advantage?", "BRIEF", "Key strengths"),
    ("Why offshore vs onshore development?", "BRIEF", "Pros explained briefly"),
    ("What do clients say about you?", "BRIEF", "Mention testimonials/ratings"),
    ("How long have you been in business?", "BRIEF", "Direct answer"),
    ("What's your track record?", "BRIEF", "Stats/experience briefly"),
    ("Do you have any certifications?", "BRIEF", "Direct answer"),
]

# PROCESS_TIMELINE - Mostly DEFER or PARTIAL_FOLLOWUP (need project context)
timeline_mapping = [
    ("How long does a typical project take?", "DEFER", "Depends on project - ask about theirs"),
    ("What's the timeline for building an MVP?", "PARTIAL_FOLLOWUP", "Range, ask about their MVP complexity"),
    ("How fast can you start?", "BRIEF", "Typically X weeks"),
    ("What's your development lifecycle?", "BRIEF", "Agile/iterative approach"),
    ("How long does the discovery phase take?", "BRIEF", "Typically X weeks"),
    ("When can we expect the first deliverables?", "PARTIAL_FOLLOWUP", "Depends on scope - ask about priorities"),
    ("How do you handle tight deadlines?", "BRIEF", "Explain approach"),
    ("What's the process from start to finish?", "PARTIAL_FOLLOWUP", "High-level overview, ask about their stage"),
    ("How long would a mobile app take to build?", "DEFER", "Depends on complexity - ask about app"),
    ("Can you speed up development if needed?", "BRIEF", "Yes, explain how"),
]

# TRUST_CREDIBILITY - Mostly BRIEF (direct answers build trust)
trust_mapping = [
    ("Can you show me some examples of your work?", "BRIEF", "Yes, offer portfolio/case studies"),
    ("Do you have case studies I can review?", "BRIEF", "Yes, offer to share"),
    ("Have you worked with companies like ours?", "PARTIAL_FOLLOWUP", "Ask about their company type first"),
    ("What's your success rate?", "BRIEF", "Stats briefly"),
    ("Can I talk to your past clients?", "BRIEF", "Yes, can arrange references"),
    ("How many projects have you completed?", "BRIEF", "Direct number"),
    ("How do you protect our intellectual property?", "BRIEF", "IP protection approach"),
    ("What about data security?", "BRIEF", "Security measures briefly"),
    ("Do you sign NDAs?", "BRIEF", "Yes, standard practice"),
    ("What happens if something goes wrong?", "BRIEF", "Explain risk mitigation"),
    ("How do you ensure quality?", "BRIEF", "QA process"),
    ("Where is your team located?", "BRIEF", "Direct answer"),
]

# FIT_QUALIFICATION - Mostly PARTIAL_FOLLOWUP (qualify the lead)
fit_mapping = [
    ("Is this suitable for a startup like us?", "PARTIAL_FOLLOWUP", "Yes, ask about their stage/needs"),
    ("We're a small company - do you work with SMBs?", "BRIEF", "Yes, we work with all sizes"),
    ("Do you only work with large enterprises?", "BRIEF", "No, work with all sizes"),
    ("We don't have technical knowledge - is that okay?", "BRIEF", "Yes, we guide you through"),
    ("I just have an idea, can you help from scratch?", "PARTIAL_FOLLOWUP", "Yes, ask about their idea"),
    ("We already have a team - can you augment it?", "PARTIAL_FOLLOWUP", "Yes, ask about gaps"),
    ("We need ongoing support - do you offer that?", "BRIEF", "Yes, support services available"),
    ("Can you handle a project of our size?", "DEFER", "Ask about their project size first"),
    ("We're on a tight budget - can you still help?", "PARTIAL_FOLLOWUP", "Options available - ask about budget range"),
    ("We need someone long-term - is that possible?", "BRIEF", "Yes, long-term partnerships"),
    ("Do you work with non-tech founders?", "BRIEF", "Yes, we translate tech for you"),
]

# VAGUE_EXPLORATORY - Mostly DEFER (need to understand intent)
vague_mapping = [
    ("I'm just looking around", "DEFER", "Welcome! Ask what brought them here"),
    ("Not sure yet", "DEFER", "That's okay - ask what they're thinking about"),
    ("Just exploring options", "DEFER", "Great - ask what options they're considering"),
    ("Tell me more about what you do", "PARTIAL_FOLLOWUP", "Brief intro, ask what they're looking for"),
    ("How does this work?", "DEFER", "Ask what specifically they want to know"),
    ("I might need something custom", "PARTIAL_FOLLOWUP", "We can help - ask about their idea"),
    ("We're thinking about building something", "DEFER", "Exciting - ask what they're thinking"),
    ("Can you help with our project?", "DEFER", "Likely yes - ask about the project"),
    ("I have an idea", "PARTIAL_FOLLOWUP", "We love ideas - ask them to share"),
    ("We need software help", "DEFER", "We're here - ask what kind of help"),
    ("What can you help us with?", "PARTIAL_FOLLOWUP", "Overview, then ask about their needs"),
    ("I'm not sure what I need", "DEFER", "That's normal - ask about their challenge"),
    ("Maybe", "DEFER", "Ask what would help them decide"),
    ("Yeah", "DEFER", "Continue conversation, ask follow-up"),
    ("Okay", "DEFER", "Acknowledge, move conversation forward"),
    ("Hmm", "DEFER", "Ask what they're thinking"),
    ("Interesting", "DEFER", "Ask what caught their attention"),
    ("Go on", "PARTIAL_FOLLOWUP", "Continue with relevant info, then ask question"),
]

# Combine all mappings
all_mappings = [
    ("SERVICE_DISCOVERY", service_discovery_mapping),
    ("CAPABILITY_INQUIRY", capability_mapping),
    ("INDUSTRY_FIT", industry_mapping),
    ("PRICING_COST", pricing_mapping),
    ("ENGAGEMENT_MODEL", engagement_mapping),
    ("COMPARISON_DIFFERENTIATION", comparison_mapping),
    ("PROCESS_TIMELINE", timeline_mapping),
    ("TRUST_CREDIBILITY", trust_mapping),
    ("FIT_QUALIFICATION", fit_mapping),
    ("VAGUE_EXPLORATORY", vague_mapping),
]

# Build final strategy map
strategy_map = {}
for intent_name, mappings in all_mappings:
    strategy_map[intent_name] = []
    for question, strategy, rationale in mappings:
        strategy_map[intent_name].append({
            "question": question,
            "strategy": strategy,
            "rationale": rationale
        })

# ============================================================
# Print output
# ============================================================
print("=" * 70)
print("STEP 4: RESPONSE STRATEGY MAPPING")
print("=" * 70)

print("\nSTRATEGY DEFINITIONS:")
for strat, desc in STRATEGIES.items():
    print(f"  [{strat}] - {desc}")

print("\n" + "=" * 70)

# Count strategies
strategy_counts = {"BRIEF": 0, "PARTIAL_FOLLOWUP": 0, "DEFER": 0}
total = 0

for intent_name, mappings in all_mappings:
    print(f"\n[{intent_name}]")
    print("-" * 50)
    for q, s, r in mappings[:5]:  # Show first 5
        print(f"  Q: {q[:50]}...")
        print(f"     -> {s}: {r}")
        strategy_counts[s] += 1
        total += 1
    remaining = len(mappings) - 5
    if remaining > 0:
        for q, s, r in mappings[5:]:
            strategy_counts[s] += 1
            total += 1
        print(f"  ... and {remaining} more questions mapped")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
TOTAL QUESTIONS MAPPED: {total}

Strategy Distribution:
  - BRIEF (direct answer): {strategy_counts['BRIEF']} ({100*strategy_counts['BRIEF']/total:.1f}%)
  - PARTIAL_FOLLOWUP (engage): {strategy_counts['PARTIAL_FOLLOWUP']} ({100*strategy_counts['PARTIAL_FOLLOWUP']/total:.1f}%)
  - DEFER (clarify first): {strategy_counts['DEFER']} ({100*strategy_counts['DEFER']/total:.1f}%)

This distribution FAVORS ENGAGEMENT:
  - {100*(strategy_counts['PARTIAL_FOLLOWUP']+strategy_counts['DEFER'])/total:.1f}% of responses will include a follow-up question
  - Only {100*strategy_counts['BRIEF']/total:.1f}% are direct answers without engagement
""")

print("=" * 70)
print("STEP 4 COMPLETE - STOPPING AS INSTRUCTED")
print("=" * 70)
print("\nAwaiting instruction to proceed to STEP 5.")

# Save for next step
with open('step4_strategy_map.json', 'w', encoding='utf-8') as f:
    json.dump(strategy_map, f, indent=2, ensure_ascii=False)

print(f"\nStrategy map saved to: step4_strategy_map.json")
