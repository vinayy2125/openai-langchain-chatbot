"""
STEP 3: EXTRACT USER-INTENT QUESTIONS
- Convert retained topics into realistic customer questions
- Questions must reflect how users actually ask
- Group by intent category
"""
import json

# Load retained pages
with open('step2_retained_pages.json', 'r', encoding='utf-8') as f:
    retained_pages = json.load(f)

# Define intent categories and extract questions based on page content/category
INTENT_QUESTIONS = {
    "SERVICE_DISCOVERY": {
        "description": "User wants to understand what services DITS offers",
        "questions": []
    },
    "CAPABILITY_INQUIRY": {
        "description": "User wants to know if DITS can handle specific tech/requirements",
        "questions": []
    },
    "INDUSTRY_FIT": {
        "description": "User wants to know if DITS has experience in their industry",
        "questions": []
    },
    "PRICING_COST": {
        "description": "User wants to understand pricing, costs, budget expectations",
        "questions": []
    },
    "ENGAGEMENT_MODEL": {
        "description": "User wants to understand how to work together",
        "questions": []
    },
    "COMPARISON_DIFFERENTIATION": {
        "description": "User wants to compare options or understand what makes DITS different",
        "questions": []
    },
    "PROCESS_TIMELINE": {
        "description": "User wants to understand how work gets done and timelines",
        "questions": []
    },
    "TRUST_CREDIBILITY": {
        "description": "User wants proof points, case studies, experience validation",
        "questions": []
    },
    "FIT_QUALIFICATION": {
        "description": "User wants to know if they're a good fit (size, budget, needs)",
        "questions": []
    },
    "VAGUE_EXPLORATORY": {
        "description": "User is early-stage, vague, just exploring",
        "questions": []
    }
}

# ============================================================
# SERVICE_DISCOVERY - What do you do?
# ============================================================
INTENT_QUESTIONS["SERVICE_DISCOVERY"]["questions"] = [
    # Core services
    "What kind of software development services do you offer?",
    "Do you build custom software applications?",
    "Can you help us with mobile app development?",
    "Do you do web application development?",
    "What about cloud services?",
    "Do you offer AI or chatbot development?",
    "Can you modernize our old legacy systems?",
    "Do you provide QA and testing services?",
    "What is full-stack development and do you do it?",
    "Do you do backend development?",
    "Can you help with cross-platform mobile apps?",
    "Do you offer IT consulting?",
    "What is product engineering?",
    "Can you help with digital transformation?",
    "Do you build SaaS products?",
    "Can you help us build an MVP?",
    "Do you develop AI agents?",
    "What about API development?",
]

# ============================================================
# CAPABILITY_INQUIRY - Can you do X?
# ============================================================
INTENT_QUESTIONS["CAPABILITY_INQUIRY"]["questions"] = [
    # Technology specific
    "Do you work with React?",
    "Can your team handle .NET development?",
    "Do you have experience with Node.js?",
    "What about Python development?",
    "Do you work with AWS or Azure?",
    "Can you integrate with our existing systems?",
    "Do you have expertise in MongoDB?",
    "Can you work with PostgreSQL?",
    "Do you do PHP or Laravel development?",
    "What technologies do you specialize in?",
    "Can you build HIPAA-compliant solutions?",
    "Do you have experience with HL7 integration?",
    "Can you handle complex enterprise systems?",
    "Do you build AI-powered features?",
    "Can you handle real-time applications?",
    "Do you have IoT development experience?",
]

# ============================================================
# INDUSTRY_FIT - Do you know my industry?
# ============================================================
INTENT_QUESTIONS["INDUSTRY_FIT"]["questions"] = [
    # Healthcare
    "Do you have healthcare software experience?",
    "Have you built HIPAA-compliant apps before?",
    "Can you develop remote patient monitoring systems?",
    "Do you understand healthcare regulations?",
    # Fintech
    "Do you work with fintech companies?",
    "Have you built financial applications?",
    "Do you understand payment processing?",
    # Real Estate
    "Do you have real estate software experience?",
    "Can you build property management platforms?",
    # EdTech
    "Have you worked on education platforms?",
    "Can you build e-learning solutions?",
    # Retail
    "Do you develop retail software?",
    "Can you build e-commerce solutions?",
    # Logistics
    "Do you have logistics software experience?",
    "Can you build fleet management systems?",
    # Other industries
    "Do you work with insurance companies?",
    "Have you done automotive software?",
    "Do you have mining industry experience?",
    "What about agriculture software?",
    "Do you work with IoT projects?",
    "Have you built workflow automation tools?",
]

# ============================================================
# PRICING_COST - How much does it cost?
# ============================================================
INTENT_QUESTIONS["PRICING_COST"]["questions"] = [
    "How much does custom software development cost?",
    "What's your pricing model?",
    "Can you give me a rough estimate?",
    "How much would a mobile app cost?",
    "What's the cost of building an MVP?",
    "How much cheaper is offshore development?",
    "What are your hourly rates?",
    "Do you offer fixed-price projects?",
    "How much does a dedicated team cost?",
    "What's included in your pricing?",
    "Are there any hidden costs?",
    "How do you handle budget overruns?",
    "Can you work within our budget?",
    "What's the minimum project size you take?",
    "How much would AI chatbot development cost?",
    "What factors affect the cost?",
]

# ============================================================
# ENGAGEMENT_MODEL - How do we work together?
# ============================================================
INTENT_QUESTIONS["ENGAGEMENT_MODEL"]["questions"] = [
    "How does working with you actually work?",
    "What engagement models do you offer?",
    "What's a dedicated team model?",
    "How does the fixed-price model work?",
    "Can we hire developers on an hourly basis?",
    "Do you work as an extension of our team?",
    "How do we communicate during the project?",
    "What's your development process?",
    "How do you handle project management?",
    "Can we scale the team up or down?",
    "What time zone do you work in?",
    "How often do we get updates?",
    "Who will be our point of contact?",
    "Can we interview the developers?",
    "How do you ensure code quality?",
]

# ============================================================
# COMPARISON_DIFFERENTIATION - Why you vs others?
# ============================================================
INTENT_QUESTIONS["COMPARISON_DIFFERENTIATION"]["questions"] = [
    "What makes you different from other development companies?",
    "Why should we choose offshore over local developers?",
    "How are you different from freelancers?",
    "What's the advantage of working with you?",
    "Why hire a dedicated team instead of in-house?",
    "How do you compare to other offshore companies?",
    "What's your competitive advantage?",
    "Why offshore vs onshore development?",
    "What do clients say about you?",
    "How long have you been in business?",
    "What's your track record?",
    "Do you have any certifications?",
]

# ============================================================
# PROCESS_TIMELINE - How long will it take?
# ============================================================
INTENT_QUESTIONS["PROCESS_TIMELINE"]["questions"] = [
    "How long does a typical project take?",
    "What's the timeline for building an MVP?",
    "How fast can you start?",
    "What's your development lifecycle?",
    "How long does the discovery phase take?",
    "When can we expect the first deliverables?",
    "How do you handle tight deadlines?",
    "What's the process from start to finish?",
    "How long would a mobile app take to build?",
    "Can you speed up development if needed?",
]

# ============================================================
# TRUST_CREDIBILITY - Can we trust you?
# ============================================================
INTENT_QUESTIONS["TRUST_CREDIBILITY"]["questions"] = [
    "Can you show me some examples of your work?",
    "Do you have case studies I can review?",
    "Have you worked with companies like ours?",
    "What's your success rate?",
    "Can I talk to your past clients?",
    "How many projects have you completed?",
    "How do you protect our intellectual property?",
    "What about data security?",
    "Do you sign NDAs?",
    "What happens if something goes wrong?",
    "How do you ensure quality?",
    "Where is your team located?",
]

# ============================================================
# FIT_QUALIFICATION - Are we a good fit?
# ============================================================
INTENT_QUESTIONS["FIT_QUALIFICATION"]["questions"] = [
    "Is this suitable for a startup like us?",
    "We're a small company - do you work with SMBs?",
    "Do you only work with large enterprises?",
    "We don't have technical knowledge - is that okay?",
    "I just have an idea, can you help from scratch?",
    "We already have a team - can you augment it?",
    "We need ongoing support - do you offer that?",
    "Can you handle a project of our size?",
    "We're on a tight budget - can you still help?",
    "We need someone long-term - is that possible?",
    "Do you work with non-tech founders?",
]

# ============================================================
# VAGUE_EXPLORATORY - Just browsing / unclear
# ============================================================
INTENT_QUESTIONS["VAGUE_EXPLORATORY"]["questions"] = [
    "I'm just looking around",
    "Not sure yet",
    "Just exploring options",
    "Tell me more about what you do",
    "How does this work?",
    "I might need something custom",
    "We're thinking about building something",
    "Can you help with our project?",
    "I have an idea",
    "We need software help",
    "What can you help us with?",
    "I'm not sure what I need",
    "Maybe",
    "Yeah",
    "Okay",
    "Hmm",
    "Interesting",
    "Go on",
]

# ============================================================
# Print output
# ============================================================
print("=" * 70)
print("STEP 3: USER-INTENT QUESTIONS EXTRACTED")
print("=" * 70)

total_questions = 0
for intent, data in INTENT_QUESTIONS.items():
    questions = data["questions"]
    total_questions += len(questions)
    print(f"\n[{intent}]")
    print(f"Description: {data['description']}")
    print(f"Questions: {len(questions)}")
    print("-" * 50)
    for q in questions[:8]:  # Show first 8
        print(f"   ? {q}")
    if len(questions) > 8:
        print(f"   ... and {len(questions) - 8} more")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
INTENT CATEGORIES: {len(INTENT_QUESTIONS)}
TOTAL QUESTIONS EXTRACTED: {total_questions}

Distribution:
  - SERVICE_DISCOVERY: {len(INTENT_QUESTIONS['SERVICE_DISCOVERY']['questions'])} questions
  - CAPABILITY_INQUIRY: {len(INTENT_QUESTIONS['CAPABILITY_INQUIRY']['questions'])} questions
  - INDUSTRY_FIT: {len(INTENT_QUESTIONS['INDUSTRY_FIT']['questions'])} questions
  - PRICING_COST: {len(INTENT_QUESTIONS['PRICING_COST']['questions'])} questions
  - ENGAGEMENT_MODEL: {len(INTENT_QUESTIONS['ENGAGEMENT_MODEL']['questions'])} questions
  - COMPARISON_DIFFERENTIATION: {len(INTENT_QUESTIONS['COMPARISON_DIFFERENTIATION']['questions'])} questions
  - PROCESS_TIMELINE: {len(INTENT_QUESTIONS['PROCESS_TIMELINE']['questions'])} questions
  - TRUST_CREDIBILITY: {len(INTENT_QUESTIONS['TRUST_CREDIBILITY']['questions'])} questions
  - FIT_QUALIFICATION: {len(INTENT_QUESTIONS['FIT_QUALIFICATION']['questions'])} questions
  - VAGUE_EXPLORATORY: {len(INTENT_QUESTIONS['VAGUE_EXPLORATORY']['questions'])} questions
""")

print("=" * 70)
print("STEP 3 COMPLETE - STOPPING AS INSTRUCTED")
print("=" * 70)
print("\nAwaiting instruction to proceed to STEP 4.")

# Save for next step
with open('step3_intent_questions.json', 'w', encoding='utf-8') as f:
    json.dump(INTENT_QUESTIONS, f, indent=2, ensure_ascii=False)

print(f"\nQuestions saved to: step3_intent_questions.json")
