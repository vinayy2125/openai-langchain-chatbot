"""
STEP 5: AUTHOR TRAINING EXAMPLES (JSONL)

This script converts the Step 4 strategy map into training examples.
Each example follows the exact format required for fine-tuning:

{
  "messages": [
    {"role": "system", "content": "State: <STATE_NAME>\nRules: Be concise. Be consultative. Avoid over-explaining."},
    {"role": "user", "content": "<realistic user question>"},
    {"role": "assistant", "content": "<2–5 sentence guided response>"}
  ]
}

CONSTRAINTS:
- Assistant response: 2–5 sentences max
- Conversational, not brochure-like
- No exhaustive lists
- No technical dumps
- Ask follow-up questions only if strategy requires it
"""

import json
from typing import Dict, List
from datetime import datetime

# Load Step 4 strategy map
with open('step4_strategy_map.json', 'r', encoding='utf-8') as f:
    STRATEGY_MAP = json.load(f)

# State names for different intent categories
STATE_MAPPING = {
    "SERVICE_DISCOVERY": "GENERAL_INQUIRY",
    "CAPABILITY_INQUIRY": "CAPABILITY_CHECK",
    "INDUSTRY_FIT": "INDUSTRY_FIT",
    "PRICING_COST": "PRICING_INQUIRY",
    "ENGAGEMENT_MODEL": "ENGAGEMENT_MODEL",
    "COMPARISON_DIFFERENTIATION": "DIFFERENTIATION",
    "PROCESS_TIMELINE": "TIMELINE_INQUIRY",
    "TRUST_CREDIBILITY": "TRUST_BUILDING",
    "FIT_QUALIFICATION": "FIT_CHECK",
    "VAGUE_EXPLORATORY": "EXPLORATORY"
}

# Response templates by strategy
# BRIEF = Direct answer, 2-3 sentences
# PARTIAL_FOLLOWUP = Partial answer + follow-up question, 3-4 sentences
# DEFER = Ask clarifying question, 2-3 sentences

def generate_brief_response(question: str, category: str, rationale: str) -> str:
    """Generate a brief, direct response (2-3 sentences)."""
    
    responses = {
        # SERVICE_DISCOVERY - BRIEF
        "Do you do web application development?": "Yes, we build custom web applications using modern frameworks like React, Angular, and .NET. Our team handles everything from frontend to backend development.",
        
        "Do you provide QA and testing services?": "Yes, we offer comprehensive QA and testing services including automated testing, manual testing, and performance testing. Quality is built into our development process.",
        
        "What is full-stack development and do you do it?": "Full-stack development means handling both frontend (what users see) and backend (server, database) work. Yes, we specialize in full-stack development across multiple technology stacks.",
        
        "Do you do backend development?": "Yes, we develop robust backend systems using Node.js, .NET, Python, and PHP. We focus on scalable, secure architectures.",
        
        "What about API development?": "Yes, we build APIs and microservices that are secure, well-documented, and designed for scalability.",
        
        # CAPABILITY_INQUIRY - BRIEF
        "Do you work with React?": "Yes, React is one of our core frontend technologies. We've delivered numerous production applications with it.",
        
        "Can your team handle .NET development?": "Yes, we have strong .NET expertise across the stack including ASP.NET, .NET Core, and related technologies.",
        
        "Do you have experience with Node.js?": "Yes, Node.js is part of our core backend stack. We use it for APIs, real-time applications, and microservices.",
        
        "What about Python development?": "Yes, we work with Python for backend development, data processing, and AI/ML projects.",
        
        "Do you work with AWS or Azure?": "Yes, we work with both AWS and Azure. Our team is experienced in cloud architecture, deployment, and management on both platforms.",
        
        "Do you have expertise in MongoDB?": "Yes, we use MongoDB for projects requiring flexible, document-based data storage.",
        
        "Can you work with PostgreSQL?": "Yes, PostgreSQL is one of our primary relational database choices.",
        
        "Do you do PHP or Laravel development?": "Yes, we have experienced PHP and Laravel developers for web application projects.",
        
        "Do you have experience with HL7 integration?": "Yes, we've integrated HL7 in healthcare projects. This is part of our healthcare compliance expertise.",
        
        "Can you handle real-time applications?": "Yes, we build real-time applications using WebSockets, Socket.io, and similar technologies.",
        
        # INDUSTRY_FIT - BRIEF
        "Have you built HIPAA-compliant apps before?": "Yes, we've built multiple HIPAA-compliant applications. We understand the security and privacy requirements.",
        
        "Do you understand healthcare regulations?": "Yes, we have experience with HIPAA, HL7, and related healthcare compliance standards.",
        
        "Have you built financial applications?": "Yes, we've developed fintech solutions including payment systems and financial management platforms.",
        
        "Do you understand payment processing?": "Yes, we've integrated payment gateways and built secure transaction systems.",
        
        "Can you build fleet management systems?": "Yes, we've built fleet management and logistics tracking solutions.",
        
        "Have you done automotive software?": "Yes, we have experience in automotive software development.",
        
        "Do you have mining industry experience?": "Yes, we've worked on software solutions for the mining industry.",
        
        "What about agriculture software?": "Yes, we've developed agricultural technology solutions.",
        
        # PRICING_COST - BRIEF
        "Do you offer fixed-price projects?": "Yes, we offer fixed-price engagements for projects with well-defined scope. It provides budget certainty.",
        
        "How much cheaper is offshore development?": "Offshore development typically offers 40-60% cost savings compared to onshore rates, while maintaining quality through structured processes.",
        
        "What's included in your pricing?": "Our pricing typically includes development, project management, QA, and regular communication. Specific inclusions depend on the engagement model.",
        
        "Are there any hidden costs?": "No, we're transparent about pricing. All costs are discussed upfront before starting.",
        
        "How do you handle budget overruns?": "We use agile methodologies with regular checkpoints to catch scope changes early. Any potential overruns are flagged and discussed before proceeding.",
        
        "What's the minimum project size you take?": "We typically work on projects starting from around $10K, but it depends on the scope and engagement model.",
        
        "What factors affect the cost?": "Main factors include project complexity, timeline, technology stack, and team size. We can discuss your specific situation.",
        
        # ENGAGEMENT_MODEL - BRIEF
        "What engagement models do you offer?": "We offer three main models: dedicated teams for ongoing work, fixed-price for defined projects, and hourly for flexible engagements.",
        
        "What's a dedicated team model?": "A dedicated team works exclusively on your project, functions as an extension of your team, and scales with your needs.",
        
        "How does the fixed-price model work?": "Fixed-price means we agree on scope and price upfront. You get budget certainty and we deliver the defined requirements.",
        
        "Can we hire developers on an hourly basis?": "Yes, hourly engagement works well for projects with evolving scope or when you need flexibility.",
        
        "Do you work as an extension of our team?": "Yes, that's exactly our approach. We integrate with your processes, tools, and workflows.",
        
        "How do we communicate during the project?": "We use your preferred tools — Slack, Teams, email, daily standups. Communication is built into our process.",
        
        "How do you handle project management?": "We assign a dedicated project manager who coordinates the team, manages timelines, and ensures clear communication.",
        
        "Can we scale the team up or down?": "Yes, our models are designed for flexibility. You can adjust team size based on project needs.",
        
        "What time zone do you work in?": "Our teams are in India (IST), but we overlap with US, UK, and other time zones as needed.",
        
        "How often do we get updates?": "Typically daily standups and weekly progress reports, but we adjust to your preferences.",
        
        "Who will be our point of contact?": "You'll have a dedicated project manager or tech lead as your primary contact.",
        
        "Can we interview the developers?": "Absolutely. You can interview and approve team members before they start.",
        
        "How do you ensure code quality?": "We use code reviews, automated testing, CI/CD pipelines, and follow coding standards consistently.",
        
        # COMPARISON_DIFFERENTIATION - BRIEF
        "How are you different from freelancers?": "We're a structured team with project management, QA, and backup resources. Freelancers work alone — we provide continuity and accountability.",
        
        "Why hire a dedicated team instead of in-house?": "Dedicated teams offer faster ramp-up, lower overhead costs, and easier scaling compared to building in-house.",
        
        "What's your competitive advantage?": "We combine offshore cost efficiency with a consultative approach — we don't just build, we help you make better decisions.",
        
        "Why offshore vs onshore development?": "Offshore offers 40-60% cost savings while providing access to a larger talent pool. With proper processes, quality is maintained.",
        
        "What do clients say about you?": "Clients appreciate our communication, ownership, and problem-solving. We'd be happy to share references.",
        
        "How long have you been in business?": "We've been operating for over 10 years, with experience across multiple industries and technologies.",
        
        "What's your track record?": "We've completed 500+ projects across various industries with a strong client retention rate.",
        
        "Do you have any certifications?": "Yes, our team holds relevant certifications in cloud platforms, security, and specific technologies as applicable.",
        
        # PROCESS_TIMELINE - BRIEF
        "How fast can you start?": "We can typically start within 1-2 weeks after finalizing requirements. For urgent needs, we can expedite.",
        
        "What's your development lifecycle?": "We follow agile methodology with 2-week sprints, regular demos, and continuous feedback loops.",
        
        "How long does the discovery phase take?": "Discovery typically takes 2-4 weeks depending on complexity. It sets the foundation for everything that follows.",
        
        "How do you handle tight deadlines?": "We can accelerate by adding resources or parallelizing work, but we're always honest about what's realistic.",
        
        "Can you speed up development if needed?": "Yes, we can add team members or work in parallel tracks. We'll discuss options that maintain quality.",
        
        # TRUST_CREDIBILITY - BRIEF
        "Can you show me some examples of your work?": "Yes, we have case studies and portfolio examples. I can share relevant ones based on your industry or project type.",
        
        "Do you have case studies I can review?": "Yes, we have documented case studies. I can send you ones most relevant to your situation.",
        
        "What's your success rate?": "We have a high project completion rate and strong client retention. Most of our work comes from referrals and repeat clients.",
        
        "Can I talk to your past clients?": "Yes, we can arrange reference calls with clients in similar industries or project types.",
        
        "How many projects have you completed?": "We've completed over 500 projects across various sizes and industries.",
        
        "How do you protect our intellectual property?": "We sign NDAs, ensure code ownership transfers to you, and follow strict data handling policies.",
        
        "What about data security?": "We follow industry security practices including encryption, access controls, and secure development guidelines.",
        
        "Do you sign NDAs?": "Yes, signing NDAs is standard practice. We respect confidentiality completely.",
        
        "What happens if something goes wrong?": "We have escalation processes, backup resources, and clear SLAs. Issues are addressed quickly and transparently.",
        
        "How do you ensure quality?": "Through code reviews, automated testing, QA processes, and regular client feedback loops.",
        
        "Where is your team located?": "Our development team is based in India, with presence in the USA and Canada for client coordination.",
        
        # FIT_QUALIFICATION - BRIEF
        "We're a small company - do you work with SMBs?": "Yes, we work with companies of all sizes. Many SMBs benefit from our cost-effective approach.",
        
        "Do you only work with large enterprises?": "No, we work with startups, SMBs, and enterprises. Our engagement models fit different sizes.",
        
        "We don't have technical knowledge - is that okay?": "That's completely fine. We guide you through decisions and translate technical details into business terms.",
        
        "We need ongoing support - do you offer that?": "Yes, we provide maintenance and support services. Many clients continue with us after initial development.",
        
        "We need someone long-term - is that possible?": "Yes, long-term partnerships are actually our preference. Many clients have worked with us for years.",
        
        "Do you work with non-tech founders?": "Absolutely. We're used to working with non-technical stakeholders and explaining things clearly.",
    }
    
    return responses.get(question, f"Yes, we can help with that. Would you like to discuss the specifics?")


def generate_partial_followup_response(question: str, category: str, rationale: str) -> str:
    """Generate a partial answer with follow-up question (3-4 sentences)."""
    
    responses = {
        # SERVICE_DISCOVERY - PARTIAL_FOLLOWUP
        "What kind of software development services do you offer?": "We offer custom software development, mobile apps, web applications, cloud services, and AI solutions. Rather than listing everything, what kind of project are you thinking about?",
        
        "Do you build custom software applications?": "Yes, custom software is one of our core strengths. What kind of application are you considering — web, mobile, or something more specialized?",
        
        "Can you help us with mobile app development?": "Yes, we build mobile apps for iOS, Android, and cross-platform. Are you targeting a specific platform, or thinking about reaching users on all devices?",
        
        "What about cloud services?": "Yes, we help with cloud architecture, migration, and management. Are you looking to move to the cloud, or optimize an existing setup?",
        
        "Do you offer AI or chatbot development?": "Yes, we build AI solutions including chatbots, automation, and intelligent systems. What kind of AI use case are you exploring?",
        
        "Can you modernize our old legacy systems?": "Yes, legacy modernization is something we do often. What technology is your current system built on?",
        
        "Can you help with cross-platform mobile apps?": "Yes, we build cross-platform apps using React Native and Flutter. Do you already have a presence on one platform, or starting fresh?",
        
        "Do you offer IT consulting?": "Yes, we provide consulting to help companies make better technology decisions. What challenge are you trying to solve?",
        
        "Do you build SaaS products?": "Yes, we've built SaaS platforms from scratch to scale. Are you at the idea stage, or do you already have something in market?",
        
        "Can you help us build an MVP?": "Yes, MVPs are a great way to validate ideas quickly. How developed is your concept — do you have requirements defined?",
        
        "Do you develop AI agents?": "Yes, we build AI agents for various use cases. What would you want the agent to accomplish?",
        
        # CAPABILITY_INQUIRY - PARTIAL_FOLLOWUP
        "Can you integrate with our existing systems?": "Yes, integration is a core capability for us. What systems are you looking to connect?",
        
        "What technologies do you specialize in?": "We work across the modern stack — React, .NET, Node, Python, cloud platforms. What technology is most relevant to your project?",
        
        "Can you build HIPAA-compliant solutions?": "Yes, we have healthcare experience including HIPAA compliance. Are you in healthcare, or working with health-related data?",
        
        "Can you handle complex enterprise systems?": "Yes, we work with enterprise-scale systems regularly. What's the approximate scale of your system — users, data volume, integrations?",
        
        "Do you build AI-powered features?": "Yes, we integrate AI for recommendations, automation, and intelligent features. What kind of AI capability are you thinking about?",
        
        "Do you have IoT development experience?": "Yes, we've built IoT solutions with sensor integration and data pipelines. What kind of IoT project are you considering?",
        
        # INDUSTRY_FIT - PARTIAL_FOLLOWUP
        "Do you have healthcare software experience?": "Yes, we've built healthcare solutions including patient portals and practice management systems. What area of healthcare are you in?",
        
        "Can you develop remote patient monitoring systems?": "Yes, RPM is one of our healthcare specialties. What patient conditions or use cases are you focused on?",
        
        "Do you work with fintech companies?": "Yes, we work with fintech clients on payments, lending, and financial management. What fintech challenge are you working on?",
        
        "Do you have real estate software experience?": "Yes, we've built real estate platforms. Are you looking at property management, CRM, or marketplace features?",
        
        "Can you build property management platforms?": "Yes, we've built property management systems. What's your primary focus — residential, commercial, or mixed?",
        
        "Have you worked on education platforms?": "Yes, we've built e-learning and education management systems. What's your vision for the platform?",
        
        "Can you build e-learning solutions?": "Yes, we've built learning platforms with courses, assessments, and progress tracking. Who's your target audience?",
        
        "Do you develop retail software?": "Yes, we build retail solutions. Are you looking at B2B, B2C, or e-commerce specifically?",
        
        "Can you build e-commerce solutions?": "Yes, we build custom e-commerce platforms. What's driving the need for custom over off-the-shelf?",
        
        "Do you have logistics software experience?": "Yes, we've built logistics and fleet management solutions. What's the main logistics challenge you're facing?",
        
        "Do you work with insurance companies?": "Yes, we've built insurance software solutions. What part of the insurance workflow are you looking to improve?",
        
        "Do you work with IoT projects?": "Yes, we have IoT experience across industries. What kind of IoT application are you considering?",
        
        "Have you built workflow automation tools?": "Yes, we build workflow automation to reduce manual effort. What processes are you looking to automate?",
        
        # PRICING_COST - PARTIAL_FOLLOWUP
        "What's the cost of building an MVP?": "MVP costs vary significantly based on complexity — could be $15K-$100K+. Can you tell me more about what you're building?",
        
        "What are your hourly rates?": "Rates depend on the skills needed and engagement model. What kind of work are you planning?",
        
        "How much does a dedicated team cost?": "Team costs depend on size and skill mix. How many developers are you thinking, and what skills do you need?",
        
        "How much would AI chatbot development cost?": "Chatbot costs vary based on complexity — simple FAQ bots vs. intelligent conversational agents are very different. What would the chatbot need to do?",
        
        # ENGAGEMENT_MODEL - PARTIAL_FOLLOWUP
        "How does working with you actually work?": "We start with understanding your needs, then match the right engagement model and team. What's your situation — do you have a project planned, or exploring options?",
        
        "What's your development process?": "We follow agile with sprints, regular demos, and feedback loops. Do you have an existing process you'd want us to align with?",
        
        # COMPARISON_DIFFERENTIATION - PARTIAL_FOLLOWUP
        "What makes you different from other development companies?": "We focus on being consultative, not just building what we're told. We help clients make better decisions. What matters most to you in a development partner?",
        
        "Why should we choose offshore over local developers?": "Offshore offers cost savings and access to a larger talent pool, while quality depends on the partner you choose. What's your main concern about offshore?",
        
        "What's the advantage of working with you?": "We combine offshore efficiency with consultative partnership — we care about your outcomes, not just deliverables. What's most important to you?",
        
        "How do you compare to other offshore companies?": "We focus on long-term relationships and actually helping clients succeed, not just completing tasks. What's your experience been with offshore teams?",
        
        # PROCESS_TIMELINE - PARTIAL_FOLLOWUP
        "What's the timeline for building an MVP?": "MVPs typically take 8-16 weeks depending on complexity. How urgently do you need to launch?",
        
        "When can we expect the first deliverables?": "First deliverables usually come within the first 2-3 weeks. What would be most valuable to see early?",
        
        "What's the process from start to finish?": "We go from discovery through design, development, testing, and launch iteratively. Where are you in your planning process?",
        
        # TRUST_CREDIBILITY - PARTIAL_FOLLOWUP
        "Have you worked with companies like ours?": "We might have, but I'd need to understand your situation better. What industry are you in, and what size is your company?",
        
        # FIT_QUALIFICATION - PARTIAL_FOLLOWUP
        "Is this suitable for a startup like us?": "Yes, we work with startups regularly. What stage are you at — pre-revenue, early traction, or scaling?",
        
        "I just have an idea, can you help from scratch?": "Absolutely. We help clients go from idea to product. Have you done any validation of the idea yet?",
        
        "We already have a team - can you augment it?": "Yes, team augmentation is one of our core models. What skills are you looking to add?",
        
        "We're on a tight budget - can you still help?": "Possibly — we can discuss options like phased delivery or MVP-first approach. What's your approximate budget range?",
        
        # VAGUE_EXPLORATORY - PARTIAL_FOLLOWUP
        "Tell me more about what you do": "We help companies build software — from mobile apps to enterprise systems, AI to cloud. What brought you here today?",
        
        "I might need something custom": "We specialize in custom solutions. What's the problem you're trying to solve?",
        
        "I have an idea": "Ideas are exciting! Would you like to share what you're thinking about?",
        
        "What can you help us with?": "We help companies build and improve software products. What challenge are you facing right now?",
        
        "Go on": "Happy to continue. What specific area would you like to know more about?",
    }
    
    return responses.get(question, "That's a good question. Could you tell me more about your specific situation so I can give you a better answer?")


def generate_defer_response(question: str, category: str, rationale: str) -> str:
    """Generate a deferring response that asks for clarification (2-3 sentences)."""
    
    responses = {
        # SERVICE_DISCOVERY - DEFER
        "Can you help with digital transformation?": "Digital transformation means different things to different companies. What does it mean for you — is it about technology, processes, or customer experience?",
        
        # PRICING_COST - DEFER
        "How much does custom software development cost?": "It really depends on what you're building. Can you tell me more about your project so I can give you a realistic range?",
        
        "Can you give me a rough estimate?": "I'd need to understand your project first. What are you looking to build?",
        
        "How much would a mobile app cost?": "Mobile apps can range from $25K to $500K+ depending on complexity. What kind of app are you thinking about?",
        
        "Can you work within our budget?": "It depends on what you need to build. What's your budget range, and what are you hoping to accomplish?",
        
        # PROCESS_TIMELINE - DEFER
        "How long does a typical project take?": "Timelines vary widely based on scope. Can you tell me more about your project?",
        
        "How long would a mobile app take to build?": "It depends on the app's complexity. What features are you planning to include?",
        
        # FIT_QUALIFICATION - DEFER
        "Can you handle a project of our size?": "We work on projects of various sizes. What's the approximate scope you're looking at?",
        
        # VAGUE_EXPLORATORY - DEFER
        "I'm just looking around": "Welcome! What originally brought you here — any specific challenge or goal you're exploring?",
        
        "Not sure yet": "That's completely fine. What's been on your mind that made you start looking?",
        
        "Just exploring options": "Makes sense. What options are you considering, and what would help you decide?",
        
        "How does this work?": "Good question. What specifically would you like to understand — our process, pricing, or something else?",
        
        "We're thinking about building something": "Exciting! What kind of thing are you thinking about building?",
        
        "Can you help with our project?": "Most likely, yes. What's the project about?",
        
        "We need software help": "We're here to help. What kind of software challenge are you facing?",
        
        "I'm not sure what I need": "That's okay — many conversations start that way. What problem are you trying to solve?",
        
        "Maybe": "No pressure. What would help you make a decision?",
        
        "Yeah": "Great. What would you like to know more about?",
        
        "Okay": "Perfect. Shall I continue, or is there something specific on your mind?",
        
        "Hmm": "Take your time. What are you thinking?",
        
        "Interesting": "What caught your attention? I can elaborate on anything.",
    }
    
    return responses.get(question, "I'd like to understand your situation better first. Could you tell me what you're trying to accomplish?")


def generate_training_example(category: str, item: Dict) -> Dict:
    """Generate a single training example in JSONL format."""
    
    question = item['question']
    strategy = item['strategy']
    rationale = item['rationale']
    
    state_name = STATE_MAPPING.get(category, "GENERAL_INQUIRY")
    
    # Select response generator based on strategy
    if strategy == "BRIEF":
        response = generate_brief_response(question, category, rationale)
    elif strategy == "PARTIAL_FOLLOWUP":
        response = generate_partial_followup_response(question, category, rationale)
    elif strategy == "DEFER":
        response = generate_defer_response(question, category, rationale)
    else:
        response = "I'd be happy to help with that. Could you tell me more?"
    
    # Build the training example
    system_content = f"State: {state_name}\nRules: Be concise. Be consultative. Avoid over-explaining."
    
    if strategy == "PARTIAL_FOLLOWUP":
        system_content += " Ask a follow-up question to understand context."
    elif strategy == "DEFER":
        system_content += " Ask clarifying question first."
    
    return {
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]
    }


def main():
    print("=" * 70)
    print("STEP 5: AUTHOR TRAINING EXAMPLES (JSONL)")
    print("=" * 70)
    
    training_examples = []
    category_counts = {}
    strategy_counts = {"BRIEF": 0, "PARTIAL_FOLLOWUP": 0, "DEFER": 0}
    
    # Generate examples for each category
    for category, items in STRATEGY_MAP.items():
        category_examples = []
        
        for item in items:
            example = generate_training_example(category, item)
            category_examples.append(example)
            strategy_counts[item['strategy']] = strategy_counts.get(item['strategy'], 0) + 1
        
        training_examples.extend(category_examples)
        category_counts[category] = len(category_examples)
        
        print(f"\n[{category}] Generated {len(category_examples)} examples")
    
    # Save to JSONL file
    output_file = 'step5_website_training_data.jsonl'
    with open(output_file, 'w', encoding='utf-8') as f:
        for example in training_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 70)
    print("GENERATION SUMMARY")
    print("=" * 70)
    
    print(f"\n[TOTAL EXAMPLES]: {len(training_examples)}")
    
    print("\n[BY CATEGORY]:")
    for cat, count in sorted(category_counts.items()):
        print(f"   {cat}: {count}")
    
    print("\n[BY STRATEGY]:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"   {strategy}: {count}")
    
    print(f"\n[OUTPUT FILE]: {output_file}")
    
    print("\n" + "=" * 70)
    print("SAMPLE EXAMPLES (first 3)")
    print("=" * 70)
    
    for i, example in enumerate(training_examples[:3]):
        print(f"\n--- Example {i+1} ---")
        print(f"System: {example['messages'][0]['content'][:80]}...")
        print(f"User: {example['messages'][1]['content']}")
        print(f"Assistant: {example['messages'][2]['content'][:100]}...")
    
    print("\n" + "=" * 70)
    print("STEP 5 COMPLETE - STOPPING AS INSTRUCTED")
    print("=" * 70)
    print("\nAwaiting instruction to proceed to STEP 6 (Dataset Size Control).")


if __name__ == "__main__":
    main()
