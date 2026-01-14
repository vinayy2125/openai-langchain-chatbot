"""
Large Training Dataset Generator for DITSTEK Chatbot
Generates 2000+ fine-tuning examples from scraped website data

Author: Generated for DITSTEK chatbot improvement
Date: 2026-01-14
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_EXAMPLES = 2100  # Target 2000+ with buffer
TRAIN_SPLIT = 0.85  # 85% train, 15% validation

# Paths
SCRAPED_DATA_PATH = Path(__file__).parent.parent / "scraped_data" / "scrape_20251229_131253.json"
EXISTING_GOOD_DATA = Path(__file__).parent.parent / "step5_website_training_data.jsonl"
OUTPUT_TRAIN = Path(__file__).parent / "train_large.jsonl"
OUTPUT_VALIDATION = Path(__file__).parent / "validation_large.jsonl"

# =============================================================================
# CANONICAL SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are DITSTEK's AI assistant - a consultative partner helping visitors explore technology solutions.

Core behaviors:
- Answer questions directly and substantively with specific details
- When context would help, ask ONE focused follow-up question
- Guide conversations naturally toward understanding user needs
- Be concise but informative (2-4 sentences + optional follow-up)

Knowledge scope:
- You know DITSTEK's services, technologies, industries, and case studies
- You can share project examples and specific capabilities
- For pricing specifics, suggest discussing with the team

Follow-up guidelines:
- Ask follow-ups when user input is vague, exploratory, or could use clarification
- Skip follow-ups for specific factual questions with complete answers
- Never ask multiple questions in one response
- Make follow-ups feel natural and consultative

Response style:
- Professional yet conversational
- Consultative, not salesy
- Grounded in real capabilities
- No marketing fluff or buzzwords"""

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ServiceInfo:
    name: str
    description: str
    technologies: List[str]
    industries: List[str]
    questions: List[str]
    follow_ups: List[str]

@dataclass
class IndustryInfo:
    name: str
    description: str
    solutions: List[str]
    challenges: List[str]
    questions: List[str]


# =============================================================================
# DITSTEK KNOWLEDGE BASE (Extracted from website)
# =============================================================================

SERVICES = {
    "custom_software": ServiceInfo(
        name="Custom Software Development",
        description="We build tailored software solutions from scratch, designed specifically for your business needs. Our team handles everything from requirements analysis to deployment and maintenance.",
        technologies=["React", ".NET", "Node.js", "Python", "PHP", "Laravel"],
        industries=["Healthcare", "Fintech", "Real Estate", "Retail", "Logistics"],
        questions=[
            "Do you build custom software?",
            "Can you develop software from scratch?",
            "I need a custom application built",
            "We need bespoke software for our business",
            "Can you create a custom solution for us?",
        ],
        follow_ups=[
            "What kind of software are you looking to build?",
            "What problem are you trying to solve?",
            "Is this for internal use or customer-facing?",
            "Do you have any existing systems it needs to integrate with?",
        ]
    ),
    "mobile_apps": ServiceInfo(
        name="Mobile App Development",
        description="We develop native and cross-platform mobile applications for iOS and Android. Our team uses React Native and Flutter for cross-platform solutions, and Swift/Kotlin for native apps.",
        technologies=["React Native", "Flutter", "Swift", "Kotlin", "iOS", "Android"],
        industries=["Healthcare", "Retail", "Fintech", "Education", "Logistics"],
        questions=[
            "Do you build mobile apps?",
            "Can you develop an app for iPhone and Android?",
            "I need a mobile application",
            "We want to create an app",
            "Can you help with mobile app development?",
            "Do you do iOS development?",
            "Can you build Android apps?",
        ],
        follow_ups=[
            "Are you targeting iOS, Android, or both?",
            "Is this a consumer app or for internal use?",
            "Do you have an existing web platform to integrate with?",
            "What's the core functionality you need?",
        ]
    ),
    "web_development": ServiceInfo(
        name="Web Application Development",
        description="We create modern, responsive web applications using cutting-edge frameworks. From simple websites to complex enterprise portals, we handle full-stack development.",
        technologies=["React", "Angular", "Vue.js", ".NET", "Node.js", "PostgreSQL", "MongoDB"],
        industries=["All industries"],
        questions=[
            "Do you do web development?",
            "Can you build a web application?",
            "I need a website built",
            "We're looking for web developers",
            "Can you create a web portal?",
        ],
        follow_ups=[
            "What kind of web application do you have in mind?",
            "Is this a customer-facing site or an internal tool?",
            "Do you have specific technology preferences?",
        ]
    ),
    "cloud_services": ServiceInfo(
        name="Cloud & DevOps Services",
        description="We help businesses migrate to the cloud, optimize infrastructure, and implement DevOps practices. We work with AWS, Azure, and Google Cloud to build scalable, reliable systems.",
        technologies=["AWS", "Azure", "Google Cloud", "Kubernetes", "Docker", "Jenkins", "Terraform"],
        industries=["All industries"],
        questions=[
            "Do you offer cloud services?",
            "Can you help with cloud migration?",
            "We need to move to AWS",
            "Do you do DevOps?",
            "Can you help with our infrastructure?",
            "We need help with scalability",
        ],
        follow_ups=[
            "Are you looking to migrate existing systems or build new?",
            "What cloud platform are you considering?",
            "What's driving the move to cloud?",
        ]
    ),
    "ai_ml": ServiceInfo(
        name="AI & Machine Learning Solutions",
        description="We develop AI-powered solutions including chatbots, automation systems, and intelligent features. Our team builds custom ML models and integrates AI into existing workflows.",
        technologies=["Python", "TensorFlow", "PyTorch", "OpenAI", "LangChain", "NLP"],
        industries=["Healthcare", "Fintech", "Retail", "Manufacturing"],
        questions=[
            "Do you do AI development?",
            "Can you build a chatbot?",
            "We want to add AI to our product",
            "Do you work with machine learning?",
            "Can you help with automation?",
            "We're interested in AI solutions",
        ],
        follow_ups=[
            "What would you want the AI to accomplish?",
            "Is there a specific process you're looking to automate?",
            "Do you have training data available?",
        ]
    ),
    "legacy_modernization": ServiceInfo(
        name="Legacy Application Modernization",
        description="We help businesses modernize outdated systems without disrupting operations. Our approach includes gradual migration, refactoring, and re-platforming strategies.",
        technologies=["All modern stacks"],
        industries=["Healthcare", "Finance", "Manufacturing", "Government"],
        questions=[
            "Can you modernize our old system?",
            "We have a legacy application that needs updating",
            "Our software is outdated",
            "We need to migrate from an old platform",
            "Can you help upgrade our existing system?",
        ],
        follow_ups=[
            "What technology is your current system built on?",
            "What's driving the modernization need?",
            "Are there specific pain points with the current system?",
        ]
    ),
    "dedicated_teams": ServiceInfo(
        name="Dedicated Development Teams",
        description="We provide dedicated development teams that work exclusively on your project. Teams integrate with your processes and function as an extension of your in-house team.",
        technologies=["All stacks"],
        industries=["All industries"],
        questions=[
            "Can we hire a dedicated team?",
            "Do you offer staff augmentation?",
            "We need developers to extend our team",
            "Can you provide dedicated resources?",
            "Do you do team extension?",
        ],
        follow_ups=[
            "Are you extending an existing team or starting fresh?",
            "What skills are you looking for?",
            "How many developers do you need?",
        ]
    ),
    "mvp_development": ServiceInfo(
        name="MVP Development",
        description="We help startups and enterprises validate ideas through rapid MVP development. Our lean approach focuses on core features to get to market quickly.",
        technologies=["React", "Node.js", "React Native", "AWS"],
        industries=["Startups", "All industries"],
        questions=[
            "Can you help build an MVP?",
            "We have a startup idea",
            "I want to validate my product concept",
            "We need a minimum viable product",
            "Can you help launch quickly?",
        ],
        follow_ups=[
            "How developed is your concept?",
            "Do you have requirements defined?",
            "What's your timeline for launch?",
        ]
    ),
    "saas_development": ServiceInfo(
        name="SaaS Development",
        description="We build scalable SaaS platforms from architecture to launch. Our experience includes multi-tenant systems, subscription management, and enterprise-grade security.",
        technologies=["React", ".NET", "Node.js", "AWS", "PostgreSQL"],
        industries=["All industries"],
        questions=[
            "Do you build SaaS products?",
            "Can you develop a SaaS platform?",
            "We want to create a subscription-based product",
            "We need a multi-tenant application",
        ],
        follow_ups=[
            "Are you at the idea stage or already have something?",
            "Who's your target market?",
            "What problem does your SaaS solve?",
        ]
    ),
    "qa_testing": ServiceInfo(
        name="QA & Testing Services",
        description="We provide comprehensive testing including automated testing, manual QA, performance testing, and security audits. Quality is built into our development process.",
        technologies=["Selenium", "Cypress", "Jest", "JMeter"],
        industries=["All industries"],
        questions=[
            "Do you offer QA services?",
            "Can you help with testing?",
            "We need automated testing",
            "Do you do security testing?",
        ],
        follow_ups=[
            "What type of testing do you need most?",
            "Do you have an existing test framework?",
        ]
    ),
}

INDUSTRIES = {
    "healthcare": IndustryInfo(
        name="Healthcare",
        description="We build HIPAA-compliant healthcare solutions including patient portals, practice management systems, remote patient monitoring, and telehealth platforms.",
        solutions=["Patient portals", "Practice management", "Remote patient monitoring", "Telehealth", "EHR integration", "Medical billing"],
        challenges=["HIPAA compliance", "HL7 integration", "Data security", "Interoperability"],
        questions=[
            "Do you have healthcare experience?",
            "Can you build HIPAA-compliant apps?",
            "We need a patient portal",
            "Do you understand medical regulations?",
            "Can you integrate with EHR systems?",
        ]
    ),
    "fintech": IndustryInfo(
        name="Fintech",
        description="We develop secure fintech solutions including payment systems, lending platforms, and financial management tools with strong security and compliance focus.",
        solutions=["Payment processing", "Lending platforms", "Financial dashboards", "Trading systems", "Banking apps"],
        challenges=["PCI compliance", "Security", "Regulatory requirements", "Real-time processing"],
        questions=[
            "Do you work with fintech companies?",
            "Can you build payment systems?",
            "We need a lending platform",
            "Do you understand financial regulations?",
        ]
    ),
    "real_estate": IndustryInfo(
        name="Real Estate",
        description="We create real estate technology solutions including property management platforms, CRM systems, and marketplace applications.",
        solutions=["Property management", "Real estate CRM", "Listing platforms", "Virtual tours", "Tenant portals"],
        challenges=["MLS integration", "Document management", "Multi-property management"],
        questions=[
            "Do you have real estate experience?",
            "Can you build a property management system?",
            "We need a real estate platform",
        ]
    ),
    "education": IndustryInfo(
        name="EdTech",
        description="We build e-learning platforms, learning management systems, and educational tools that engage learners and simplify administration.",
        solutions=["Learning management systems", "E-learning platforms", "Assessment tools", "Virtual classrooms", "Student portals"],
        challenges=["Engagement", "Accessibility", "Content management", "Progress tracking"],
        questions=[
            "Do you build education platforms?",
            "Can you create an e-learning system?",
            "We need an LMS",
        ]
    ),
    "retail": IndustryInfo(
        name="Retail & E-commerce",
        description="We develop retail solutions including e-commerce platforms, inventory management, and omnichannel experiences.",
        solutions=["E-commerce platforms", "Inventory management", "POS systems", "Order management", "Customer loyalty"],
        challenges=["Inventory tracking", "Payment integration", "Multi-channel sync"],
        questions=[
            "Do you build e-commerce platforms?",
            "Can you help with retail software?",
            "We need an online store",
        ]
    ),
    "logistics": IndustryInfo(
        name="Logistics & Transportation",
        description="We build logistics solutions including fleet management, route optimization, and supply chain tracking systems.",
        solutions=["Fleet management", "Route optimization", "Warehouse management", "Shipment tracking", "Load management"],
        challenges=["Real-time tracking", "Route efficiency", "Driver management"],
        questions=[
            "Do you have logistics experience?",
            "Can you build a fleet management system?",
            "We need supply chain software",
        ]
    ),
    "iot": IndustryInfo(
        name="IoT",
        description="We develop IoT solutions including sensor integration, data pipelines, and monitoring dashboards for connected devices.",
        solutions=["Sensor integration", "Data collection", "Monitoring dashboards", "Device management", "Predictive maintenance"],
        challenges=["Connectivity", "Data volume", "Real-time processing", "Device security"],
        questions=[
            "Do you work on IoT projects?",
            "Can you help with connected devices?",
            "We need sensor integration",
        ]
    ),
}

TECHNOLOGIES = {
    "react": {
        "name": "React",
        "description": "React is one of our core frontend technologies. We've delivered numerous production applications using React, including complex dashboards, customer portals, and e-commerce platforms.",
        "questions": ["Do you work with React?", "Can your team do React development?", "Do you have React experience?"],
    },
    "dotnet": {
        "name": ".NET",
        "description": "We have strong .NET expertise across the stack including ASP.NET Core, .NET Framework, and related Microsoft technologies. We use it for enterprise applications and APIs.",
        "questions": ["Do you do .NET development?", "Can you work with ASP.NET?", "Do you have Microsoft stack experience?"],
    },
    "nodejs": {
        "name": "Node.js",
        "description": "Node.js is part of our core backend stack. We use it extensively for APIs, real-time applications, and microservices architectures.",
        "questions": ["Do you use Node.js?", "Can you build with Node?", "Do you have JavaScript backend experience?"],
    },
    "python": {
        "name": "Python",
        "description": "We work with Python for backend development, data processing, and AI/ML projects. It's our primary language for machine learning work.",
        "questions": ["Do you work with Python?", "Can you do Python development?"],
    },
    "aws": {
        "name": "AWS",
        "description": "We're experienced with AWS services including EC2, Lambda, S3, RDS, and many others. We help with cloud architecture, migration, and optimization.",
        "questions": ["Do you work with AWS?", "Can you help with Amazon cloud?", "Do you have AWS experience?"],
    },
    "azure": {
        "name": "Azure",
        "description": "We work with Microsoft Azure for cloud solutions, particularly for enterprises using the Microsoft ecosystem.",
        "questions": ["Do you work with Azure?", "Can you help with Microsoft cloud?"],
    },
    "react_native": {
        "name": "React Native",
        "description": "We use React Native for cross-platform mobile development, allowing code sharing between iOS and Android while maintaining native performance.",
        "questions": ["Do you use React Native?", "Can you build cross-platform apps?"],
    },
    "flutter": {
        "name": "Flutter",
        "description": "Flutter is another cross-platform framework we use, particularly for apps requiring custom UI or specific performance characteristics.",
        "questions": ["Do you work with Flutter?", "Can you build Flutter apps?"],
    },
}

# =============================================================================
# CASE STUDIES (Extracted from website)
# =============================================================================

CASE_STUDIES = [
    {
        "name": "Medical Practice Management System",
        "industry": "Healthcare",
        "description": "Built a comprehensive practice management system for a medical clinic chain, including appointment scheduling, patient records, and billing integration.",
        "technologies": ["React", ".NET", "SQL Server"],
        "results": "Reduced administrative time by 40% and improved patient satisfaction scores.",
    },
    {
        "name": "Remote Patient Monitoring Platform",
        "industry": "Healthcare",
        "description": "Developed a HIPAA-compliant remote patient monitoring solution with IoT device integration, real-time alerts, and care team dashboards.",
        "technologies": ["React Native", "Node.js", "AWS", "IoT"],
        "results": "Enabled monitoring of 5,000+ patients with 99.9% uptime.",
    },
    {
        "name": "Property Management Platform",
        "industry": "Real Estate",
        "description": "Created a multi-property management platform with tenant portals, maintenance tracking, and automated rent collection.",
        "technologies": ["React", "Node.js", "PostgreSQL"],
        "results": "Managing 10,000+ units with 50% reduction in manual work.",
    },
    {
        "name": "Fintech Lending Platform",
        "industry": "Fintech",
        "description": "Built a peer-to-peer lending platform with credit scoring, automated underwriting, and regulatory compliance.",
        "technologies": [".NET", "React", "Azure"],
        "results": "Processed $50M+ in loans with automated compliance checks.",
    },
    {
        "name": "E-learning Platform",
        "industry": "Education",
        "description": "Developed an interactive e-learning platform with video courses, assessments, and progress tracking.",
        "technologies": ["React", "Node.js", "AWS", "MongoDB"],
        "results": "Serving 100,000+ learners with 95% completion rate for started courses.",
    },
    {
        "name": "Fleet Management System",
        "industry": "Logistics",
        "description": "Created a real-time fleet tracking and management system with route optimization and driver management.",
        "technologies": ["React Native", "Node.js", "PostgreSQL", "Google Maps"],
        "results": "Tracking 500+ vehicles with 15% fuel savings through route optimization.",
    },
    {
        "name": "Inventory Management SaaS",
        "industry": "Retail",
        "description": "Built a cloud-based inventory management platform for retail chains with multi-location sync and forecasting.",
        "technologies": ["React", ".NET", "Azure", "SQL Server"],
        "results": "Used by 200+ retail locations with 30% reduction in stockouts.",
    },
]

# =============================================================================
# ENGAGEMENT MODELS
# =============================================================================

ENGAGEMENT_MODELS = {
    "dedicated_team": {
        "name": "Dedicated Team",
        "description": "A dedicated team works exclusively on your project, functions as an extension of your team, and scales with your needs. It's ideal for long-term projects or ongoing development.",
        "benefits": ["Full control", "Team stability", "Flexible scaling", "Deep product knowledge"],
        "when_to_use": "Long-term projects, ongoing development, complex products",
    },
    "fixed_price": {
        "name": "Fixed Price",
        "description": "Fixed-price engagement means we agree on scope and price upfront. You get budget certainty and we deliver the defined requirements.",
        "benefits": ["Budget certainty", "Clear deliverables", "Defined timeline"],
        "when_to_use": "Well-defined projects, MVPs, specific features",
    },
    "hourly": {
        "name": "Hourly",
        "description": "Hourly engagement provides flexibility for evolving scope or when you need resources for varied tasks.",
        "benefits": ["Maximum flexibility", "Pay for what you use", "Easy to adjust"],
        "when_to_use": "Evolving requirements, maintenance, support",
    },
}

# =============================================================================
# QUESTION TEMPLATES
# =============================================================================

EXPLORATORY_QUESTIONS = [
    "What services do you offer?",
    "Tell me about DITSTEK",
    "What can you help with?",
    "I'm looking for a development partner",
    "We need software help",
    "I'm just exploring options",
    "What makes you different?",
    "Why should I choose you?",
    "How do you work?",
    "Tell me about your process",
]

PRICING_QUESTIONS = [
    "How much does it cost?",
    "What are your rates?",
    "How much for a mobile app?",
    "What's the cost of an MVP?",
    "Do you offer fixed pricing?",
    "What's your pricing model?",
    "Is offshore development cheaper?",
    "What's the minimum project size?",
]

TIMELINE_QUESTIONS = [
    "How long does a project take?",
    "What's the timeline for an MVP?",
    "How fast can you start?",
    "When can we see first deliverables?",
    "What's your development process?",
]

TRUST_QUESTIONS = [
    "Can you show me your work?",
    "Do you have case studies?",
    "Have you worked with companies like ours?",
    "What's your track record?",
    "Can I talk to your past clients?",
    "How do you protect our IP?",
    "Do you sign NDAs?",
]

ACKNOWLEDGMENT_INPUTS = [
    "okay", "ok", "sure", "yes", "yeah", "yep", "alright", "sounds good",
    "great", "perfect", "thanks", "thank you", "got it", "I see", "interesting",
    "that makes sense", "understood", "cool", "nice", "hmm", "go on", "continue",
]

# =============================================================================
# EXAMPLE GENERATORS
# =============================================================================

class TrainingDataGenerator:
    def __init__(self):
        self.examples = []
        self.scraped_data = None
        self.load_scraped_data()
    
    def load_scraped_data(self):
        """Load scraped website data for grounding."""
        try:
            with open(SCRAPED_DATA_PATH, "r", encoding="utf-8") as f:
                self.scraped_data = json.load(f)
            print(f"Loaded {len(self.scraped_data.get('pages', []))} pages from scraped data")
        except Exception as e:
            print(f"Warning: Could not load scraped data: {e}")
            self.scraped_data = {"pages": []}
    
    def create_example(self, user_input: str, assistant_response: str) -> Dict:
        """Create a training example with the canonical format."""
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    
    def create_multi_turn_example(self, turns: List[Tuple[str, str]]) -> Dict:
        """Create a multi-turn conversation example."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, assistant_msg in turns:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        return {"messages": messages}
    
    def generate_service_examples(self) -> List[Dict]:
        """Generate examples for each service area."""
        examples = []
        
        for service_key, service in SERVICES.items():
            # Direct questions about the service
            for question in service.questions:
                # Version with follow-up
                response = f"{service.description} "
                if service.follow_ups:
                    follow_up = random.choice(service.follow_ups)
                    response += follow_up
                examples.append(self.create_example(question, response.strip()))
                
                # Sometimes create version without follow-up (for specific questions)
                if random.random() < 0.3:
                    response_no_followup = service.description
                    examples.append(self.create_example(question, response_no_followup))
            
            # Technology-specific questions
            for tech in service.technologies[:3]:  # Limit to avoid explosion
                tech_question = f"Do you use {tech} for {service.name.lower()}?"
                tech_response = f"Yes, {tech} is one of the technologies we use for {service.name.lower()}. {service.description[:100]}..."
                examples.append(self.create_example(tech_question, tech_response))
        
        return examples
    
    def generate_industry_examples(self) -> List[Dict]:
        """Generate examples for each industry."""
        examples = []
        
        for industry_key, industry in INDUSTRIES.items():
            for question in industry.questions:
                response = f"Yes, we have significant {industry.name.lower()} experience. {industry.description}"
                # Add solution examples
                solutions_text = ", ".join(industry.solutions[:3])
                response += f" We've built solutions including {solutions_text}."
                
                # Add follow-up for vague questions
                if "experience" in question.lower():
                    response += f" What {industry.name.lower()} challenge are you working on?"
                
                examples.append(self.create_example(question, response))
            
            # Challenge-specific questions
            for challenge in industry.challenges[:2]:
                challenge_q = f"Can you handle {challenge.lower()} requirements?"
                challenge_r = f"Yes, {challenge} is something we're experienced with in our {industry.name.lower()} work. We understand the requirements and have implemented compliant solutions."
                examples.append(self.create_example(challenge_q, challenge_r))
        
        return examples
    
    def generate_technology_examples(self) -> List[Dict]:
        """Generate examples for technology questions."""
        examples = []
        
        for tech_key, tech in TECHNOLOGIES.items():
            for question in tech["questions"]:
                response = tech["description"]
                examples.append(self.create_example(question, response))
        
        # Cross-technology questions
        tech_list = list(TECHNOLOGIES.keys())
        for _ in range(30):
            techs = random.sample(tech_list, 2)
            question = f"Do you work with {TECHNOLOGIES[techs[0]]['name']} and {TECHNOLOGIES[techs[1]]['name']}?"
            response = f"Yes, we work with both {TECHNOLOGIES[techs[0]]['name']} and {TECHNOLOGIES[techs[1]]['name']}. We often combine them in full-stack projects."
            examples.append(self.create_example(question, response))
        
        return examples
    
    def generate_case_study_examples(self) -> List[Dict]:
        """Generate examples featuring case studies."""
        examples = []
        
        for case in CASE_STUDIES:
            # Direct case study questions
            questions = [
                f"Have you done {case['industry'].lower()} projects?",
                f"Can you show me a {case['industry'].lower()} example?",
                f"What {case['industry'].lower()} work have you done?",
            ]
            
            for q in questions:
                response = f"Yes, we've worked on several {case['industry']} projects. One example is our {case['name']}: {case['description']} {case['results']}"
                examples.append(self.create_example(q, response))
            
            # Technology-specific case study
            if case['technologies']:
                tech = random.choice(case['technologies'])
                tech_q = f"Have you built {case['industry'].lower()} applications with {tech}?"
                tech_r = f"Yes, for example our {case['name']} used {', '.join(case['technologies'])}. {case['description']} {case['results']}"
                examples.append(self.create_example(tech_q, tech_r))
        
        return examples
    
    def generate_pricing_examples(self) -> List[Dict]:
        """Generate pricing-related examples."""
        examples = []
        
        pricing_responses = {
            "How much does it cost?": "Pricing depends on project scope and complexity. I can help you understand what's involved, and then you could discuss specifics with our team. What kind of project are you considering?",
            "What are your rates?": "Rates vary based on the engagement model and skills required. We offer dedicated teams, fixed-price projects, and hourly arrangements. What type of engagement fits your needs?",
            "How much for a mobile app?": "Mobile app costs vary significantly based on complexity - anywhere from $25K for a simple app to $500K+ for complex platforms. What kind of app are you thinking about?",
            "What's the cost of an MVP?": "MVPs typically range from $15K to $100K+ depending on complexity and features. The goal is to validate quickly with minimal investment. How developed is your concept?",
            "Do you offer fixed pricing?": "Yes, we offer fixed-price engagements for projects with well-defined scope. It provides budget certainty. Do you have clear requirements defined?",
            "What's your pricing model?": "We offer three main models: dedicated teams for ongoing work, fixed-price for defined projects, and hourly for flexible engagements. Which sounds closest to your situation?",
            "Is offshore development cheaper?": "Offshore typically offers 40-60% cost savings while maintaining quality through structured processes. The key is choosing the right partner.",
            "What's the minimum project size?": "We typically work on projects starting around $10K, but it depends on scope and engagement model. What are you looking to build?",
        }
        
        for question, response in pricing_responses.items():
            examples.append(self.create_example(question, response))
            
            # Variations
            variations = [
                question.replace("How much", "What's the cost"),
                question.replace("What", "Tell me what"),
            ]
            for var in variations:
                if var != question:
                    examples.append(self.create_example(var, response))
        
        return examples
    
    def generate_timeline_examples(self) -> List[Dict]:
        """Generate timeline-related examples."""
        examples = []
        
        timeline_responses = {
            "How long does a project take?": "Timelines vary based on scope and complexity. A simple MVP might take 8-12 weeks, while larger projects span several months. What kind of project are you planning?",
            "What's the timeline for an MVP?": "MVPs typically take 8-16 weeks depending on complexity. The focus is on getting to market quickly with core features. How urgently do you need to launch?",
            "How fast can you start?": "We can typically start within 1-2 weeks after finalizing requirements. For urgent needs, we can expedite. What's your timeline looking like?",
            "When can we see first deliverables?": "First deliverables usually come within 2-3 weeks of starting development. What would be most valuable to see early?",
            "What's your development process?": "We follow agile methodology with 2-week sprints, regular demos, and continuous feedback loops. Do you have an existing process you'd want us to align with?",
        }
        
        for question, response in timeline_responses.items():
            examples.append(self.create_example(question, response))
        
        return examples
    
    def generate_engagement_model_examples(self) -> List[Dict]:
        """Generate engagement model examples."""
        examples = []
        
        for model_key, model in ENGAGEMENT_MODELS.items():
            questions = [
                f"Tell me about the {model['name'].lower()} model",
                f"How does {model['name'].lower()} work?",
                f"What is {model['name'].lower()}?",
            ]
            
            for q in questions:
                response = f"{model['description']} Benefits include {', '.join(model['benefits'][:3])}."
                examples.append(self.create_example(q, response))
        
        # Comparison questions
        examples.append(self.create_example(
            "What engagement models do you offer?",
            "We offer three main models: dedicated teams for long-term projects, fixed-price for well-defined scope, and hourly for flexible engagements. Each has different benefits depending on your situation. What's your project like?"
        ))
        
        return examples
    
    def generate_trust_examples(self) -> List[Dict]:
        """Generate trust-building examples."""
        examples = []
        
        trust_responses = {
            "Can you show me your work?": "Yes, we have case studies and portfolio examples across industries. I can share relevant ones based on your project type. What industry or technology is most relevant?",
            "Do you have case studies?": "Yes, we have documented case studies for projects in healthcare, fintech, real estate, and more. What type of project are you considering?",
            "Have you worked with companies like ours?": "We might have - we work with startups, SMBs, and enterprises across various industries. Tell me more about your company and I can share relevant examples.",
            "What's your track record?": "We've completed 500+ projects across various industries with a strong client retention rate. Most of our work comes from referrals and repeat clients.",
            "Can I talk to your past clients?": "Absolutely. We can arrange reference calls with clients in similar industries or project types. What kind of reference would be most helpful?",
            "How do you protect our IP?": "We sign NDAs, ensure code ownership transfers to you, and follow strict data handling policies. IP protection is standard in all our engagements.",
            "Do you sign NDAs?": "Yes, signing NDAs is standard practice. We take confidentiality seriously and can sign before any detailed discussions.",
        }
        
        for question, response in trust_responses.items():
            examples.append(self.create_example(question, response))
        
        return examples
    
    def generate_exploratory_examples(self) -> List[Dict]:
        """Generate examples for exploratory conversations."""
        examples = []
        
        exploratory_responses = {
            "What services do you offer?": "We help companies build software - from mobile apps to enterprise systems, AI to cloud solutions. We focus on custom development, team extension, and modernization. What brings you here today?",
            "Tell me about DITSTEK": "DITSTEK is a technology services company with 10+ years of experience building custom software solutions. We work with startups and enterprises across healthcare, fintech, and other industries. What would you like to know more about?",
            "What can you help with?": "We help companies build and improve software products - mobile apps, web platforms, AI solutions, cloud infrastructure. What challenge are you facing?",
            "I'm looking for a development partner": "We'd be happy to help. We work as long-term partners for many clients. What kind of project or situation are you looking at?",
            "We need software help": "We're here to help with a range of software challenges. What's the situation you're dealing with?",
            "I'm just exploring options": "Totally understand - exploration is a good first step. What originally made you start looking for development help?",
            "What makes you different?": "We focus on being consultative partners, not just code factories. We help clients make better decisions, not just execute tasks. We also offer offshore efficiency with onshore communication quality.",
            "Why should I choose you?": "We combine offshore cost efficiency with a consultative approach - we don't just build what you ask, we help you figure out what's best for your situation. What matters most to you in a partner?",
            "How do you work?": "We start by understanding your needs, then propose the right engagement model and team. We use agile processes with regular communication. Would you like details on any specific aspect?",
            "Tell me about your process": "We follow agile methodology with discovery, design, development, and delivery phases. Regular demos and feedback loops ensure alignment. What part of the process interests you most?",
        }
        
        for question, response in exploratory_responses.items():
            examples.append(self.create_example(question, response))
            
        return examples
    
    def generate_acknowledgment_examples(self) -> List[Dict]:
        """Generate examples for handling acknowledgments."""
        examples = []
        
        # Acknowledgment responses that continue the conversation
        acknowledgment_responses = [
            "Is there anything specific you'd like to know more about?",
            "Happy to elaborate on any of that. What interests you most?",
            "Shall I continue, or is there something specific on your mind?",
            "What else would you like to explore?",
            "Anything else I can help clarify?",
            "What questions do you have?",
        ]
        
        for ack in ACKNOWLEDGMENT_INPUTS:
            response = random.choice(acknowledgment_responses)
            examples.append(self.create_example(ack, response))
        
        return examples
    
    def generate_multi_turn_examples(self) -> List[Dict]:
        """Generate multi-turn conversation examples."""
        examples = []
        
        # Service exploration flow
        examples.append(self.create_multi_turn_example([
            ("What kind of development do you do?", 
             "We specialize in custom software development across web, mobile, cloud, and AI. What type of project are you considering?"),
            ("We need a mobile app",
             "We build mobile apps for iOS, Android, and cross-platform. Our team uses React Native and Flutter for cross-platform, native Swift/Kotlin when needed. Are you targeting both platforms?"),
            ("Yes, both iOS and Android",
             "Great - cross-platform development is usually the most cost-effective approach for reaching both audiences. What would the app need to do?"),
        ]))
        
        # Healthcare project flow
        examples.append(self.create_multi_turn_example([
            ("Do you have healthcare experience?",
             "Yes, we've built multiple HIPAA-compliant healthcare solutions including patient portals and remote monitoring systems. What area of healthcare are you in?"),
            ("We're a medical clinic looking for patient management software",
             "That's right in our wheelhouse. We've built practice management systems with scheduling, patient records, and billing integration. What's driving your need for new software?"),
            ("Our current system is outdated and hard to use",
             "That's a common situation. We can help modernize while minimizing disruption to your operations. Would you like to discuss how we'd approach this?"),
        ]))
        
        # Pricing discussion flow
        examples.append(self.create_multi_turn_example([
            ("How much would a web application cost?",
             "Web application costs vary widely based on complexity - from $25K for simpler apps to hundreds of thousands for enterprise systems. Can you tell me more about what you're building?"),
            ("A customer portal with account management",
             "Customer portals typically fall in the $50K-$150K range depending on features and integrations. The main cost drivers are complexity of features, third-party integrations, and security requirements. Do you have specific features in mind?"),
            ("User authentication, dashboard, and payment processing",
             "Those are core features we've implemented many times. Payment processing adds some complexity for security and compliance. Would you like to discuss this in more detail with our team?"),
        ]))
        
        # Add more multi-turn flows
        for _ in range(50):
            service = random.choice(list(SERVICES.values()))
            industry = random.choice(list(INDUSTRIES.values()))
            
            examples.append(self.create_multi_turn_example([
                (random.choice(service.questions),
                 f"{service.description} {random.choice(service.follow_ups)}"),
                (f"Yes, for {industry.name.lower()}",
                 f"We have experience in {industry.name.lower()}. {industry.description} What specific challenge are you facing?"),
            ]))
        
        return examples
    
    def generate_cta_examples(self) -> List[Dict]:
        """Generate examples that show natural CTA introduction after engagement."""
        examples = []
        
        # CTA after showing interest
        cta_contexts = [
            ("This sounds like what we need",
             "I'm glad this resonates. The next step would be to discuss your specific requirements in detail. Would you like to schedule a call with our team to explore this further?"),
            ("How do we get started?",
             "Great question! We typically start with a discovery call to understand your needs, then provide a proposal. Would you like to schedule a call to discuss your project?"),
            ("I want to move forward",
             "Excellent! The next step is a detailed discussion with our team to scope out your project. Should I connect you with someone to set that up?"),
            ("Let's do it",
             "Perfect! To get started, we'll need to understand your requirements in more detail. Would you like to schedule a call with our team?"),
        ]
        
        for user_msg, bot_msg in cta_contexts:
            examples.append(self.create_example(user_msg, bot_msg))
        
        return examples
    
    def generate_rejection_examples(self) -> List[Dict]:
        """Generate examples for handling rejections gracefully."""
        examples = []
        
        rejection_responses = {
            "Not interested": "No problem at all. Feel free to come back if anything changes. Is there anything else I can help you with today?",
            "This isn't for us": "Understood - we're not a fit for every situation. If your needs change, we're here. Anything else I can help clarify?",
            "Too expensive": "I understand budget is important. Our pricing varies by approach - there might be options like phased delivery or MVP-first that could work. Would any of those be worth exploring?",
            "We'll think about it": "Of course, take your time. If any questions come up as you're thinking it through, I'm happy to help.",
            "Maybe later": "Sounds good. When you're ready, just come back and we can pick up where we left off.",
            "No thanks": "No problem. Is there anything else I can help with?",
        }
        
        for user_msg, bot_msg in rejection_responses.items():
            examples.append(self.create_example(user_msg, bot_msg))
        
        return examples
    
    def generate_meta_examples(self) -> List[Dict]:
        """Generate examples for meta questions about the bot."""
        examples = []
        
        meta_responses = {
            "Are you a bot?": "Yes, I'm DITSTEK's AI assistant. I'm here to help you learn about our services and answer questions. How can I help?",
            "Who are you?": "I'm an AI assistant for DITSTEK, here to help with questions about our technology services and capabilities.",
            "Are you real?": "I'm an AI assistant - not human, but designed to be genuinely helpful. What would you like to know about DITSTEK?",
            "Can I talk to a human?": "Of course! I can help connect you with our team. Would you like to schedule a call?",
            "I want to speak to someone": "Absolutely - I can help arrange that. Would you like to schedule a call with our team?",
        }
        
        for user_msg, bot_msg in meta_responses.items():
            examples.append(self.create_example(user_msg, bot_msg))
        
        return examples
    
    def generate_off_topic_examples(self) -> List[Dict]:
        """Generate examples for off-topic questions."""
        examples = []
        
        off_topic_responses = {
            "What's the weather?": "I don't have access to weather information - I'm focused on helping with questions about DITSTEK's technology services. What can I help you with there?",
            "Tell me a joke": "I'm better at software than comedy! But I'm happy to help with questions about technology services. What brings you here?",
            "What time is it?": "I don't have access to real-time information. I'm here to help with questions about DITSTEK's services. How can I assist?",
            "What's the capital of France?": "Paris! But I'm actually here to help with technology and software questions. What brings you to DITSTEK?",
        }
        
        for user_msg, bot_msg in off_topic_responses.items():
            examples.append(self.create_example(user_msg, bot_msg))
        
        return examples
    
    def generate_all_examples(self) -> List[Dict]:
        """Generate all training examples."""
        print("Generating training examples...")
        
        all_examples = []
        
        # Generate examples from each category
        generators = [
            ("Service inquiries", self.generate_service_examples),
            ("Industry examples", self.generate_industry_examples),
            ("Technology examples", self.generate_technology_examples),
            ("Case studies", self.generate_case_study_examples),
            ("Pricing", self.generate_pricing_examples),
            ("Timeline", self.generate_timeline_examples),
            ("Engagement models", self.generate_engagement_model_examples),
            ("Trust building", self.generate_trust_examples),
            ("Exploratory", self.generate_exploratory_examples),
            ("Acknowledgments", self.generate_acknowledgment_examples),
            ("Multi-turn", self.generate_multi_turn_examples),
            ("CTA", self.generate_cta_examples),
            ("Rejections", self.generate_rejection_examples),
            ("Meta questions", self.generate_meta_examples),
            ("Off-topic", self.generate_off_topic_examples),
        ]
        
        for name, generator in generators:
            examples = generator()
            print(f"  {name}: {len(examples)} examples")
            all_examples.extend(examples)
        
        # Deduplicate based on user input
        seen_inputs = set()
        unique_examples = []
        for ex in all_examples:
            user_input = ex["messages"][1]["content"].lower().strip()
            if user_input not in seen_inputs:
                seen_inputs.add(user_input)
                unique_examples.append(ex)
        
        print(f"\nTotal unique examples: {len(unique_examples)}")
        
        # If we need more examples, generate variations
        if len(unique_examples) < TARGET_EXAMPLES:
            print(f"\nGenerating {TARGET_EXAMPLES - len(unique_examples)} additional variations...")
            unique_examples.extend(self.generate_variations(unique_examples, TARGET_EXAMPLES - len(unique_examples)))
        
        return unique_examples[:TARGET_EXAMPLES]
    
    def generate_variations(self, existing: List[Dict], count: int) -> List[Dict]:
        """Generate variations of existing examples."""
        variations = []
        
        # Variation templates
        prefixes = ["", "Hi, ", "Hello, ", "Hey, ", "Quick question - ", "I was wondering, "]
        suffixes = ["", "?", " please", " for me"]
        
        for _ in range(count):
            base = random.choice(existing)
            user_msg = base["messages"][1]["content"]
            assistant_msg = base["messages"][2]["content"]
            
            # Add variation
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes) if not user_msg.endswith("?") else ""
            varied_user = prefix + user_msg + suffix
            
            variations.append(self.create_example(varied_user.strip(), assistant_msg))
        
        return variations
    
    def save_examples(self, examples: List[Dict]):
        """Save examples to train and validation files."""
        random.shuffle(examples)
        
        split_idx = int(len(examples) * TRAIN_SPLIT)
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        # Save training set
        with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
            for ex in train_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        
        # Save validation set
        with open(OUTPUT_VALIDATION, "w", encoding="utf-8") as f:
            for ex in val_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        
        print(f"\nSaved {len(train_examples)} training examples to {OUTPUT_TRAIN}")
        print(f"Saved {len(val_examples)} validation examples to {OUTPUT_VALIDATION}")
        
        return train_examples, val_examples


def main():
    """Main entry point."""
    print("=" * 60)
    print("DITSTEK Training Data Generator")
    print("=" * 60)
    
    generator = TrainingDataGenerator()
    examples = generator.generate_all_examples()
    train, val = generator.save_examples(examples)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Total examples: {len(train) + len(val)}")
    print(f"Training: {len(train)}")
    print(f"Validation: {len(val)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
