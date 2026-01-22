"""
Training Data Generator V2 - From Scraped Website Content
Generates 2000+ fine-tuning examples DIRECTLY from scraped DITSTEK website content

This version extracts REAL content from the scraped data including:
- Actual client testimonials and names
- Real project descriptions
- Actual service descriptions
- Real industry information
"""

import json
import random
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION
# =============================================================================

TARGET_EXAMPLES = 50000  # No limit - generate from ALL website content
TRAIN_SPLIT = 0.85

# Paths
SCRAPED_DATA_PATH = Path(__file__).parent.parent / "scraped_data" / "scrape_20251229_131253.json"
OUTPUT_TRAIN = Path(__file__).parent / "train.jsonl"
OUTPUT_VALIDATION = Path(__file__).parent / "validation.jsonl"


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


class ScrapedDataProcessor:
    """Process scraped website data to extract training content."""
    
    def __init__(self):
        self.scraped_data = None
        self.pages = []
        self.chunks = []
        self.services = []
        self.industries = []
        self.testimonials = []
        self.case_studies = []
        self.technologies = []
        self.faqs = []
        self.about_team = []  # NEW: Company meta info (founders, team, about us)
        
    def load_data(self) -> bool:
        """Load and parse scraped data."""
        try:
            with open(SCRAPED_DATA_PATH, "r", encoding="utf-8") as f:
                self.scraped_data = json.load(f)
            self.pages = self.scraped_data.get("pages", [])
            print(f"Loaded {len(self.pages)} pages from scraped data")
            self._extract_all_content()
            return True
        except Exception as e:
            print(f"Error loading scraped data: {e}")
            return False
    
    def _extract_all_content(self):
        """Extract all content types from pages."""
        for page in self.pages:
            chunks = page.get("chunks", [])
            url = page.get("url", "")
            title = page.get("title", "")
            
            for chunk in chunks:
                self.chunks.append({
                    "content": chunk,
                    "url": url,
                    "title": title
                })
                
                # Categorize chunks
                chunk_lower = chunk.lower()
                
                # Testimonials (contain client names)
                if any(name in chunk for name in ["Dave Armstrong", "Tom Rupsis", "Lionel Hagege", 
                                                   "Daniel Gochin", "Erin McCutcheon", "Kingman Ho"]):
                    self.testimonials.append(chunk)
                
                # Services
                if any(s in chunk_lower for s in ["development", "services", "we offer", "we provide", "we build"]):
                    self.services.append(chunk)
                
                # Industries
                if any(i in chunk_lower for i in ["healthcare", "fintech", "real estate", "edtech", 
                                                   "retail", "logistics", "iot", "mining"]):
                    self.industries.append(chunk)
                
                # Technologies
                if any(t in chunk_lower for t in ["react", ".net", "node", "python", "angular", 
                                                   "aws", "azure", "flutter"]):
                    self.technologies.append(chunk)
                
                # Case studies / portfolio
                if "case stud" in chunk_lower or "portfolio" in chunk_lower:
                    self.case_studies.append(chunk)
                
                # FAQs
                if "?" in chunk and len(chunk) < 500:
                    self.faqs.append(chunk)
                
                # NEW: About/Team/Leadership
                if any(t in chunk_lower for t in ["founder", "co-founder", "ceo", "cto", "leadership", 
                                                   "team", "about us", "our story", "our mission",
                                                   "our vision", "headquarter", "office", "years of experience",
                                                   "established", "founded in", "saarthak", "mohali"]):
                    self.about_team.append(chunk)
        
        print(f"Extracted:")
        print(f"  - {len(self.chunks)} total chunks")
        print(f"  - {len(self.services)} service chunks")
        print(f"  - {len(self.industries)} industry chunks")
        print(f"  - {len(self.technologies)} technology chunks")
        print(f"  - {len(self.testimonials)} testimonial chunks")
        print(f"  - {len(self.case_studies)} case study chunks")
        print(f"  - {len(self.faqs)} FAQ chunks")
        print(f"  - {len(self.about_team)} about/team chunks")


class TrainingDataGenerator:
    """Generate training examples from scraped content."""
    
    def __init__(self, processor: ScrapedDataProcessor):
        self.processor = processor
        self.examples = []
        
        # Expanded question templates for different content types
        self.service_questions = [
            "What is {topic}?",
            "Tell me about {topic}",
            "Do you offer {topic}?",
            "Can you help with {topic}?",
            "What {topic} services do you provide?",
            "How does {topic} work at DITSTEK?",
            "I need help with {topic}",
            "Looking for {topic} services",
            "Can you do {topic}?",
            "What's your approach to {topic}?",
            "How do you handle {topic}?",
            "Interested in {topic}",
            "Need {topic} for my business",
            "Tell me more about your {topic} capabilities",
            "What experience do you have with {topic}?",
        ]
        
        self.industry_questions = [
            "Do you have {industry} experience?",
            "What {industry} solutions do you offer?",
            "Can you build {industry} software?",
            "Tell me about your {industry} work",
            "Have you worked in the {industry} space?",
            "Looking for {industry} development",
            "Need a {industry} application",
            "Can you help with {industry} projects?",
            "What {industry} projects have you done?",
            "Do you understand {industry} requirements?",
            "Any {industry} case studies?",
            "How do you approach {industry} software?",
        ]
        
        self.followup_questions = [
            "What specific challenge are you trying to solve?",
            "What's driving this initiative?",
            "Are you starting from scratch or modernizing an existing system?",
            "What's your timeline for this project?",
            "What business problem are you looking to address?",
            "Is this for internal use or customer-facing?",
            "Do you have existing systems this needs to integrate with?",
            "What scale are you expecting?",
            "What's the primary goal you're trying to achieve?",
            "Who are the main users of this system?",
            "What's your budget range for this project?",
            "Do you have specific requirements already defined?",
            "What's most important - speed, quality, or cost?",
            "Are there any compliance requirements we should know about?",
            "What does success look like for this project?",
        ]
        
        # Question starters for variations
        self.question_starters = [
            "", "Hi, ", "Hello, ", "Hey, ", "Quick question - ", 
            "I was wondering, ", "Can you tell me ", "I'd like to know ",
            "Could you explain ", "Please tell me about ", "I'm curious about ",
            "We're interested in ", "Our company needs ", "I'm looking for info on ",
        ]
    
    def create_example(self, user_input: str, assistant_response: str) -> Dict:
        """Create a training example."""
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_response}
            ]
        }
    
    def create_multi_turn(self, turns: List[Tuple[str, str]]) -> Dict:
        """Create a multi-turn conversation example."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user, assistant in turns:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        return {"messages": messages}
    
    def extract_section_content(self, chunk: str) -> Tuple[str, str]:
        """Extract section title and content from a chunk."""
        # Parse the chunk format: [Page: ...] [Section: ...] ## Title \n\n Content
        lines = chunk.split("\n")
        title = ""
        content = ""
        
        for i, line in enumerate(lines):
            if line.startswith("## "):
                title = line[3:].strip()
                # Content is everything after the title
                content = "\n".join(lines[i+1:]).strip()
                break
        
        return title, content
    
    def clean_content(self, text: str) -> str:
        """Clean up content for training."""
        # Remove markdown headers
        text = re.sub(r'^##\s+', '', text, flags=re.MULTILINE)
        # Remove [Page:...] [Section:...] metadata
        text = re.sub(r'\[Page:.*?\]', '', text)
        text = re.sub(r'\[Type:.*?\]', '', text)
        text = re.sub(r'\[Section:.*?\]', '', text)
        text = re.sub(r'\[Source:.*?\]', '', text)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def generate_from_services(self) -> List[Dict]:
        """Generate examples from service chunks."""
        examples = []
        
        # Extract unique service topics
        service_topics = {
            "custom software development": [],
            "web development": [],
            "mobile app development": [],
            "cloud services": [],
            "SaaS development": [],
            "MVP development": [],
            "legacy modernization": [],
            "API development": [],
            "full-stack development": [],
            "QA testing": [],
            "AI development": [],
            "dedicated team": [],
            "DevOps": [],
        }
        
        # Categorize ALL service chunks - no limit
        for chunk in self.processor.services:  # Use ALL chunks
            chunk_lower = chunk.lower()
            for topic in service_topics:
                if topic.lower() in chunk_lower or topic.replace("-", " ") in chunk_lower:
                    service_topics[topic].append(chunk)
        
        # Generate Q&A pairs for each service
        for topic, chunks in service_topics.items():
            if not chunks:
                continue
                
            for chunk in chunks[:100]:  # Max 100 per topic for variety
                title, content = self.extract_section_content(chunk)
                if not content or len(content) < 50:
                    continue
                
                clean = self.clean_content(content)[:400]  # Limit length
                
                # Generate questions with multiple templates
                for q_template in random.sample(self.service_questions, min(5, len(self.service_questions))):
                    question = q_template.format(topic=topic)
                    
                    # Add random prefix for variation
                    if random.random() < 0.3:
                        question = random.choice(self.question_starters) + question
                    
                    # Response with optional follow-up
                    response = clean
                    if random.random() < 0.6:  # 60% get follow-up
                        followup = random.choice(self.followup_questions)
                        response = f"{clean} {followup}"
                    
                    examples.append(self.create_example(question, response))
        
        print(f"  Generated {len(examples)} service examples")
        return examples
    
    def generate_from_industries(self) -> List[Dict]:
        """Generate examples from industry chunks."""
        examples = []
        
        industry_topics = {
            "healthcare": [],
            "fintech": [],
            "real estate": [],
            "edtech": [],
            "retail": [],
            "logistics": [],
            "IoT": [],
            "mining": [],
            "agriculture": [],
            "automotive": [],
            "insurance": [],
        }
        
        for chunk in self.processor.industries:  # Use ALL industry chunks
            chunk_lower = chunk.lower()
            for industry in industry_topics:
                if industry.lower() in chunk_lower:
                    industry_topics[industry].append(chunk)
        
        for industry, chunks in industry_topics.items():
            if not chunks:
                continue
            
            for chunk in chunks[:100]:  # Max 100 per industry
                title, content = self.extract_section_content(chunk)
                if not content or len(content) < 50:
                    continue
                
                clean = self.clean_content(content)[:400]
                
                for q_template in random.sample(self.industry_questions, min(5, len(self.industry_questions))):
                    question = q_template.format(industry=industry)
                    
                    # Add random prefix
                    if random.random() < 0.3:
                        question = random.choice(self.question_starters) + question
                    
                    response = f"Yes, we have significant {industry} experience. {clean}"
                    if random.random() < 0.5:
                        followup = random.choice(self.followup_questions)
                        response = f"{response} {followup}"
                    
                    examples.append(self.create_example(question, response))
        
        print(f"  Generated {len(examples)} industry examples")
        return examples
    
    def generate_from_technologies(self) -> List[Dict]:
        """Generate examples from technology mentions."""
        examples = []
        
        tech_patterns = {
            "React": "React JS is one of our core frontend technologies. We've delivered numerous production applications using React.",
            ".NET": "We have strong .NET expertise across the stack including ASP.NET Core and related Microsoft technologies.",
            "Node": "Node.js is part of our core backend stack for APIs and real-time applications.",
            "Python": "We use Python for backend development, data processing, and AI/ML projects.",
            "Angular": "Angular is one of our frontend framework options for enterprise applications.",
            "AWS": "We're experienced with AWS services and help with cloud architecture and migration.",
            "Azure": "We work with Microsoft Azure, particularly for enterprises using the Microsoft ecosystem.",
            "Flutter": "We use Flutter for cross-platform mobile development with custom UI.",
            "Vue": "Vue.js is another frontend framework we use for web applications.",
            "PHP": "We have PHP expertise, particularly with Laravel for web applications.",
            "Laravel": "Laravel is our primary PHP framework for backend development.",
            "MongoDB": "We use MongoDB for applications requiring flexible document-based data storage.",
            "PostgreSQL": "PostgreSQL is our preferred relational database for production applications.",
        }
        
        for tech, description in tech_patterns.items():
            questions = [
                f"Do you work with {tech}?",
                f"Do you have {tech} experience?",
                f"Can you build with {tech}?",
            ]
            
            for q in questions:
                response = description
                if random.random() < 0.3:  # Less frequent follow-ups for tech questions
                    response += " What are you looking to build?"
                examples.append(self.create_example(q, response))
        
        print(f"  Generated {len(examples)} technology examples")
        return examples
    
    def generate_from_chunks(self) -> List[Dict]:
        """Generate examples from general content chunks."""
        examples = []
        
        # Use ALL chunks for general Q&A - no sampling limit
        all_chunks = self.processor.chunks  # Use everything
        
        for chunk_data in all_chunks:
            chunk = chunk_data["content"]
            title, content = self.extract_section_content(chunk)
            
            if not title or not content or len(content) < 100:
                continue
            
            clean = self.clean_content(content)[:350]
            
            # Create a question from the section title
            title_clean = re.sub(r'^[\d.]+', '', title).strip()
            if len(title_clean) < 10:
                continue
            
            questions = [
                f"What about {title_clean.lower()}?",
                f"Tell me about {title_clean.lower()}",
                f"Can you explain {title_clean.lower()}?",
            ]
            
            question = random.choice(questions)
            response = clean
            
            if random.random() < 0.4:
                response += f" {random.choice(self.followup_questions)}"
            
            examples.append(self.create_example(question, response))
        
        print(f"  Generated {len(examples)} general chunk examples")
        return examples
    
    def generate_exploratory(self) -> List[Dict]:
        """Generate exploratory conversation examples."""
        examples = []
        
        exploratory = [
            ("What services do you offer?",
             "We offer end-to-end software development services including custom software, web and mobile apps, cloud solutions, and AI development. We serve industries like healthcare, fintech, and real estate. What brings you here today?"),
            ("Tell me about DITSTEK",
             "DITSTEK Innovations is an offshore software development company with 8+ years of experience. We've helped clients across USA, UK, Canada, Australia, and Middle East build scalable software solutions. We specialize in custom development, team extension, and digital transformation. What would you like to know more about?"),
            ("What can you help with?",
             "We help companies build and improve software products - from web platforms and mobile apps to AI solutions and cloud infrastructure. We work with startups launching MVPs to enterprises modernizing legacy systems. What challenge are you facing?"),
            ("I'm looking for a development partner",
             "We work as long-term technology partners for many clients, providing dedicated teams or project-based engagement. We've served clients across multiple industries with custom software solutions. What kind of project or situation are you looking at?"),
            ("Why should I choose DITSTEK?",
             "We combine offshore cost efficiency with a consultative approach - we help you figure out what's best for your situation, not just execute tasks. We have 8+ years of experience across healthcare, fintech, real estate and more. What matters most to you in a partner?"),
            ("How do you work?",
             "We start with understanding your needs, then propose the right team and engagement model. We use agile methodology with regular communication and demos. Our teams integrate with your processes and work in your timezone overlap. Would you like details on any specific aspect?"),
            ("What makes you different?",
             "We focus on being consultative partners, not just a code factory. We bring industry-specific expertise, clear communication, and accountability for results. We've maintained long-term relationships with clients across multiple continents. What's most important for your project?"),
        ]
        
        for q, a in exploratory:
            examples.append(self.create_example(q, a))
            # Add variations
            if random.random() < 0.5:
                examples.append(self.create_example(f"Hi, {q.lower()}", a))
        
        print(f"  Generated {len(examples)} exploratory examples")
        return examples
    
    def generate_pricing_timeline(self) -> List[Dict]:
        """Generate pricing and timeline examples."""
        examples = []
        
        qa_pairs = [
            ("How much does it cost?",
             "Pricing depends on project scope and complexity. We offer different engagement models - dedicated teams, fixed-price projects, and hourly arrangements. I can help you understand what's involved, then you could discuss specifics with our team. What kind of project are you considering?"),
            ("What are your rates?",
             "Rates vary based on the engagement model and skills required. Offshore development typically offers 40-60% cost savings compared to onshore. We can tailor the engagement to your budget and needs. What type of project are you thinking about?"),
            ("How much for a mobile app?",
             "Mobile app costs vary significantly based on complexity - from $25K for simpler apps to $500K+ for complex platforms. Key factors include features, integrations, and platforms targeted. What kind of app are you thinking about?"),
            ("What's the cost of an MVP?",
             "MVPs typically range from $15K to $100K+ depending on complexity. The goal is to validate quickly with minimal investment. How developed is your concept?"),
            ("How long does a project take?",
             "Timelines vary by scope. A typical MVP takes 8-16 weeks, while larger projects span several months. We focus on iterative delivery so you see progress quickly. What kind of project are you planning?"),
            ("How fast can you start?",
             "We can typically start within 1-2 weeks after finalizing requirements. For urgent needs, we can expedite. What's your timeline looking like?"),
            ("What's your development process?",
             "We follow agile methodology with 2-week sprints, regular demos, and continuous feedback. We adapt to client preferences and can align with your existing processes. Do you have specific process requirements?"),
        ]
        
        for q, a in qa_pairs:
            examples.append(self.create_example(q, a))
        
        print(f"  Generated {len(examples)} pricing/timeline examples")
        return examples
    
    def generate_engagement_models(self) -> List[Dict]:
        """Generate engagement model examples."""
        examples = []
        
        models = [
            ("dedicated team",
             "Our dedicated team model provides a team that works exclusively on your project. They integrate with your processes and function as an extension of your in-house team. It includes super control, guaranteed security, and flexible scaling. Are you extending an existing team or starting fresh?"),
            ("fixed price",
             "Fixed price model works when you have well-defined scope. We agree on requirements and price upfront, giving you budget certainty. It's ideal for specific features or MVPs with clear specifications. Do you have requirements documented?"),
            ("hourly",
             "Hourly engagement provides maximum flexibility for evolving scope or varied tasks. You pay only for time used with transparent monthly billing. It's great for ongoing support or when requirements may change. What kind of work do you need help with?"),
        ]
        
        for model, description in models:
            questions = [
                f"What is the {model} model?",
                f"Tell me about {model}",
                f"How does {model} work?",
            ]
            for q in questions:
                examples.append(self.create_example(q, description))
        
        # Comparison question
        examples.append(self.create_example(
            "What engagement models do you offer?",
            "We offer three main models: dedicated teams for ongoing projects where you need full control, fixed-price for well-defined scope with budget certainty, and hourly for flexible engagements. Each has different benefits depending on your situation. What's your project like?"
        ))
        
        print(f"  Generated {len(examples)} engagement model examples")
        return examples
    
    def generate_trust_building(self) -> List[Dict]:
        """Generate trust-building examples."""
        examples = []
        
        trust_qa = [
            ("Can you show me your work?",
             "Yes, we have case studies and portfolio across healthcare, fintech, real estate, and other industries. We've successfully served clients across USA, UK, Canada, Australia, and Middle East. What industry or type of project is most relevant for you?"),
            ("Do you have case studies?",
             "Yes, we have documented case studies across multiple industries. We've helped startups launch MVPs and enterprises modernize legacy systems. What type of project are you considering?"),
            ("What's your track record?",
             "We have 8+ years of experience with clients across multiple continents. Most of our work comes from referrals and repeat clients, which speaks to our reliability. What industry are you in?"),
            ("How do you protect our IP?",
             "We sign NDAs, ensure code ownership transfers to you, and follow strict data handling policies. IP protection and data security are standard in all our engagements."),
            ("Do you sign NDAs?",
             "Yes, signing NDAs is standard practice before any detailed discussions. We take confidentiality seriously."),
            ("Can I talk to your past clients?",
             "Absolutely. We can arrange reference calls with clients in similar industries or project types. What kind of reference would be most helpful?"),
        ]
        
        for q, a in trust_qa:
            examples.append(self.create_example(q, a))
        
        print(f"  Generated {len(examples)} trust-building examples")
        return examples
    
    def generate_acknowledgments(self) -> List[Dict]:
        """Generate acknowledgment handling examples."""
        examples = []
        
        acks = ["okay", "ok", "sure", "yes", "yeah", "alright", "sounds good",
                "great", "perfect", "thanks", "thank you", "got it", "I see",
                "interesting", "makes sense", "understood", "cool", "nice"]
        
        responses = [
            "Is there anything specific you'd like to know more about?",
            "What else would you like to explore?",
            "Shall I continue, or is there something specific on your mind?",
            "What questions do you have?",
            "Anything else I can help clarify?",
        ]
        
        for ack in acks:
            response = random.choice(responses)
            examples.append(self.create_example(ack, response))
        
        print(f"  Generated {len(examples)} acknowledgment examples")
        return examples
    
    def generate_cta_examples(self) -> List[Dict]:
        """Generate CTA handling examples."""
        examples = []
        
        cta_pairs = [
            ("This sounds like what we need",
             "I'm glad this resonates. The next step would be to discuss your specific requirements in detail. Would you like to schedule a call with our team to explore this further?"),
            ("How do we get started?",
             "Great question! We typically start with a discovery call to understand your needs, then provide a proposal. Would you like to schedule a call to discuss your project?"),
            ("I want to move forward",
             "Excellent! The next step is a detailed discussion with our team to scope out your project. Should I connect you with someone to set that up?"),
            ("Let's do it",
             "Perfect! To get started, we'll need to understand your requirements in more detail. Would you like to schedule a call with our team?"),
            ("I'm interested",
             "Great to hear! I can help connect you with our team for a more detailed discussion. Would you like to schedule a call?"),
        ]
        
        for q, a in cta_pairs:
            examples.append(self.create_example(q, a))
        
        print(f"  Generated {len(examples)} CTA examples")
        return examples
    
    def generate_rejections(self) -> List[Dict]:
        """Generate rejection handling examples."""
        examples = []
        
        rejection_pairs = [
            ("Not interested",
             "No problem at all. Feel free to come back if anything changes. Is there anything else I can help you with today?"),
            ("This isn't for us",
             "Understood - we're not a fit for every situation. If your needs change, we're here. Anything else I can help clarify?"),
            ("Too expensive",
             "I understand budget is important. Our pricing varies by approach - there might be options like phased delivery or MVP-first that could work. Would any of those be worth exploring?"),
            ("We'll think about it",
             "Of course, take your time. If any questions come up as you're thinking it through, I'm happy to help."),
            ("Maybe later",
             "Sounds good. When you're ready, just come back and we can pick up where we left off."),
            ("No thanks",
             "No problem. Is there anything else I can help with?"),
        ]
        
        for q, a in rejection_pairs:
            examples.append(self.create_example(q, a))
        
        print(f"  Generated {len(examples)} rejection examples")
        return examples
    
    def generate_meta_examples(self) -> List[Dict]:
        """Generate meta questions about the bot."""
        examples = []
        
        meta_pairs = [
            ("Are you a bot?",
             "Yes, I'm DITSTEK's AI assistant. I'm here to help you learn about our services and answer questions. How can I help?"),
            ("Who are you?",
             "I'm an AI assistant for DITSTEK, here to help with questions about our technology services and capabilities."),
            ("Are you real?",
             "I'm an AI assistant - not human, but designed to be genuinely helpful. What would you like to know about DITSTEK?"),
            ("Can I talk to a human?",
             "Of course! I can help connect you with our team. Would you like to schedule a call?"),
            ("I want to speak to someone",
             "Absolutely - I can help arrange that. Would you like to schedule a call with our team?"),
        ]
        
        for q, a in meta_pairs:
            examples.append(self.create_example(q, a))
        
        print(f"  Generated {len(examples)} meta examples")
        return examples
    
    def generate_off_topic(self) -> List[Dict]:
        """Generate off-topic handling examples."""
        examples = []
        
        off_topic_pairs = [
            ("What's the weather?",
             "I don't have access to weather information - I'm focused on helping with questions about DITSTEK's technology services. What can I help you with there?"),
            ("Tell me a joke",
             "I'm better at software than comedy! But I'm happy to help with questions about technology services. What brings you here?"),
            ("What time is it?",
             "I don't have access to real-time information. I'm here to help with questions about DITSTEK's services. How can I assist?"),
        ]
        
        for q, a in off_topic_pairs:
            examples.append(self.create_example(q, a))
        
        print(f"  Generated {len(examples)} off-topic examples")
        return examples
    
    def generate_multi_turn(self) -> List[Dict]:
        """Generate multi-turn conversation examples."""
        examples = []
        
        # Service exploration flow
        examples.append(self.create_multi_turn([
            ("What kind of development do you do?",
             "We specialize in custom software development across web, mobile, cloud, and AI. We serve industries like healthcare, fintech, and real estate. What type of project are you considering?"),
            ("We need a mobile app",
             "Great - we build mobile apps for iOS, Android, and cross-platform using React Native and Flutter. Are you targeting both platforms?"),
            ("Yes, both iOS and Android",
             "Cross-platform development is usually the most cost-effective approach for reaching both audiences. What would the app need to do?"),
        ]))
        
        # Healthcare exploration
        examples.append(self.create_multi_turn([
            ("Do you have healthcare experience?",
             "Yes, we've built multiple HIPAA-compliant healthcare solutions including patient portals and remote patient monitoring systems. We're proficient in HL7 integration and understand healthcare compliance. What area of healthcare are you in?"),
            ("We're a medical clinic looking for patient management software",
             "That's right in our wheelhouse. We've built practice management systems with scheduling, patient records, and billing integration. What's driving your need for new software?"),
            ("Our current system is outdated and hard to use",
             "That's a common situation. We specialize in legacy modernization - upgrading systems while minimizing disruption to operations. Would you like to discuss how we'd approach this?"),
        ]))
        
        # Pricing discussion
        examples.append(self.create_multi_turn([
            ("How much would a web application cost?",
             "Web application costs vary by complexity - from $25K for simpler apps to hundreds of thousands for enterprise systems. Can you tell me more about what you're building?"),
            ("A customer portal with account management",
             "Customer portals typically fall in the $50K-$150K range depending on features. Main cost drivers are feature complexity, integrations, and security requirements. Do you have specific features in mind?"),
            ("User authentication, dashboard, and payment processing",
             "Those are core features we've implemented many times. Payment processing adds complexity for security and compliance. Would you like to discuss this in more detail with our team?"),
        ]))
        
        # Technology discussion
        examples.append(self.create_multi_turn([
            ("What technologies do you use?",
             "We work with modern stacks including React, Angular, Vue for frontend; .NET, Node.js, Python for backend; and AWS, Azure for cloud. We choose based on project needs. What are you building?"),
            ("A SaaS platform",
             "For SaaS, we typically use React or Angular for the frontend, .NET or Node.js for backend, with PostgreSQL for data and AWS or Azure for hosting. We build with multi-tenancy and scalability in mind. Are you at the idea stage or already have something?"),
            ("We have an MVP but need to scale",
             "That's a common transition point. We can help assess your current architecture and plan the scaling path - whether that's optimizing the existing codebase or re-architecting specific components. What's the main bottleneck you're facing?"),
        ]))
        
        print(f"  Generated {len(examples)} multi-turn examples")
        return examples
    
    def generate_variations(self, examples: List[Dict], target_count: int) -> List[Dict]:
        """Generate variations to reach target count."""
        variations = []
        variation_count = target_count - len(examples)
        
        if variation_count <= 0:
            return variations
        
        prefixes = ["", "Hi, ", "Hello, ", "Hey, ", "Quick question - ", "I was wondering, "]
        suffixes = ["", " please", " for me"]
        
        for _ in range(variation_count):
            base = random.choice(examples)
            if len(base["messages"]) != 3:  # Skip multi-turn for variations
                continue
                
            user_msg = base["messages"][1]["content"]
            assistant_msg = base["messages"][2]["content"]
            
            prefix = random.choice(prefixes)
            suffix = random.choice(suffixes) if not user_msg.endswith("?") else ""
            varied_user = prefix + user_msg + suffix
            
            variations.append(self.create_example(varied_user.strip(), assistant_msg))
        
        print(f"  Generated {len(variations)} variations")
        return variations
    def generate_from_about_team(self) -> List[Dict]:
        """Generate examples from about/team/leadership content."""
        examples = []
        
        # Question templates for company meta info
        about_questions = [
            "Who founded DITSTEK?",
            "Who are the co-founders?",
            "Tell me about DITSTEK's leadership",
            "Who runs DITSTEK?",
            "Where is DITSTEK located?",
            "Where is your office?",
            "Tell me about your team",
            "What's DITSTEK's story?",
            "How long has DITSTEK been in business?",
            "What's DITSTEK's mission?",
            "Tell me about your company",
            "Who is on your leadership team?",
            "Where is DITSTEK headquartered?",
            "What is DITSTEK's vision?",
        ]
        
        for chunk in self.processor.about_team:
            title, content = self.extract_section_content(chunk)
            if not content or len(content) < 50:
                continue
            
            clean = self.clean_content(content)[:400]
            
            # Generate Q&A pairs
            for q in random.sample(about_questions, min(3, len(about_questions))):
                # Add random prefix for variation
                if random.random() < 0.3:
                    q = random.choice(self.question_starters) + q
                
                response = clean
                # Less frequent follow-ups for factual company questions
                if random.random() < 0.2:
                    response += f" {random.choice(self.followup_questions)}"
                
                examples.append(self.create_example(q, response))
        
        print(f"  Generated {len(examples)} about/team examples")
        return examples
    
    def generate_all(self) -> List[Dict]:
        """Generate all training examples."""
        print("\nGenerating training examples from scraped content...")
        
        all_examples = []
        
        # Generate from different sources
        all_examples.extend(self.generate_from_services())
        all_examples.extend(self.generate_from_industries())
        all_examples.extend(self.generate_from_technologies())
        all_examples.extend(self.generate_from_chunks())
        all_examples.extend(self.generate_from_about_team())  # NEW
        all_examples.extend(self.generate_exploratory())
        all_examples.extend(self.generate_pricing_timeline())
        all_examples.extend(self.generate_engagement_models())
        all_examples.extend(self.generate_trust_building())
        all_examples.extend(self.generate_acknowledgments())
        all_examples.extend(self.generate_cta_examples())
        all_examples.extend(self.generate_rejections())
        all_examples.extend(self.generate_meta_examples())
        all_examples.extend(self.generate_off_topic())
        all_examples.extend(self.generate_multi_turn())
        
        # Deduplicate using FULL signature (system + user + assistant)
        def get_signature(ex):
            msgs = ex["messages"]
            return f"{msgs[0]['content']}|{msgs[1]['content']}|{msgs[2]['content']}"
        
        seen = set()
        unique = []
        for ex in all_examples:
            sig = get_signature(ex)
            if sig not in seen:
                seen.add(sig)
                unique.append(ex)
        
        print(f"\nUnique examples (pre-variation): {len(unique)}")
        
        # Add variations if needed
        if len(unique) < TARGET_EXAMPLES:
            variations = self.generate_variations(unique, TARGET_EXAMPLES)
            # Deduplicate variations too
            for ex in variations:
                sig = get_signature(ex)
                if sig not in seen:
                    seen.add(sig)
                    unique.append(ex)
        
        print(f"Unique examples (post-variation): {len(unique)}")
        
        return unique[:TARGET_EXAMPLES]
    
    def save(self, examples: List[Dict]):
        """Save examples to files."""
        random.shuffle(examples)
        
        split_idx = int(len(examples) * TRAIN_SPLIT)
        train = examples[:split_idx]
        validation = examples[split_idx:]
        
        with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
            for ex in train:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        
        with open(OUTPUT_VALIDATION, "w", encoding="utf-8") as f:
            for ex in validation:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        
        print(f"\nSaved {len(train)} training examples to {OUTPUT_TRAIN}")
        print(f"Saved {len(validation)} validation examples to {OUTPUT_VALIDATION}")
        
        return train, validation


def main():
    print("=" * 60)
    print("DITSTEK Training Data Generator V2")
    print("Generating from SCRAPED WEBSITE CONTENT")
    print("=" * 60)
    
    # Load and process scraped data
    processor = ScrapedDataProcessor()
    if not processor.load_data():
        print("Failed to load scraped data. Exiting.")
        return
    
    # Generate training data
    generator = TrainingDataGenerator(processor)
    examples = generator.generate_all()
    train, val = generator.save(examples)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print(f"Total examples: {len(train) + len(val)}")
    print(f"Training: {len(train)}")
    print(f"Validation: {len(val)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
