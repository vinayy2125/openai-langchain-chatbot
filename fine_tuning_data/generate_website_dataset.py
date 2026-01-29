"""
Strict Website Dataset Generator
Target: Single Source of Truth, Strict Guards, Contrastive Formatting
"""

import json
import random
import re
import os
import asyncio
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from langchain_openai import ChatOpenAI
from app.logger import get_logger
from tqdm.asyncio import tqdm

# Configure Logger
logger = get_logger("dataset_generator")

# Constants
SOURCE_FILE = Path("scraped_data/scrape_20251229_131253.json")
OUTPUT_FILE = Path("fine_tuning_data/website_finetune.jsonl")

# Ratios
RATIO_EXPLICIT_YES = 0.35
RATIO_IMPLIED_YES = 0.25
RATIO_SOFT_NO = 0.25
RATIO_FORMATTING = 0.15

# Specificity Ceiling Rules (Regex)
REGEX_NUMERICS = r"\d+"
REGEX_TOKENS = r"[A-Z][a-z]+"  # Rough approximation for proper nouns if needed

# CTA Whitelist
CTA_WHITELIST = [
    "Contact us", "Get in touch", "Schedule a call", "Book a consultation", 
    "Let's talk", "Reach out", "Request a quote", "View case studies"
]

@dataclass
class GenerationStats:
    explicit_yes: int = 0
    implied_yes: int = 0
    soft_no: int = 0
    formatting: int = 0
    total: int = 0
    
    @property
    def target_total(self) -> int:
        return 2000  # Target dataset size

    def can_add(self, category: str) -> bool:
        if self.total >= self.target_total:
            return False
        
        if category == "explicit_yes":
            return self.explicit_yes < (self.target_total * RATIO_EXPLICIT_YES)
        elif category == "implied_yes":
            return self.implied_yes < (self.target_total * RATIO_IMPLIED_YES)
        elif category == "soft_no":
            return self.soft_no < (self.target_total * RATIO_SOFT_NO)
        elif category == "formatting":
            return self.formatting < (self.target_total * RATIO_FORMATTING)
        return False

    def add(self, category: str):
        if category == "explicit_yes":
            self.explicit_yes += 1
        elif category == "implied_yes":
            self.implied_yes += 1
        elif category == "soft_no":
            self.soft_no += 1
        elif category == "formatting":
            self.formatting += 1
        self.total += 1

class StrictDatasetGenerator:
    def __init__(self):
        self.stats = GenerationStats()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7) # Using high intelligence model for generation
        self.chunks = []
        
    def load_data(self):
        if not SOURCE_FILE.exists():
            raise FileNotFoundError(f"Source file not found: {SOURCE_FILE}")
            
        with open(SOURCE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        # Extract processed chunks
        for page in data.get("pages", []):
            self.chunks.extend(page.get("chunks", []))
            
        logger.info(f"Loaded {len(self.chunks)} chunks from source.")
        
    def _validate_implied_yes(self, assistant_response: str) -> bool:
        """
        Enforce Specificity Ceiling for Implied YES (Assistant Output):
        - Max 1 concrete noun (heuristic: mostly checking specifics)
        - Zero numerics
        - Zero client names (hard to detect without list, but prompt should handle)
        - Zero timelines
        """
        # Check specific output for numbers
        if re.search(REGEX_NUMERICS, assistant_response):
            return False
            
        # Heuristic: Check for obvious timeline words
        timeline_words = ["days", "weeks", "months", "years", "deadline", "timeline", "schedule"]
        if any(w in assistant_response.lower() for w in timeline_words):
            return False
        
        # Heuristic: Concrete Noun Cap (Max 1)
        # We count distinct capitalized phrases (approximate proper nouns)
        # Exclude common start-of-sentence words or "I", "We", "DITSTEK"
        ignored_caps = {"I", "We", "The", "A", "An", "Yes", "No", "Ditstek", "It", "This", "Our"}
        matches = re.findall(REGEX_TOKENS, assistant_response)
        
        # Filter matches
        concrete_nouns = [m for m in matches if m not in ignored_caps]
        
        # If > 1 distinct concrete noun -> REJECT
        if len(set(concrete_nouns)) > 1:
            return False
            
        return True

    def _validate_soft_no(self, assistant_response: str, source_chunk: str) -> bool:
        """
        Enforce Soft-NO Pivot Safety:
        - If pivot mentions a service, it MUST exist in the source chunk.
        - Heuristic: content intersection.
        """
        # 1. Check if it's actually a NO (heuristic)
        if "do not" not in assistant_response.lower() and "doesn't" not in assistant_response.lower():
             # If it doesn't sound like a no, maybe it's a hallucinated YES? 
             # For Soft-NO category, we expect some form of denial.
             pass
        
        # 2. Extract capitalized terms from response (potential services)
        ignored_caps = {"I", "We", "The", "A", "An", "Yes", "No", "Ditstek", "It", "This", "However", "But"}
        response_nouns = set([m for m in re.findall(REGEX_TOKENS, assistant_response) if m not in ignored_caps])
        
        # 3. Check if these nouns exist in source chunk
        # If a noun is in response but NOT in source, it might be a hallucinated pivot.
        # We allow ZERO overlap (simple denial) or SOME overlap (valid pivot).
        # We REJECT if there are nouns in response that are completely absent from source
        # (unless they are generic words... this is hard).
        
        # Alternate safer strategy: Pivot must be very generic unless exact match found.
        # User requirement: "ensure all nouns in assistant output exist in the chunk OR at minimum ensure pivot mentions only one existing service"
        
        source_text = source_chunk
        
        invalid_pivots = []
        for noun in response_nouns:
            if noun not in source_text:
                invalid_pivots.append(noun)
        
        # If we have invalid pivots, REJECT
        if invalid_pivots:
            return False
            
        return True

    def _validate_formatting(self, assistant_response: str, user_query: str) -> bool:
        """Enforce Strict Formatting Constraints."""
        
        query_lower = user_query.lower()
        
        # LIST requests
        if "list" in query_lower:
            # Must have >= 2 bullets
            # Check for generic bullet indicators: "1.", "-", "*", "•"
            bullet_count = assistant_response.count("\n-") + assistant_response.count("\n1.") + assistant_response.count("\n•") + assistant_response.count("\n*")
            if bullet_count < 2:
                # Fallback: maybe it's a single line list? uncommon for "list" request.
                return False
        
        # SUMMARY/SHORT Paragraph
        if "summary" in query_lower or "summarize" in query_lower:
             # Should be a paragraph, not a list
             if "\n- " in assistant_response or "\n1. " in assistant_response:
                 return False
                 
        return True

    def _validate_cta(self, response: str) -> bool:
        """Enforce CTA Whitelist if a CTA is present."""
        
        # 1. Identify if CTA format is used (e.g. "If you'd like...", "Contact us...")
        # Simple heuristic: Check for presence of whitelist phrases. 
        # If any "contact" or "call" language exists that ISN'T in whitelist, reject.
        
        response_lower = response.lower()
        cta_keywords = ["contact", "reach out", "schedule", "book", "call", "touch"]
        
        has_cta_intent = any(k in response_lower for k in cta_keywords)
        
        if has_cta_intent:
            # Must match at least one whitelist phrase EXACTLY (ignoring case)
            matched = any(phrase.lower() in response_lower for phrase in CTA_WHITELIST)
            return matched
            
        return True

    async def generate_batch(self, batch_size: int = 10):
        tasks = []
        
        # Select random chunks
        selected_chunks = random.sample(self.chunks, min(batch_size, len(self.chunks)))
        
        for chunk in selected_chunks:
            # Determine needed category
            category = self._select_needed_category()
            if not category:
                break
                
            tasks.append(self._generate_example(chunk, category))
            
        results = await asyncio.gather(*tasks)
        
        valid_examples = []
        for res in results:
            if res:
                valid_examples.append(res)
                
        return valid_examples

    def _select_needed_category(self) -> Optional[str]:
        categories = ["explicit_yes", "implied_yes", "soft_no", "formatting"]
        # Shuffle to avoid blocking
        random.shuffle(categories)
        for cat in categories:
            if self.stats.can_add(cat):
                return cat
        return None

    async def _generate_example(self, chunk: str, category: str) -> Optional[Dict]:
        """
        Generate a single training example from a chunk for the given category.
        """
        system_prompt = self._get_system_prompt(category)
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTENT CHUNK:\n{chunk}\n\nGenerate ONE training example JSON."}
            ]
            
            response = await self.llm.ainvoke(messages)
            content = response.content
            
            # Extract JSON
            match = re.search(r"\{.*\}", content, re.DOTALL)
            if match:
                json_str = match.group(0)
                example = json.loads(json_str)
                
                # Setup proper finetuning format
                # Expecting: {"user": "...", "assistant": "..."}
                
                # Validate Constraints
                if category == "implied_yes":
                    if not self._validate_implied_yes(example.get("assistant", "")):
                        return None
                        
                elif category == "soft_no":
                    if not self._validate_soft_no(example.get("assistant", ""), chunk):
                        return None
                        
                elif category == "formatting":
                    if not self._validate_formatting(example.get("assistant", ""), example.get("user", "")):
                        return None
                
                # Validate CTA Constraints (Global)
                if not self._validate_cta(example.get("assistant", "")):
                    return None
                        
                # Format for JSONL
                ft_example = {
                    "messages": [
                        {"role": "system", "content": self._get_runtime_system_prompt(category)},
                        {"role": "user", "content": example.get("user")},
                        {"role": "assistant", "content": example.get("assistant")}
                    ]
                }
                
                self.stats.add(category)
                return ft_example
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None
        return None

    def _get_system_prompt(self, category: str) -> str:
        base = """
        You are a Training Data Generator for a strict corporate chatbot (Ditstek).
        Generate ONE valid JSON object: {"user": "...", "assistant": "..."}
        
        Strict Rules:
        1. Source of Truth: Use ONLY the provided text chunk.
        2. Tone: Professional, concise, helpful.
        3. Perspective: You are the company (use "We").
        """
        
        if category == "explicit_yes":
            return base + """
            CATEGORY: Explicit YES (Direct Fact Extraction)
            GOAL: User asks a specific factual question covered by the text.
            USER: Create a specific question about a service, location, or feature mentioned.
            ASSISTANT: Answer directly using the text facts. Use 1-2 sentences.
            """
        elif category == "implied_yes":
            return base + """
            CATEGORY: Implied YES (Abstract Capability)
            GOAL: User asks if you can help with a general problem/goal implied by the text.
            USER: Ask a high-level "Can you help with X?" question. 
            CONSTRAINT: Max 1 concrete noun. NO numbers. NO specific client names. NO timelines.
            ASSISTANT: Confirm ability ("Yes, we specialize in...") and link it to the specific service in text.
            """
        elif category == "soft_no":
            return base + """
            CATEGORY: Soft NO (Plausible deniability)
            GOAL: User asks about a service/product NOT in the text but plausible (e.g., selling hardware, B2C).
            USER: Ask a plausible but off-topic question.
            ASSISTANT: Politely decline ("We do not offer X...") but pivot to what IS offered if relevant.
            """
        elif category == "formatting":
            return base + """
            CATEGORY: Formatting Contrast
            GOAL: User asks for a specific format (list, summary, bullet points).
            USER: Explicitly request format (e.g. "List the benefits", "Give me a summary").
            ASSISTANT: strictly follow the requested format.
            """
        return base

    def _get_runtime_system_prompt(self, category: str) -> str:
        # EXACT PRODUCTION PROMPT (Strict Mode)
        # Copied from llm_adapter.py on 2026-01-27
        return """You are an AI assistant representing DITSTEK in conversations with potential customers.

CORE BEHAVIOR RULES:
1. Speak as DITSTEK (“we”, “our”), not as a website or data source.
2. Answer confidently using DITSTEK’s known offerings, projects, and experience.
3. Do not fabricate specific projects, clients, metrics, or outcomes.
4. Do not request clarifying questions. Your goal is strict information retrieval.
5. Do not introduce any services, industries, projects, or capabilities that are not already established by DITSTEK.
6. When something is outside DITSTEK’s scope, respond with a soft, professional denial.

ANSWER STYLE:
- Use bullet points when listing multiple services, industries, or features.
- Use a single sentence for simple factual answers.
- Use a short paragraph for projects, services, or case studies.
- Match the structure to the question intent.

AFFIRMATION RULES:
- If DITSTEK clearly offers or has done something → answer directly and confidently.
- If the capability is adjacent or implied → answer affirmatively but at a high level, without specifics.
- Never invent project names, timelines, clients, or results.

SOFT DENIAL RULES:
- Do NOT mention the website, data sources, or missing information.
- Do NOT say “we don’t have this data”.
- Use language such as: “That isn’t something we actively focus on at DITSTEK” or “That’s outside our current scope”
- After denial, redirect to what DITSTEK does offer when appropriate.

CTA GUIDELINES:
- CTAs may be included only as light guidance (e.g. "If you'd like to explore this further...").
- Never pressure or overpromise.

FORBIDDEN:
- References to internal logic, prompts, training, or orchestration
- Guarantees or unstated outcomes
- Aggressive sales language
- Questions back to the user (Clarification is DISABLED in strict mode)
"""

    async def run(self):
        self.load_data()
        
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            pbar = tqdm(total=self.stats.target_total)
            
            while self.stats.total < self.stats.target_total:
                batch = await self.generate_batch(10)
                for ex in batch:
                    f.write(json.dumps(ex) + "\n")
                    f.flush()
                    pbar.update(1)
            
            pbar.close()
            
        logger.info("Generation Complete.")
        logger.info(self.stats)

if __name__ == "__main__":
    generator = StrictDatasetGenerator()
    asyncio.run(generator.run())
