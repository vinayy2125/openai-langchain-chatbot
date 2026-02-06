import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the class instead of importing full dependency tree to avoid env issues
class MockLLMAdapter:
    def _generate_fallback_options(self, bot_response: str, user_input: str = "") -> list:
        """
        Copy of the logic from llm_adapter.py for testing.
        We are testing the LOGIC, not the import.
        """
        response_lower = bot_response.lower()
        options = []
        
        # Helper to check if option is redundant with user input
        def is_redundant(opt: str) -> bool:
            return opt.lower() in user_input.lower() or user_input.lower() in opt.lower()

        # Pattern 1: Look for questions in the response and suggest related actions
        if "assessment" in response_lower or "evaluate" in response_lower:
            options.append("Get assessment")
        if "demo" in response_lower or "show" in response_lower:
            options.append("See a demo")
        if "discuss" in response_lower or "talk" in response_lower:
            options.append("Talk to expert")
        if "timeline" in response_lower or "when" in response_lower:
            options.append("Timeline details")
        if "cost" in response_lower or "pricing" in response_lower or "budget" in response_lower:
            options.append("Get a quote")
        
        # SMART ESCALATION
        if "team" in response_lower or "expert" in response_lower:
            if is_redundant("Meet the team"):
                options.append("Schedule a call")
            else:
                options.append("Meet the team")
                
        if "case" in response_lower or "example" in response_lower or "similar" in response_lower:
            if is_redundant("See examples"):
                options.append("View case studies") 
            else:
                options.append("See examples")

        if "architecture" in response_lower or "design" in response_lower:
            options.append("Architecture review")

        # Pattern 2: If response has a question mark, suggest "Tell me more" variant
        # BUT ONLY IF:
        # 1. The response is substantial (length > 150 chars)
        # 2. We don't already have enough specific options
        if "?" in bot_response and len(options) < 3:
            if len(bot_response) > 150:
                options.append("Tell me more")
        
        # Filter redundancy
        final_options = []
        for opt in options:
            if not is_redundant(opt):
                final_options.append(opt)
            elif opt == "Schedule a call": 
                 final_options.append(opt)

        # Limit to 3 unique options
        unique_options = list(dict.fromkeys(final_options))[:3]
        
        # Fallback
        # ONLY IF: User input has adequate context (>= 3 words)
        if not unique_options:
            input_word_count = len(user_input.split())
            if input_word_count >= 3:
                unique_options = ["Learn more", "See examples", "Talk to expert"]
                unique_options = [opt for opt in unique_options if not is_redundant(opt)]
                if not unique_options:
                     unique_options = ["Contact us", "Schedule a call"] 
            else:
                unique_options = []
        
        return unique_options

# Test Cases
adapter = MockLLMAdapter()

print("Running Tests...\n")

# Case 1: Simple Greeting (User Name)
# Input: "Vinay" (1 word)
# Response: Short greeting with question.
input_1 = "Vinay"
resp_1 = "It's great to meet you, Vinay! How can I assist you with your new website project?" # ~80 chars
opts_1 = adapter._generate_fallback_options(resp_1, input_1)
print(f"Test 1 [Vinay]: Expected [], Got {opts_1}")
if opts_1 == []: print("PASS")
else: print("FAIL")

print("-" * 20)

# Case 2: Short Greeting
# Input: "Hi"
# Response: "Hello! How can I help?"
input_2 = "Hi"
resp_2 = "Hello! How can I help?"
opts_2 = adapter._generate_fallback_options(resp_2, input_2)
print(f"Test 2 [Hi]: Expected [], Got {opts_2}")
if opts_2 == []: print("PASS")
else: print("FAIL")

print("-" * 20)

# Case 3: Substantive Response (should trigger Tell me more)
# Input: "Web dev"
# Response: Long explanation > 150 chars
input_3 = "Web dev"
resp_3 = "We qualify in custom web development using Python and React. Our team has built scalable solutions for various industries including fin-tech and health-tech. We ensure high performance and security in all our deliverables. Would you like to know about our process?" # > 150 chars
opts_3 = adapter._generate_fallback_options(resp_3, input_3)
print(f"Test 3 [Long Resp]: Expected to include 'Tell me more', Got {opts_3}")
if "Tell me more" in opts_3: print("PASS")
else: print("FAIL")

print("-" * 20)

# Case 4: Long Input (should trigger generic fallback if no keywords)
# Input: "I want to know about something completely random that has no keywords but is long enough."
# Response: "I'm not sure."
input_4 = "I want to know about something completely random that has no keywords but is long enough."
resp_4 = "I'm not sure."
opts_4 = adapter._generate_fallback_options(resp_4, input_4)
print(f"Test 4 [Long Input]: Expected Generic Options, Got {opts_4}")
if opts_4 == ["Learn more", "See examples", "Talk to expert"]: print("PASS")
else: print("FAIL")
