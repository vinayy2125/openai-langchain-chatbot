from typing import Optional, List, Dict, Any
import re

def format_response(
    response: str,
    query: str,
    conversation_history: Optional[List[Dict[str, Any]]] = None
) -> str:
    """
    Format responses in a ChatGPT-like style, adapting to query complexity.
    """
    # Check if it's a simple query needing a concise response
    simple_patterns = [
        r'^what is',
        r'^who is',
        r'^is it',
        r'^does',
        r'^can you',
        r'^tell me',
    ]
    
    is_simple_query = any(re.match(pattern, query.lower()) for pattern in simple_patterns)
    
    if is_simple_query and len(response.split()) > 30:
        # Extract first relevant sentence
        sentences = response.split('.')
        for sentence in sentences:
            if len(sentence.split()) >= 5:  # Ensure it's a complete thought
                return sentence.strip() + '.'
    
    # For technical responses, improve formatting
    tech_terms = ['api', 'backend', 'database', 'framework', 'integration', 'server']
    if any(term in query.lower() for term in tech_terms):
        # Convert paragraphs to bullet points if they're not already
        if not response.strip().startswith('- ') and len(response.split()) > 50:
            paragraphs = response.split('\n\n')
            formatted = []
            for para in paragraphs:
                if para.strip():
                    if not para.strip().startswith('- '):
                        formatted.append(f"- {para.strip()}")
                    else:
                        formatted.append(para.strip())
            return '\n'.join(formatted)
    
    # For list-like responses that aren't properly formatted
    list_indicators = ['first', 'second', 'next', 'finally', 'additionally']
    if any(indicator in response.lower() for indicator in list_indicators):
        if not response.strip().startswith('- '):
            sentences = response.split('. ')
            formatted = []
            for sentence in sentences:
                if any(indicator in sentence.lower() for indicator in list_indicators):
                    formatted.append(f"- {sentence.strip()}")
                else:
                    formatted.append(sentence.strip())
            return '\n'.join(formatted)
    
    return response