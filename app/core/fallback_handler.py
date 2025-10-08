import re
from typing import Optional, List, Dict, Any

def get_smart_fallback(query: str, context_chunks: Optional[List[str]] = None, conversation_history: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    Generate smart fallback responses when exact information isn't available.
    Uses partial matches and related information to provide helpful responses.
    """
    # Extract key terms for matching
    query_terms = set(re.findall(r'\w+', query.lower()))
    tech_terms = {
        'backend': ['development', 'server', 'api', 'database', 'nodejs', 'python', 'java'],
        'frontend': ['ui', 'interface', 'react', 'angular', 'vue', 'web'],
        'database': ['sql', 'mysql', 'postgresql', 'mongodb', 'nosql', 'data'],
        'ai': ['ml', 'machine learning', 'artificial intelligence', 'gpt', 'model'],
        'healthcare': ['medical', 'health', 'clinical', 'patient', 'hospital']
    }

    # Check for related terms
    matched_categories = []
    for category, terms in tech_terms.items():
        if any(term in query_terms for term in terms):
            matched_categories.append(category)

    if matched_categories:
        # Return category-specific response
        if 'backend' in matched_categories:
            return ("While I don't have specific information about that, I can tell you about our backend development expertise. "
                   "We specialize in Node.js, Python, and Java development with modern frameworks like Express.js and Django.")
        elif 'healthcare' in matched_categories:
            return ("Let me share relevant information about our healthcare development experience. "
                   "We've worked on various healthcare projects involving patient management systems and clinical applications.")
        elif 'ai' in matched_categories:
            return ("While I don't have exact details for that, I can discuss our AI integration capabilities. "
                   "We have experience implementing machine learning models and AI services in various applications.")
    
    # Check context for partial matches if available
    if context_chunks:
        partial_matches = []
        for chunk in context_chunks:
            if any(term in chunk.lower() for term in query_terms):
                partial_matches.append(chunk)
        
        if partial_matches:
            return ("While I don't have the exact information you're looking for, I found some relevant details that might help: "
                   f"{partial_matches[0]}")

    # Default engaging fallback
    return ("I'd be happy to help you with that. Could you tell me more about your specific requirements? "
            "For example, are you interested in our development services, healthcare solutions, or something else?")