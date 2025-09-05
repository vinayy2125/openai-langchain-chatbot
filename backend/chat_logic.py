from dotenv import load_dotenv
import os    
from backend.retriever import retriever
from typing import List, Tuple, Any
from backend.search_client import search_site
from crawler.scraper import scrape_url
from backend.services.chatbot_optimizer import OptimizedChatbot
from backend.llm_client import llm  # Ensure llm is initialized before importing here

load_dotenv()

def _dedupe_chunks(docs) -> List[Tuple[str, dict]]:
    """
    De-duplicate retrieved documents and keep metadata for traceability.

    Args:
        docs: List of documents, each with `page_content` and `metadata`.

    Returns:
        List of tuples: (text, metadata)
    """
    seen = set()
    unique = []
    
    for d in docs:
        # Extract text and metadata
        text = d.page_content.strip() if hasattr(d, "page_content") else str(d)
        metadata = getattr(d, "metadata", {}) if hasattr(d, "metadata") else {}

        # Skip empty or duplicate texts
        if not text or text in seen:
            continue

        seen.add(text)
        unique.append((text, metadata))
    
    return unique


def _maybe_expand_queries(query: str) -> List[str]:
    # Lightweight RAG fusion: expand the query to reduce “same answer” effect
    return list(dict.fromkeys([
        query,
        f"Details about {query}",
        f"In-depth explanation of {query}",
    ]))

# Initialize the optimized chatbot (do this once at application startup)

def build_chatbot_response(query: str, chat_history: list, site: str = "ditstek.com", detailed: bool = False):
    """
    Enhanced chatbot response function with all optimizations.
    """
    optimized_chatbot = OptimizedChatbot(llm, model="gpt-4o-mini")  # Ensure model matches llm

    try:
        response, success = optimized_chatbot.get_detailed_response(
            query=query,
            chat_history=chat_history,
            site=site,
            detailed=detailed  # ✅ fixed comma + pass flag
        )

        if success:
            return response, True
        else:
            return _fallback_to_original(query, chat_history, site)

    except Exception as e:
        print(f"[ERROR] Optimized chatbot failed: {e}")
        return _fallback_to_original(query, chat_history, site)


# ✅ ADD - Fallback function (simplified version of your original logic)
def _fallback_to_original(query: str, chat_history: list, site: str):
    """Fallback to simplified original behavior"""
    try:
        # Simplified version of your original logic
        variant_queries = _maybe_expand_queries(query)
        pooled_docs = []
        for q in variant_queries:
            pooled_docs.extend(retriever.get_relevant_documents(q))
        
        unique_texts = _dedupe_chunks(pooled_docs)
        context_text = "\n\n---\n\n".join(unique_texts[:8])  # Reduced chunks for fallback
        
        if not context_text.strip():
            search_results = search_site(query, site)
            scraped_texts = []
            for res in search_results:
                url = res.get("url")
                title = res.get("title") or url
                if url:
                    text = scrape_url(url)
                    if text:
                        scraped_texts.append(f"[{title}]({url}): {text}")
            context_text = "\n\n".join(scraped_texts[:5])
            if not context_text.strip():
                return (
                    "No relevant content found. Please visit the website directly "
                    f"[{site}](https://{site}).",
                    True
                )
        
        history_text = "\n".join(
            f"{'User' if role == 'user' else 'Assistant'}: {msg}"
            for role, msg in chat_history
        )
        
        # Simple fallback prompt
        fallback_prompt = f"""
You are a helpful assistant.
Conversation: {history_text}
Context: {context_text}
Question: {query}
Please provide a helpful answer.
"""
        
        raw_answer = llm.invoke(fallback_prompt)
        answer = raw_answer.content if hasattr(raw_answer, 'content') else str(raw_answer)
        
        return answer, True if answer.strip() else ("No response generated.", False)
        
    except Exception as e:
        return f"I apologize, but I'm experiencing technical difficulties: {str(e)}", False